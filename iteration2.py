"""Итерация 2: исправленные данные, смесь для счётных данных, динамика, кластеры как признаки.

Шаги:
  1. Диагностика старого пайплайна (утечка таргета, дубли, коды оценок).
  2. Кластеризация: K-means, GMM (EM, гауссовская смесь) и CountMixture (EM, мультиномиально-
     пуассоновская смесь) для k = 2..8 — внутренние метрики, устойчивость, связь с отчислением.
  3. Прогноз (набор 2023 г.): признаки за семестры 1..H, исходы — после H. Цели: отчисление/академ,
     долги в следующих семестрах, «красный диплом». Наборы признаков: оценки, +динамика,
     +вероятности кластеров (обучаются внутри фолда), +анкетные. Модели: логистическая регрессия,
     градиентный бустинг. Валидация — GroupKFold по учебным группам.
  4. Выгрузка агрегатов и параметров моделей в reports/iteration2/results.json и сборка
     интерактивного отчёта reports/iteration2/index.html (шаблон report_template.html).

Запуск:
    python features_v2.py      # один раз: data/panel_v2.parquet
    python iteration2.py       # ~5–10 минут
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, average_precision_score, brier_score_loss,
                             normalized_mutual_info_score, precision_recall_curve,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from count_mixture import CountMixture
from features_v2 import (BASE, DEMOGRAPHICS, DYNAMICS, TARGETS, clustering_dataset,
                         prediction_dataset)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
KS = range(2, 9)
GMM_FEATURES = ["share_5", "share_3", "log_problems"]
GMM_REG = 0.05
CLUSTER_K = 5                    # k для признаков-кластеров: минимум ICL у CountMixture
N_FOLDS = 5
THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)
PROFILE_COLS = ["gpa", "share_5", "share_4", "share_3", "failed", "debts", "attrition",
                "al", "expelled", "paid", "master"]


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def counts(df):
    return df[["n5", "n4", "n3"]].values, df[["failed", "debts"]].values, df["exposure"].values


# ── 1. диагностика старого пайплайна ──────────────────────────────────────────
def diagnose_old(cohort_path: Path, diplom_path: Path) -> dict:
    raw = pd.read_parquet(cohort_path)
    st = raw.drop_duplicates("student_uuid")
    out = {
        "rows": len(raw), "students": len(st),
        "dup_factor": len(raw) / len(st),
        "auc_n3_only": roc_auc_score(st["target"], -st["n_3"]),
        "auc_n3_n4_rule": roc_auc_score(st["target"], ((st["n_3"] < 5) & (st["n_4"] < 15)).astype(int)),
        "rule_agreement": float((((st["n_3"] < 5) & (st["n_4"] < 15)).astype(int) == st["target"]).mean()),
        "target_rate": float(st["target"].mean()),
        "al_in_target": int(((st["final_state"].str.replace("\xa0", " ") == "в академическом отпуске")
                             & (st["target"] == 1)).sum()),
        "gpa_min": float(st["gpa_1_4"].min()),
    }
    d = pd.read_csv(diplom_path, usecols=["mark", "test_type", "term", "student_uuid"], low_memory=False)
    d = d[d["student_uuid"].isin(st["student_uuid"]) & d["term"].between(1, 4) & (d["mark"] == 3)]
    out["code3_credit_share"] = float((d["test_type"] == 2).mean())
    return out


# ── 2. кластеризация ──────────────────────────────────────────────────────────
class Clusterer:
    """Единый интерфейс к трём методам: fit на таблице студентов, proba на ней же."""

    def __init__(self, method, k, seed=RANDOM_STATE, n_init=None):
        self.method, self.k, self.seed, self.n_init = method, k, seed, n_init

    def fit(self, df):
        if self.method == "lca":
            self.m = CountMixture(self.k, n_init=self.n_init or 8, random_state=self.seed).fit(*counts(df))
        else:
            self.sc = StandardScaler().fit(df[GMM_FEATURES])
            X = self.sc.transform(df[GMM_FEATURES])
            if self.method == "kmeans":
                self.m = KMeans(self.k, n_init=self.n_init or 10, random_state=self.seed).fit(X)
            else:
                self.m = GaussianMixture(self.k, covariance_type="full", reg_covar=GMM_REG,
                                         n_init=self.n_init or 3, max_iter=500,
                                         random_state=self.seed).fit(X)
        return self

    def proba(self, df):
        if self.method == "lca":
            return self.m.predict_proba(*counts(df))
        X = self.sc.transform(df[GMM_FEATURES])
        if self.method == "gmm":
            return self.m.predict_proba(X)
        return np.eye(self.k)[self.m.predict(X)]


def gpa_order(P, gpa):
    """Порядок компонент по возрастанию среднего балла отнесённых студентов (пустые — в начало)."""
    k = P.shape[1]
    return np.argsort(pd.Series(gpa).groupby(P.argmax(1)).mean().reindex(range(k)).fillna(0).values)


def relabel_by_gpa(labels, gpa):
    order = pd.Series(gpa).groupby(labels).mean().sort_values().index
    return pd.Series(labels).map({old: new for new, old in enumerate(order)}).values


def cluster_profile(df, labels):
    g = df.assign(c=labels).groupby("c")
    prof = g[PROFILE_COLS].mean()
    prof.insert(0, "size", g.size())
    return [{"cluster": int(c), **{k: r(v) for k, v in row.items()}} for c, row in prof.iterrows()]


def clustering_experiment(df, n_boot):
    Xs = StandardScaler().fit_transform(df[GMM_FEATURES])
    rng = np.random.default_rng(RANDOM_STATE)
    sil_idx = rng.choice(len(df), min(6000, len(df)), replace=False)
    boots = [rng.choice(len(df), int(0.8 * len(df)), replace=False) for _ in range(n_boot)]
    att = df["attrition"].values
    results, params = [], {}
    for method in ["kmeans", "gmm", "lca"]:
        for k in KS:
            c = Clusterer(method, k).fit(df)
            P = c.proba(df)
            order = gpa_order(P, df["gpa"].values)
            P = P[:, order]
            lab = P.argmax(1)
            risk_rate = pd.Series(att).groupby(lab).mean()
            score = risk_rate.reindex(lab).values           # риск кластера как оценка риска студента
            top = int(risk_rate.idxmax())
            ari = [adjusted_rand_score(lab, Clusterer(method, k, seed=b, n_init=2 if method != "kmeans" else 3)
                                       .fit(df.iloc[idx]).proba(df).argmax(1))
                   for b, idx in enumerate(boots)]
            row = dict(method=method, k=k,
                       silhouette=r(silhouette_score(Xs[sil_idx], lab[sil_idx])),
                       mean_max_proba=r(P.max(1).mean()),
                       uncertain_share=r((P.max(1) < 0.8).mean()),
                       boot_ari=r(np.mean(ari)), boot_ari_min=r(np.min(ari)),
                       nmi_status=r(normalized_mutual_info_score(df["final_state"], lab)),
                       auc_attrition=r(roc_auc_score(att, score)),
                       risk_cluster_rate=r(risk_rate.max()),
                       risk_cluster_recall=r((lab[att == 1] == top).mean()),
                       risk_cluster_size=int((lab == top).sum()),
                       profile=cluster_profile(df, lab))
            if method == "lca":
                row["bic"], row["icl"] = r(c.m.bic(), 1), r(c.m.icl(*counts(df)), 1)
                params[f"lca_{k}"] = dict(w=c.m.weights_[order].tolist(), p=c.m.p_[order].tolist(),
                                          lam=c.m.lam_[order].tolist())
            elif method == "gmm":
                X = c.sc.transform(df[GMM_FEATURES])
                row["bic"] = r(c.m.bic(X), 1)
                row["icl"] = r(c.m.bic(X) - 2 * (P * np.log(P + 1e-300)).sum(), 1)
                params[f"gmm_{k}"] = dict(w=c.m.weights_[order].tolist(), mu=c.m.means_[order].tolist(),
                                          cov=c.m.covariances_[order].tolist(),
                                          mean=c.sc.mean_.tolist(), scale=c.sc.scale_.tolist())
            else:
                params[f"kmeans_{k}"] = dict(centers=c.m.cluster_centers_[order].tolist(),
                                             mean=c.sc.mean_.tolist(), scale=c.sc.scale_.tolist())
            results.append(row)
            print(f"  {method:6s} k={k}  sil={row['silhouette']:.3f}  maxp={row['mean_max_proba']:.3f}  "
                  f"ARI={row['boot_ari']:.3f}  AUC(att)={row['auc_attrition']:.3f}")
    agree = {}
    for k in KS:
        labs = {m: relabel_by_gpa(Clusterer(m, k).fit(df).proba(df).argmax(1), df["gpa"].values)
                for m in ["kmeans", "gmm", "lca"]}
        agree[k] = {f"{a}_{b}": r(adjusted_rand_score(labs[a], labs[b]))
                    for a, b in itertools.combinations(labs, 2)}
    hist = {f: np.histogram(df[f], bins=20)[0].tolist() for f in ["gpa", "share_5", "share_3"]}
    hist["edges"] = {f: np.histogram(df[f], bins=20)[1].round(3).tolist() for f in ["gpa", "share_5", "share_3"]}
    hist["problems"] = (df["failed"] + df["debts"]).clip(upper=10).value_counts().sort_index().tolist()
    return results, params, agree, hist


# ── 3. прогноз ────────────────────────────────────────────────────────────────
def make_model(name):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.15,
                                          random_state=RANDOM_STATE)


def threshold_table(y, p):
    rows = []
    for t in THRESHOLDS:
        pred = p >= t
        rows.append([int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())])
    return rows   # [TP, FP] для каждого порога; FN и TN восстанавливаются из числа классов


def curve(x, y, n=120):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return [[r(x[i], 4), r(y[i], 4)] for i in idx]


def prediction_experiment(horizons):
    out, calc = [], {}
    for H in horizons:
        df = prediction_dataset(PANEL, H)
        cl_cols = [f"p_cluster_{i}" for i in range(1, CLUSTER_K)]   # P(кластер 0) линейно зависима
        blocks = {"dyn": DYNAMICS, "cl": cl_cols, "demo": DEMOGRAPHICS}
        toggles = ["dyn", "cl", "demo"] if H > 1 else ["cl", "demo"]
        for target in TARGETS:
            d = df[df[target].notna()].reset_index(drop=True)
            dfolds = list(GroupKFold(N_FOLDS).split(d, groups=d["group"]))
            y = d[target].astype(int).values
            # Вероятности кластеров — внутри фолда: смесь обучается на обучающей части и применяется
            # к обеим частям. Так номера кластеров согласованы между train и test (нет label switching).
            fold_cl = []
            for tr, te in dfolds:
                m = CountMixture(CLUSTER_K, n_init=5, random_state=RANDOM_STATE).fit(*counts(d.iloc[tr]))
                fold_cl.append((m.predict_proba(*counts(d.iloc[tr]))[:, 1:], m.predict_proba(*counts(d.iloc[te]))[:, 1:]))

            def xy(i, feats):
                tr, te = dfolds[i]
                own = [f for f in feats if f not in cl_cols]
                Xtr, Xte = d.loc[tr, own].copy(), d.loc[te, own].copy()
                if len(own) < len(feats):
                    Xtr[cl_cols], Xte[cl_cols] = fold_cl[i]
                return Xtr[feats], Xte[feats]
            for on in itertools.product([0, 1], repeat=len(toggles)):
                active = [t for t, o in zip(toggles, on) if o]
                feats = BASE + sum((blocks[t] for t in active), [])
                for model in ["logreg", "hgb"]:
                    p = np.zeros(len(d))
                    for i, (tr, te) in enumerate(dfolds):
                        Xtr, Xte = xy(i, feats)
                        p[te] = make_model(model).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
                    fpr, tpr, _ = roc_curve(y, p)
                    prec, rec, _ = precision_recall_curve(y, p)
                    out.append(dict(H=H, target=target, model=model, sets=active,
                                    n=len(y), pos=int(y.sum()),
                                    roc_auc=r(roc_auc_score(y, p)), pr_auc=r(average_precision_score(y, p)),
                                    brier=r(brier_score_loss(y, p)),
                                    roc=curve(fpr, tpr), pr=curve(rec[::-1], prec[::-1]),
                                    thr=threshold_table(y, p)))
                    print(f"  H={H} {target:10s} {model:6s} {'+'.join(active) or 'base':12s} "
                          f"ROC-AUC={out[-1]['roc_auc']:.3f}  PR-AUC={out[-1]['pr_auc']:.3f}")

            # важность признаков для полного набора
            feats = BASE + sum((blocks[t] for t in toggles), [])
            tr, te = dfolds[0]
            Xtr, Xte = xy(0, feats)
            for model in ["logreg", "hgb"]:
                m = make_model(model).fit(Xtr, y[tr])
                pi = permutation_importance(m, Xte, y[te], scoring="roc_auc",
                                            n_repeats=5, random_state=RANDOM_STATE)
                out.append(dict(H=H, target=target, model=model, kind="importance",
                                features=feats, importance=[r(v) for v in pi.importances_mean]))

            # параметры логистической регрессии для калькулятора (оценки + динамика + анкетные)
            feats = BASE + (DYNAMICS if H > 1 else []) + DEMOGRAPHICS
            m = make_model("logreg").fit(d[feats], y)
            sc, lr = m.steps[0][1], m.steps[1][1]
            calc[f"{target}_{H}"] = dict(features=feats, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                                         coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]),
                                         base_rate=r(y.mean()), gpa_median=r(d["gpa"].median()))
        out.append(dict(H=H, kind="dataset", n=len(df),
                        rates={t: r(df[t].mean()) for t in TARGETS},
                        status={k: int(v) for k, v in df["final_state"].value_counts().items()}))
    return out, calc


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    global PANEL
    PANEL = pd.read_parquet(args.panel)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("1. Диагностика старого пайплайна")
    old = diagnose_old(Path(args.cohort), Path(args.diplom))
    print(json.dumps(old, ensure_ascii=False, indent=1))

    print("2. Кластеризация")
    cdf = clustering_dataset(PANEL)
    clus, params, agree, hist = clustering_experiment(cdf, args.boot)

    print("3. Прогноз")
    pred, calc = prediction_experiment([1, 2])

    res = dict(old=old, clustering=dict(n=len(cdf), base_attrition=r(cdf["attrition"].mean()),
                                         results=clus, params=params, agreement=agree, hist=hist,
                                         features=GMM_FEATURES),
               prediction=pred, calculator=calc, targets=TARGETS,
               thresholds=THRESHOLDS.tolist(), cluster_k=CLUSTER_K)
    js = json.dumps(res, ensure_ascii=False, separators=(",", ":"), default=float)
    (out / "results.json").write_text(js, encoding="utf-8")
    pd.DataFrame([{k: v for k, v in x.items() if k != "profile"} for x in clus]).to_csv(
        out / "clustering_metrics.csv", index=False)
    pd.DataFrame([{k: v for k, v in x.items() if k not in ("roc", "pr", "thr")}
                  for x in pred if "roc_auc" in x]).assign(sets=lambda t: t.sets.map("+".join)).to_csv(
        out / "prediction_metrics.csv", index=False)
    build_report(out, js)


def build_report(out: Path, js: str):
    tpl = (out / "report_template.html")
    if not tpl.exists():
        print("Шаблон отчёта не найден — results.json сохранён, HTML не собран")
        return
    html = tpl.read_text(encoding="utf-8").replace("/*__DATA__*/null", js)
    # шаблон без <!doctype>/<head> (так его публикует Artifact); для локального просмотра добавляем
    head = ('<!doctype html>\n<html lang="ru"><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n')
    (out / "index.html").write_text(head + html, encoding="utf-8")
    print(f"Отчёт: {out / 'index.html'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel_v2.parquet")
    ap.add_argument("--cohort", default="data/df_features.parquet")
    ap.add_argument("--diplom", default="data/diplom.csv")
    ap.add_argument("--out", default="reports/iteration2")
    ap.add_argument("--boot", type=int, default=10, help="число бутстрэп-подвыборок")
    ap.add_argument("--report-only", action="store_true", help="только пересобрать HTML из results.json")
    a = ap.parse_args()
    if a.report_only:
        o = Path(a.out)
        build_report(o, (o / "results.json").read_text(encoding="utf-8"))
    else:
        main(a)
