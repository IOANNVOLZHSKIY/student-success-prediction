"""Кластеризация студентов по успеваемости за 1–4 семестры: K-means и GMM (EM-алгоритм).

Признаки строятся заново из data/diplom.csv (по одной строке на студента), т.к. в df_raw.parquet
оценка «Зачтено» (код 3) смешана с «Удовлетворительно», а gpa_1_4 усредняет служебные коды
(перенос, неявка, не аттестован). Отчёт: reports/clustering/REPORT.md.

Запуск:
    python clustering.py --k-gmm 3 --cov full  # вариант из отчёта
    python clustering.py                       # k выбирается автоматически (силуэт / BIC)
    python clustering.py --k-max 8 --boot 10   # быстрее

Результаты: reports/clustering/*.csv, *.png; метки студентов — data/cluster_labels.parquet.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
CREDIT = 2                       # test_type «зачёт»: 3 = «Зачтено», 2 = «Не зачтено»
BAD_MARKS = {-4, -2, 0, 1, 2}    # не аттестован, недопуск, неявка, неуд / не зачтено

# gpa не используется: на 98 % (R²) линейно выражается через share_5 и share_3.
# credit_rate не используется: у 98.8 % студентов равен 1.
FEATURES = {
    "share_5":      "Доля «отлично»",
    "share_3":      "Доля «удовлетворительно»",
    "log_problems": "log(1 + неудачные попытки и задолженности)",
}
PROFILE_ONLY = ["gpa", "credit_rate", "log_failed", "log_debts"]


# ── признаки ──────────────────────────────────────────────────────────────────
def build_features(diplom_path: Path, cohort_path: Path, min_graded: int) -> pd.DataFrame:
    cohort = (pd.read_parquet(cohort_path,
                              columns=["student_uuid", "faculty_uuid", "studytype",
                                       "foreign", "final_state", "target"])
                .drop_duplicates("student_uuid"))

    d = pd.read_csv(diplom_path, low_memory=False,
                    usecols=["id", "student_uuid", "discipline_name", "mark", "test_type", "term"])
    d = d[d["student_uuid"].isin(cohort["student_uuid"]) & d["term"].between(1, 4)]

    # Итог по дисциплине — последняя попытка (по id). Флаг not_valid не используется: им помечаются
    # и пересданные попытки, и все записи отчисленных студентов, которые иначе выпали бы из выборки.
    key = ["student_uuid", "discipline_name", "term", "test_type"]
    last = d.sort_values("id").drop_duplicates(key, keep="last")
    graded = last[(last["test_type"] != CREDIT) & last["mark"].between(2, 5)]
    credit = last[(last["test_type"] == CREDIT) & last["mark"].isin([2, 3])]

    g = graded.groupby("student_uuid")["mark"]
    feats = pd.DataFrame({
        "n_graded": g.size(),
        "gpa":      g.mean(),
        "share_5":  g.apply(lambda m: (m == 5).mean()),
        "share_4":  g.apply(lambda m: (m == 4).mean()),
        "share_3":  g.apply(lambda m: (m == 3).mean()),
    })
    feats["credit_rate"] = credit.groupby("student_uuid")["mark"].apply(lambda m: (m == 3).mean())
    n_bad = d[d["mark"].isin(BAD_MARKS)].groupby("student_uuid").size()
    feats["n_debts"] = last[last["mark"].isin(BAD_MARKS)].groupby("student_uuid").size()
    feats["n_failed"] = n_bad.sub(feats["n_debts"], fill_value=0)   # неудачи, исправленные пересдачей
    feats["n_terms"] = d.groupby("student_uuid")["term"].nunique()

    feats[["n_failed", "n_debts"]] = feats[["n_failed", "n_debts"]].fillna(0)
    feats["credit_rate"] = feats["credit_rate"].fillna(feats["credit_rate"].median())
    feats["log_failed"] = np.log1p(feats["n_failed"])
    feats["log_debts"] = np.log1p(feats["n_debts"])
    feats["log_problems"] = np.log1p(feats["n_failed"] + feats["n_debts"])

    df = cohort.merge(feats.reset_index(), on="student_uuid", how="left")
    print(f"Студентов в когорте: {len(df):,}")
    df = df[df["n_graded"] >= min_graded].reset_index(drop=True)
    print(f"С ≥{min_graded} оценками за 1–4 семестры: {len(df):,}")
    return df


# ── подбор числа кластеров ────────────────────────────────────────────────────
def silhouette(X, labels, n=8000):
    return silhouette_score(X, labels, sample_size=min(n, len(X)), random_state=RANDOM_STATE)


def scan_kmeans(X, ks):
    rows = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette(X, km.labels_),
                         calinski_harabasz=calinski_harabasz_score(X, km.labels_),
                         davies_bouldin=davies_bouldin_score(X, km.labels_)))
        print(f"  K-means k={k:2d}  sil={rows[-1]['silhouette']:.3f}  DB={rows[-1]['davies_bouldin']:.3f}")
    return pd.DataFrame(rows)


def scan_gmm(X, ks, reg_covar, cov_types=("full", "diag", "tied", "spherical")):
    rows = []
    for cov in cov_types:
        for k in ks:
            gm = GaussianMixture(k, covariance_type=cov, n_init=3, max_iter=500,
                                 reg_covar=reg_covar, random_state=RANDOM_STATE).fit(X)
            labels = gm.predict(X)
            n_used = len(np.unique(labels))
            rows.append(dict(covariance=cov, k=k, bic=gm.bic(X), aic=gm.aic(X),
                             loglik=gm.score(X) * len(X),
                             silhouette=silhouette(X, labels) if n_used > 1 else np.nan,
                             mean_max_proba=gm.predict_proba(X).max(1).mean(),
                             converged=gm.converged_))
        best = min((r for r in rows if r["covariance"] == cov), key=lambda r: r["bic"])
        print(f"  GMM {cov:9s} лучший по BIC: k={best['k']}")
    return pd.DataFrame(rows)


def elbow_k(ks, values):
    """Точка максимальной кривизны (наибольшее расстояние до хорды)."""
    x, y = np.asarray(ks, float), np.asarray(values, float)
    x, y = (x - x.min()) / np.ptp(x), (y - y.min()) / np.ptp(y)
    dist = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return int(ks[int(np.argmax(dist))])


# ── оценка качества выбранного разбиения ──────────────────────────────────────
def bootstrap_ari(X, make_model, ref_labels, n_boot, frac=0.8):
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    for b in range(n_boot):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        scores.append(adjusted_rand_score(ref_labels, make_model(b).fit(X[idx]).predict(X)))
    return np.array(scores)


def null_silhouette(X, k, n_rep=5):
    """Силуэт K-means на данных, где столбцы перемешаны независимо: структура маргиналов
    сохраняется, связи между признаками разрушаются."""
    rng = np.random.default_rng(RANDOM_STATE)
    out = []
    for r in range(n_rep):
        Xn = np.column_stack([rng.permutation(col) for col in X.T])
        out.append(silhouette(Xn, KMeans(k, n_init=5, random_state=r).fit_predict(Xn)))
    return np.array(out)


def order_by_gpa(df, col):
    """Переименовывает кластеры по возрастанию среднего балла: 0 — самый слабый."""
    order = df.groupby(col)["gpa"].mean().sort_values().index
    return df[col].map({old: new for new, old in enumerate(order)})


def profile(df, col):
    num = ["gpa", "share_5", "share_4", "share_3", "credit_rate", "n_failed", "n_debts", "n_graded"]
    prof = df.groupby(col)[num].mean()
    prof.insert(0, "size", df.groupby(col).size())
    prof.insert(1, "share", prof["size"] / len(df))
    states = pd.crosstab(df[col], df["final_state"], normalize="index").add_prefix("state: ")
    prof["target_rate"] = df.groupby(col)["target"].mean()
    prof["paid_share"] = df.groupby(col)["studytype"].apply(lambda s: (s == "платная").mean())
    prof["foreign_share"] = df.groupby(col)["foreign"].mean()
    return prof.join(states)


# ── графики ───────────────────────────────────────────────────────────────────
def plot_selection(km_scan, gm_scan, k_km, k_gm, cov, out):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(km_scan.k, km_scan.inertia, "o-")
    ax[0].axvline(k_km, ls="--", c="grey")
    ax[0].set(title="K-means: метод локтя", xlabel="k", ylabel="Инерция (WCSS)")
    ax[1].plot(km_scan.k, km_scan.silhouette, "o-", label="K-means")
    g = gm_scan[gm_scan.covariance == cov]
    ax[1].plot(g.k, g.silhouette, "s-", label=f"GMM ({cov})")
    ax[1].set(title="Силуэт", xlabel="k", ylabel="silhouette")
    ax[1].legend()
    for c, gg in gm_scan.groupby("covariance"):
        ax[2].plot(gg.k, gg.bic, "o-", label=c)
    ax[2].axvline(k_gm, ls="--", c="grey")
    ax[2].set(title="GMM: BIC (меньше — лучше)", xlabel="k", ylabel="BIC")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_pca(X, labels_km, labels_gm, out, n=6000):
    pca = PCA(2, random_state=RANDOM_STATE).fit(X)
    idx = np.random.default_rng(RANDOM_STATE).choice(len(X), min(n, len(X)), replace=False)
    P = pca.transform(X[idx])
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for a, lab, name in [(ax[0], labels_km, "K-means"), (ax[1], labels_gm, "GMM")]:
        sc = a.scatter(P[:, 0], P[:, 1], c=lab[idx], cmap="viridis", s=5, alpha=.6)
        a.set(title=f"{name}: проекция на PC1–PC2",
              xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})",
              ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
        a.legend(*sc.legend_elements(), title="Кластер", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_profiles(df, X, col, title, out):
    Z = pd.DataFrame(X, columns=list(FEATURES)).groupby(df[col].values).mean()
    fig, ax = plt.subplots(figsize=(1.3 * len(FEATURES) + 2, 0.6 * len(Z) + 1.8))
    im = ax.imshow(Z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(FEATURES)), list(FEATURES), rotation=30, ha="right")
    sizes = df[col].value_counts().sort_index()
    ax.set_yticks(range(len(Z)), [f"{c} (n={sizes[c]:,})" for c in Z.index])
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, f"{Z.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Среднее (z-оценка)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_outcomes(df, col, title, out):
    ct = pd.crosstab(df[col], df["final_state"], normalize="index")
    ax = ct.plot(kind="bar", stacked=True, figsize=(8, 4.5), colormap="tab10")
    ax.set(title=title, xlabel="Кластер", ylabel="Доля студентов")
    ax.legend(title="Статус", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()


def plot_distributions(df, out):
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(4.7 * len(FEATURES), 3.8))
    for a, f in zip(np.ravel(axes), FEATURES):
        a.hist(df[f], bins=40, color="steelblue")
        a.set_title(FEATURES[f], fontsize=9)
    fig.suptitle("Распределения признаков кластеризации")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────
def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = build_features(Path(args.diplom), Path(args.cohort), args.min_graded)
    X = StandardScaler().fit_transform(df[list(FEATURES)])
    plot_distributions(df, out / "feature_distributions.png")
    df[list(FEATURES) + PROFILE_ONLY].corr().round(3).to_csv(out / "feature_corr.csv")

    ks = range(2, args.k_max + 1)
    print("K-means: перебор k")
    km_scan = scan_kmeans(X, ks)
    print("GMM: перебор k и типа ковариации")
    gm_scan = scan_gmm(X, range(1, args.k_max + 1), args.reg_covar)
    km_scan.to_csv(out / "kmeans_scan.csv", index=False)
    gm_scan.to_csv(out / "gmm_scan.csv", index=False)

    # K-means: k по силуэту; GMM: k и ковариация по BIC
    k_km = args.k_kmeans or int(km_scan.loc[km_scan.silhouette.idxmax(), "k"])
    best = gm_scan.loc[gm_scan.bic.idxmin()]
    cov = args.cov or best.covariance
    k_gm = args.k_gmm or int(gm_scan[gm_scan.covariance == cov].set_index("k").bic.idxmin())
    k_elbow = elbow_k(list(km_scan.k), km_scan.inertia)
    print(f"Выбрано: K-means k={k_km} (локоть: {k_elbow}), GMM k={k_gm}, cov={cov}")
    plot_selection(km_scan, gm_scan, k_km, k_gm, cov, out / "model_selection.png")

    km = KMeans(k_km, n_init=20, random_state=RANDOM_STATE).fit(X)
    gm = GaussianMixture(k_gm, covariance_type=cov, n_init=5, max_iter=500,
                         reg_covar=args.reg_covar, random_state=RANDOM_STATE).fit(X)
    df["km"] = km.labels_
    df["gmm"] = gm.predict(X)
    proba = gm.predict_proba(X)
    df["gmm_proba"] = proba.max(1)
    df["km"], df["gmm"] = order_by_gpa(df, "km"), order_by_gpa(df, "gmm")

    print("Бутстрэп-устойчивость…")
    ari_km = bootstrap_ari(X, lambda b: KMeans(k_km, n_init=5, random_state=b), km.labels_, args.boot)
    ari_gm = bootstrap_ari(X, lambda b: GaussianMixture(k_gm, covariance_type=cov, n_init=2,
                                                        max_iter=500, reg_covar=args.reg_covar,
                                                        random_state=b),
                           gm.predict(X), args.boot)
    null_sil = null_silhouette(X, k_km)

    status = df["final_state"]
    summary = {
        "n_students": len(df),
        "features": list(FEATURES),
        "min_graded": args.min_graded, "reg_covar": args.reg_covar,
        "kmeans": {
            "k": k_km, "k_elbow": k_elbow,
            "silhouette": silhouette(X, df.km),
            "silhouette_null_mean": null_sil.mean(), "silhouette_null_max": null_sil.max(),
            "davies_bouldin": davies_bouldin_score(X, df.km),
            "calinski_harabasz": calinski_harabasz_score(X, df.km),
            "bootstrap_ari_mean": ari_km.mean(), "bootstrap_ari_min": ari_km.min(),
            "nmi_final_state": normalized_mutual_info_score(status, df.km),
            "nmi_target": normalized_mutual_info_score(df.target, df.km),
        },
        "gmm": {
            "k": k_gm, "covariance": cov,
            "bic": gm.bic(X), "aic": gm.aic(X),
            "silhouette": silhouette(X, df.gmm),
            "davies_bouldin": davies_bouldin_score(X, df.gmm),
            "mean_max_proba": float(proba.max(1).mean()),
            "share_uncertain_lt_0_8": float((proba.max(1) < .8).mean()),
            "bootstrap_ari_mean": ari_gm.mean(), "bootstrap_ari_min": ari_gm.min(),
            "nmi_final_state": normalized_mutual_info_score(status, df.gmm),
            "nmi_target": normalized_mutual_info_score(df.target, df.gmm),
        },
        "ari_kmeans_vs_gmm": adjusted_rand_score(df.km, df.gmm),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")

    profile(df, "km").round(3).to_csv(out / "profile_kmeans.csv")
    profile(df, "gmm").round(3).to_csv(out / "profile_gmm.csv")
    pd.crosstab(df.km, df.gmm).to_csv(out / "kmeans_vs_gmm.csv")

    plot_pca(X, df.km.values, df.gmm.values, out / "pca_clusters.png")
    plot_profiles(df, X, "km", f"Профили кластеров K-means (k={k_km})", out / "profile_kmeans.png")
    plot_profiles(df, X, "gmm", f"Профили кластеров GMM (k={k_gm}, {cov})", out / "profile_gmm.png")
    plot_outcomes(df, "km", "Статус студентов по кластерам K-means", out / "outcomes_kmeans.png")
    plot_outcomes(df, "gmm", "Статус студентов по кластерам GMM", out / "outcomes_gmm.png")

    labels_path = Path(args.labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(labels_path, index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"Готово: {out}/, метки — {labels_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--diplom", default="data/diplom.csv")
    ap.add_argument("--cohort", default="data/df_features.parquet")
    ap.add_argument("--out", default="reports/clustering")
    ap.add_argument("--labels", default="data/cluster_labels.parquet")
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--min-graded", type=int, default=3,
                    help="минимум оценок за экзамены/КП/КР, чтобы студент попал в выборку")
    ap.add_argument("--reg-covar", type=float, default=0.05,
                    help="добавка к диагонали ковариаций GMM (в z-единицах); защищает от вырождения "
                         "компонент на точечных массах вроде «0 задолженностей»")
    ap.add_argument("--boot", type=int, default=20, help="число бутстрэп-подвыборок")
    ap.add_argument("--k-kmeans", type=int, help="зафиксировать k для K-means")
    ap.add_argument("--k-gmm", type=int, help="зафиксировать k для GMM")
    ap.add_argument("--cov", choices=["full", "diag", "tied", "spherical"],
                    help="зафиксировать тип ковариации GMM")
    main(ap.parse_args())
