"""Исправленное построение признаков и целевых переменных (итерация 2).

Отличия от data_load.py / feature_engineering.py:
  * одна строка на студента (в df_raw.parquet каждый студент повторён в среднем 19 раз);
  * экзамены/диф. зачёты/КП/КР и зачёты разделены (код 3 = и «удовл.», и «зачтено»);
  * итог по дисциплине — последняя попытка; неудачные попытки до неё — «пересданные неудачи»;
  * признаки берутся строго за семестры 1..H, целевые переменные — по тому, что было после H.

Запуск:
    python features_v2.py          # data/panel_v2.parquet — сводка «студент × семестр»
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CREDIT = 2                       # test_type «зачёт»
BAD_MARKS = [-4, -2, 0, 1, 2]    # не аттестован, недопуск, неявка, неуд / не зачтено
ATTRITION = ["отчислен", "в академическом отпуске"]
LAST_TERM = 3                    # последний семестр с данными у набора 2023 г. (срез — зима 2024/25)

TERM_COUNTS = ["n5", "n4", "n3", "n_credit", "failed", "debts"]


# ── панель «студент × семестр» ────────────────────────────────────────────────
def build_panel(diplom_path: Path, cohort_path: Path) -> pd.DataFrame:
    cohort = (pd.read_parquet(cohort_path,
                              columns=["student_uuid", "group", "faculty_uuid", "studytype",
                                       "foreign", "military", "term", "final_state"])
                .drop_duplicates("student_uuid")
                .rename(columns={"term": "cur_term"}))
    # в выгрузке «в академическом отпуске» записано с неразрывным пробелом
    cohort["final_state"] = cohort["final_state"].str.replace("\xa0", " ")

    d = pd.read_csv(diplom_path, low_memory=False,
                    usecols=["id", "student_uuid", "discipline_name", "mark", "test_type", "term"])
    d = d[d["student_uuid"].isin(cohort["student_uuid"]) & d["term"].between(1, 8)]

    key = ["student_uuid", "discipline_name", "term", "test_type"]
    d = d.sort_values("id")
    d["is_last"] = ~d.duplicated(key, keep="last")
    d["bad"] = d["mark"].isin(BAD_MARKS)
    last = d[d["is_last"]]
    graded = last["test_type"] != CREDIT

    agg = pd.DataFrame({
        "n5":       (graded & (last["mark"] == 5)),
        "n4":       (graded & (last["mark"] == 4)),
        "n3":       (graded & (last["mark"] == 3)),
        "n_credit": (~graded & (last["mark"] == 3)),
        "debts":    last["bad"],
    }).astype(int).groupby([last["student_uuid"], last["term"]]).sum()
    agg["failed"] = d[d["bad"] & ~d["is_last"]].groupby(["student_uuid", "term"]).size()
    panel = agg.fillna(0).astype(int).reset_index()

    panel = panel.merge(cohort, on="student_uuid", how="left")
    print(f"Панель: {len(panel):,} строк «студент × семестр», {panel.student_uuid.nunique():,} студентов")
    return panel


# ── признаки за семестры 1..H ─────────────────────────────────────────────────
def _summ(t: pd.DataFrame) -> pd.DataFrame:
    """Производные признаки из счётчиков n5, n4, n3, n_credit, failed, debts."""
    out = pd.DataFrame(index=t.index)
    n = t["n5"] + t["n4"] + t["n3"]
    out["n_graded"] = n
    safe = n.where(n > 0)
    out["share_5"] = (t["n5"] / safe).fillna(0)
    out["share_4"] = (t["n4"] / safe).fillna(0)
    out["share_3"] = (t["n3"] / safe).fillna(0)
    out["gpa"] = ((5 * t["n5"] + 4 * t["n4"] + 3 * t["n3"]) / safe)
    out["log_failed"] = np.log1p(t["failed"])
    out["log_debts"] = np.log1p(t["debts"])
    out["log_problems"] = np.log1p(t["failed"] + t["debts"])
    out["exposure"] = n + t["n_credit"] + t["debts"]
    return out


BASE = ["gpa", "share_5", "share_3", "log_failed", "log_debts"]
DYNAMICS = ["d_gpa", "d_share_5", "log_failed_last", "log_debts_last", "problem_terms"]
DEMOGRAPHICS = ["paid", "foreign", "military", "master"]


def student_features(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Одна строка на студента: счётчики и признаки за семестры 1..horizon + динамика."""
    p = panel[panel["term"] <= horizon]
    counts = p.groupby("student_uuid")[TERM_COUNTS].sum()
    df = counts.join(_summ(counts))
    df["gpa"] = df["gpa"].fillna(df["gpa"].median())

    # динамика: последний наблюдаемый семестр против первого
    first = p[p["term"] == 1].set_index("student_uuid")[TERM_COUNTS]
    lastt = p[p["term"] == horizon].set_index("student_uuid")[TERM_COUNTS]
    s1 = _summ(first.reindex(df.index, fill_value=0))
    sl = _summ(lastt.reindex(df.index, fill_value=0))
    s1["gpa"], sl["gpa"] = s1["gpa"].fillna(df["gpa"]), sl["gpa"].fillna(df["gpa"])
    df["d_gpa"] = sl["gpa"] - s1["gpa"]
    df["d_share_5"] = sl["share_5"] - s1["share_5"]
    df["log_failed_last"] = sl["log_failed"]
    df["log_debts_last"] = sl["log_debts"]
    df["problem_terms"] = (p.assign(pr=(p["failed"] + p["debts"]) > 0)
                            .groupby("student_uuid")["pr"].sum()
                            .reindex(df.index, fill_value=0))
    for t in range(1, horizon + 1):   # счётчики по семестрам — для калькулятора в отчёте
        tt = p[p["term"] == t].set_index("student_uuid")[TERM_COUNTS].reindex(df.index, fill_value=0)
        df[[f"{c}_t{t}" for c in TERM_COUNTS]] = tt.values

    info = (panel.drop_duplicates("student_uuid")
                 .set_index("student_uuid")[["group", "faculty_uuid", "studytype", "foreign",
                                             "military", "cur_term", "final_state"]])
    df = df.join(info)
    df["paid"] = (df["studytype"] == "платная").astype(int)
    df["master"] = df["group"].str.contains(r"\d+МВ?\s*$", regex=True).astype(int)
    return df.reset_index()


# ── целевые переменные (после семестра H) ─────────────────────────────────────
TARGETS = {
    "attrition":  "Отчисление или академ. отпуск",
    "debts_next": "Задолженность в следующих семестрах",
    "excellent":  "«Красный диплом» в следующих семестрах",
}


def prediction_dataset(panel: pd.DataFrame, horizon: int, min_graded: int = 3) -> pd.DataFrame:
    """Набор 2023 г. (сейчас 4-й семестр): признаки за 1..H, исходы — после H.

    Берутся только студенты, у которых есть записи после семестра H, т.е. на момент окончания
    семестра H они ещё учились: событие (отчисление, академ) произошло позже признаков.
    """
    df = student_features(panel, horizon)
    df = df[(df["cur_term"] == 4) & (df["n_graded"] >= min_graded)]

    after = panel[(panel["term"] > horizon) & (panel["term"] <= LAST_TERM)]
    nxt = after.groupby("student_uuid")[TERM_COUNTS].sum()
    nxt = nxt.join(_summ(nxt))
    df = df[df["student_uuid"].isin(nxt.index)].copy()
    nxt = nxt.reindex(df["student_uuid"])

    df["attrition"] = df["final_state"].isin(ATTRITION).astype(int)
    df["debts_next"] = (nxt["debts"].values > 0).astype(int)
    exc = ((nxt["n3"] == 0) & (nxt["share_5"] >= 0.75) & (nxt["debts"] == 0)).values
    ok = (nxt["n_graded"] >= 2).values          # иначе «красный диплом» не определён
    df["excellent"] = np.where(ok, exc.astype(float), np.nan)
    return df.reset_index(drop=True)


def clustering_dataset(panel: pd.DataFrame, min_graded: int = 3) -> pd.DataFrame:
    """Все студенты когорты, семестры 1–4 — как в итерации 1."""
    df = student_features(panel, 4)
    df = df[df["n_graded"] >= min_graded].reset_index(drop=True)
    df["attrition"] = df["final_state"].isin(ATTRITION).astype(int)
    df["al"] = (df["final_state"] == ATTRITION[1]).astype(int)
    df["expelled"] = (df["final_state"] == ATTRITION[0]).astype(int)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--diplom", default="data/diplom.csv")
    ap.add_argument("--cohort", default="data/df_features.parquet")
    ap.add_argument("--out", default="data/panel_v2.parquet")
    args = ap.parse_args()
    panel = build_panel(Path(args.diplom), Path(args.cohort))
    panel.to_parquet(args.out, index=False)
    for h in (1, 2):
        ds = prediction_dataset(panel, h)
        print(f"H={h}: {len(ds):,} студентов; " + ", ".join(
            f"{t}={ds[t].mean():.3f} (n={ds[t].notna().sum():,})" for t in TARGETS))
    print(f"Кластеризация: {len(clustering_dataset(panel)):,} студентов")
