from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CSV_FILES = {
    "students":   "students.csv",
    "contingent": "contingent.csv",
    "diplom":     "diplom.csv",
    "pass_df":    "pass.csv",
    "perenos":    "perenos.csv",
}

MARK_GOOD = 4  # «4»
MARK_BAD  = 3  # «3»

def read_csvs(root: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key, fname in CSV_FILES.items():
        path = root / fname
        print(f"-> {path} ...", end="")
        df = pd.read_csv(path, low_memory=False)
        print(f" (rows: {len(df):,})")
        out[key] = df
    return out


def _ensure_col(df: pd.DataFrame, desired: str, candidates: list[str]) -> pd.DataFrame:
    """Если нужного столбца нет, но есть один из candidates, переименовываем."""
    if desired in df.columns:
        return df.copy()
    for cand in candidates:
        if cand in df.columns:
            return df.rename(columns={cand: desired}).copy()
    raise KeyError(f"Столбец '{desired}' не найден и ни один из {candidates}")


def _aggregate_marks(pass_df: pd.DataFrame,
                     diplom: pd.DataFrame,
                     students: pd.DataFrame,
                     max_term: int) -> pd.DataFrame:
    pass_df = _ensure_col(pass_df, "diplom_id", ["diplom", "diplom_uuid", "id"])

    dipl_key = (diplom[["id", "student_uuid", "session_id"]]
                .rename(columns={"id": "diplom_id"})
                .drop_duplicates("diplom_id", keep="first"))

    marks = (pass_df
             .merge(dipl_key, on="diplom_id", how="left")
             .merge(students[["student_uuid", "session_id", "term"]]
                    .drop_duplicates(["student_uuid", "session_id"], keep="first"),
                    on=["student_uuid", "session_id"], how="left"))

    marks = marks[marks["term"].between(1, max_term)].copy()

    marks["is_3"]    = (marks["mark"] == MARK_BAD).astype(int)
    marks["is_4"]    = (marks["mark"] == MARK_GOOD).astype(int)
    marks["is_fail"] = ((marks["mark"] < 3) | (marks.get("not_valid", 0) == 1)).astype(int)

    agg = (marks.groupby("student_uuid", as_index=False)
                .agg(gpa=("mark", "mean"),
                     n4=("is_4", "sum"),
                     n3=("is_3", "sum"),
                     fails=("is_fail", "sum")))

    suf = f"1_{max_term}"
    return agg.rename(columns={
        "gpa":   f"gpa_{suf}",
        "n4":    f"n_4_{suf}",
        "n3":    f"n_3_{suf}",
        "fails": "fail_cnt" if max_term == 4 else f"fail_cnt_{suf}",
    })


def _last_state(contingent: pd.DataFrame, max_term: int):
    subset = contingent[contingent["term"].between(1, max_term)].copy()
    last_rows = (subset.sort_values(["student_uuid", "term"])\
                       .drop_duplicates("student_uuid", keep="last"))

    demo_cols = ["student_uuid", "group", "faculty_uuid", "department_uuid",
                 "studytype", "disabled", "foreign", "military",
                 "dormitory_uuid", "term"]
    demo = last_rows[[c for c in demo_cols if c in last_rows.columns]]\
           .drop_duplicates("student_uuid", keep="first")

    final_state = last_rows[["student_uuid", "state"]]\
                  .rename(columns={"state": "final_state"})
    return demo, final_state

def build_df(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    students, contingent, diplom = dfs["students"].copy(), dfs["contingent"].copy(), dfs["diplom"].copy()
    pass_df, perenos = dfs["pass_df"].copy(), dfs["perenos"].copy()

    for df in (students, contingent):
        if "student_uuid" not in df.columns and "uuid" in df.columns:
            df.rename(columns={"uuid": "student_uuid"}, inplace=True)

    agg_1_4 = _aggregate_marks(pass_df, diplom, students, 4)
    agg_1_8 = _aggregate_marks(pass_df, diplom, students, 8)

    demo_4,  final_state_4 = _last_state(contingent, 4)
    _, final_state_8 = _last_state(contingent, 8)
    final_state_8 = final_state_8.rename(columns={"final_state": "final_state_8"})

    debts = (_ensure_col(perenos, "diplom_id", ["diplom", "diplom_uuid", "id"])
             .merge(diplom[["id", "student_uuid"]].rename(columns={"id": "diplom_id"}),
                    on="diplom_id", how="left")
             .merge(students[["student_uuid", "term"]], on="student_uuid", how="left"))
    debts = (debts[debts["term"].between(1, 4)]
             .assign(has_debt=1)
             .drop_duplicates("diplom_id", keep="first")[["diplom_id", "has_debt"]])

    base = (demo_4
            .merge(agg_1_4, on="student_uuid", how="left")
            .merge(final_state_4, on="student_uuid", how="left")
            .merge(agg_1_8[["student_uuid", "n_4_1_8", "n_3_1_8"]], on="student_uuid", how="left")
            .merge(diplom[["id", "student_uuid"]].rename(columns={"id": "diplom_id"}),
                   on="student_uuid", how="left")
            .merge(debts, on="diplom_id", how="left")
            .merge(final_state_8, on="student_uuid", how="left"))

    base["has_debt"] = base["has_debt"].fillna(0)

    def make_target(row):
        if row["final_state_8"] in ("отчислен", "академический отпуск"):
            return 0
        return int((row.get("n_3_1_8", 0) < 5) and (row.get("n_4_1_8", 0) < 15))

    base["target"] = base.apply(make_target, axis=1).astype(int)
    base.drop(columns=["final_state_8", "n_3_1_8", "n_4_1_8"], inplace=True)

    cols = ["student_uuid", "group", "faculty_uuid", "department_uuid",
            "studytype", "disabled", "foreign", "military",
            "dormitory_uuid", "term",
            "gpa_1_4", "n_4_1_4", "n_3_1_4",
            "has_debt", "final_state", "target"]
    for col in cols:
        if col not in base.columns:
            base[col] = pd.NA

    return base[cols]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="папка с исходными CSV")
    parser.add_argument("--out", default="data/df_raw.parquet", help="куда сохранить parquet")
    args = parser.parse_args()

    dfs = read_csvs(Path(args.root))
    df = build_df(dfs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"df_raw.parquet -Ю {out_path}  (rows: {len(df):,})")

if __name__ == "__main__":
    main()