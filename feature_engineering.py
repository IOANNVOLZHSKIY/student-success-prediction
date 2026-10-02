from __future__ import annotations
import argparse, warnings
from pathlib import Path
import pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from scipy import sparse

def to_csr(x):
    """dense → CSR-sparse"""
    return sparse.csr_matrix(x)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",  default="data/df_raw.parquet")
    ap.add_argument("--out",  default="data/df_features.parquet")
    ap.add_argument("--pipe", default="models/preprocess.joblib")
    args = ap.parse_args()

    df = pd.read_parquet(args.raw)
    print("df_raw:", df.shape)

    rename_map = {"n_4_1_4": "n_4", "n_3_1_4": "n_3"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ── NUMERIC ----------------------------------------------------
    num_cols = [c for c in [
        "gpa_1_4", "n_3", "n_4", "fail_cnt_1_4", "has_debt",
        "disabled", "foreign", "military"
    ] if c in df.columns]

    extra = ["group_size", "group_foreign_share", "group_avg_gpa"]
    num_cols += [c for c in extra if c in df.columns]

    # ── CATEGORICAL ------------------------------------------------
    cat_cols = [c for c in ["group", "faculty_uuid", "department_uuid", "studytype"] if c in df.columns]

    # ── NaN handling ----------------------------------------------
    if num_cols:
        df[num_cols] = df[num_cols].fillna(0)
    for c in cat_cols:
        df[c] = (
            df[c].astype("category")
                  .cat.add_categories("missing")
                  .fillna("missing")
        )

    num_pipe = Pipeline([
        ("to_sparse", FunctionTransformer(to_csr, accept_sparse=True))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols)
    ], sparse_threshold=1.0).fit(df[num_cols + cat_cols])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pipe).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    joblib.dump(pre, args.pipe)

    print(f"df_features.parquet  rows: {len(df):,}")
    print("num_cols:", num_cols)
    print("cat_cols:", cat_cols)

if __name__ == "__main__":
    main()