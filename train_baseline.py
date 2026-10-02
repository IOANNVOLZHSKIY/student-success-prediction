from __future__ import annotations
import argparse, warnings
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

import xgboost as xgb
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from scipy import sparse
warnings.filterwarnings("ignore", category=UserWarning,
                        message="X does not have valid feature names")

def to_csr(x): return sparse.csr_matrix(x)

def eval_metrics(y_true, y_prob, thr: float = .5):
    y_pred = (y_prob >= thr).astype(int)
    return dict(
        roc_auc=roc_auc_score(y_true, y_prob),
        pr_auc=average_precision_score(y_true, y_prob),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
    )

def main(args):
    df = pd.read_parquet(args.data)
    X, y = df.drop(columns=["target"]), df["target"].astype(int)

    groups = X["group"]

    cv = GroupKFold(n_splits=3)
    tr_idx, val_idx = next(cv.split(X, y, groups))
    X_tr, X_val, y_tr, y_val = X.iloc[tr_idx], X.iloc[val_idx], y.iloc[tr_idx], y.iloc[val_idx]

    prep: Pipeline = joblib.load(args.pipe)
    X_tr_t, X_val_t = prep.transform(X_tr), prep.transform(X_val)

    pos_ratio = y_tr.mean()
    scale_pos = (1 - pos_ratio) / pos_ratio

    models = {
        "logreg": {"model": LogisticRegression(
            max_iter=5000, solver="lbfgs", class_weight="balanced", n_jobs=-1),
            "early": False},
        "xgb": {"model": XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
            scale_pos_weight=scale_pos, eval_metric="aucpr", tree_method="hist",
            device="cuda", random_state=42, n_jobs=-1, verbosity=0,
            early_stopping_rounds=100),
            "early": True},
    }
    if HAS_LGBM:
        models["lgbm"] = {"model": LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, device="gpu",
            gpu_platform_id=0, gpu_device_id=0, class_weight="balanced",
            random_state=42, n_jobs=-1, min_child_samples=20,
            early_stopping_rounds=100, verbose=-1),
            "early": True}

    best_name, best_pr, best_pipe, metrics = None, -np.inf, None, []
    for name, cfg in tqdm(models.items(), desc="Training models", unit="model"):
        clf, early = cfg["model"], cfg["early"]

        if name == "logreg":
            pipe = Pipeline([("prep", prep), ("clf", clf)])
            pipe.fit(X_tr, y_tr)
            y_prob = pipe.predict_proba(X_val)[:, 1]
        else:
            clf.fit(X_tr_t, y_tr, eval_set=[(X_val_t, y_val)])
            pipe = Pipeline([("prep", prep), ("clf", clf)])
            y_prob = clf.predict_proba(X_val_t)[:, 1]

        m = eval_metrics(y_val, y_prob) | {"model": name}
        metrics.append(m)
        print(f"{name:6s}  PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}")
        if m["pr_auc"] > best_pr:
            best_name, best_pr, best_pipe = name, m["pr_auc"], pipe

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, args.model)
    pd.DataFrame(metrics).to_csv(args.report, index=False)
    print(f"\nBest model: {best_name} (PR-AUC={best_pr:.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default="data/df_features.parquet")
    ap.add_argument("--pipe",  default="models/preprocess.joblib")
    ap.add_argument("--model", default="models/best_model.joblib")
    ap.add_argument("--report", default="reports/metrics_baseline.csv")
    main(ap.parse_args())