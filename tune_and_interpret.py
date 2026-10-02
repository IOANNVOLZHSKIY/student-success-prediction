from __future__ import annotations

import argparse
import json
import warnings
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import optuna
import pandas as pd
import shap
import scipy
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
import scipy.sparse as sp
from scipy import sparse

import numpy.core.multiarray as _multi
sys.modules['numpy._core.multiarray'] = _multi

warnings.filterwarnings("ignore", message="No further splits with positive gain")
warnings.filterwarnings("ignore", message="X does not have valid feature names")
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pre_utils
import feature_engineering

DATA_PATH = Path("data/df_features.parquet")
PREP_PATH = Path("models/preprocess.joblib")
MODEL_OUT = Path("models/best_model.joblib")
THRESH_OUT = Path("models/final_threshold.json")
SHAP_BAR_OUT = Path("reports/shap_summary_bar.png")
SHAP_SWARM_OUT = Path("reports/shap_summary_swarm.png")
RANDOM_STATE = 42
TARGET_RECALL = 0.90
N_JOBS = -1

def to_csr(x):
    """Конвертирует dense-матрицу (или DataFrame) в CSR-разреженный формат."""
    return sparse.csr_matrix(x)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lightgbm"], default="lightgbm")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--skip-shap", action="store_true", help="Не строить SHAP‑плоты")
    return p.parse_args()

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    df = pd.read_parquet(DATA_PATH)
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])
    groups = X.get("student_uuid")
    return X, y, groups

def get_estimator(params: dict) -> LGBMClassifier:
    return LGBMClassifier(
        **params,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        verbose=-1,  # без служебного вывода
    )

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, groups, prep) -> float:
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 400, 800),
        num_leaves=trial.suggest_int("num_leaves", 31, 255),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 3.0),
    )
    pipe = Pipeline([("prep", prep), ("clf", get_estimator(params))])
    cv = GroupKFold(n_splits=3)
    scores: List[float] = []
    for tr_idx, val_idx in cv.split(X, y, groups):
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[val_idx], p))
    return float(np.mean(scores))

def calibrate_threshold(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, groups) -> float:
    cv = GroupKFold(n_splits=3)
    y_true: list[int] = []
    proba: list[float] = []
    for tr_idx, val_idx in cv.split(X, y, groups):
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        proba.extend(pipe.predict_proba(X.iloc[val_idx])[:, 1])
        y_true.extend(y.iloc[val_idx])
    prec, rec, thr = precision_recall_curve(y_true, proba)
    idx = np.where(rec >= TARGET_RECALL)[0]
    return float(thr[idx[-1]]) if len(idx) else 0.5

def shap_analysis(model, X):
    import numpy as np
    import shap
    from scipy import sparse
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import OneHotEncoder

    preprocessor = None
    classifier = model
    try:
        preprocessor = model.named_steps['prep']
        classifier = model.named_steps['clf']
    except AttributeError:
        preprocessor = None

    X_processed = X
    if preprocessor is not None:
        X_processed = preprocessor.transform(X)
        if sparse.issparse(X_processed):
            X_processed = X_processed.toarray()

    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if transformer == 'passthrough':
            feature_names.extend(cols)
            continue
        if hasattr(transformer, 'named_steps'):
            for step_name, step_obj in transformer.named_steps.items():
                if isinstance(step_obj, OneHotEncoder):
                    ohe = step_obj
                    break
        else:
            ohe = transformer if isinstance(transformer, OneHotEncoder) else None

        if name == 'num':
               feature_names.extend(list(cols))
        elif name == 'cat':
            if ohe is not None:
                for col, categories in zip(cols, ohe.categories_):
                    for cat in categories:
                        feature_names.append(f"{col}={cat}")
            else:
                feature_names.extend(list(cols))
                
    explainer = shap.Explainer(classifier, X_processed, feature_names=feature_names)
    shap_values = explainer(X_processed)
    
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    if values.ndim == 3:
        values = values[:, :, 1]

    shap.summary_plot(values, features=X_processed, feature_names=feature_names, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_bar.png")
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    print("Loading data...")
    X, y, groups = load_data()
    prep = joblib.load(PREP_PATH)
    print(f"Tuning with {args.n_trials} trials…")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X, y, groups, prep), n_trials=args.n_trials, show_progress_bar=True)
    print("Best ROC-AUC:", study.best_value)
    print("Best params:", study.best_params)
    print("Calibrating threshold...")
    pipe_cal = Pipeline([("prep", prep), ("clf", get_estimator(study.best_params))])
    threshold = calibrate_threshold(pipe_cal, X, y, groups)
    print(f"Threshold = {threshold:.4f}")
    print("Fitting final model...")
    final_pipe = Pipeline([("prep", prep), ("clf", get_estimator(study.best_params))])
    final_pipe.fit(X, y)
    if not args.skip_shap:
        print("Running SHAP analysis...")
        shap_analysis(final_pipe, X.sample(min(5000, len(X)), random_state=RANDOM_STATE))
    MODEL_OUT.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(final_pipe, MODEL_OUT)
    with open(THRESH_OUT, "w") as f:
        json.dump({"threshold": threshold}, f)
    print("Done! Artifacts saved", MODEL_OUT, THRESH_OUT, ("+ SHAP plots" if not args.skip_shap else ""))