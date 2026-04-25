"""
Medical-feature-focused training for the Cardio dataset.

Why this exists:
- The generic `ml_system.train` path supports many datasets.
- This script is intentionally opinionated for medical tabular data and adds:
  - leakage column removal
  - robust BP sanity checks
  - Age (days->years) conversion
  - medically meaningful feature engineering (BMI, MAP, pulse pressure, risk flags)
  - categorical handling for ordinal labs (cholesterol/gluc as categories, not linear)
  - StratifiedKFold ROC-AUC evaluation
  - optional hyperparameter tuning (Optuna if installed, otherwise RandomizedSearchCV)

Usage (recommended):
  ..\\r1\\Scripts\\python.exe -m ml_system.cardio_medical_train ml_system\\data\\cardio_train_clean.csv --delimiter ";"

Optional tuning:
  ..\\r1\\Scripts\\python.exe -m ml_system.cardio_medical_train ml_system\\data\\cardio_train_clean.csv --delimiter ";" --tune --tune-iter 50
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from lightgbm import LGBMClassifier
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'lightgbm'. Install requirements.txt before running this script."
    ) from exc


LEAKAGE_COLS = {"prediction", "probability", "prediction_thresholded"}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a medical-feature-focused LightGBM cardio model (ROC-AUC CV).")
    p.add_argument("csv_path", help="Path to cardio CSV (can include leakage columns; they will be dropped).")
    p.add_argument("--delimiter", default=";", help="CSV delimiter (default: ';').")
    p.add_argument("--target", default="cardio", help="Target column (default: cardio).")
    p.add_argument("--drop-cols", default="id", help="Comma-separated columns to drop (default: id).")
    p.add_argument("--cv", type=int, default=5, help="Stratified CV folds (default: 5).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--tune", action="store_true", help="Tune hyperparameters (Optuna if available, else random search).")
    p.add_argument("--tune-iter", type=int, default=50, help="Number of tuning trials/iterations (default: 50).")
    p.add_argument("--save-model", default="ml_system/models/cardio_lgbm_medical.joblib", help="Where to save model.")
    p.add_argument("--save-metrics", default="ml_system/models/cardio_lgbm_medical.metrics.json", help="Where to save metrics.")
    return p


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def clean_and_engineer_cardio(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Clean and engineer medically meaningful features from the raw cardio schema.

    Returns a dataframe that includes engineered columns and the target column.
    """
    required = ["age", "gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
    missing = [c for c in required + [target] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    out = df[required + [target]].copy()
    out = _coerce_numeric(out, required + [target])

    # Fix common BP data issue where systolic/diastolic were swapped.
    swapped = out["ap_hi"].notna() & out["ap_lo"].notna() & (out["ap_hi"] < out["ap_lo"])
    if swapped.any():
        hi = out.loc[swapped, "ap_hi"].copy()
        out.loc[swapped, "ap_hi"] = out.loc[swapped, "ap_lo"]
        out.loc[swapped, "ap_lo"] = hi

    # Age conversion: source dataset stores age in days (typically ~10k-30k). Convert to years.
    age_raw = out["age"].astype(float)
    out["age_years"] = np.where(age_raw > 200, age_raw / 365.25, age_raw).astype(float)

    # Basic physiological constraints / outlier cleanup.
    before = len(out)
    out = out[
        out[target].notna()
        & out["age_years"].between(18, 100)
        & out["height"].between(120, 220)
        & out["weight"].between(30, 250)
        & out["ap_hi"].between(70, 250)
        & out["ap_lo"].between(40, 160)
        & (out["ap_hi"] > out["ap_lo"])
    ].copy()
    logging.info("Dropped %s rows by basic physiological constraints.", before - len(out))

    # Medical features.
    height_m = out["height"] / 100.0
    out["bmi"] = out["weight"] / (height_m * height_m)
    out["pulse_pressure"] = out["ap_hi"] - out["ap_lo"]
    out["map"] = (out["ap_hi"] + 2.0 * out["ap_lo"]) / 3.0
    out["bp_ratio"] = out["ap_hi"] / (out["ap_lo"] + 1.0)

    before2 = len(out)
    out = out[out["bmi"].between(15, 60)].copy()
    logging.info("Dropped %s rows by BMI constraints.", before2 - len(out))

    # Risk flags / groupings (interpretable for thesis writing).
    out["obesity"] = (out["bmi"] >= 30).astype(int)
    out["hypertension"] = ((out["ap_hi"] >= 140) | (out["ap_lo"] >= 90)).astype(int)
    out["high_cholesterol"] = (out["cholesterol"] > 1).astype(int)
    out["high_glucose"] = (out["gluc"] > 1).astype(int)

    out["age_group"] = pd.cut(
        out["age_years"],
        bins=[-float("inf"), 40, 60, float("inf")],
        labels=["young", "middle", "old"],
    ).astype("category")
    out["bp_stage"] = pd.cut(
        out["ap_hi"],
        bins=[-float("inf"), 120, 130, 140, 180, float("inf")],
        labels=["normal", "elevated", "stage1", "stage2", "crisis"],
    ).astype("category")

    # Treat ordinal lab levels as categorical to avoid linear misuse.
    out["cholesterol"] = out["cholesterol"].round().astype("Int64").astype("category")
    out["gluc"] = out["gluc"].round().astype("Int64").astype("category")
    out["gender"] = out["gender"].round().astype("Int64").astype("category")

    # Simple additive risk proxy (transparent, not "magic").
    out["risk_score"] = (
        out["smoke"].fillna(0)
        + out["alco"].fillna(0)
        + (out["cholesterol"].cat.codes.replace(-1, 0))  # 0/1/2 for 1/2/3
        + (out["gluc"].cat.codes.replace(-1, 0))
    ).astype(float)

    # Interactions (can help linear models; sometimes helps boosting as well).
    out["age_x_ap_hi"] = out["age_years"] * out["ap_hi"]
    out["bmi_x_ap_hi"] = out["bmi"] * out["ap_hi"]
    out["chol_x_ap_hi"] = out["cholesterol"].cat.codes.replace(-1, 0).astype(float) * out["ap_hi"]
    out["gluc_x_bmi"] = out["gluc"].cat.codes.replace(-1, 0).astype(float) * out["bmi"]

    return out


def build_pipeline(X: pd.DataFrame, seed: int) -> Pipeline:
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                cat_cols,
            ),
        ],
        sparse_threshold=0.3,
    )

    model = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,  # Windows-safe (avoids multiprocessing permission errors in some setups)
    )

    return Pipeline([("preprocessor", pre), ("model", model)])


def _cv_auc(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> Tuple[float, float]:
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(X_tr, y_tr)
        prob = pipeline.predict_proba(X_te)[:, 1]
        aucs.append(float(roc_auc_score(y_te, prob)))
    return float(np.mean(aucs)), float(np.std(aucs))


def tune_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold, seed: int, n_iter: int) -> Pipeline:
    # Try OptunaSearchCV if optuna is installed; otherwise fall back to RandomizedSearchCV.
    param_space: Dict[str, Any] = {
        "model__n_estimators": [300, 600, 900, 1200],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.08],
        "model__num_leaves": [31, 63, 127, 255],
        "model__min_child_samples": [10, 20, 40, 80],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
    }

    try:
        import optuna  # type: ignore
        from optuna.integration import OptunaSearchCV  # type: ignore

        _ = optuna  # silence lint
        search = OptunaSearchCV(
            pipeline,
            param_space,
            n_trials=int(n_iter),
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            random_state=seed,
        )
        search.fit(X, y)
        logging.info("Optuna best params: %s", search.best_params_)
        return search.best_estimator_
    except Exception:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_space,
            n_iter=int(n_iter),
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            random_state=seed,
        )
        search.fit(X, y)
        logging.info("Random search best params: %s", search.best_params_)
        return search.best_estimator_


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    df = pd.read_csv(args.csv_path, delimiter=str(args.delimiter), low_memory=False)

    # Drop known leakage columns.
    leak = [c for c in LEAKAGE_COLS if c in df.columns and c != args.target]
    if leak:
        logging.warning("Dropping leakage columns: %s", leak)
        df = df.drop(columns=leak)

    drop_cols = [c.strip() for c in (args.drop_cols or "").split(",") if c.strip()]
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df = clean_and_engineer_cardio(df, target=str(args.target))
    y = df[str(args.target)].astype(int)
    X = df.drop(columns=[str(args.target)])

    cv = StratifiedKFold(n_splits=int(args.cv), shuffle=True, random_state=int(args.seed))
    pipe = build_pipeline(X, seed=int(args.seed))

    if bool(args.tune):
        pipe = tune_pipeline(pipe, X, y, cv=cv, seed=int(args.seed), n_iter=int(args.tune_iter))

    mean_auc, std_auc = _cv_auc(pipe, X, y, cv=cv)
    metrics = {
        "task_type": "classification",
        "model_type": "lgbm_medical",
        "roc_auc_cv_mean": float(mean_auc),
        "roc_auc_cv_std": float(std_auc),
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "tuned": bool(args.tune),
        "notes": "This evaluation uses leakage-safe pipeline-in-CV. Cholesterol/gluc treated as categorical (one-hot).",
    }

    Path(args.save_metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)

    # Fit final model on all data for saving.
    pipe.fit(X, y)
    import joblib

    joblib.dump({"pipeline": pipe, "metrics": metrics}, args.save_model)
    Path(args.save_metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logging.info("Saved model: %s", args.save_model)
    logging.info("Saved metrics: %s", args.save_metrics)
    logging.info("ROC-AUC (CV mean ± std): %.4f ± %.4f", mean_auc, std_auc)


if __name__ == "__main__":
    main()

