import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import shap
except ImportError as exc:  # pragma: no cover
    missing = getattr(exc, "name", "required package")
    raise SystemExit(
        f"Missing dependency '{missing}'. Install project requirements before running train.py."
    ) from exc
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml_system.config import Config
from ml_system.preprocess import (
    auto_feature_select,
    build_preprocessor,
    infer_task_type,
    optimize_memory,
    recommend_targets,
    split_features,
)
from ml_system.utils import create_version_dir, dataframe_schema, hash_dataframe, save_json, setup_logging

try:
    from scipy import sparse
except Exception:  # pragma: no cover
    sparse = None

try:  # LightGBM is optional; only needed for model_type=lgbm or model_type=stack.
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover
    LGBMClassifier = None


@dataclass
class Artifacts:
    pipeline: Pipeline
    feature_names: List[str]
    raw_features: List[str]
    task_type: str
    shap_background: Optional[pd.DataFrame]


class Trainer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.artifacts: Optional[Artifacts] = None
        self._shap_explainer = None

    def _build_lgbm_classifier(self) -> Any:
        if LGBMClassifier is None:  # pragma: no cover
            raise SystemExit("LightGBM is not installed. Install it with: pip install lightgbm")
        return LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

    def _build_stacking_classifier(self, num_classes: Optional[int]) -> Any:
        # Keep scope safe: support the common binary classification case used in this repo.
        if num_classes and num_classes > 2:
            raise ValueError("model_type=stack currently supports only binary classification.")

        estimators: List[tuple[str, Any]] = [
            (
                "xgb",
                XGBClassifier(
                    # Keep stacking reasonably fast: the base models are trained multiple times via CV.
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    tree_method="hist",
                    eval_metric="logloss",
                    booster=self.config.booster,
                    random_state=self.config.random_state,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    # NOTE: Some environments restrict joblib thread pools; safe default avoids PermissionError.
                    n_jobs=1,
                    random_state=self.config.random_state,
                ),
            ),
        ]

        if LGBMClassifier is not None:
            estimators.append(("lgbm", self._build_lgbm_classifier()))
        else:
            logging.warning("LightGBM not installed; stacking will use only XGBoost + RandomForest.")

        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000),
            stack_method="predict_proba",
            n_jobs=1,
            cv=3,
            passthrough=False,
        )

    def _build_model(self, task_type: str, num_classes: Optional[int]) -> Any:
        model_type = (self.config.model_type or "xgboost").lower()

        if task_type == "classification":
            if model_type in {"stack", "stacking"}:
                return self._build_stacking_classifier(num_classes=num_classes)
            if model_type == "xgboost":
                return XGBClassifier(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    tree_method="hist",
                    eval_metric="logloss",
                    booster=self.config.booster,
                    num_class=num_classes if num_classes and num_classes > 2 else None,
                )
            if model_type == "logreg":
                return LogisticRegression(max_iter=2000)
            if model_type == "rf":
                return RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    # NOTE: Some environments restrict joblib thread pools; safe default avoids PermissionError.
                    n_jobs=1,
                    random_state=self.config.random_state,
                )
            if model_type == "lgbm":
                return self._build_lgbm_classifier()
            if model_type == "svm":
                # RBF SVC can be very slow on larger datasets. Use a calibrated linear SVM so we still have predict_proba.
                base = LinearSVC(C=0.5, max_iter=2000, dual="auto", random_state=self.config.random_state)
                # Calibrated probabilities (needed for AUC + API output). Keep CV small for speed.
                return CalibratedClassifierCV(base, method="sigmoid", cv=2)
            if model_type == "mlp":
                return MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=400,
                    early_stopping=True,
                    random_state=self.config.random_state,
                )
            raise ValueError(f"Unsupported model_type for classification: {model_type}")

        # Regression / multi-target regression (kept mostly for backward compatibility).
        if model_type != "xgboost":
            # If you want non-XGBoost regression later, we can add it, but keep scope safe for now.
            raise ValueError(
                f"model_type={model_type} is currently only supported for classification. "
                "Use --model-type xgboost for regression."
            )

        base = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="rmse",
            booster=self.config.booster,
        )
        if task_type == "multi_regression":
            return MultiOutputRegressor(base)
        return base

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        tune: bool,
        tune_iter: int,
        cv_folds: int,
        auto_select: bool,
        group_col: Optional[str] = None,
        cardio_minimal: bool = False,
        cardio_age_bmi: bool = False,
    ) -> Dict[str, Any]:
        df = optimize_memory(df)

        # Prevent accidental leakage if a user passes prediction outputs back into training data.
        leakage_cols = {"prediction", "probability", "prediction_thresholded"}
        drop_leakage = [c for c in leakage_cols if c in df.columns and c != target_col]
        if drop_leakage:
            logging.warning("Dropping leakage columns: %s", drop_leakage)
            df = df.drop(columns=drop_leakage)
        if cardio_minimal and cardio_age_bmi:
            raise SystemExit("Use only one: --cardio-minimal or --cardio-age-bmi")
        if cardio_age_bmi:
            df = self._prepare_cardio_age_bmi(df, target_col=target_col)
        elif cardio_minimal:
            df = self._prepare_cardio_minimal(df, target_col=target_col)

        feature_report = {}
        if auto_select:
            df, feature_report = auto_feature_select(df, target_col)
        groups = None
        if group_col:
            if group_col not in df.columns:
                raise KeyError(f"Group column '{group_col}' not found.")
            groups = df[group_col].copy()
            df = df.drop(columns=[group_col])
        if "," in target_col:
            target_cols = [c.strip() for c in target_col.split(",") if c.strip()]
        else:
            target_cols = [target_col]
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Target columns not found: {', '.join(missing)}")
        if len(target_cols) > 1:
            X = df.drop(columns=target_cols)
            y = df[target_cols]
            numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
            categorical_cols = [c for c in X.columns if c not in numeric_cols]
        else:
            X, y, numeric_cols, categorical_cols = split_features(df, target_cols[0])

        if len(target_cols) > 1:
            task_type = "multi_regression"
        else:
            task_type = infer_task_type(y)
        num_classes = int(y.nunique()) if task_type == "classification" else None

        preprocessor = build_preprocessor(
            numeric_cols,
            categorical_cols,
            normalize_numeric=self.config.normalize_numeric,
            booster=self.config.booster,
        )
        model = self._build_model(task_type, num_classes=num_classes)

        if groups is not None:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            train_idx, test_idx = next(splitter.split(X, y, groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            groups_train = groups.iloc[train_idx]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y if task_type == "classification" else None,
            )
            groups_train = None

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        if tune:
            model_type = (self.config.model_type or "xgboost").lower()
            if model_type == "xgboost":
                param_space = {
                    "model__n_estimators": [200, 400, 600],
                    "model__max_depth": [3, 5, 7, 9],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__subsample": [0.7, 0.8, 1.0],
                    "model__colsample_bytree": [0.7, 0.8, 1.0],
                }
            elif model_type == "lgbm":
                param_space = {
                    "model__n_estimators": [200, 400, 800],
                    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                    "model__num_leaves": [15, 31, 63, 127],
                    "model__min_child_samples": [10, 20, 40],
                    "model__subsample": [0.7, 0.8, 1.0],
                    "model__colsample_bytree": [0.7, 0.8, 1.0],
                }
            elif model_type == "logreg":
                param_space = {
                    "model__C": [0.1, 0.3, 1.0, 3.0, 10.0],
                    "model__penalty": ["l2"],
                    "model__solver": ["lbfgs", "liblinear"],
                }
            elif model_type == "rf":
                param_space = {
                    "model__n_estimators": [300, 600, 900],
                    "model__max_depth": [None, 8, 12, 16],
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", "log2", None],
                }
            elif model_type == "svm":
                param_space = {
                    "model__C": [0.5, 1.0, 2.0, 5.0, 10.0],
                    "model__gamma": ["scale", "auto"],
                    "model__kernel": ["rbf"],
                }
            elif model_type == "mlp":
                param_space = {
                    "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
                    "model__alpha": [1e-5, 1e-4, 1e-3],
                    "model__learning_rate_init": [1e-4, 5e-4, 1e-3],
                }
            else:
                logging.warning("Tuning not configured for model_type=%s. Skipping tuning.", model_type)
                param_space = {}

            if task_type == "classification" and y.nunique(dropna=True) == 2 and hasattr(pipeline, "predict_proba"):
                scoring = "roc_auc"
            else:
                scoring = "f1_weighted" if task_type == "classification" else "neg_mean_squared_error"
            try:
                import optuna  # type: ignore
                from optuna.integration import OptunaSearchCV  # type: ignore

                if param_space:
                    search = OptunaSearchCV(
                        pipeline,
                        param_space,
                        n_trials=tune_iter,
                        scoring=scoring,
                        cv=GroupKFold(3) if groups_train is not None else 3,
                        n_jobs=1,
                        random_state=self.config.random_state,
                    )
                    if groups_train is not None:
                        search.fit(X_train, y_train, groups=groups_train)
                    else:
                        search.fit(X_train, y_train)
                    pipeline = search.best_estimator_
            except Exception:
                if param_space:
                    search = RandomizedSearchCV(
                        pipeline,
                        param_distributions=param_space,
                        n_iter=tune_iter,
                        scoring=scoring,
                        cv=GroupKFold(3) if groups_train is not None else 3,
                        n_jobs=1,
                        random_state=self.config.random_state,
                    )
                    if groups_train is not None:
                        search.fit(X_train, y_train, groups=groups_train)
                    else:
                        search.fit(X_train, y_train)
                    pipeline = search.best_estimator_
        else:
            pipeline.fit(X_train, y_train)

        feature_names = (
            pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        )

        background_sample = X_train.sample(
            n=min(self.config.shap_max_samples, len(X_train)),
            random_state=self.config.random_state,
        )

        self.artifacts = Artifacts(
            pipeline=pipeline,
            feature_names=feature_names,
            raw_features=X.columns.tolist(),
            task_type=task_type,
            shap_background=background_sample,
        )

        metrics = self.evaluate(X_test, y_test)
        metrics["feature_selection"] = feature_report

        if cv_folds >= 3:
            cv_scores = self.cross_validate(pipeline, X, y, task_type, cv_folds, groups)
            metrics["cross_validation"] = cv_scores
        return metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        if not self.artifacts:
            raise RuntimeError("Model has not been trained yet.")

        pipeline = self.artifacts.pipeline
        task_type = self.artifacts.task_type
        y_pred = pipeline.predict(X_test)

        results: Dict[str, Any] = {
            "task_type": task_type,
            "model_type": (self.config.model_type or "xgboost"),
            "normalize_numeric": bool(self.config.normalize_numeric),
        }

        if task_type == "classification":
            results["accuracy"] = float(accuracy_score(y_test, y_pred))
            results["precision"] = float(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            results["recall"] = float(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            results["f1_score"] = float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            )

            # Binary-focused metrics (important for class=1 screening recall).
            try:
                results["precision_pos"] = float(
                    precision_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0)
                )
                results["recall_pos"] = float(
                    recall_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0)
                )
                results["f1_pos"] = float(
                    f1_score(y_test, y_pred, pos_label=1, average="binary", zero_division=0)
                )
            except Exception:
                pass

            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)
                if y_prob.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                    auc_score = roc_auc_score(y_test, y_prob[:, 1])
                    results["roc_auc"] = float(auc_score)
                    self._plot_roc_curve(fpr, tpr, auc_score)
                    results["threshold_sweep"] = self._threshold_sweep(y_test, y_prob[:, 1])
        else:
            if task_type == "multi_regression":
                results["targets"] = {}
                for idx, col in enumerate(y_test.columns):
                    y_true = y_test.iloc[:, idx]
                    y_hat = y_pred[:, idx]
                    results["targets"][col] = {
                        "mae": float(mean_absolute_error(y_true, y_hat)),
                        "mse": float(mean_squared_error(y_true, y_hat)),
                        "rmse": float(np.sqrt(mean_squared_error(y_true, y_hat))),
                        "r2": float(r2_score(y_true, y_hat)),
                    }
            else:
                results["mae"] = float(mean_absolute_error(y_test, y_pred))
                results["mse"] = float(mean_squared_error(y_test, y_pred))
                results["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                results["r2"] = float(r2_score(y_test, y_pred))

        return results

    def cross_validate(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        cv_folds: int,
        groups: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        if task_type == "classification":
            if y.nunique(dropna=True) == 2:
                scoring = {
                    "accuracy": "accuracy",
                    "precision": "precision",
                    "recall": "recall",
                    "f1": "f1",
                    "roc_auc": "roc_auc",
                }
            else:
                scoring = {"accuracy": "accuracy", "f1": "f1_weighted"}
        else:
            scoring = {"mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error", "r2": "r2"}
        cv_strategy = GroupKFold(cv_folds) if groups is not None else cv_folds
        if task_type == "multi_regression":
            raise ValueError("Cross-validation is not supported for multi-target regression.")
        cv = cross_validate(
            pipeline,
            X,
            y,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=1,
            groups=groups if groups is not None else None,
        )
        results = {}
        for key, values in cv.items():
            if key.startswith("test_"):
                metric = key.replace("test_", "")
                mean_val = float(np.mean(values))
                if metric in {"mae", "mse"}:
                    mean_val = abs(mean_val)
                results[f"{metric}_mean"] = mean_val
        if "mse_mean" in results:
            results["rmse_mean"] = float(np.sqrt(results["mse_mean"]))
        return results

    def _prepare_cardio_minimal(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Cardio minimal feature set + feature engineering.

        Uses only:
        weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active + target cardio

        Adds engineered features:
        - pulse_pressure = ap_hi - ap_lo
        - MAP = (ap_hi + 2 * ap_lo) / 3
        - hypertension = 1 if ap_hi >= 140 or ap_lo >= 90 else 0
        - cholesterol_high = 1 if cholesterol > 1 else 0
        - gluc_high = 1 if gluc > 1 else 0

        Removes invalid BP rows (examples):
        - ap_hi < ap_lo
        - ap_hi > 250
        - ap_lo < 40
        """
        required = ["weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
        missing = [c for c in required + [target_col] if c not in df.columns]
        if missing:
            raise SystemExit(f"--cardio-minimal requires missing columns: {missing}")

        out = df[required + [target_col]].copy()

        # Coerce numeric safely (handles strings, blanks, etc.).
        for c in required + [target_col]:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        before = len(out)
        out = out[
            (out["ap_hi"].notna())
            & (out["ap_lo"].notna())
            & (out["ap_hi"] >= out["ap_lo"])
            & (out["ap_hi"] <= 250)
            & (out["ap_lo"] >= 40)
        ].copy()
        dropped = before - len(out)
        if dropped > 0:
            logging.info("Cardio minimal: dropped %s invalid BP rows.", dropped)

        # Drop missing target if any.
        out = out[out[target_col].notna()].copy()

        out["pulse_pressure"] = out["ap_hi"] - out["ap_lo"]
        out["MAP"] = (out["ap_hi"] + 2 * out["ap_lo"]) / 3.0
        out["hypertension"] = ((out["ap_hi"] >= 140) | (out["ap_lo"] >= 90)).astype(int)
        out["cholesterol_high"] = (out["cholesterol"] > 1).astype(int)
        out["gluc_high"] = (out["gluc"] > 1).astype(int)

        return out

    def _prepare_cardio_age_bmi(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Cardio dataset with strong emphasis on Age + BMI (plus BP and risk factors).

        Required base columns:
        age, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, + target.

        Engineering:
        - age_years (auto converts if age looks like days)
        - BMI
        - pulse_pressure, map, bp_ratio
        - risk_score (simple additive lifestyle/metabolic risk proxy)
        - age_group (young/middle/old)
        - bp_stage category (Normal / Elevated / Stage1 / Stage2 / Crisis)
        - interactions: age_years*ap_hi, BMI*ap_hi, ap_hi*cholesterol, BMI*gluc
        - risk flags: hypertension, obesity, high_cholesterol, high_glucose

        Notes:
        - Drops any model-output leakage columns earlier in train().
        - Treats cholesterol/gluc as categorical (not continuous) to avoid linear misuse.
        """
        required = ["age", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
        missing = [c for c in required + [target_col] if c not in df.columns]
        if missing:
            raise SystemExit(f"--cardio-age-bmi requires missing columns: {missing}")

        out = df[required + [target_col]].copy()
        for c in required + [target_col]:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        before = len(out)

        # If ap_hi/ap_lo appear swapped, fix it instead of dropping the row.
        swapped_mask = out["ap_hi"].notna() & out["ap_lo"].notna() & (out["ap_hi"] < out["ap_lo"])
        if swapped_mask.any():
            hi = out.loc[swapped_mask, "ap_hi"].copy()
            out.loc[swapped_mask, "ap_hi"] = out.loc[swapped_mask, "ap_lo"]
            out.loc[swapped_mask, "ap_lo"] = hi

        out = out[
            out[target_col].notna()
            & out["age"].notna()
            & out["height"].notna()
            & out["weight"].notna()
            & out["ap_hi"].notna()
            & out["ap_lo"].notna()
            & (out["ap_hi"] > out["ap_lo"])
            & (out["ap_hi"] <= 250)
            & (out["ap_lo"] >= 40)
            & (out["ap_lo"] <= 160)
            & (out["height"] >= 120)
            & (out["height"] <= 220)
            & (out["weight"] >= 30)
            & (out["weight"] <= 250)
        ].copy()
        dropped = before - len(out)
        if dropped > 0:
            logging.info("Cardio age+bmi: dropped %s invalid rows.", dropped)

        # Force float dtype (pandas may downcast to int, which breaks in-place float assignment).
        age_years = out["age"].astype(float)
        # age is typically stored in days in this dataset; if values look like days, convert to years.
        age_years = np.where(age_years > 200, age_years / 365.25, age_years)
        out["age_years"] = age_years.astype(float)

        height_m = out["height"] / 100.0
        out["BMI"] = out["weight"] / (height_m * height_m)
        out["pulse_pressure"] = out["ap_hi"] - out["ap_lo"]
        out["map"] = (out["ap_hi"] + 2.0 * out["ap_lo"]) / 3.0
        out["bp_ratio"] = out["ap_hi"] / (out["ap_lo"] + 1.0)
        # Simple additive proxy. Kept intentionally transparent for thesis writing.
        out["risk_score"] = (
            out["smoke"].fillna(0)
            + out["alco"].fillna(0)
            + (out["cholesterol"].fillna(1) - 1)
            + (out["gluc"].fillna(1) - 1)
        )

        before2 = len(out)
        out = out[(out["BMI"] >= 15) & (out["BMI"] <= 60)].copy()
        dropped2 = before2 - len(out)
        if dropped2 > 0:
            logging.info("Cardio age+bmi: dropped %s BMI outliers.", dropped2)

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

        # Interactions: explicit multiplicative terms can help linear models, and sometimes boost trees too.
        out["age_x_ap_hi"] = out["age_years"] * out["ap_hi"]
        out["bmi_x_ap_hi"] = out["BMI"] * out["ap_hi"]
        out["chol_x_ap_hi"] = out["cholesterol"] * out["ap_hi"]
        out["gluc_x_bmi"] = out["gluc"] * out["BMI"]

        out["hypertension"] = ((out["ap_hi"] >= 140) | (out["ap_lo"] >= 90)).astype(int)
        out["obesity"] = (out["BMI"] >= 30).astype(int)
        out["high_cholesterol"] = (out["cholesterol"] > 1).astype(int)
        out["high_glucose"] = (out["gluc"] > 1).astype(int)

        # Treat these as categorical (not linear continuous). The preprocessor will one-hot encode them.
        out["cholesterol"] = out["cholesterol"].round().astype("Int64").astype("category")
        out["gluc"] = out["gluc"].round().astype("Int64").astype("category")

        keep = [
            "age_years",
            "age_group",
            "BMI",
            "height",
            "weight",
            "ap_hi",
            "ap_lo",
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "age_x_ap_hi",
            "bmi_x_ap_hi",
            "chol_x_ap_hi",
            "gluc_x_bmi",
            "pulse_pressure",
            "map",
            "bp_ratio",
            "risk_score",
            "bp_stage",
            "hypertension",
            "obesity",
            "high_cholesterol",
            "high_glucose",
            target_col,
        ]
        return out[keep].copy()

    def _plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc_score: float) -> None:
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, linewidth=2.5, color="#1f77b4", label=f"AUC = {auc_score:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Binary Classification)")
        plt.legend(loc="lower right", frameon=True)
        sns.despine()
        plt.tight_layout()
        plt.show()

    def _threshold_sweep(self, y_true: Any, prob_pos: Any) -> Dict[str, Any]:
        """
        Sweep thresholds (with emphasis on 0.30-0.45) and report best thresholds for:
        - recall_pos (maximize recall for class 1)
        - f1_pos
        """
        y_true_arr = np.asarray(y_true).astype(int)
        prob_arr = np.asarray(prob_pos).astype(float)
        thresholds = np.linspace(0.25, 0.55, 31)

        best_recall = {"threshold": 0.5, "recall_pos": -1.0, "precision_pos": 0.0, "f1_pos": 0.0}
        best_f1 = {"threshold": 0.5, "f1_pos": -1.0, "precision_pos": 0.0, "recall_pos": 0.0}

        for t in thresholds:
            pred = (prob_arr >= float(t)).astype(int)
            prec = precision_score(y_true_arr, pred, pos_label=1, average="binary", zero_division=0)
            rec = recall_score(y_true_arr, pred, pos_label=1, average="binary", zero_division=0)
            f1v = f1_score(y_true_arr, pred, pos_label=1, average="binary", zero_division=0)

            if rec > best_recall["recall_pos"]:
                best_recall = {
                    "threshold": float(t),
                    "recall_pos": float(rec),
                    "precision_pos": float(prec),
                    "f1_pos": float(f1v),
                }
            if f1v > best_f1["f1_pos"]:
                best_f1 = {
                    "threshold": float(t),
                    "f1_pos": float(f1v),
                    "precision_pos": float(prec),
                    "recall_pos": float(rec),
                }

        return {"best_recall_pos": best_recall, "best_f1_pos": best_f1}

    def _init_shap(self, X_sample: pd.DataFrame) -> None:
        if not self.artifacts:
            raise RuntimeError("Model has not been trained yet.")
        pipeline = self.artifacts.pipeline
        model = pipeline.named_steps["model"]
        X_transformed = pipeline.named_steps["preprocessor"].transform(X_sample)
        if sparse is not None and sparse.issparse(X_transformed):
            X_transformed = X_transformed.toarray()
        self._shap_explainer = shap.TreeExplainer(model, data=X_transformed)

    def shap_global_importance(self, X: pd.DataFrame, out_dir: Optional[Path] = None) -> None:
        if not self.artifacts:
            raise RuntimeError("Model has not been trained yet.")

        X_sample = X.sample(
            n=min(self.config.shap_max_samples, len(X)),
            random_state=self.config.random_state,
        )
        self._init_shap(X_sample)

        pipeline = self.artifacts.pipeline
        X_transformed = pipeline.named_steps["preprocessor"].transform(X_sample)
        if sparse is not None and sparse.issparse(X_transformed):
            X_transformed = X_transformed.toarray()
        shap_values = self._shap_explainer(X_transformed)

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            shap_values,
            features=X_transformed,
            feature_names=self.artifacts.feature_names,
            plot_type="bar",
            show=False,
        )
        plt.title("Global Feature Importance (SHAP)")
        plt.tight_layout()
        if out_dir is not None:
            plt.savefig(out_dir / "shap_global_importance_bar.png", dpi=180)
        plt.close()

        plt.figure(figsize=(9, 6))
        shap.summary_plot(
            shap_values,
            features=X_transformed,
            feature_names=self.artifacts.feature_names,
            show=False,
        )
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        if out_dir is not None:
            plt.savefig(out_dir / "shap_summary.png", dpi=180)
        plt.close()

    def save(self, path: Path, metadata: Dict[str, Any]) -> Path:
        if not self.artifacts:
            raise RuntimeError("Model has not been trained yet.")
        payload = {
            "pipeline": self.artifacts.pipeline,
            "feature_names": self.artifacts.feature_names,
            "raw_features": self.artifacts.raw_features,
            "task_type": self.artifacts.task_type,
            "shap_background": self.artifacts.shap_background,
            "metadata": metadata,
            "schema": metadata.get("schema", {}),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)
        return path


def _prompt_target_column(columns: List[str]) -> str:
    print("Columns detected:")
    for idx, col in enumerate(columns, start=1):
        print(f"{idx}. {col}")
    while True:
        target = input("Enter the target column name: ").strip()
        if target in columns:
            return target
        print("Invalid column name. Please try again.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a model on a CSV file (XGBoost / Logistic Regression / Random Forest / SVM / MLP / LightGBM / Stacking)."
    )
    parser.add_argument("csv_path", help="Path to the input CSV file.")
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: ,). Use ';' for cardio_train.csv.",
    )
    parser.add_argument("--target", help="Target column name (skip prompt).")
    parser.add_argument(
        "--group-col",
        dest="group_col",
        help="Group column for leakage-safe splitting (e.g., uid).",
    )
    parser.add_argument(
        "--drop-cols",
        dest="drop_cols",
        help="Comma-separated columns to drop before training.",
    )
    parser.add_argument("--save-model", dest="save_model", help="Path to save model.")
    parser.add_argument(
        "--model-type",
        default="xgboost",
        choices=["xgboost", "logreg", "rf", "svm", "mlp", "lgbm", "stack"],
        help="Model family to train (default: xgboost).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize numeric features. Recommended for logreg/svm/mlp.",
    )
    parser.add_argument(
        "--cardio-minimal",
        action="store_true",
        help=(
            "Use a minimal cardio feature set (weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active) "
            "+ engineered features (pulse_pressure, MAP, hypertension, cholesterol_high, gluc_high). "
            "Also drops leakage columns like prediction/probability if present."
        ),
    )
    parser.add_argument(
        "--cardio-age-bmi",
        action="store_true",
        help=(
            "Use cardio features with emphasis on Age + BMI. Adds: age_years, BMI, age_group, interactions, and risk flags. "
            "Removes invalid BP / BMI outliers and drops leakage columns if present."
        ),
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Generate SHAP plots after training (can be slow). Saved under the model version folder.",
    )
    parser.add_argument(
        "--shap-max-samples",
        type=int,
        default=5000,
        help="Max rows to sample for SHAP (default: 5000).",
    )
    parser.add_argument(
        "--booster",
        default="gbtree",
        choices=["gbtree", "gblinear", "dart"],
        help="XGBoost booster to use.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable basic hyperparameter tuning.",
    )
    parser.add_argument(
        "--tune-iter",
        type=int,
        default=15,
        help="Number of random search iterations (only with --tune).",
    )
    parser.add_argument(
        "--auto-target",
        action="store_true",
        help="Auto-pick a recommended target column.",
    )
    parser.add_argument(
        "--auto-features",
        action="store_true",
        help="Drop low-variance, high-missing, or duplicate columns.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="Enable k-fold cross-validation (e.g., 5).",
    )
    return parser


def _parse_target_cols(target: str) -> List[str]:
    if "," in target:
        return [c.strip() for c in target.split(",") if c.strip()]
    return [target]


def main() -> None:
    setup_logging()
    sns.set_theme(style="whitegrid")

    args = build_arg_parser().parse_args()

    # Some model families need scaling to behave correctly.
    needs_scaling = args.model_type in {"logreg", "svm", "mlp"}
    config = Config(
        data_path=Path(args.csv_path),
        target=args.target,
        normalize_numeric=args.normalize or needs_scaling,
        model_type=args.model_type,
        booster=args.booster,
        shap_max_samples=args.shap_max_samples,
    )

    df = pd.read_csv(config.data_path, low_memory=False, delimiter=args.delimiter)
    if args.auto_target:
        candidates = recommend_targets(df)
        if not candidates:
            raise SystemExit("No suitable target columns found.")
        target_col = candidates[0][0]
        logging.info("Auto target selected: %s", target_col)
    else:
        target_col = config.target or _prompt_target_column(df.columns.tolist())

    if args.drop_cols:
        drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
        missing = [c for c in drop_cols if c not in df.columns]
        if missing:
            logging.warning("Drop columns not found: %s", ", ".join(missing))
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    trainer = Trainer(config)
    metrics = trainer.train(
        df,
        target_col,
        tune=args.tune,
        tune_iter=args.tune_iter,
        cv_folds=args.cv,
        auto_select=args.auto_features,
        group_col=args.group_col,
        cardio_minimal=args.cardio_minimal,
        cardio_age_bmi=args.cardio_age_bmi,
    )
    logging.info("Model performance:")
    logging.info(metrics)

    target_cols = _parse_target_cols(target_col)
    data_hash = hash_dataframe(df)
    schema = dataframe_schema(df)
    version_dir = create_version_dir(Path("ml_system/models"))
    metadata = {
        "target": target_cols if len(target_cols) > 1 else target_col,
        "data_hash": data_hash,
        "rows": len(df),
        "columns": len(df.columns),
        "metrics": metrics,
        "model_type": config.model_type,
        "schema": schema,
        "group_col": args.group_col,
        "drop_cols": drop_cols if args.drop_cols else None,
    }

    model_path = Path(args.save_model) if args.save_model else config.model_path
    trainer.save(model_path, metadata)
    save_json(config.metrics_path, metrics)
    save_json(version_dir / "metadata.json", metadata)
    logging.info("Saved model to: %s", model_path)
    logging.info("Saved metrics to: %s", config.metrics_path)
    logging.info("Saved model version to: %s", version_dir)

    # SHAP is optional (and can be slow). We save the model first so you can always stop SHAP safely.
    if args.shap:
        if len(target_cols) == 1:
            logging.info("Generating SHAP plots (this can be slow)...")
            trainer.shap_global_importance(df.drop(columns=target_cols), out_dir=version_dir)
            logging.info("Saved SHAP plots under: %s", version_dir)
        else:
            logging.info("Skipping SHAP for multi-target regression.")
    else:
        logging.info("Skipping SHAP (use --shap to generate SHAP plots).")


if __name__ == "__main__":
    main()
