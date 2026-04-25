import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier


DEFAULT_XPT_PATH = Path(r"C:\Users\Sajjad Alam PC\Desktop\R1\project1\BPX_J.xpt")


def load_xpt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_sas(path, format="xport")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    required = ["SEQN", "BPXSY1", "BPXDI1"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required].copy()
    df = df.dropna(subset=["BPXSY1", "BPXDI1"]).copy()

    df["pulse_pressure"] = df["BPXSY1"] - df["BPXDI1"]
    df["mean_arterial_pressure"] = (2 * df["BPXDI1"] + df["BPXSY1"]) / 3.0

    df["hypertension"] = np.where(
        (df["BPXSY1"] >= 140) | (df["BPXDI1"] >= 90), 1, 0
    )

    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_cols = ["BPXSY1", "BPXDI1", "pulse_pressure", "mean_arterial_pressure"]
    X = df[feature_cols].copy()
    y = df["hypertension"].astype(int)
    return X, y


def leakage_checks(df: pd.DataFrame, feature_cols: List[str]) -> None:
    rule_based = np.where((df["BPXSY1"] >= 140) | (df["BPXDI1"] >= 90), 1, 0)
    if "hypertension" in df.columns:
        match_ratio = float((df["hypertension"].values == rule_based).mean())
        if match_ratio == 1.0 and any(
            col in feature_cols
            for col in ["BPXSY1", "BPXDI1", "pulse_pressure", "mean_arterial_pressure"]
        ):
            print(
                "\nLeakage warning: The target is directly defined by BP features "
                "that are also in the model inputs. Perfect scores are expected but "
                "do not generalize. Consider predicting hypertension from other "
                "clinical/demographic features (age, BMI, meds) instead."
            )


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    debug: bool,
) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_curve(y_test, y_prob)
    _plot_feature_importance(model, X_test.columns.tolist())

    if debug:
        print("\nDebug Checks:")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
        print(f"y_train distribution:\n{y_train.value_counts(dropna=False)}")
        print(f"y_test distribution:\n{y_test.value_counts(dropna=False)}")
        print("\nFirst 10 y_test vs y_pred:")
        print(y_test.head(10).to_list())
        print(y_pred[:10].tolist())
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Train vs test performance check
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"\nTrain accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")

        # Label shuffle test
        y_test_shuffled = np.random.permutation(y_test)
        shuffled_acc = accuracy_score(y_test_shuffled, y_pred)
        print(f"Shuffled label accuracy (should drop): {shuffled_acc:.4f}")

        # Cross-validation check
        cv_scores = cross_val_score(model, X_full, y_full, cv=5)
        print(f"Cross-val scores: {cv_scores}")

    return metrics


def _plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def _plot_roc_curve(y_true: pd.Series, y_prob: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def _plot_feature_importance(model: XGBClassifier, feature_names: list) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances[order], y=[feature_names[i] for i in order], palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def shap_explain(model: XGBClassifier, X: pd.DataFrame) -> None:
    background = X.sample(n=min(500, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model, data=background)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=True)


def predict_patient(
    model: XGBClassifier,
    systolic: float,
    diastolic: float,
    background: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    if systolic is None or diastolic is None:
        raise ValueError("Both systolic and diastolic values are required.")
    if systolic <= 0 or diastolic <= 0:
        raise ValueError("Blood pressure values must be positive.")

    pulse_pressure = systolic - diastolic
    mean_arterial_pressure = (2 * diastolic + systolic) / 3.0

    X_new = pd.DataFrame(
        [
            {
                "BPXSY1": systolic,
                "BPXDI1": diastolic,
                "pulse_pressure": pulse_pressure,
                "mean_arterial_pressure": mean_arterial_pressure,
            }
        ]
    )
    pred = int(model.predict(X_new)[0])
    prob = float(model.predict_proba(X_new)[0][1])

    bg = background if background is not None else X_new
    explainer = shap.TreeExplainer(model, data=bg)
    shap_values = explainer(X_new)
    contributions = dict(zip(X_new.columns, shap_values.values[0].tolist()))

    return {
        "prediction": pred,
        "probability": prob,
        "contributions": {k: float(v) for k, v in contributions.items()},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NHANES Hypertension Model Trainer")
    parser.add_argument("--xpt", type=str, default=str(DEFAULT_XPT_PATH))
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--debug", action="store_true", help="Run debug checks for perfect scores.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sns.set_theme(style="whitegrid")

    df_raw = load_xpt(Path(args.xpt))
    df = preprocess(df_raw)
    X, y = build_features(df)
    leakage_checks(df, X.columns.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train)
    evaluate_model(
        model,
        X_test,
        y_test,
        X_train,
        y_train,
        X,
        y,
        debug=args.debug,
    )
    shap_explain(model, X_test.sample(n=min(500, len(X_test)), random_state=42))

    background = X_train.sample(n=min(500, len(X_train)), random_state=42)
    example = predict_patient(model, systolic=150, diastolic=95, background=background)
    print("\nExample prediction:")
    print(example)


if __name__ == "__main__":
    main()
