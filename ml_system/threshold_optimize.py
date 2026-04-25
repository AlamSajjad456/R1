import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

try:
    import joblib
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    missing = getattr(exc, "name", "required package")
    raise SystemExit(
        f"Missing dependency '{missing}'. Install project requirements before running threshold_optimize.py."
    ) from exc
from sklearn.metrics import accuracy_score, f1_score


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find best probability threshold for accuracy or F1."
    )
    parser.add_argument("csv_path", help="Path to the dataset CSV.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument(
        "--model",
        default="ml_system/models/model.joblib",
        help="Path to the trained model pipeline.",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: ,). Use ';' for cardio.",
    )
    parser.add_argument(
        "--drop-cols",
        default="",
        help="Comma-separated columns to drop before prediction.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Threshold step size (default: 0.01).",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional path to save threshold metrics as JSON.",
    )
    return parser


def _thresholds(step: float) -> List[float]:
    step = max(step, 0.001)
    return [round(t, 4) for t in np.arange(0.05, 0.96, step)]


def _best_thresholds(y_true: np.ndarray, probs: np.ndarray, step: float) -> Tuple[dict, dict, list]:
    best_acc = {"threshold": 0.5, "accuracy": 0.0}
    best_f1 = {"threshold": 0.5, "f1": 0.0}
    curve = []
    for t in _thresholds(step):
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        curve.append({"threshold": t, "accuracy": acc, "f1": f1})
        if acc > best_acc["accuracy"]:
            best_acc = {"threshold": t, "accuracy": acc}
        if f1 > best_f1["f1"]:
            best_f1 = {"threshold": t, "f1": f1}
    return best_acc, best_f1, curve


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_arg_parser().parse_args()

    df = pd.read_csv(args.csv_path, delimiter=args.delimiter, low_memory=False)
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if args.target not in df.columns:
        raise SystemExit(f"Target column not found: {args.target}")

    X = df.drop(columns=[args.target])
    y = df[args.target].to_numpy()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X)[:, 1]
    else:
        preds = pipeline.predict(X)
        probs = preds.astype(float)

    best_acc, best_f1, curve = _best_thresholds(y, probs, args.step)

    logging.info("Best accuracy threshold: %s (accuracy=%.4f)", best_acc["threshold"], best_acc["accuracy"])
    logging.info("Best F1 threshold: %s (f1=%.4f)", best_f1["threshold"], best_f1["f1"])

    if args.out_json:
        out = {
            "best_accuracy": best_acc,
            "best_f1": best_f1,
            "curve": curve,
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
