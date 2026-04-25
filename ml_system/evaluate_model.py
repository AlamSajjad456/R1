import argparse
from pathlib import Path

try:
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
except ImportError as exc:  # pragma: no cover
    missing = getattr(exc, "name", "required package")
    raise SystemExit(
        f"Missing dependency '{missing}'. Install project requirements before running evaluate_model.py."
    ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved classification model on a labeled CSV.")
    parser.add_argument("csv_path", help="Path to labeled CSV.")
    parser.add_argument("--model", default="ml_system/models/model.joblib", help="Path to model payload/joblib.")
    parser.add_argument("--target", default="cardio", help="Target column in the CSV.")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default: ,). Use ';' for cardio.")
    parser.add_argument("--drop-cols", default="id", help="Comma-separated columns to drop.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for class=1.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    df = pd.read_csv(args.csv_path, delimiter=args.delimiter, low_memory=False)

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    if args.target not in df.columns:
        raise SystemExit(f"Target column not found: {args.target}")

    y_true = df[args.target].astype(int).to_numpy()
    X = df.drop(columns=[args.target])

    payload = joblib.load(Path(args.model))
    pipeline = payload["pipeline"] if isinstance(payload, dict) and "pipeline" in payload else payload

    if not hasattr(pipeline, "predict_proba"):
        raise SystemExit("Model does not support predict_proba; AUC cannot be computed.")

    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= float(args.threshold)).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "threshold": float(args.threshold),
        "rows": int(len(df)),
    }
    print(metrics)


if __name__ == "__main__":
    main()
