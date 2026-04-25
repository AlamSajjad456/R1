import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ml_system.config import Config
from ml_system.preprocess import optimize_memory, recommend_targets, auto_feature_select
from ml_system.train import Trainer
from ml_system.utils import create_version_dir, dataframe_schema, hash_dataframe, save_json, setup_logging


def _find_csv_files(data_root: Path) -> List[Path]:
    return [p for p in data_root.rglob("*.csv") if "00_shapefiles" not in str(p)]


def _auto_pick_key(files: List[Path]) -> str:
    candidates = ["district", "tehsil"]
    scores: Dict[str, int] = {c: 0 for c in candidates}
    for path in files:
        try:
            cols = pd.read_csv(path, nrows=0).columns.str.lower().tolist()
        except Exception:
            continue
        for c in candidates:
            if c in cols:
                scores[c] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        raise ValueError(
            "Could not find a common key. Provide --key (e.g., district or tehsil)."
        )
    return best


def _aggregate_by_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df[key].duplicated().any():
        numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        non_numeric = [c for c in df.columns if c not in numeric_cols and c != key]
        agg = {c: "sum" for c in numeric_cols}
        for c in non_numeric:
            agg[c] = "first"
        return df.groupby(key, as_index=False).agg(agg)
    return df


def _prepare_table(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if key not in df.columns:
        raise KeyError(f"Key '{key}' not found.")
    df = _aggregate_by_key(df, key)

    renamed = {}
    for col in df.columns:
        if col == key:
            continue
        renamed[col] = f"{prefix}__{col}"
    return df.rename(columns=renamed)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multiple CSV tables and train a single model."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root folder containing CSV tables.",
    )
    parser.add_argument(
        "--key",
        help="Join key column (e.g., district or tehsil).",
    )
    parser.add_argument(
        "--target",
        help="Target column after merge (use prefixed name if needed).",
    )
    parser.add_argument(
        "--save-model",
        dest="save_model",
        default="ml_system/models/model.joblib",
        help="Path to save model.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize numeric features.",
    )
    parser.add_argument(
        "--booster",
        default="gbtree",
        choices=["gbtree", "gblinear", "dart"],
        help="XGBoost booster to use.",
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


def main() -> None:
    setup_logging()

    args = build_arg_parser().parse_args()
    data_root = Path(args.data_root)
    files = _find_csv_files(data_root)
    if not files:
        raise SystemExit("No CSV files found.")

    key = args.key or _auto_pick_key(files)
    logging.info("Using join key: %s", key)

    merged: pd.DataFrame | None = None
    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            logging.warning("Skipping %s (%s)", path, exc)
            continue

        if key not in df.columns:
            logging.info("Skipping %s (missing key)", path.name)
            continue

        prefix = path.stem
        table = _prepare_table(df, key, prefix)
        table = optimize_memory(table)

        if merged is None:
            merged = table
        else:
            merged = merged.merge(table, on=key, how="outer")

        logging.info("Merged %s (rows=%s, cols=%s)", path.name, table.shape[0], table.shape[1])

    if merged is None:
        raise SystemExit("No compatible tables found for merge.")

    logging.info("Final merged shape: %s", merged.shape)

    target_col = args.target
    if args.auto_target:
        candidates = recommend_targets(merged)
        if not candidates:
            raise SystemExit("No suitable target columns found.")
        target_col = candidates[0][0]
        logging.info("Auto target selected: %s", target_col)
    if not target_col:
        raise SystemExit("Provide --target or use --auto-target.")
    if target_col not in merged.columns:
        raise SystemExit(f"Target column not found: {target_col}")

    before = merged.shape[0]
    merged = merged[merged[target_col].notna()].copy()
    after = merged.shape[0]
    if after == 0:
        raise SystemExit("No rows left after dropping missing target values.")
    if after < before:
        logging.info("Dropped %s rows with missing target.", before - after)

    config = Config(
        data_path=data_root,
        target=args.target,
        normalize_numeric=args.normalize,
        booster=args.booster,
    )
    trainer = Trainer(config)
    if args.auto_features:
        merged, feature_report = auto_feature_select(merged, target_col)
        logging.info("Feature selection report: %s", feature_report)
    metrics = trainer.train(
        merged,
        target_col,
        tune=False,
        tune_iter=0,
        cv_folds=args.cv,
        auto_select=False,
    )
    logging.info("Model performance: %s", metrics)

    model_path = Path(args.save_model)
    data_hash = hash_dataframe(merged)
    schema = dataframe_schema(merged)
    version_dir = create_version_dir(Path("ml_system/models"))
    metadata = {
        "target": target_col,
        "data_hash": data_hash,
        "rows": len(merged),
        "columns": len(merged.columns),
        "metrics": metrics,
        "schema": schema,
    }
    trainer.save(model_path, metadata)
    save_json(Path("ml_system/models/metrics.json"), metrics)
    save_json(version_dir / "metadata.json", metadata)
    logging.info("Saved model to: %s", model_path)
    logging.info("Saved model version to: %s", version_dir)


if __name__ == "__main__":
    main()
