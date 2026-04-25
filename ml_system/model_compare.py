import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    missing = getattr(exc, "name", "required package")
    raise SystemExit(
        f"Missing dependency '{missing}'. Install project requirements before running model_compare.py."
    ) from exc

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml_system.config import Config
from ml_system.train import Trainer
from ml_system.utils import save_json, setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and compare multiple model families on the same dataset."
    )
    p.add_argument("csv_path", help="Path to the input CSV.")
    p.add_argument("--delimiter", default=",", help="CSV delimiter (default: ,).")
    p.add_argument("--target", required=True, help="Target column name (e.g., cardio).")
    p.add_argument("--drop-cols", dest="drop_cols", default="", help="Comma-separated columns to drop.")
    p.add_argument(
        "--models",
        default="logreg,rf,svm,xgboost,mlp",
        help="Comma-separated model types to compare (default: logreg,rf,svm,xgboost,mlp). Also supports: lgbm, stack.",
    )
    p.add_argument("--cv", type=int, default=0, help="Enable k-fold cross-validation per model (e.g., 5).")
    p.add_argument("--tune", action="store_true", help="Enable randomized tuning for each model (slower).")
    p.add_argument("--tune-iter", type=int, default=15, help="Random search iterations per model (with --tune).")
    p.add_argument(
        "--tune-models",
        default="",
        help="Optional comma-separated list of model types to tune (e.g., xgboost,lgbm). If set, only these will be tuned.",
    )
    p.add_argument(
        "--cardio-minimal",
        action="store_true",
        help=(
            "Use only weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active + engineered features. "
            "Also drops leakage columns (prediction/probability) if present."
        ),
    )
    p.add_argument(
        "--cardio-age-bmi",
        action="store_true",
        help=(
            "Use Age + BMI engineered features (age_years, BMI, age_group, interactions, and risk flags). "
            "Drops leakage columns if present."
        ),
    )
    return p


def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()

    df = pd.read_csv(args.csv_path, low_memory=False, delimiter=args.delimiter)
    drop_cols = [c.strip() for c in (args.drop_cols or "").split(",") if c.strip()]
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    models = [m.strip().lower() for m in (args.models or "").split(",") if m.strip()]
    if not models:
        raise SystemExit("No models specified. Use --models logreg,rf,svm,xgboost,mlp")

    tune_models = {m.strip().lower() for m in (args.tune_models or "").split(",") if m.strip()}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("ml_system/models") / "compare" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    logging.info("Comparing models: %s", ", ".join(models))
    logging.info("Output dir: %s", out_dir)

    for model_type in models:
        needs_scaling = model_type in {"logreg", "svm", "mlp"}
        config = Config(
            data_path=Path(args.csv_path),
            target=args.target,
            normalize_numeric=needs_scaling,
            model_type=model_type,
        )
        trainer = Trainer(config)
        logging.info("Training model_type=%s ...", model_type)
        metrics = trainer.train(
            df.copy(),
            target_col=args.target,
            tune=(bool(args.tune) and ((not tune_models) or (model_type in tune_models))),
            tune_iter=args.tune_iter,
            cv_folds=int(args.cv or 0),
            auto_select=False,
            group_col=None,
            cardio_minimal=args.cardio_minimal,
            cardio_age_bmi=args.cardio_age_bmi,
        )

        model_path = out_dir / f"{model_type}.joblib"
        trainer.save(model_path, metadata={"target": args.target, "model_type": model_type, "metrics": metrics})
        save_json(out_dir / f"{model_type}.metrics.json", metrics)

        row: Dict[str, Any] = {"model_type": model_type}
        # Some metrics keys are task-dependent; keep best-effort.
        for k in ("accuracy", "f1_score", "roc_auc"):
            if k in metrics:
                row[k] = float(metrics[k])
        results.append(row)

    save_json(out_dir / "compare_summary.json", {"results": results})
    logging.info("Compare done. Summary saved to: %s", out_dir / "compare_summary.json")


if __name__ == "__main__":
    main()
