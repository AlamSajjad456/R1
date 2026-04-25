import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import joblib
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    missing = getattr(exc, "name", "required package")
    raise SystemExit(
        f"Missing dependency '{missing}'. Install project requirements before running predict.py."
    ) from exc

try:
    from scipy import sparse
except Exception:  # pragma: no cover
    sparse = None


def load_artifacts(model_path: Path) -> Dict[str, Any]:
    payload = joblib.load(model_path)
    pipeline = payload["pipeline"]
    raw_features = payload.get("raw_features", [])
    if not raw_features:
        raw_features = list(getattr(pipeline, "feature_names_in_", []))
    return {
        "pipeline": pipeline,
        "feature_names": payload["feature_names"],
        "raw_features": raw_features,
        "task_type": payload["task_type"],
        "shap_background": payload.get("shap_background"),
        "schema": payload.get("schema", {}),
        "metadata": payload.get("metadata", {}),
    }


def parse_kv_list(items: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid input: {item}. Expected key=value.")
        key, value = item.split("=", 1)
        parsed[key] = _convert_value(value)
    return parsed


def _convert_value(value: str) -> Any:
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def validate_schema(
    df: pd.DataFrame, expected_schema: Dict[str, str], strict: bool
) -> Dict[str, List[str]]:
    if not expected_schema:
        return {"missing": [], "extra": []}
    expected_cols = list(expected_schema.keys())
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]
    if strict and missing:
        raise ValueError(f"Missing columns: {missing}")
    return {"missing": missing, "extra": extra}


def align_to_features(df: pd.DataFrame, raw_features: List[str]) -> pd.DataFrame:
    if raw_features:
        return df.reindex(columns=raw_features, fill_value=np.nan)
    return df


def predict_sample(artifacts: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = artifacts["pipeline"]
    task_type = artifacts["task_type"]
    feature_names = artifacts["feature_names"]
    raw_features = artifacts.get("raw_features") or list(
        getattr(pipeline, "feature_names_in_", [])
    )

    input_df = pd.DataFrame([sample])
    input_df = align_to_features(input_df, raw_features)
    prediction = pipeline.predict(input_df)[0]
    if isinstance(prediction, (np.generic,)):
        prediction = prediction.item()

    result: Dict[str, Any] = {"prediction": prediction}
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(input_df)[0]
        result["probability_score"] = [float(x) for x in proba.tolist()]

    # SHAP explainability is optional; keep prediction fast when SHAP isn't installed.
    try:
        import shap  # type: ignore
    except Exception:
        return result

    # SHAP can fail for some model families (e.g., SVM/MLP/Stacking). Keep inference working.
    try:
        X_transformed = pipeline.named_steps["preprocessor"].transform(input_df)
        if sparse is not None and sparse.issparse(X_transformed):
            X_transformed = X_transformed.toarray()

        background = artifacts.get("shap_background")
        if isinstance(background, pd.DataFrame) and not background.empty:
            bg = background.copy()
            if raw_features:
                for col in raw_features:
                    if col not in bg.columns:
                        bg[col] = np.nan
                extra_cols = [col for col in bg.columns if col not in raw_features]
                if extra_cols:
                    bg = bg.drop(columns=extra_cols)
                bg = bg[raw_features]
            bg_transformed = pipeline.named_steps["preprocessor"].transform(bg)
            if sparse is not None and sparse.issparse(bg_transformed):
                bg_transformed = bg_transformed.toarray()
            explainer = shap.TreeExplainer(pipeline.named_steps["model"], data=bg_transformed)
        else:
            explainer = shap.TreeExplainer(pipeline.named_steps["model"], data=X_transformed)
        shap_values = explainer(X_transformed)
    except Exception as exc:
        result["explanation_error"] = str(exc)
        return result

    raw_contribs = shap_values.values[0].tolist()
    contributions = {name: float(val) for name, val in zip(feature_names, raw_contribs)}

    def _base_feature(name: str) -> str:
        if name.startswith("cat__"):
            stripped = name[len("cat__") :]
            if "_" in stripped:
                return stripped.split("_", 1)[0]
            return stripped
        if name.startswith("num__"):
            return name[len("num__") :]
        return name

    aggregated: Dict[str, float] = {}
    for name, val in contributions.items():
        base = _base_feature(name)
        aggregated[base] = aggregated.get(base, 0.0) + float(val)

    top_contributions = sorted(
        aggregated.items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:10]

    base_value = shap_values.base_values[0]
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0]
    base_value = float(base_value)
    prediction_delta = float(prediction) - base_value

    result["baseline"] = base_value
    result["prediction_delta"] = prediction_delta
    result["top_contributions"] = [
        {
            "feature": feat,
            "contribution": float(contrib),
            "direction": "increased" if contrib >= 0 else "decreased",
        }
        for feat, contrib in top_contributions
    ]
    def _pretty_name(name: str) -> str:
        # Drop table prefixes like "01_district_aggregated__"
        if "__" in name:
            name = name.split("__", 1)[1]
        cleaned = name.replace("_", " ")
        return cleaned.strip()

    def _context_from_feature(name: str) -> str:
        if "__" in name:
            prefix = name.split("__", 1)[0]
            if "district" in prefix:
                return "in district"
            if "tehsil" in prefix:
                return "in tehsil"
        return ""

    def _strength(value: float) -> str:
        abs_val = abs(value)
        if abs_val >= 100000:
            return "strongly"
        if abs_val >= 20000:
            return "moderately"
        if abs_val >= 5000:
            return "slightly"
        return "a little"

    natural = []
    for item in result["top_contributions"]:
        feat = _pretty_name(item["feature"])
        context = _context_from_feature(item["feature"])
        direction = "increased" if item["contribution"] >= 0 else "reduced"
        suffix = f" {context}" if context else ""
        natural.append(
            f"{feat}{suffix} {_strength(item['contribution'])} {direction} the prediction."
        )

    result["summary"] = (
        f"Baseline was {base_value:.2f}. This sample changed it by {prediction_delta:.2f} "
        f"to predict {float(prediction):.2f}."
    )
    result["natural_explanations"] = natural

    if os.getenv("SHOW_FULL_CONTRIBUTIONS", "0") == "1":
        result["feature_contributions"] = contributions

    return result


def predict_dataframe(
    artifacts: Dict[str, Any],
    df: pd.DataFrame,
    strict_schema: bool = False,
    threshold: float | None = None,
) -> pd.DataFrame:
    pipeline = artifacts["pipeline"]
    raw_features = artifacts.get("raw_features") or list(
        getattr(pipeline, "feature_names_in_", [])
    )
    validate_schema(df, artifacts.get("schema", {}), strict_schema)
    df_aligned = align_to_features(df, raw_features)
    preds = pipeline.predict(df_aligned)
    out = pd.DataFrame({"prediction": preds})
    if (
        artifacts.get("task_type") == "classification"
        and threshold is not None
        and hasattr(pipeline, "predict_proba")
    ):
        probs = pipeline.predict_proba(df_aligned)[:, 1]
        out["probability"] = probs
        out["prediction_thresholded"] = (probs >= threshold).astype(int)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict using a saved model.")
    parser.add_argument("--model", required=True, help="Path to the saved model.")
    parser.add_argument(
        "--input-json",
        help="Path to a JSON file containing a single sample.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="Inline sample as key=value pairs.",
    )
    parser.add_argument(
        "--input-csv",
        help="Path to a CSV file for batch prediction.",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter for --input-csv (default: ,). Use ';' for cardio files.",
    )
    parser.add_argument(
        "--output-csv",
        help="Output CSV path for batch predictions.",
    )
    parser.add_argument(
        "--strict-schema",
        action="store_true",
        help="Fail if input schema is missing columns.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold to override default 0.5.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    model_path = Path(args.model)
    artifacts = load_artifacts(model_path)

    if args.input_csv:
        if not args.output_csv:
            raise SystemExit("Provide --output-csv when using --input-csv.")
        df = pd.read_csv(args.input_csv, low_memory=False, delimiter=args.delimiter)
        preds = predict_dataframe(
            artifacts, df, strict_schema=args.strict_schema, threshold=args.threshold
        )
        out = df.copy()
        out["prediction"] = preds["prediction"]
        if "probability" in preds.columns:
            out["probability"] = preds["probability"]
        if "prediction_thresholded" in preds.columns:
            out["prediction_thresholded"] = preds["prediction_thresholded"]
        out.to_csv(args.output_csv, index=False)
        print(f"Saved predictions to {args.output_csv}")
        return
    if args.input_json:
        sample = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    elif args.input:
        sample = parse_kv_list(args.input)
    else:
        raise SystemExit("Provide --input-json or --input key=value pairs.")

    result = predict_sample(artifacts, sample)
    if (
        args.threshold is not None
        and artifacts.get("task_type") == "classification"
        and "probability_score" in result
    ):
        positive_prob = result["probability_score"][1]
        result["prediction_thresholded"] = int(positive_prob >= args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
