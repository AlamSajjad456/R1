import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.responses import FileResponse, HTMLResponse, Response
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    missing = getattr(exc, "name", "required package")
    raise SystemExit(
        f"Missing dependency '{missing}'. Install project requirements before running api.py."
    ) from exc

from ml_system.predict import align_to_features, load_artifacts, predict_dataframe, predict_sample


class PredictRequest(BaseModel):
    input: Dict[str, Any] | List[Dict[str, Any]]


class PredictResponse(BaseModel):
    results: List[Dict[str, Any]]


MODEL_PATH_WITH_BP = Path(os.getenv("MODEL_PATH_WITH_BP", "ml_system/models/cardio_model.joblib"))
MODEL_PATH_NO_BP = Path(os.getenv("MODEL_PATH_NO_BP", "ml_system/models/cardio_no_bp.joblib"))
MODEL_PATH_LGBM = Path(os.getenv("MODEL_PATH_LGBM", "ml_system/models/cardio_lgbm.joblib"))
MODEL_PATH_STACK = Path(os.getenv("MODEL_PATH_STACK", "ml_system/models/cardio_stack.joblib"))
MODEL_PATH_SVM = Path(os.getenv("MODEL_PATH_SVM", "ml_system/models/cardio_svm.joblib"))
MODEL_PATH_MLP = Path(os.getenv("MODEL_PATH_MLP", "ml_system/models/cardio_mlp.joblib"))
MODEL_PATH_LOGREG = Path(os.getenv("MODEL_PATH_LOGREG", "ml_system/models/cardio_logreg.joblib"))
MODEL_PATH_FALLBACK = Path(os.getenv("MODEL_PATH", "ml_system/models/model.joblib"))

app = FastAPI(title="Auto XGBoost API", version="1.0.0")
models: Dict[str, Dict[str, Any]] = {}
WEB_ROOT = Path(__file__).resolve().parent / "web"


def _coerce_prediction_label(prediction: Any) -> str:
    try:
        pred_int = int(prediction)
    except Exception:
        return str(prediction)
    if pred_int == 1:
        return "Cardio: Yes"
    if pred_int == 0:
        return "Cardio: No"
    return str(prediction)


def _risk_label(probability: float, threshold: float) -> str:
    if probability >= max(threshold + 0.2, 0.8):
        return "High"
    if probability >= threshold:
        return "Moderate"
    if probability >= max(threshold - 0.15, 0.2):
        return "Borderline"
    return "Low"


def _augment_result(item: Dict[str, Any], threshold: Optional[float]) -> Dict[str, Any]:
    scores = item.get("probability_score")
    effective_threshold = float(threshold) if threshold is not None else 0.5
    if isinstance(scores, list) and len(scores) >= 2:
        prob_no = float(scores[0])
        prob_yes = float(scores[1])
        thresholded = item.get("prediction_thresholded")
        if thresholded is None:
            thresholded = int(prob_yes >= effective_threshold)
            item["prediction_thresholded"] = thresholded
        item["prob_no_cardio"] = prob_no
        item["prob_cardio"] = prob_yes
        item["threshold_used"] = effective_threshold
        item["prediction_label"] = _coerce_prediction_label(thresholded)
        item["risk_label"] = _risk_label(prob_yes, effective_threshold)
    else:
        item["prediction_label"] = _coerce_prediction_label(item.get("prediction"))
    return item


@app.on_event("startup")
def _load_model() -> None:
    global models
    models = {}

    if MODEL_PATH_WITH_BP.exists():
        models["with_bp"] = load_artifacts(MODEL_PATH_WITH_BP)
    if MODEL_PATH_NO_BP.exists():
        models["no_bp"] = load_artifacts(MODEL_PATH_NO_BP)
    if MODEL_PATH_LGBM.exists():
        models["lgbm"] = load_artifacts(MODEL_PATH_LGBM)
    if MODEL_PATH_STACK.exists():
        models["stack"] = load_artifacts(MODEL_PATH_STACK)
    if MODEL_PATH_SVM.exists():
        models["svm"] = load_artifacts(MODEL_PATH_SVM)
    if MODEL_PATH_MLP.exists():
        models["mlp"] = load_artifacts(MODEL_PATH_MLP)
    if MODEL_PATH_LOGREG.exists():
        models["logreg"] = load_artifacts(MODEL_PATH_LOGREG)

    # Backward compatible fallback: if nothing is configured, use MODEL_PATH.
    if not models and MODEL_PATH_FALLBACK.exists():
        models["with_bp"] = load_artifacts(MODEL_PATH_FALLBACK)

    if not models:
        raise RuntimeError(
            "No model found. Set MODEL_PATH_WITH_BP / MODEL_PATH_NO_BP or MODEL_PATH."
        )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

def _select_model_id(model: Optional[str], samples: List[Dict[str, Any]]) -> str:
    if model and model in models:
        return model

    # Auto selection based on presence of BP features.
    has_bp = any(("ap_hi" in s) or ("ap_lo" in s) for s in samples)
    if has_bp and "with_bp" in models:
        return "with_bp"
    if (not has_bp) and "no_bp" in models:
        return "no_bp"
    # Fallbacks
    if "with_bp" in models:
        return "with_bp"
    return next(iter(models.keys()))


@app.get("/schema")
def schema(model: Optional[str] = None) -> Dict[str, Any]:
    sample: List[Dict[str, Any]] = [{}]
    model_id = _select_model_id(model, sample)
    artifacts = models.get(model_id)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_id": model_id,
        "available_models": sorted(models.keys()),
        "schema": artifacts.get("schema", {}),
        "raw_features": artifacts.get("raw_features", []),
        "metadata": artifacts.get("metadata", {}),
    }


@app.get("/", response_class=HTMLResponse)
def ui() -> FileResponse:
    index_path = WEB_ROOT / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path)


@app.get("/favicon.ico")
def favicon() -> Response:
    # Avoid noisy 404s in the browser console.
    return Response(status_code=204)


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    threshold: Optional[float] = None,
    explain: bool = False,
    model: Optional[str] = None,
) -> PredictResponse:
    samples = req.input if isinstance(req.input, list) else [req.input]
    model_id = _select_model_id(model, samples)
    artifacts = models.get(model_id)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if explain:
        results = [predict_sample(artifacts, sample) for sample in samples]
    else:
        # Fast path: no SHAP, just prediction + proba.
        pipeline = artifacts["pipeline"]
        raw_features = artifacts.get("raw_features") or []
        df = pd.DataFrame(samples)
        df = align_to_features(df, raw_features)
        preds = pipeline.predict(df)
        probs = None
        if artifacts.get("task_type") == "classification" and hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(df)
        results = []
        for i, pred in enumerate(preds):
            item: Dict[str, Any] = {"prediction": int(pred) if str(pred).isdigit() else pred}
            if probs is not None:
                score = [float(x) for x in probs[i].tolist()]
                item["probability_score"] = score
                if threshold is not None and len(score) >= 2:
                    item["prediction_thresholded"] = int(score[1] >= float(threshold))
            results.append(item)
    if threshold is not None:
        for item in results:
            scores = item.get("probability_score")
            if isinstance(scores, list) and len(scores) >= 2:
                item["prediction_thresholded"] = int(float(scores[1]) >= float(threshold))

    # Always include which model produced this prediction (important in auto mode).
    for item in results:
        item["model_id"] = model_id

    results = [_augment_result(item, threshold) for item in results]
    return PredictResponse(results=results)


@app.post("/predict_csv")
async def predict_csv(
    file: UploadFile = File(...),
    delimiter: str = ",",
    threshold: Optional[float] = None,
    model: Optional[str] = None,
) -> FileResponse:
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    content = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(content), delimiter=delimiter)

    # Choose model based on provided columns when model isn't forced.
    sample_records = df.head(1).to_dict(orient="records") or [{}]
    model_id = _select_model_id(model, sample_records)
    artifacts = models.get(model_id)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    preds = predict_dataframe(artifacts, df, strict_schema=False, threshold=threshold)
    out = df.copy()
    out["prediction"] = preds["prediction"]
    if "probability" in preds.columns:
        out["probability"] = preds["probability"]
    if "prediction_thresholded" in preds.columns:
        out["prediction_thresholded"] = preds["prediction_thresholded"]
    out["model_id"] = model_id

    output_path = WEB_ROOT / "predictions.csv"
    out.to_csv(output_path, index=False)
    return FileResponse(output_path, filename="predictions.csv", media_type="text/csv")
