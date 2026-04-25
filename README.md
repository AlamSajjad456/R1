# Axel ML Studio (API-Only) — XGBoost Cardio Demo

This repository contains a small, production-style ML pipeline and a **FastAPI web app** you can use to:

- clean a dataset (optional)
- train an **XGBoost** model
- evaluate it
- run a local web UI where you enter values in a **form** and get a clear **YES/NO** result + probability
- run batch predictions on a CSV

## Introduction (What This ML Predicts)

This project trains an ML classifier to predict the **`cardio`** target from the cardio dataset:

- `cardio = 1` → higher likelihood of cardiovascular disease (YES)
- `cardio = 0` → lower likelihood (NO)

The prediction is based on common health measurements such as age, gender, height, weight, systolic/diastolic blood pressure (`ap_hi`, `ap_lo`), cholesterol, glucose, smoking, alcohol intake, and physical activity.

In the web UI you will also see a **BP Category** (Normal / Elevated / Stage 1 / Stage 2). This BP category is a simple rule-based label derived from `ap_hi` and `ap_lo` to make the result easier to explain — it is not a separate ML prediction.

Important: this is an academic/demo project and is **not medical advice**.

## What Model Is Used?

This project uses **XGBoost** (from `xgboost`) via:

- `XGBClassifier` for classification tasks
- Saved model path: `ml_system/models/model.joblib`

Notes:
- CatBoost is **not** used in this repo anymore.
- The desktop GUI (PySide6) is **not** part of this repo anymore. We use the API + web UI only.

### Optional Model Upgrades

The training script also supports multiple model families so you can compare baselines:

- `--model-type logreg` (Logistic Regression)
- `--model-type rf` (Random Forest)
- `--model-type svm` (Support Vector Machine)
- `--model-type xgboost` (XGBoost, default)
- `--model-type mlp` (Neural Network / MLP)

## Folder Map (What Matters)

- `ml_system/train.py` — train and save the XGBoost model
- `ml_system/model_compare.py` — train multiple model types and compare metrics (optional)
- `ml_system/predict.py` — CLI prediction (single sample or CSV)
- `ml_system/evaluate_model.py` — evaluate a saved model on a labeled CSV
- `ml_system/threshold_optimize.py` — find best threshold for accuracy/F1
- `ml_system/clean_cardio.py` — optional cleaning/outlier filtering for cardio dataset
- `ml_system/api.py` — FastAPI app (endpoints + serves the web UI)
- `ml_system/run_api.py` — Windows-friendly launcher (fixes import path issues)
- `ml_system/web/index.html` — the web UI (form + JSON + batch CSV)
- `ml_system/data/` — datasets (example: `cardio_train_clean.csv`)
- `ml_system/models/` — saved model + metrics

## Setup (Windows / PowerShell)

1. Create + activate a virtual environment:

```powershell
cd "C:\Users\Sajjad Alam PC\Desktop\R1\project1"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn xgboost joblib fastapi uvicorn shap matplotlib seaborn
```

If you only want API prediction (no training), `shap/matplotlib/seaborn` are optional.

Alternative (recommended for GitHub):

```powershell
python -m pip install -r requirements.txt
```

## How The Model Is Loaded

The API loads the model from:

1. `MODEL_PATH` environment variable (if set)
2. otherwise: `ml_system/models/model.joblib`

Example (PowerShell):

```powershell
$env:MODEL_PATH = "ml_system\models\model.joblib"
python ml_system/run_api.py
```

## Dataset Notes (Cardio)

If you train on the classic cardio dataset format:

- delimiter is usually `;`
- target column is `cardio` (0/1)
- `id` is a drop column
- **Important**: `age` is stored in **days** (not years).
  - The web form asks for **years** and converts to days automatically.

### Cardio Columns (Expected)

For training/prediction, the model expects the classic columns:

- `age` (days)
- `gender` (1=female, 2=male)
- `height` (cm)
- `weight` (kg)
- `ap_hi` (systolic)
- `ap_lo` (diastolic)
- `cholesterol` (1/2/3)
- `gluc` (1/2/3)
- `smoke` (0/1)
- `alco` (0/1)
- `active` (0/1)

Target:

- `cardio` (0/1)

## (Optional) Clean the Dataset

If your raw file is `ml_system/data/cardio_train.csv`, you can produce a cleaner file:

```powershell
python ml_system/clean_cardio.py "ml_system\data\cardio_train.csv" --delimiter ";" --out "ml_system\data\cardio_train_clean.csv"
```

## Train (XGBoost)

Train on the cleaned dataset:

```powershell
python ml_system/train.py "ml_system\data\cardio_train_clean.csv" --delimiter ";" --target cardio --drop-cols "id"
```

With light hyperparameter tuning (slower):

```powershell
python ml_system/train.py "ml_system\data\cardio_train_clean.csv" --delimiter ";" --target cardio --drop-cols "id" --tune --tune-iter 30
```

Outputs:

- model: `ml_system/models/model.joblib`
- metrics: `ml_system/models/metrics.json`
- versioned folder: `ml_system/models/versions/<timestamp>/`

### SHAP Plots (Optional)

SHAP explanations can be slow for some models. By default training **skips SHAP** and saves the model immediately.

To generate SHAP plots and save them inside the model version folder:

```powershell
python ml_system/train.py "ml_system\data\cardio_train_clean.csv" --delimiter ";" --target cardio --drop-cols "id" --model-type xgboost --shap
```

### Train Other Models (Optional)

Example (Random Forest):

```powershell
python ml_system/train.py "ml_system\data\cardio_train_clean.csv" --delimiter ";" --target cardio --drop-cols "id" --model-type rf
```

Example (Logistic Regression):

```powershell
python ml_system/train.py "ml_system\data\cardio_train_clean.csv" --delimiter ";" --target cardio --drop-cols "id" --model-type logreg
```

Example (Compare All Models):

```powershell
python ml_system/model_compare.py "ml_system\data\cardio_train_clean.csv" --delimiter ";" --target cardio --drop-cols "id"
```

## Evaluate a Saved Model (On a Labeled CSV)

```powershell
python ml_system/evaluate_model.py "ml_system\data\cardio_train_clean.csv" --model "ml_system\models\model.joblib" --delimiter ";" --target cardio --drop-cols "id" --threshold 0.51
```

## Find a Better Threshold (Accuracy vs F1)

```powershell
python ml_system/threshold_optimize.py "ml_system\data\cardio_train_clean.csv" --delimiter ";" --target cardio --drop-cols "id"
```

It prints the best threshold for accuracy and the best threshold for F1.

## Run the API + Web UI

Start the server:

```powershell
python ml_system/run_api.py
```

Open:
- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs` (Swagger API docs)

### Web UI Features

- **Form input** (recommended): pick Yes/No from dropdowns, age in years, and get output immediately
- **Threshold**: set e.g. `0.51` (the UI colors cards red/green based on threshold)
- **Explain**: optional SHAP explanations (if available)
- **Batch CSV**: upload a CSV and download `prediction` + `probability` columns

## API Endpoints (Details)

### `GET /schema`
Returns the required input features (used by the web form).

### `POST /predict`
Predict a single sample (or a list of samples).

Query params:
- `threshold` (optional): if provided, API returns `prediction_thresholded`
- `explain` (optional, default false): if true, tries to include SHAP drivers

Example request (single sample):

```bash
curl -X POST "http://127.0.0.1:8000/predict?threshold=0.51&explain=false" ^
  -H "Content-Type: application/json" ^
  -d "{\"input\":{\"age\":20089,\"gender\":1,\"height\":165,\"weight\":72,\"ap_hi\":130,\"ap_lo\":80,\"cholesterol\":1,\"gluc\":1,\"smoke\":0,\"alco\":0,\"active\":1}}"
```

Example response (simplified):

- `prediction` is 0/1
- `probability_score` is `[P(0), P(1)]`
- `prediction_thresholded` is based on the threshold you passed

### `POST /predict_csv`
Upload a CSV and download a CSV with predictions.

Query params:
- `delimiter` (default `;` for cardio)
- `threshold` (optional)

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict_csv?delimiter=%3B&threshold=0.51" -F "file=@ml_system/data/cardio_train_clean.csv" --output predictions.csv
```

## Predict From CLI (Optional)

Single sample (inline `key=value`):

```powershell
python ml_system/predict.py --model "ml_system\models\model.joblib" --input age=20089 gender=1 height=165 weight=72 ap_hi=130 ap_lo=80 cholesterol=1 gluc=1 smoke=0 alco=0 active=1 --threshold 0.51
```

Batch CSV prediction:

```powershell
python ml_system/predict.py --model "ml_system\models\model.joblib" --input-csv "ml_system\data\cardio_train_clean.csv" --output-csv "ml_system\data\cardio_preds.csv" --delimiter ";" --threshold 0.51
```

## Troubleshooting

- `ModuleNotFoundError: No module named 'ml_system'`
  - Always run commands from the repo root: `...\project1`
  - Or use the provided launcher: `python ml_system/run_api.py`

- `Missing dependency 'pandas'` / `joblib` / etc.
  - Install packages inside your active venv: `python -m pip install pandas joblib ...`

- Web UI shows colors (red/green) but prediction feels "wrong"
  - Check you are using the right threshold (try `0.5` or the optimized value from `threshold_optimize.py`)
  - Remember: `age` is **days** in the API. The web form converts years → days automatically.
