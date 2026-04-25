# Machine Learning System (Auto XGBoost)

This document covers everything related to the machine learning code inside `ml_system/`.

If you are using the **cardio demo + FastAPI form UI**, follow the repo root guide:

- `README.md`

## What This System Does

- Trains tabular models on CSV data (classification or regression).
- Handles missing values and categorical encoding.
- Evaluates performance with standard metrics.
- Explains predictions using SHAP.
- Saves the trained model for reuse.
- Provides a FastAPI server + Tailwind UI for predictions.

## Folder Structure

- `ml_system/train.py` — Train on a single CSV.
- `ml_system/merge_train.py` — Merge all CSVs in `data/` and train one large model.
- `ml_system/predict.py` — Predict using a saved model (CLI).
- `ml_system/api.py` — FastAPI server for predictions + UI.
- `ml_system/preprocess.py` — preprocessing utilities.
- `ml_system/config.py` — training configuration.
- `ml_system/utils.py` — logging + JSON helpers.
- `ml_system/web/index.html` — Tailwind UI.
- `ml_system/models/` — saved model + metrics.

## Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn joblib fastapi uvicorn
```

Optional (for LightGBM + stacking ensembles):

```bash
pip install lightgbm
```

## Train a Model (Single CSV)

```bash
python -m ml_system.train data\01\01_district_aggregated.csv --target male --save-model ml_system\models\model.joblib
```

Optional hyperparameter tuning:

```bash
python -m ml_system.train data\01\01_district_aggregated.csv --target male --tune --tune-iter 20
```

Auto target + feature selection + cross-validation:

```bash
python -m ml_system.train data\01\01_district_aggregated.csv --auto-target --auto-features --cv 5
```

## Cardio Minimal Feature Set (Leakage-Safe)

If you are training from an output CSV that already contains predictions (e.g., \texttt{cardio\_out.csv}), make sure you do not leak \texttt{prediction}/\texttt{probability} columns into training.

This repo supports a minimal cardio feature set + feature engineering:

```bash
python -m ml_system.train "ml_system\data\cardio_out.csv" --delimiter "," --target cardio --drop-cols "id" --cardio-minimal --model-type xgboost --tune --tune-iter 20
```

## Train One Large Model (Merge All CSVs)

This scans all CSVs under `data/`, merges them by a shared key (e.g. `district`), and trains one model.

```bash
python -m ml_system.merge_train --data-root data --key district --target 01_district_aggregated__male
```

Auto target + feature selection + cross-validation:

```bash
python -m ml_system.merge_train --data-root data --key district --auto-target --auto-features --cv 5
```

Important:
- After merge, columns are prefixed with the filename.
- Example target name: `01_district_aggregated__male`

## Run API + UI

```bash
set MODEL_PATH=ml_system\models\model.joblib
python -m uvicorn ml_system.api:app --host 0.0.0.0 --port 8000
```

If you have multiple cardio models, you can also set:

- `MODEL_PATH_WITH_BP` (default: `ml_system/models/cardio_model.joblib`)
- `MODEL_PATH_NO_BP` (default: `ml_system/models/cardio_no_bp.joblib`)
- `MODEL_PATH_LOGREG` (default: `ml_system/models/cardio_logreg.joblib`)
- `MODEL_PATH_LGBM` (default: `ml_system/models/cardio_lgbm.joblib`)
- `MODEL_PATH_STACK` (default: `ml_system/models/cardio_stack.joblib`)
- `MODEL_PATH_SVM` (default: `ml_system/models/cardio_svm.joblib`)
- `MODEL_PATH_MLP` (default: `ml_system/models/cardio_mlp.joblib`)

Open the UI:

```
http://localhost:8000/
```

## Make Predictions (API)

### Single sample

```json
{
  "input": {
    "district": "Lahore",
    "area_sqkm": 1772.0,
    "all_sex": 11021000,
    "female": 5312000,
    "transgender": 1200,
    "sex_ratio": 102,
    "pop_density_sqkm": 6220,
    "urban_proportion": 82.5,
    "avg_hhsize": 6.5,
    "population1998": 6318000,
    "pop_growth_avg_1998_2017": 3.1
  }
}
```

## Batch Prediction (CSV Upload)

```bash
curl -X POST http://localhost:8000/predict_csv -F "file=@data\\01\\01_district_aggregated.csv" --output predictions.csv
```

## Auto EDA Report

```bash
python -m ml_system.eda data\\01\\01_district_aggregated.csv --out-dir ml_system\\reports
```

### Batch compare multiple districts

```json
{
  "input": [
    { "district": "Lahore", "area_sqkm": 1772.0, "all_sex": 11021000 },
    { "district": "Faisalabad", "area_sqkm": 5856.0, "all_sex": 7870000 }
  ]
}
```

## Make Predictions (CLI)

```bash
python -m ml_system.predict --model ml_system\models\model.joblib --input district=Lahore area_sqkm=1772.0 all_sex=11021000 female=5312000 transgender=1200 sex_ratio=102 urban_proportion=82.5 avg_hhsize=6.5 population1998=6318000 pop_growth_avg_1998_2017=3.1
```

## Where Evaluation Metrics Are Stored

After training, metrics are saved to:

```
ml_system/models/metrics.json
```

## Model Versioning

Each training run saves a version folder:

```
ml_system/models/versions/YYYYMMDD_HHMMSS/
```

Each version includes:
- `metadata.json` (target, data hash, metrics)


View in PowerShell:

```bash
type ml_system\models\metrics.json
```

## Understanding the Output

- **Prediction**: the model’s estimated value for your target.
- **Baseline**: average prediction from training data.
- **Delta**: how your inputs shifted the prediction.
- **Top Drivers**: the most influential features in this prediction.

## Troubleshooting

- If the API says missing columns, re-train the model after merging.
- If the UI button doesn’t work, open the page from `http://localhost:8000/` not a file path.
- If the output is too long, only the top drivers are shown by default.
