---
title: CVD ML Predictor
emoji: ❤️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# CVD ML Predictor

Cardiovascular disease prediction API with multiple models (XGBoost, LightGBM, SVM, MLP, Logistic Regression).

This folder contains the ML code + the FastAPI app.

For the full, detailed, GitHub-ready procedure (install → clean → train → evaluate → run API), read:

- `README.md` (repo root)

## Quick Start (Cardio)

Train:

```powershell
python ml_system/train.py "ml_system\data\cardio_train_clean.csv" --delimiter ";" --target cardio --drop-cols "id"
```

Run API:

```powershell
python ml_system/run_api.py
```

Open web UI:

- `http://127.0.0.1:8000/`

API docs:

- `http://127.0.0.1:8000/docs`
