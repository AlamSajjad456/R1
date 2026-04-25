from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_path: Path | None = None
    target: str | None = None
    test_size: float = 0.2
    random_state: int = 42
    normalize_numeric: bool = False
    # Used by train.py to pick the estimator family.
    # For now, the API loads whatever is saved to `model_path`.
    model_type: str = "xgboost"
    booster: str = "gbtree"
    shap_max_samples: int = 5000
    model_path: Path = Path("ml_system/models/model.joblib")
    metrics_path: Path = Path("ml_system/models/metrics.json")
