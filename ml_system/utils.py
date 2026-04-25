import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def hash_dataframe(df: pd.DataFrame, sample_size: int = 1000) -> str:
    if df.empty:
        return "empty"
    sample = df.head(sample_size)
    data_bytes = pd.util.hash_pandas_object(sample, index=True).values.tobytes()
    cols_bytes = ",".join(df.columns).encode("utf-8")
    h = hashlib.md5()
    h.update(cols_bytes)
    h.update(data_bytes)
    h.update(str(len(df)).encode("utf-8"))
    return h.hexdigest()


def create_version_dir(root: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = root / "versions" / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataframe_schema(df: pd.DataFrame) -> Dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}
