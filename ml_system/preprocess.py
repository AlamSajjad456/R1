from typing import List, Tuple, Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["int", "int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float", "float64", "float32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("category")
    return df


def infer_task_type(target: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(target):
        return "classification"
    if isinstance(target.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(
        target
    ):
        return "classification"
    if pd.api.types.is_numeric_dtype(target):
        unique_count = target.nunique(dropna=True)
        if unique_count <= 20 and (target.dropna() % 1 == 0).all():
            return "classification"
        return "regression"
    return "classification"


def split_features(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return X, y, numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    normalize_numeric: bool,
    booster: str,
) -> ColumnTransformer:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if normalize_numeric or booster == "gblinear":
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_cols),
            ("cat", Pipeline(steps=categorical_steps), categorical_cols),
        ],
        sparse_threshold=0.3,
    )
    return preprocessor


def recommend_targets(df: pd.DataFrame, top_k: int = 10) -> List[Tuple[str, float]]:
    scores: List[Tuple[str, float]] = []
    for col in df.columns:
        series = df[col]
        missing_ratio = series.isna().mean()
        if missing_ratio > 0.4:
            continue
        if pd.api.types.is_numeric_dtype(series):
            unique = series.nunique(dropna=True)
            if unique < 5:
                continue
            score = (1 - missing_ratio) * (1 + min(unique, 100) / 100)
            scores.append((col, score))
        elif pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            unique = series.nunique(dropna=True)
            if 2 <= unique <= 50:
                score = (1 - missing_ratio) * (1 + min(unique, 50) / 50)
                scores.append((col, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def auto_feature_select(
    df: pd.DataFrame,
    target_col: str,
    missing_threshold: float = 0.4,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    dropped: Dict[str, List[str]] = {"missing": [], "low_variance": [], "duplicate": []}
    features = df.drop(columns=[target_col])

    missing_ratio = features.isna().mean()
    drop_missing = missing_ratio[missing_ratio > missing_threshold].index.tolist()
    if drop_missing:
        features = features.drop(columns=drop_missing)
        dropped["missing"] = drop_missing

    low_variance = []
    for col in features.columns:
        if features[col].nunique(dropna=True) <= 1:
            low_variance.append(col)
    if low_variance:
        features = features.drop(columns=low_variance)
        dropped["low_variance"] = low_variance

    try:
        dup_cols = features.T.duplicated().to_numpy()
        dup_names = features.columns[dup_cols].tolist()
        if dup_names:
            features = features.drop(columns=dup_names)
            dropped["duplicate"] = dup_names
    except Exception:
        pass

    cleaned = pd.concat([features, df[target_col]], axis=1)
    return cleaned, dropped
