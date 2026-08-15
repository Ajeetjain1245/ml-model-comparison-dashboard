"""
Leak-free data preprocessing pipelines with ColumnTransformer.
"""
from typing import Tuple, List, Optional, Any, Dict
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str], Optional[LabelEncoder]]:
    """
    Separates X and y, encodes target appropriately, and identifies feature types.
    Does NOT transform or fit features here to prevent data leakage!
    """
    df_clean = df.dropna(subset=[target_col]).copy()
    X_raw = df_clean.drop(columns=[target_col]).copy()
    y_raw = df_clean[target_col].copy()

    # Identify column types
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    label_encoder = None
    if task_type == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(y_raw.median()).to_numpy(dtype=float)

    return X_raw, y, num_cols, cat_cols, label_encoder


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Builds an unfitted ColumnTransformer with median imputation + standard scaling
    for numeric columns and mode imputation + OneHotEncoding for categorical columns.
    """
    transformers = []

    if num_cols:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return preprocessor


def get_feature_names_out(fitted_preprocessor: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    """
    Extracts human-readable feature names after one-hot encoding & numerical scaling.
    """
    try:
        return fitted_preprocessor.get_feature_names_out().tolist()
    except Exception:
        # Fallback manual reconstruction
        names = list(num_cols)
        if cat_cols and "cat" in fitted_preprocessor.named_transformers_:
            cat_trans = fitted_preprocessor.named_transformers_["cat"]
            if hasattr(cat_trans, "named_steps") and "onehot" in cat_trans.named_steps:
                ohe = cat_trans.named_steps["onehot"]
                if hasattr(ohe, "get_feature_names_out"):
                    names.extend(ohe.get_feature_names_out(cat_cols).tolist())
                else:
                    names.extend(cat_cols)
            else:
                names.extend(cat_cols)
        return names
