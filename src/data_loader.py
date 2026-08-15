"""
Data loading and sample dataset utilities.
"""
from typing import Tuple, Dict, Any, Optional
import io
import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_wine,
    fetch_california_housing,
    load_diabetes,
)


def get_sample_datasets_info() -> Dict[str, Dict[str, Any]]:
    """Returns metadata about available built-in sample datasets."""
    return {
        "Breast Cancer (Binary Classification)": {
            "loader": _load_breast_cancer_df,
            "type": "classification",
            "description": "30 numeric features predicting malignant vs benign tumors.",
            "default_target": "diagnosis",
        },
        "Wine Recognition (Multiclass Classification)": {
            "loader": _load_wine_df,
            "type": "classification",
            "description": "13 chemical constituents from 3 different wine cultivars.",
            "default_target": "cultivar",
        },
        "Iris Benchmark (Multiclass Classification)": {
            "loader": _load_iris_df,
            "type": "classification",
            "description": "4 sepal/petal measurements predicting 3 iris species.",
            "default_target": "species",
        },
        "Titanic Survival (Mixed Tabular & Missing Data)": {
            "loader": _load_titanic_df,
            "type": "classification",
            "description": "Mixed numerical & categorical passenger data with missing values.",
            "default_target": "Survived",
        },
        "California Housing (Regression)": {
            "loader": _load_housing_df,
            "type": "regression",
            "description": "8 demographic & geographical features predicting median home values.",
            "default_target": "MedHouseVal",
        },
        "Diabetes Progression (Regression)": {
            "loader": _load_diabetes_df,
            "type": "regression",
            "description": "10 baseline variables predicting quantitative disease progression.",
            "default_target": "progression",
        },
    }


def _load_breast_cancer_df() -> Tuple[pd.DataFrame, str]:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    target_col = "diagnosis"
    df[target_col] = df["target"].map({0: "Malignant", 1: "Benign"}).astype(str)
    df = df.drop(columns=["target"])
    return df, target_col


def _load_wine_df() -> Tuple[pd.DataFrame, str]:
    data = load_wine(as_frame=True)
    df = data.frame.copy()
    target_col = "cultivar"
    df[target_col] = df["target"].map({0: "Class 0", 1: "Class 1", 2: "Class 2"}).astype(str)
    df = df.drop(columns=["target"])
    return df, target_col


def _load_iris_df() -> Tuple[pd.DataFrame, str]:
    data = load_iris(as_frame=True)
    df = data.frame.copy()
    target_col = "species"
    df[target_col] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"}).astype(str)
    df = df.drop(columns=["target"])
    return df, target_col


def _load_titanic_df() -> Tuple[pd.DataFrame, str]:
    # Curated Titanic dataset sample with mixed types
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        df = pd.read_csv(url)
        df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
        df["Survived"] = df["Survived"].map({0: "Died", 1: "Survived"})
        return df, "Survived"
    except Exception:
        # Fallback local synthetic mini-titanic if network fails
        np.random.seed(42)
        n = 300
        pclass = np.random.choice([1, 2, 3], size=n, p=[0.25, 0.25, 0.5])
        sex = np.random.choice(["male", "female"], size=n, p=[0.6, 0.4])
        age = np.random.normal(30, 14, size=n).clip(1, 80)
        age[np.random.choice(n, 30, replace=False)] = np.nan
        fare = (pclass == 1) * np.random.exponential(60, n) + (pclass == 2) * np.random.exponential(25, n) + (pclass == 3) * np.random.exponential(12, n)
        embarked = np.random.choice(["S", "C", "Q"], size=n, p=[0.7, 0.2, 0.1])
        embarked[np.random.choice(n, 5, replace=False)] = np.nan
        surv_prob = 0.3 + 0.4 * (sex == "female") + 0.2 * (pclass == 1) - 0.1 * (pclass == 3)
        survived = (np.random.rand(n) < surv_prob.clip(0.05, 0.95)).astype(int)
        df = pd.DataFrame({
            "Pclass": pclass,
            "Sex": sex,
            "Age": np.round(age, 1),
            "SibSp": np.random.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]),
            "Parch": np.random.choice([0, 1, 2], size=n, p=[0.8, 0.15, 0.05]),
            "Fare": np.round(fare, 2),
            "Embarked": embarked,
            "Survived": ["Survived" if s == 1 else "Died" for s in survived],
        })
        return df, "Survived"


def _load_housing_df() -> Tuple[pd.DataFrame, str]:
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    # Sample down to 2,000 rows for lightning-fast training
    if len(df) > 2000:
        df = df.sample(2000, random_state=42).reset_index(drop=True)
    return df, "MedHouseVal"


def _load_diabetes_df() -> Tuple[pd.DataFrame, str]:
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df.rename(columns={"target": "progression"}, inplace=True)
    return df, "progression"


def load_uploaded_csv(file_buffer) -> pd.DataFrame:
    """Safely loads an uploaded CSV file with encoding fallbacks."""
    try:
        return pd.read_csv(file_buffer)
    except UnicodeDecodeError:
        file_buffer.seek(0)
        return pd.read_csv(file_buffer, encoding="latin-1")
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file: {str(e)}")


def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Intelligently determines whether a target is Classification or Regression.
    """
    if target_col not in df.columns:
        return "classification"
    
    target_series = df[target_col].dropna()
    n_unique = target_series.nunique()
    
    # If string, boolean, or categorical dtype -> classification
    if target_series.dtype == "object" or target_series.dtype == "bool" or pd.api.types.is_categorical_dtype(target_series):
        return "classification"
    
    # If numeric but low cardinality (<= 15 distinct values) and integer-like -> classification
    if pd.api.types.is_numeric_dtype(target_series):
        is_integer_like = np.all(np.mod(target_series, 1) == 0)
        if n_unique <= 15 and is_integer_like:
            return "classification"
        return "regression"
        
    return "classification"
