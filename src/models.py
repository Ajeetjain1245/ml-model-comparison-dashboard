"""
Model Registry for Classification and Regression tasks.
"""
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# High-contrast, vibrant model colors tailored for light & dark themes
MODEL_COLORS = {
    # Classification
    "Logistic Regression": "#2563eb",   # Vibrant Royal Blue
    "K-Nearest Neighbors": "#059669",   # Emerald Green
    "Decision Tree": "#d97706",         # Warm Amber
    "Random Forest": "#7c3aed",         # Deep Violet
    "SVM (RBF)": "#e11d48",             # Crimson Rose
    "Gradient Boosting": "#0d9488",     # Teal
    "Extra Trees": "#0284c7",           # Sky Blue
    "XGBoost": "#ea580c",               # Vivid Orange
    # Regression
    "Ridge Regression": "#2563eb",
    "Linear Regression": "#3b82f6",
    "KNN Regressor": "#059669",
    "Decision Tree Regressor": "#d97706",
    "Random Forest Regressor": "#7c3aed",
    "SVR (RBF)": "#e11d48",
    "Gradient Boosting Regressor": "#0d9488",
    "Extra Trees Regressor": "#0284c7",
    "XGBoost Regressor": "#ea580c",
}


def get_classification_models(balance_weights: bool = False, random_state: int = 42) -> Dict[str, Any]:
    """Returns the pool of classification models with optional balanced class weights."""
    cw = "balanced" if balance_weights else None
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight=cw,
            random_state=random_state,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance" if balance_weights else "uniform",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight=cw,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight=cw,
            random_state=random_state,
            n_jobs=-1,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            probability=True,
            class_weight=cw,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.1,
            max_depth=4,
            random_state=random_state,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=150,
            class_weight=cw,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        )

    return models


def get_regression_models(random_state: int = 42) -> Dict[str, Any]:
    """Returns the pool of regression models."""
    models = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=8, random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150, max_depth=12, random_state=random_state, n_jobs=-1
        ),
        "SVR (RBF)": SVR(kernel="rbf", C=1.0),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=120, learning_rate=0.1, max_depth=4, random_state=random_state
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=150, random_state=random_state, n_jobs=-1
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost Regressor"] = XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state,
            verbosity=0,
        )

    return models
