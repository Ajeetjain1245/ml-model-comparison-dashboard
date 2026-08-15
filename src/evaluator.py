"""
Leak-free training, cross-validation, and metrics evaluation engine with SMOTE & Imbalance handling.
"""
from typing import Dict, Any, List, Optional, Callable
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.inspection import permutation_importance
from .preprocessor import build_preprocessor, get_feature_names_out
from .imbalance import apply_imbalance_handling


def train_and_evaluate_all(
    X_df: pd.DataFrame,
    y: np.ndarray,
    num_cols: List[str],
    cat_cols: List[str],
    active_models: Dict[str, Any],
    task_type: str = "classification",
    imbalance_strategy: str = "None",
    test_size: float = 0.20,
    cv_folds: int = 5,
    random_state: int = 42,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Executes leak-free training, holdout evaluation, and cross-validation with SMOTE/resampling.
    """
    stratify = y if (task_type == "classification" and len(np.unique(y)) > 1) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    n_models = len(active_models)
    results = {}
    fitted_pipelines = {}
    transformed_feature_names = []

    if task_type == "classification":
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        n_classes = len(np.unique(y))
    else:
        cv_splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        n_classes = 0

    for idx, (name, base_model) in enumerate(active_models.items()):
        if progress_callback:
            progress_callback(idx, n_models, f"Training & validating {name} ({idx+1}/{n_models})...")

        # 1. Fit Preprocessor ONLY on X_train (Zero Data Leakage)
        preprocessor = build_preprocessor(num_cols, cat_cols)
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)

        if not transformed_feature_names:
            transformed_feature_names = get_feature_names_out(preprocessor, num_cols, cat_cols)

        # 2. Apply Imbalance Resampling (SMOTE / Oversample / Undersample) on X_train_trans
        if task_type == "classification" and imbalance_strategy != "None":
            X_train_fit, y_train_fit = apply_imbalance_handling(
                X_train_trans, y_train, strategy=imbalance_strategy, random_state=random_state
            )
        else:
            X_train_fit, y_train_fit = X_train_trans, y_train

        # 3. Fit Model
        model = clone(base_model)
        model.fit(X_train_fit, y_train_fit)

        # 4. Assemble complete deployable pipeline
        full_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        fitted_pipelines[name] = full_pipeline

        # 5. Holdout Evaluation
        y_pred = model.predict(X_test_trans)

        if task_type == "classification":
            y_prob = None
            roc_auc = np.nan
            try:
                if hasattr(model, "predict_proba"):
                    y_prob_full = model.predict_proba(X_test_trans)
                    if n_classes == 2:
                        y_prob = y_prob_full[:, 1]
                        roc_auc = roc_auc_score(y_test, y_prob)
                    else:
                        roc_auc = roc_auc_score(
                            y_test, y_prob_full, multi_class="ovr", average="weighted"
                        )
                        y_prob = y_prob_full
            except Exception:
                pass

            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

            # Leak-free cross validation with imbalance handling
            cv_acc_scores = []
            cv_f1_scores = []
            for tr_idx, val_idx in cv_splitter.split(X_df, y):
                X_cv_tr, X_cv_val = X_df.iloc[tr_idx], X_df.iloc[val_idx]
                y_cv_tr, y_cv_val = y[tr_idx], y[val_idx]

                cv_prep = build_preprocessor(num_cols, cat_cols)
                X_cv_tr_trans = cv_prep.fit_transform(X_cv_tr)
                X_cv_val_trans = cv_prep.transform(X_cv_val)

                if imbalance_strategy != "None":
                    X_cv_tr_trans, y_cv_tr = apply_imbalance_handling(
                        X_cv_tr_trans, y_cv_tr, strategy=imbalance_strategy, random_state=random_state
                    )

                cv_m = clone(base_model)
                cv_m.fit(X_cv_tr_trans, y_cv_tr)
                cv_pred = cv_m.predict(X_cv_val_trans)

                cv_acc_scores.append(accuracy_score(y_cv_val, cv_pred))
                cv_f1_scores.append(f1_score(y_cv_val, cv_pred, average="weighted", zero_division=0))

            cv_acc = np.mean(cv_acc_scores)
            cv_f1 = np.mean(cv_f1_scores)

            feat_imp = _extract_feature_importance(model, full_pipeline, X_test, y_test)

            results[name] = {
                "pipeline": full_pipeline,
                "model": model,
                "y_test": y_test,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "feature_importance": feat_imp,
                "Accuracy": round(acc, 4),
                "Balanced Acc": round(bal_acc, 4),
                "F1 Score": round(f1, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "ROC AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
                "CV Accuracy": round(cv_acc, 4),
                "CV F1 Score": round(cv_f1, 4),
            }

        else: # Regression
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            try:
                mape = mean_absolute_percentage_error(y_test, y_pred)
            except Exception:
                mape = np.nan

            cv_r2_scores = []
            cv_rmse_scores = []
            for tr_idx, val_idx in cv_splitter.split(X_df, y):
                X_cv_tr, X_cv_val = X_df.iloc[tr_idx], X_df.iloc[val_idx]
                y_cv_tr, y_cv_val = y[tr_idx], y[val_idx]

                cv_prep = build_preprocessor(num_cols, cat_cols)
                X_cv_tr_trans = cv_prep.fit_transform(X_cv_tr)
                X_cv_val_trans = cv_prep.transform(X_cv_val)

                cv_m = clone(base_model)
                cv_m.fit(X_cv_tr_trans, y_cv_tr)
                cv_pred = cv_m.predict(X_cv_val_trans)

                cv_r2_scores.append(r2_score(y_cv_val, cv_pred))
                cv_rmse_scores.append(np.sqrt(mean_squared_error(y_cv_val, cv_pred)))

            cv_r2 = np.mean(cv_r2_scores)
            cv_rmse = np.mean(cv_rmse_scores)

            feat_imp = _extract_feature_importance(model, full_pipeline, X_test, y_test, is_classifier=False)

            results[name] = {
                "pipeline": full_pipeline,
                "model": model,
                "y_test": y_test,
                "y_pred": y_pred,
                "feature_importance": feat_imp,
                "R2 Score": round(r2, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "MAPE": round(mape, 4) if not np.isnan(mape) else None,
                "CV R2": round(cv_r2, 4),
                "CV RMSE": round(cv_rmse, 4),
            }

    if progress_callback:
        progress_callback(n_models, n_models, "All models successfully trained & evaluated!")

    leaderboard = _create_leaderboard_df(results, task_type)

    return {
        "results": results,
        "pipelines": fitted_pipelines,
        "leaderboard": leaderboard,
        "transformed_feature_names": transformed_feature_names,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "task_type": task_type,
        "n_classes": n_classes,
        "imbalance_strategy": imbalance_strategy,
    }


def _extract_feature_importance(
    model: Any,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    is_classifier: bool = True,
) -> Optional[np.ndarray]:
    """Extracts tree-based feature importances, linear coefficients, or permutation importances."""
    if hasattr(model, "feature_importances_"):
        return np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            return np.mean(np.abs(coef), axis=0)
        return np.abs(coef)
    else:
        try:
            sample_size = min(len(X_test), 150)
            X_sample = X_test.iloc[:sample_size]
            y_sample = y_test[:sample_size]
            scoring = "f1_weighted" if is_classifier else "r2"
            pi = permutation_importance(
                pipeline, X_sample, y_sample, n_repeats=3, random_state=42, scoring=scoring, n_jobs=-1
            )
            return pi.importances_mean
        except Exception:
            return None


def _create_leaderboard_df(results: Dict[str, Any], task_type: str) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        row = {"Model": name}
        if task_type == "classification":
            for metric in [
                "Accuracy", "Balanced Acc", "F1 Score", "Precision", "Recall",
                "ROC AUC", "CV Accuracy", "CV F1 Score"
            ]:
                row[metric] = r.get(metric, np.nan)
        else:
            for metric in ["R2 Score", "RMSE", "MAE", "MAPE", "CV R2", "CV RMSE"]:
                row[metric] = r.get(metric, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_col = "Accuracy" if task_type == "classification" else "R2 Score"
    df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    df.index += 1
    return df
