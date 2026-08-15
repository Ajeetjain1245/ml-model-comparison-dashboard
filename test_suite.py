"""
Comprehensive Automated Test Suite for ML Model Comparison Studio v3.1
"""
import sys
import unittest
import numpy as np
import pandas as pd

from src.data_loader import (
    get_sample_datasets_info,
    detect_task_type,
)
from src.preprocessor import prepare_features_and_target, build_preprocessor, get_feature_names_out
from src.models import (
    get_classification_models,
    get_regression_models,
)
from src.imbalance import (
    smote_resample,
    random_oversample,
    random_undersample,
    apply_imbalance_handling,
)
from src.evaluator import train_and_evaluate_all
from src.visualizer import (
    plot_leaderboard_bar,
    plot_radar_chart,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_feature_importance,
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_target_distribution,
)
from src.exporter import (
    serialize_pipeline_to_bytes,
    generate_python_inference_code,
)


class TestMLStudio(unittest.TestCase):

    def test_01_data_loader_and_samples(self):
        """Test loading built-in benchmark datasets and task detection."""
        samples = get_sample_datasets_info()
        self.assertIn("Breast Cancer (Binary Classification)", samples)
        self.assertIn("Titanic Survival (Mixed Tabular & Missing Data)", samples)
        self.assertIn("California Housing (Regression)", samples)

        df_bc, target_bc = samples["Breast Cancer (Binary Classification)"]["loader"]()
        self.assertEqual(target_bc, "diagnosis")
        self.assertGreater(len(df_bc), 100)
        self.assertEqual(detect_task_type(df_bc, target_bc), "classification")

        df_tit, target_tit = samples["Titanic Survival (Mixed Tabular & Missing Data)"]["loader"]()
        self.assertEqual(target_tit, "Survived")
        self.assertEqual(detect_task_type(df_tit, target_tit), "classification")

        df_house, target_house = samples["California Housing (Regression)"]["loader"]()
        self.assertEqual(detect_task_type(df_house, target_house), "regression")

    def test_02_imbalance_resampling_engine(self):
        """Test SMOTE, Random Oversampling, and Undersampling engines."""
        # Create severely imbalanced synthetic dataset: 100 Class 0, 10 Class 1
        np.random.seed(42)
        X_imb = np.random.randn(110, 4)
        y_imb = np.array([0] * 100 + [1] * 10)

        # 1. Test SMOTE
        X_smote, y_smote = smote_resample(X_imb, y_imb, k_neighbors=3, random_state=42)
        self.assertEqual(len(y_smote), 200)  # 100 + 100
        self.assertEqual(np.sum(y_smote == 0), 100)
        self.assertEqual(np.sum(y_smote == 1), 100)

        # 2. Test Random Oversampling
        X_over, y_over = random_oversample(X_imb, y_imb, random_state=42)
        self.assertEqual(len(y_over), 200)

        # 3. Test Random Undersampling
        X_under, y_under = random_undersample(X_imb, y_imb, random_state=42)
        self.assertEqual(len(y_under), 20)  # 10 + 10

        # 4. Test apply_imbalance_handling wrapper
        X_res, y_res = apply_imbalance_handling(X_imb, y_imb, strategy="SMOTE (Synthetic Minority Over-sampling)")
        self.assertEqual(len(y_res), 200)

    def test_03_classification_evaluation_with_smote(self):
        """Test full training, leak-free CV, and metrics with SMOTE enabled."""
        samples = get_sample_datasets_info()
        df, target = samples["Breast Cancer (Binary Classification)"]["loader"]()
        df_sub = df.sample(120, random_state=42)

        X_df, y, num_cols, cat_cols, le = prepare_features_and_target(df_sub, target, "classification")
        
        models = {
            "Logistic Regression": get_classification_models()["Logistic Regression"],
            "Random Forest": get_classification_models()["Random Forest"],
        }

        output = train_and_evaluate_all(
            X_df=X_df,
            y=y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            active_models=models,
            task_type="classification",
            imbalance_strategy="SMOTE (Synthetic Minority Over-sampling)",
            test_size=0.25,
            cv_folds=3,
        )

        self.assertIn("results", output)
        self.assertIn("leaderboard", output)
        results = output["results"]
        self.assertIn("Logistic Regression", results)
        self.assertGreaterEqual(results["Logistic Regression"]["Accuracy"], 0.5)
        self.assertIn("CV Accuracy", results["Logistic Regression"])

    def test_04_regression_evaluation(self):
        """Test full training and metrics on regression."""
        samples = get_sample_datasets_info()
        df, target = samples["California Housing (Regression)"]["loader"]()
        df_sub = df.sample(120, random_state=42)

        X_df, y, num_cols, cat_cols, le = prepare_features_and_target(df_sub, target, "regression")
        
        models = {
            "Ridge Regression": get_regression_models()["Ridge Regression"],
            "KNN Regressor": get_regression_models()["KNN Regressor"],
        }

        output = train_and_evaluate_all(
            X_df=X_df,
            y=y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            active_models=models,
            task_type="regression",
            test_size=0.25,
            cv_folds=3,
        )

        results = output["results"]
        self.assertIn("Ridge Regression", results)
        self.assertIn("R2 Score", results["Ridge Regression"])
        self.assertIn("RMSE", results["Ridge Regression"])

    def test_05_visualizations_render(self):
        """Test generation of all Plotly figures without exceptions."""
        samples = get_sample_datasets_info()
        df, target = samples["Breast Cancer (Binary Classification)"]["loader"]()
        df_sub = df.sample(100, random_state=42)
        X_df, y, num_cols, cat_cols, le = prepare_features_and_target(df_sub, target, "classification")

        models = {"Logistic Regression": get_classification_models()["Logistic Regression"]}
        output = train_and_evaluate_all(X_df, y, num_cols, cat_cols, models, "classification", cv_folds=2)

        results = output["results"]
        df_lead = output["leaderboard"]
        class_names = le.classes_.tolist()

        fig_bar = plot_leaderboard_bar(df_lead, "Accuracy", "classification")
        self.assertIsNotNone(fig_bar)

        fig_radar = plot_radar_chart(results, "classification")
        self.assertIsNotNone(fig_radar)

        fig_cm = plot_confusion_matrix(results["Logistic Regression"]["y_test"], results["Logistic Regression"]["y_pred"], "Logistic Regression", class_names)
        self.assertIsNotNone(fig_cm)

        fig_roc = plot_roc_curves(results, class_names, 2)
        self.assertIsNotNone(fig_roc)

        fig_corr = plot_correlation_heatmap(df_sub)
        self.assertIsNotNone(fig_corr)

        fig_dist = plot_target_distribution(df_sub, target, "classification")
        self.assertIsNotNone(fig_dist)

    def test_06_exporter(self):
        """Test model pipeline serialization and python code generator."""
        samples = get_sample_datasets_info()
        df, target = samples["Breast Cancer (Binary Classification)"]["loader"]()
        df_sub = df.sample(80, random_state=42)
        X_df, y, num_cols, cat_cols, le = prepare_features_and_target(df_sub, target, "classification")

        models = {"Decision Tree": get_classification_models()["Decision Tree"]}
        output = train_and_evaluate_all(X_df, y, num_cols, cat_cols, models, "classification", cv_folds=2)

        pipeline = output["pipelines"]["Decision Tree"]
        b_data = serialize_pipeline_to_bytes(pipeline, "Decision Tree", le, num_cols, cat_cols, "classification")
        self.assertGreater(len(b_data), 100)

        code = generate_python_inference_code("Decision Tree", num_cols, cat_cols, "classification")
        self.assertIn("joblib.load", code)
        self.assertIn("pipeline.predict", code)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMLStudio)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
