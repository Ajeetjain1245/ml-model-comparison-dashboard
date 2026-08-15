# ⚡ ML Model Comparison & Evaluation Studio

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> 🔗 **Live Demo App:** [ml-dashboard-ajeet.streamlit.app](https://ml-dashboard-ajeet.streamlit.app)

An enterprise-grade, end-to-end Machine Learning benchmarking studio built with **Streamlit**, **Scikit-Learn**, and **Plotly**. Train, compare, and diagnose multiple classification and regression models simultaneously with **zero data leakage**, built-in **k-NN SMOTE resampling**, interactive visualizers, a live **What-If inference simulator**, and one-click **production `.joblib` model export**.

---

## 🌟 Key Features

- 🛡️ **Zero Data Leakage Pipeline**: Preprocessing (`StandardScaler`, `SimpleImputer`, `OneHotEncoder`) encapsulated in `ColumnTransformer` and fit strictly on training splits and inside cross-validation folds.
- ⚖️ **Robust Imbalance & Resampling Engine**: Pure k-NN **SMOTE (Synthetic Minority Over-sampling)**, Random Over/Under-sampling, and cost-sensitive class weighting.
- 🎯 **Dual Task Paradigm**: Intelligent auto-detection for both **Classification** (Accuracy, Balanced Acc, F1, Precision, Recall, ROC-AUC) and **Regression** ($R^2$, RMSE, MAE, MAPE).
- 📊 **Interactive Plotly Visualizations**: Modern, light-themed responsive charts:
  - 🏆 Multidimensional Performance Radar
  - 🔲 Confusion Matrix (toggle raw counts vs. row-normalized percentages)
  - 📈 Overlaid ROC & Precision-Recall Curves with probability threshold tooltips
  - 🌿 Feature Importance & Model-Agnostic Permutation Importance
  - 🔍 Automated Exploratory Data Analysis (Correlation Heatmaps, Target Distributions)
- 🎮 **Live What-If Prediction Simulator**: Test real-time predictions with dynamic sliders and probability confidence meters.
- 💾 **Model Serialization & Export**: Download fitted `.joblib` pipeline bundles and auto-generated standalone Python inference scripts.
- ✨ **1-Click Benchmark Datasets**: Built-in loaders for *Breast Cancer*, *Wine*, *Titanic*, *Iris*, *California Housing*, and *Diabetes*.

---

## 🏗️ Architecture & Technical Flow

```
ml-model-comparison-dashboard/
│
├── app.py                      # Main Streamlit dashboard application (Modular UI & Tabs)
├── ml_dashboard.py             # Entry point / backward compatibility wrapper
├── requirements.txt            # Project dependencies
├── test_suite.py               # Automated unit tests
├── ajeet.md                    # Detailed architectural breakdown & interview guide
│
└── src/
    ├── __init__.py             # Package initialization
    ├── ui_theme.py             # Modern Light Theme design system & components
    ├── imbalance.py            # Zero-dependency SMOTE & resampling strategies
    ├── data_loader.py          # Benchmark datasets & CSV ingestion
    ├── preprocessor.py         # Sklearn ColumnTransformer (Leak-free preprocessing)
    ├── models.py               # Classification and Regression model registries
    ├── evaluator.py            # Leak-free cross-validation & evaluation engine
    ├── visualizer.py           # Plotly interactive visualizations
    ├── simulator.py            # Live What-If inference simulator
    └── exporter.py             # .joblib serializer & Python script generator
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Ajeetjain1245/ml-model-comparison-dashboard.git
cd ml-model-comparison-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```

### 4. Run Automated Test Suite
```bash
python test_suite.py
```

---

## 📋 Supported Algorithms

| Paradigm | Supported Models |
| :--- | :--- |
| **Classification** | Logistic Regression · Random Forest · Gradient Boosting · XGBoost · SVM (RBF) · K-Nearest Neighbors · Decision Tree · Extra Trees |
| **Regression** | Ridge Regression · Linear Regression · Random Forest Regressor · Gradient Boosting Regressor · XGBoost Regressor · KNN Regressor · SVR · Extra Trees Regressor |

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
