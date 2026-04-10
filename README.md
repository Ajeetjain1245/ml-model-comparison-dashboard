# ML Model Comparison Dashboard

An interactive machine learning dashboard built using Streamlit to compare multiple models on any dataset.

##  Features

* Upload CSV dataset
* Automatic data preprocessing (missing values, encoding, scaling)
* Multiple ML models (Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Gradient Boosting, XGBoost)
* Handles imbalanced data using SMOTE
* Advanced evaluation metrics:

  * Accuracy
  * F1 Score
  * Precision & Recall
  * ROC-AUC
* Visualizations:

  * Confusion Matrix
  * ROC Curves
  * Feature Importance
  * Performance Charts
* Model comparison leaderboard

##  Tech Stack

* Python
* Scikit-learn
* Streamlit
* Pandas, NumPy
* Matplotlib, Seaborn
* XGBoost

##  How to Run

```bash
pip install -r requirements.txt
streamlit run ml_dashboard.py
```

##  Use Case

This tool helps compare multiple machine learning models quickly and understand their performance using advanced metrics and visualizations.
