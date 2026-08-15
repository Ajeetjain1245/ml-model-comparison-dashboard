"""
ML Model Comparison Studio · v3.1
====================================
Enterprise-grade machine learning benchmarking studio with leak-free preprocessing,
zero-dependency SMOTE & resampling, interactive Plotly visualizations, live What-If
simulator, and production artifact export.
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

from src.ui_theme import (
    apply_theme,
    render_hero_header,
    render_winner_spotlight,
    render_podium_cards,
)
from src.data_loader import (
    get_sample_datasets_info,
    load_uploaded_csv,
    detect_task_type,
)
from src.preprocessor import prepare_features_and_target
from src.models import (
    get_classification_models,
    get_regression_models,
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
from src.simulator import render_prediction_simulator
from src.exporter import render_export_tab


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION & THEME INJECTION
# ╚══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ML Model Comparison Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()


# ╔══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: DATA & MODEL CONFIGURATION
# ╚══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Studio Controls")
    st.caption("v3.1 · Automated Benchmarking & Explainability")
    st.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)

    # 1. Dataset Source
    st.markdown("### 📁 1. Dataset Source")
    data_source = st.radio(
        "Choose source:",
        options=["✨ Built-in Benchmark", "📂 Upload Custom CSV"],
        index=0,
        label_visibility="collapsed",
    )

    df_raw = None
    target_col = None
    sample_datasets = get_sample_datasets_info()

    if data_source == "✨ Built-in Benchmark":
        selected_sample_name = st.selectbox(
            "Select Benchmark Dataset:",
            options=list(sample_datasets.keys()),
            index=0,
        )
        sample_meta = sample_datasets[selected_sample_name]
        df_raw, target_col = sample_meta["loader"]()
        st.caption(f"ℹ️ {sample_meta['description']}")

    else:
        uploaded_file = st.file_uploader("Upload CSV dataset:", type=["csv"])
        if uploaded_file is not None:
            try:
                df_raw = load_uploaded_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    if df_raw is not None:
        st.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)
        st.markdown("### 🎯 2. Target & Task")
        cols = df_raw.columns.tolist()
        default_target_idx = cols.index(target_col) if target_col in cols else len(cols) - 1
        target_col = st.selectbox("Target (Label) Column:", options=cols, index=default_target_idx)

        # Auto-detect task with manual toggle
        auto_task = detect_task_type(df_raw, target_col)
        task_type = st.radio(
            "Task Paradigm:",
            options=["classification", "regression"],
            index=0 if auto_task == "classification" else 1,
            format_func=lambda x: "🎯 Classification" if x == "classification" else "📈 Regression",
            horizontal=True,
        )

        st.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)
        st.markdown("### ⚖️ 3. Imbalance Handling")
        if task_type == "classification":
            imbalance_strategy = st.selectbox(
                "Sampling & Balancing Strategy:",
                options=[
                    "None",
                    "SMOTE (Synthetic Minority Over-sampling)",
                    "Balanced Class Weights",
                    "Random Over-sampling",
                    "Random Under-sampling",
                ],
                index=1,  # Default to SMOTE
                help="Leak-free resampling applied strictly within training folds.",
            )
            # Use class weights if selected
            use_cw = (imbalance_strategy == "Balanced Class Weights")
            available_models = get_classification_models(balance_weights=use_cw)
        else:
            imbalance_strategy = "None"
            st.caption("Imbalance handling applies to classification tasks.")
            available_models = get_regression_models()

        st.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)
        st.markdown("### 🔧 4. Validation Settings")
        test_size = st.slider(
            "Test Split Fraction", min_value=0.10, max_value=0.40, value=0.20, step=0.05,
            help="Fraction of data held out for unbiased test evaluation."
        )
        cv_folds = st.slider(
            "Cross-Validation Folds (k)", min_value=2, max_value=10, value=5,
            help="Number of folds for leak-free cross-validation."
        )

        st.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)
        st.markdown("### 🤖 5. Active Model Pool")
        selected_model_flags = {}
        for m_name in available_models.keys():
            selected_model_flags[m_name] = st.checkbox(m_name, value=True)

        active_models = {k: available_models[k] for k, v in selected_model_flags.items() if v}

        st.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)
        run_btn = st.button("🚀 Run Model Benchmark", use_container_width=True)

    st.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)
    st.caption("ML Comparison Studio · Streamlit + Scikit-Learn + Plotly")


# ╔══════════════════════════════════════════════════════════════════════════════
# MAIN VIEWPORT
# ╚══════════════════════════════════════════════════════════════════════════════
render_hero_header()

if df_raw is None:
    st.markdown(
        """<div class="studio-card">
<h3 style="margin-top:0; color:#0f172a !important;">👋 Welcome to ML Model Comparison Studio</h3>
<p style="color:#475569; line-height:1.6; margin-bottom:0;">
Train, compare, and diagnose multiple machine learning algorithms side-by-side with rigorous
leak-free cross-validation, built-in SMOTE oversampling, interactive Plotly visualizers,
and a live What-If inference simulator.
</p>
</div>""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("📁 **1. Choose Dataset**\n\nPick a built-in benchmark dataset (e.g. Breast Cancer, Wine, Titanic) or upload your custom CSV.")
    with c2:
        st.info("⚙️ **2. Configure Pipeline**\n\nChoose target feature, select SMOTE or class weighting, and pick algorithms.")
    with c3:
        st.info("🚀 **3. Benchmark & Deploy**\n\nExplore podium leaderboards, ROC/PR curves, feature importance, test live predictions, and export `.joblib`.")
    st.stop()


# ╔══════════════════════════════════════════════════════════════════════════════
# DATASET OVERVIEW & EDA
# ╚══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📋 Dataset Profile & Exploratory Analysis")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows", f"{df_raw.shape[0]:,}")
k2.metric("Features", f"{df_raw.shape[1] - 1:,}")
k3.metric("Missing Values", f"{df_raw.isnull().sum().sum():,}")
k4.metric("Target Feature", target_col)
k5.metric("Task Mode", task_type.capitalize())

with st.expander("🔍 Inspect Raw Dataset & Descriptive Statistics", expanded=False):
    t_data, t_stats = st.tabs(["Raw Data (First 100 Rows)", "Summary Statistics"])
    with t_data:
        st.dataframe(df_raw.head(100), use_container_width=True)
    with t_stats:
        st.dataframe(df_raw.describe(include="all"), use_container_width=True)

# Visual EDA Row
eda_col1, eda_col2 = st.columns([1, 1.2])
with eda_col1:
    st.plotly_chart(plot_target_distribution(df_raw, target_col, task_type), use_container_width=True)
with eda_col2:
    corr_fig = plot_correlation_heatmap(df_raw)
    if corr_fig:
        st.plotly_chart(corr_fig, use_container_width=True)
    else:
        st.info("No numerical correlation matrix available.")

st.markdown("<hr>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING TRIGGER
# ╚══════════════════════════════════════════════════════════════════════════════
if run_btn:
    if not active_models:
        st.error("Please select at least one model from the sidebar to train.")
        st.stop()

    pb = st.progress(0.0)
    status_box = st.empty()

    def update_progress(current_idx: int, total_idx: int, message: str):
        fraction = max(0.0, min(1.0, current_idx / max(1, total_idx)))
        pb.progress(fraction)
        status_box.markdown(f"⚙️ **{message}**")

    try:
        X_df, y, num_cols, cat_cols, label_encoder = prepare_features_and_target(
            df=df_raw,
            target_col=target_col,
            task_type=task_type,
        )

        eval_output = train_and_evaluate_all(
            X_df=X_df,
            y=y,
            num_cols=num_cols,
            cat_cols=cat_cols,
            active_models=active_models,
            task_type=task_type,
            imbalance_strategy=imbalance_strategy,
            test_size=test_size,
            cv_folds=cv_folds,
            progress_callback=update_progress,
        )

        # Cache in session state
        st.session_state["eval_output"] = eval_output
        st.session_state["num_cols"] = num_cols
        st.session_state["cat_cols"] = cat_cols
        st.session_state["label_encoder"] = label_encoder
        st.session_state["task_type"] = task_type
        st.session_state["target_col"] = target_col
        st.session_state["df_raw"] = df_raw
        st.session_state["imbalance_strategy"] = imbalance_strategy

        status_box.success("🎉 All models successfully trained and evaluated!")

    except Exception as e:
        status_box.error(f"❌ Training pipeline error: {str(e)}")
        st.stop()


# ╔══════════════════════════════════════════════════════════════════════════════
# RESULTS DASHBOARD
# ╚══════════════════════════════════════════════════════════════════════════════
if "eval_output" not in st.session_state:
    st.info("👈 Choose your settings in the sidebar and click **🚀 Run Model Benchmark** to evaluate models.")
    st.stop()

eval_data = st.session_state["eval_output"]
results = eval_data["results"]
pipelines = eval_data["pipelines"]
df_leaderboard = eval_data["leaderboard"]
transformed_feat_names = eval_data["transformed_feature_names"]
num_cols = st.session_state["num_cols"]
cat_cols = st.session_state["cat_cols"]
label_encoder = st.session_state["label_encoder"]
curr_task_type = st.session_state["task_type"]
n_classes = eval_data["n_classes"]

best_row = df_leaderboard.iloc[0]
best_model_name = best_row["Model"]

# Winner Spotlight Card
render_winner_spotlight(best_row, curr_task_type)


# ── Interactive Tabs ──────────────────────────────────────────────────────────
tab_arena, tab_diag, tab_fi, tab_sim, tab_exp = st.tabs([
    "🏆 Leaderboard Arena",
    "📈 Deep Diagnostics",
    "🌿 Feature Importance",
    "🎯 What-If Simulator",
    "💾 Export & Deploy",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: LEADERBOARD ARENA
# ══════════════════════════════════════════════════════════════════════════════
with tab_arena:
    # 1. Podium Cards
    render_podium_cards(df_leaderboard, curr_task_type)

    st.markdown("### 📊 Complete Leaderboard Matrix")
    metric_cols = [c for c in df_leaderboard.columns if c != "Model"]

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #064e3b; color: #34d399; font-weight: bold' if v else '' for v in is_max]

    styled_df = df_leaderboard.style.apply(highlight_max, subset=metric_cols).format({c: "{:.4f}" for c in metric_cols})
    st.dataframe(styled_df, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    col_rad, col_bar = st.columns([1, 1.4])

    with col_rad:
        st.plotly_chart(plot_radar_chart(results, curr_task_type), use_container_width=True)

    with col_bar:
        selected_metric = st.selectbox("Rank models by metric:", options=metric_cols, index=0)
        st.plotly_chart(plot_leaderboard_bar(df_leaderboard, selected_metric, curr_task_type), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: DEEP DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    if curr_task_type == "classification":
        class_names = [str(c) for c in label_encoder.classes_] if label_encoder else [str(c) for c in np.unique(eval_data["y_test"])]

        # ROC & PR Curves
        st.markdown("### 📈 Decision Threshold Curves (ROC & PR)")
        c_roc, c_pr = st.columns(2)
        with c_roc:
            roc_fig = plot_roc_curves(results, class_names, n_classes)
            if roc_fig:
                st.plotly_chart(roc_fig, use_container_width=True)
            else:
                st.info("Probability estimates not available for ROC curves.")
        with c_pr:
            pr_fig = plot_precision_recall_curves(results, n_classes)
            if pr_fig:
                st.plotly_chart(pr_fig, use_container_width=True)
            else:
                st.caption("Precision-Recall curve available for binary classification.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🔲 Confusion Matrices")
        norm_cm = st.checkbox("Show Row-Normalized Percentages (%) in Confusion Matrix", value=False)
        cm_cols = st.columns(min(len(results), 3))
        for idx, (name, r) in enumerate(results.items()):
            with cm_cols[idx % len(cm_cols)]:
                cm_fig = plot_confusion_matrix(
                    y_test=r["y_test"],
                    y_pred=r["y_pred"],
                    model_name=name,
                    class_names=class_names,
                    normalize=norm_cm,
                )
                st.plotly_chart(cm_fig, use_container_width=True)

    else:
        st.markdown("### 📈 Actual vs. Predicted Regression Diagnostics")
        reg_cols = st.columns(min(len(results), 2))
        for idx, (name, r) in enumerate(results.items()):
            with reg_cols[idx % len(reg_cols)]:
                avp_fig = plot_actual_vs_predicted(r["y_test"], r["y_pred"], name)
                st.plotly_chart(avp_fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_fi:
    st.markdown("### 🌿 Feature & Permutation Importance")
    st.caption("Identifies the features with the strongest decision boundary influence.")

    fi_c1, fi_c2 = st.columns([1, 3])
    with fi_c1:
        fi_model_name = st.selectbox("Select Model:", options=list(results.keys()), index=0)
        top_k = st.slider("Top Features to Display", min_value=5, max_value=30, value=15)

    with fi_c2:
        model_result = results[fi_model_name]
        imp_array = model_result.get("feature_importance")

        if imp_array is not None and len(imp_array) > 0:
            fi_fig = plot_feature_importance(
                importances=imp_array,
                feat_names=transformed_feat_names,
                model_name=fi_model_name,
                top_n=top_k,
            )
            if fi_fig:
                st.plotly_chart(fi_fig, use_container_width=True)
            else:
                st.info(f"Feature importance could not be rendered for {fi_model_name}.")
        else:
            st.info(f"Feature importance is not available for {fi_model_name}.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    render_prediction_simulator(
        pipelines=pipelines,
        num_cols=num_cols,
        cat_cols=cat_cols,
        df_raw=st.session_state["df_raw"],
        label_encoder=label_encoder,
        task_type=curr_task_type,
        best_model_name=best_model_name,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: EXPORT & DEPLOY
# ══════════════════════════════════════════════════════════════════════════════
with tab_exp:
    render_export_tab(
        results=results,
        pipelines=pipelines,
        df_leaderboard=df_leaderboard,
        num_cols=num_cols,
        cat_cols=cat_cols,
        label_encoder=label_encoder,
        task_type=curr_task_type,
        best_model_name=best_model_name,
    )
