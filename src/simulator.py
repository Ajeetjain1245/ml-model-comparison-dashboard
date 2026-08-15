"""
Interactive What-If Prediction Simulator with Clean Light Theme.
"""
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from .visualizer import apply_light_theme


def render_prediction_simulator(
    pipelines: Dict[str, Any],
    num_cols: List[str],
    cat_cols: List[str],
    df_raw: pd.DataFrame,
    label_encoder: Optional[LabelEncoder],
    task_type: str = "classification",
    best_model_name: Optional[str] = None,
):
    """
    Renders an interactive UI where users can input custom feature values
    and test real-time predictions with confidence probabilities.
    """
    st.markdown("### 🎯 Interactive What-If Prediction Simulator")
    st.caption("Adjust input parameters below to test real-time predictions across any trained model.")

    if not pipelines:
        st.info("No trained models available for simulation.")
        return

    # Model selector
    model_options = list(pipelines.keys())
    default_idx = model_options.index(best_model_name) if best_model_name in model_options else 0
    selected_model_name = st.selectbox(
        "🤖 Select Model for Inference:",
        options=model_options,
        index=default_idx,
    )
    pipeline = pipelines[selected_model_name]

    # Input form in 2 or 3 columns
    st.markdown("#### ⚙️ Feature Inputs")
    input_cols = num_cols + cat_cols
    col_layout = st.columns(3 if len(input_cols) >= 3 else 2)
    user_inputs = {}

    for i, col in enumerate(input_cols):
        with col_layout[i % len(col_layout)]:
            if col in num_cols:
                series = df_raw[col].dropna()
                min_v = float(series.min()) if len(series) > 0 else 0.0
                max_v = float(series.max()) if len(series) > 0 else 100.0
                med_v = float(series.median()) if len(series) > 0 else (min_v + max_v) / 2.0

                step = (max_v - min_v) / 100.0 if max_v != min_v else 0.1
                step = round(step, 4) if step < 1 else round(step, 2)

                user_inputs[col] = st.number_input(
                    label=f"🔢 {col}",
                    value=round(med_v, 4),
                    step=max(step, 0.0001),
                    key=f"sim_num_{col}",
                )
            else:
                options = df_raw[col].dropna().unique().tolist()
                options = [str(o) for o in options] if options else ["Unknown"]
                user_inputs[col] = st.selectbox(
                    label=f"🔤 {col}",
                    options=options,
                    index=0,
                    key=f"sim_cat_{col}",
                )

    # Convert to single-row DataFrame
    input_df = pd.DataFrame([user_inputs])

    st.markdown("<hr>", unsafe_allow_html=True)

    # Run Prediction
    try:
        raw_pred = pipeline.predict(input_df)[0]

        res_c1, res_c2 = st.columns([1, 1.5])

        with res_c1:
            st.markdown("#### 📋 Inference Result")
            if task_type == "classification":
                pred_label = label_encoder.inverse_transform([raw_pred])[0] if label_encoder else str(raw_pred)
                st.success(f"**Predicted Class:**\n### 🏷️ `{pred_label}`")
            else:
                st.success(f"**Predicted Value:**\n### 📈 `{raw_pred:,.4f}`")

        with res_c2:
            if task_type == "classification" and hasattr(pipeline, "predict_proba"):
                try:
                    probs = pipeline.predict_proba(input_df)[0]
                    classes = label_encoder.classes_ if label_encoder else [f"Class {i}" for i in range(len(probs))]

                    prob_fig = go.Figure(
                        go.Bar(
                            x=probs,
                            y=[str(c) for c in classes],
                            orientation="h",
                            marker=dict(
                                color=["#059669" if c == pred_label else "#2563eb" for c in classes],
                            ),
                            text=[f"{p * 100:.1f}%" for p in probs],
                            textposition="outside",
                            hovertemplate="Class: %{y}<br>Probability: %{x:.2%}<extra></extra>",
                        )
                    )
                    apply_light_theme(
                        prob_fig,
                        height=240,
                        title="<b>Prediction Confidence Breakdown</b>",
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    prob_fig.update_layout(
                        xaxis=dict(range=[0, 1.15], title="Confidence Probability"),
                        yaxis=dict(title=""),
                    )
                    st.plotly_chart(prob_fig, use_container_width=True)
                except Exception:
                    pass

    except Exception as e:
        st.error(f"Inference error: {str(e)}")
