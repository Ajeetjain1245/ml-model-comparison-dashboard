"""
Interactive Plotly visualizer with clean Light Theme styling.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from .models import MODEL_COLORS

LIGHT_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font=dict(family="Plus Jakarta Sans, -apple-system, sans-serif", color="#0f172a", size=12),
)


def apply_light_theme(
    fig: go.Figure,
    height: int = 400,
    title: str = "",
    margin: Optional[dict] = None,
) -> go.Figure:
    """Applies clean consistent light styling, soft gridlines, and typography to Plotly figures."""
    m = margin if margin is not None else dict(l=40, r=30, t=50, b=40)
    fig.update_layout(
        **LIGHT_LAYOUT,
        margin=m,
        height=height,
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color="#0f172a", size=14, family="Plus Jakarta Sans")))
    fig.update_xaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
    fig.update_yaxes(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
    return fig


def plot_leaderboard_bar(df_res: pd.DataFrame, metric: str, task_type: str = "classification") -> go.Figure:
    """Horizontal bar chart comparing models on a specific metric."""
    df_sorted = df_res.sort_values(by=metric, ascending=True).copy()
    colors = [MODEL_COLORS.get(name, "#2563eb") for name in df_sorted["Model"]]

    fig = go.Figure(
        go.Bar(
            x=df_sorted[metric],
            y=df_sorted["Model"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.4f}" if pd.notnull(v) else "N/A" for v in df_sorted[metric]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>" + metric + ": %{x:.4f}<extra></extra>",
        )
    )

    max_val = df_sorted[metric].max() if len(df_sorted) > 0 else 1.0
    x_range = [0, max(1.15, max_val * 1.15)] if task_type == "classification" else None

    apply_light_theme(
        fig,
        height=max(320, len(df_sorted) * 45),
        title=f"<b>Model Benchmark — {metric}</b>",
    )
    fig.update_layout(
        xaxis_title=metric,
        yaxis_title="",
        xaxis_range=x_range,
    )
    return fig


def plot_radar_chart(results: Dict[str, Any], task_type: str = "classification") -> go.Figure:
    """Interactive radar chart comparing multidimensional performance in light styling."""
    if task_type == "classification":
        categories = ["Accuracy", "Balanced Acc", "F1 Score", "Precision", "Recall", "CV Accuracy"]
    else:
        categories = ["R2 Score", "CV R2"]

    fig = go.Figure()

    for name, r in results.items():
        vals = [r.get(c, 0) or 0 for c in categories]
        vals.append(vals[0])
        col = MODEL_COLORS.get(name, "#2563eb")

        fig.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                name=name,
                line=dict(color=col, width=2.5),
                fill="toself",
                fillcolor=f"rgba({int(col[1:3], 16)}, {int(col[3:5], 16)}, {int(col[5:7], 16)}, 0.08)"
                if col.startswith("#") and len(col) == 7 else "rgba(37, 99, 235, 0.08)",
                hovertemplate="<b>" + name + "</b><br>%{theta}: %{r:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        **LIGHT_LAYOUT,
        title=dict(text="<b>Multidimensional Performance Radar</b>", font=dict(color="#0f172a", size=14)),
        polar=dict(
            bgcolor="#f8fafc",
            radialaxis=dict(visible=True, range=[0, 1] if task_type == "classification" else None, gridcolor="#e2e8f0"),
            angularaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1"),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=450,
        margin=dict(l=40, r=40, t=50, b=50),
    )
    return fig


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    class_names: List[str],
    normalize: bool = False,
) -> go.Figure:
    """Plotly interactive Confusion Matrix with light styling."""
    cm = confusion_matrix(y_test, y_pred)
    if normalize:
        cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-9)
        cm_display = np.round(cm_norm * 100, 1)
        text_template = [[f"{cm[i, j]}<br>({cm_display[i, j]:.1f}%)" for j in range(len(class_names))] for i in range(len(class_names))]
        z_vals = cm_norm
    else:
        text_template = [[f"{cm[i, j]}" for j in range(len(class_names))] for i in range(len(class_names))]
        z_vals = cm

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=[str(c) for c in class_names],
            y=[str(c) for c in class_names],
            text=text_template,
            texttemplate="%{text}",
            colorscale=[[0, "#ffffff"], [0.5, "#bae6fd"], [1.0, "#0284c7"]],
            showscale=True,
            colorbar=dict(title="Normalized" if normalize else "Count"),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
        )
    )

    apply_light_theme(fig, height=400, title=f"<b>Confusion Matrix — {model_name}</b>")
    fig.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        yaxis_autorange="reversed",
    )
    return fig


def plot_roc_curves(results: Dict[str, Any], class_names: List[str], n_classes: int) -> Optional[go.Figure]:
    """Overlaid interactive ROC Curves with light styling."""
    fig = go.Figure()
    has_valid_roc = False

    for name, r in results.items():
        y_prob = r.get("y_prob")
        y_test = r.get("y_test")
        if y_prob is None or y_test is None:
            continue

        col = MODEL_COLORS.get(name, "#2563eb")

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc_val = auc(fpr, tpr)
            has_valid_roc = True
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"{name} (AUC = {roc_auc_val:.3f})",
                    line=dict(color=col, width=2.5),
                    hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
                )
            )
        else:
            y_bin = label_binarize(y_test, classes=list(range(n_classes)))
            fpr_list, tpr_list = [], []
            for c_idx in range(n_classes):
                f, t, _ = roc_curve(y_bin[:, c_idx], y_prob[:, c_idx])
                fpr_list.append(f)
                tpr_list.append(t)
            all_fpr = np.unique(np.concatenate(fpr_list))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
            mean_tpr /= n_classes
            macro_auc = auc(all_fpr, mean_tpr)
            has_valid_roc = True

            fig.add_trace(
                go.Scatter(
                    x=all_fpr, y=mean_tpr, mode="lines", name=f"{name} (Macro AUC = {macro_auc:.3f})",
                    line=dict(color=col, width=2.5),
                    hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
                )
            )

    if not has_valid_roc:
        return None

    # Diagonal baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random Guess",
            line=dict(color="#94a3b8", dash="dash", width=1.5),
            hoverinfo="none",
        )
    )

    apply_light_theme(fig, height=480, title="<b>Receiver Operating Characteristic (ROC) Curves</b>")
    fig.update_layout(
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        xaxis_range=[-0.02, 1.02],
        yaxis_range=[-0.02, 1.05],
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
    )
    return fig


def plot_precision_recall_curves(results: Dict[str, Any], n_classes: int) -> Optional[go.Figure]:
    """Interactive Precision-Recall curves in light theme."""
    if n_classes != 2:
        return None

    fig = go.Figure()
    has_traces = False

    for name, r in results.items():
        y_prob = r.get("y_prob")
        y_test = r.get("y_test")
        if y_prob is None or y_test is None:
            continue

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        col = MODEL_COLORS.get(name, "#2563eb")
        has_traces = True

        fig.add_trace(
            go.Scatter(
                x=recall, y=precision, mode="lines", name=f"{name} (PR-AUC = {pr_auc:.3f})",
                line=dict(color=col, width=2.5),
                hovertemplate=f"<b>{name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>",
            )
        )

    if not has_traces:
        return None

    apply_light_theme(fig, height=480, title="<b>Precision-Recall (PR) Curves</b>")
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[-0.02, 1.02],
        yaxis_range=[-0.02, 1.05],
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
    )
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feat_names: List[str],
    model_name: str,
    top_n: int = 15,
) -> Optional[go.Figure]:
    """Plots top-N feature importance values in light styling."""
    if importances is None or len(importances) == 0:
        return None

    min_len = min(len(importances), len(feat_names))
    imps = importances[:min_len]
    names = feat_names[:min_len]

    clean_names = [n.replace("num__", "").replace("cat__", "") for n in names]

    df_imp = pd.DataFrame({"Feature": clean_names, "Importance": np.abs(imps)})
    df_imp = df_imp.sort_values(by="Importance", ascending=True).tail(top_n)

    col = MODEL_COLORS.get(model_name, "#2563eb")

    fig = go.Figure(
        go.Bar(
            x=df_imp["Importance"],
            y=df_imp["Feature"],
            orientation="h",
            marker=dict(color=col, line=dict(width=0)),
            text=[f"{v:.4f}" for v in df_imp["Importance"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
        )
    )

    apply_light_theme(
        fig,
        height=max(320, len(df_imp) * 28),
        title=f"<b>Top {min(top_n, len(df_imp))} Features — {model_name}</b>",
    )
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="",
    )
    return fig


def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    """Scatter plot of Actual vs Predicted values for regression."""
    fig = go.Figure()
    col = MODEL_COLORS.get(model_name, "#2563eb")

    fig.add_trace(
        go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            marker=dict(color=col, size=7, opacity=0.7),
            name="Predictions",
            hovertemplate="Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>",
        )
    )

    min_v = min(np.min(y_test), np.min(y_pred))
    max_v = max(np.max(y_test), np.max(y_pred))
    fig.add_trace(
        go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v], mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="Ideal Fit (y = x)",
        )
    )

    apply_light_theme(fig, height=420, title=f"<b>Actual vs. Predicted — {model_name}</b>")
    fig.update_layout(
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
    """Plotly interactive Correlation Matrix for EDA in light theme."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr().round(2)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            text=corr.values,
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation"),
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
        )
    )

    apply_light_theme(fig, height=max(400, min(800, num_df.shape[1] * 35)), title="<b>Feature Correlation Matrix</b>")
    return fig


def plot_target_distribution(df: pd.DataFrame, target_col: str, task_type: str) -> go.Figure:
    """Visualizes target balance or histogram for EDA in light theme."""
    if task_type == "classification":
        counts = df[target_col].value_counts().reset_index()
        counts.columns = [target_col, "Count"]
        fig = px.bar(
            counts, x=target_col, y="Count",
            text="Count", color=target_col,
            color_discrete_sequence=["#2563eb", "#059669", "#d97706", "#7c3aed", "#e11d48", "#0d9488"],
        )
        fig.update_traces(textposition="outside")
    else:
        fig = px.histogram(
            df, x=target_col, nbins=30,
            color_discrete_sequence=["#2563eb"],
            marginal="box",
        )

    apply_light_theme(fig, height=380, title=f"<b>Target Distribution ({target_col})</b>")
    return fig
