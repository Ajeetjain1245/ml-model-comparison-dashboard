"""
Modern, Clean & Friendly Light Theme Engine for ML Model Comparison Studio.
"""
import textwrap
import streamlit as st
import pandas as pd


LIGHT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Global Reset & Typography ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #f8fafc !important;
    color: #1e293b !important;
}

[data-testid="stHeader"] {
    background: rgba(248, 250, 252, 0.85) !important;
    backdrop-filter: blur(12px) !important;
}

/* ── Sidebar (Clean Crisp White/Slate) ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] * {
    color: #334155;
}

/* ── Headings ── */
h1 {
    font-size: 2.25rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #7c3aed 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin-bottom: 0.2rem !important;
}

h2 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    color: #0f172a !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #1e293b !important;
}

h4 {
    font-size: 0.92rem !important;
    color: #475569 !important;
    font-weight: 600 !important;
}

p, span, label {
    color: #334155 !important;
}

code, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
    background: #f1f5f9 !important;
    color: #0369a1 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* ── Studio Cards ── */
.studio-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
    margin-bottom: 16px;
    transition: all 0.2s ease-in-out;
}
.studio-card:hover {
    border-color: #93c5fd;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.08);
}

/* ── Winner Spotlight Card ── */
.winner-spotlight {
    background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 60%, #eff6ff 100%);
    border: 1.5px solid #10b981;
    border-radius: 16px;
    padding: 22px 26px;
    margin: 16px 0 24px 0;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.12);
    position: relative;
    overflow: hidden;
}
.winner-spotlight::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #10b981, #3b82f6, #8b5cf6);
}

/* ── Podium Cards ── */
.podium-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}
.podium-card {
    border-radius: 14px;
    padding: 18px 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.podium-card:hover {
    transform: translateY(-3px);
}
.podium-1 {
    background: linear-gradient(145deg, #fffbeb, #ffffff);
    border: 1.5px solid #f59e0b;
    box-shadow: 0 4px 16px rgba(245, 158, 11, 0.12);
}
.podium-2 {
    background: linear-gradient(145deg, #f8fafc, #ffffff);
    border: 1.5px solid #94a3b8;
    box-shadow: 0 4px 16px rgba(148, 163, 184, 0.1);
}
.podium-3 {
    background: linear-gradient(145deg, #fff7ed, #ffffff);
    border: 1.5px solid #f97316;
    box-shadow: 0 4px 16px rgba(249, 115, 22, 0.1);
}

/* ── Metric KPI Blocks ── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    border-color: #3b82f6 !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.08) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* ── Primary Action Button ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 50%, #3b82f6 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 1.8rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
    opacity: 0.95 !important;
}

/* ── Tabs Navigation ── */
[data-testid="stTabs"] {
    margin-top: 10px;
}
[data-testid="stTabs"] button {
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 10px 22px !important;
    border-radius: 8px 8px 0 0 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s ease !important;
}
[data-testid="stTabs"] button:hover {
    color: #2563eb !important;
    background: rgba(37, 99, 235, 0.04) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563eb !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #2563eb !important;
    background: rgba(37, 99, 235, 0.06) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    margin-bottom: 12px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
}

/* ── Badge Chips (Pastel Tint) ── */
.badge-chip {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 8px;
}
.chip-cyan   { background: #ecfeff; color: #0891b2; border: 1px solid #a5f3fc; }
.chip-blue   { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
.chip-purple { background: #f5f3ff; color: #7c3aed; border: 1px solid #ddd6fe; }
.chip-green  { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
.chip-amber  { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }

/* ── Dividers ── */
hr {
    border: 0 !important;
    height: 1px !important;
    background: #e2e8f0 !important;
    margin: 1.5rem 0 !important;
}
.sidebar-sep {
    border-top: 1px solid #e2e8f0;
    margin: 0.9rem 0;
    padding-top: 0.4rem;
}
</style>
"""


def apply_theme():
    """Injects the custom light studio theme into Streamlit."""
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)


def render_hero_header():
    """Renders the top branding header with live badge chips."""
    st.markdown("# ⚡ ML Model Comparison Studio")
    header_html = """<div style="margin-bottom: 12px;">
<span class="badge-chip chip-cyan">🛡️ Zero Data Leakage</span>
<span class="badge-chip chip-green">⚖️ SMOTE & Imbalance Engine</span>
<span class="badge-chip chip-blue">📊 Interactive Visuals</span>
<span class="badge-chip chip-purple">🎯 What-If Simulator & Model Export</span>
</div>"""
    st.markdown(header_html, unsafe_allow_html=True)
    st.markdown("<hr style='margin: 0.6rem 0 1.2rem 0;'>", unsafe_allow_html=True)


def render_winner_spotlight(best_row: pd.Series, task_type: str):
    """Renders a premier winner spotlight banner in modern light styling without markdown code-block triggers."""
    best_name = best_row["Model"]

    if task_type == "classification":
        metrics_html = f"""<div style="display: flex; gap: 24px; flex-wrap: wrap; margin-top: 12px;">
<div><span style="color:#64748b; font-size:0.78rem; font-weight:600;">HOLD-OUT ACCURACY</span><br><b style="font-family:'JetBrains Mono'; font-size:1.35rem; color:#059669;">{best_row['Accuracy']:.4f}</b></div>
<div><span style="color:#64748b; font-size:0.78rem; font-weight:600;">F1 SCORE (WEIGHTED)</span><br><b style="font-family:'JetBrains Mono'; font-size:1.35rem; color:#2563eb;">{best_row['F1 Score']:.4f}</b></div>
<div><span style="color:#64748b; font-size:0.78rem; font-weight:600;">CROSS-VAL ACCURACY</span><br><b style="font-family:'JetBrains Mono'; font-size:1.35rem; color:#7c3aed;">{best_row['CV Accuracy']:.4f}</b></div>
<div><span style="color:#64748b; font-size:0.78rem; font-weight:600;">PRECISION</span><br><b style="font-family:'JetBrains Mono'; font-size:1.35rem; color:#d97706;">{best_row['Precision']:.4f}</b></div>
<div><span style="color:#64748b; font-size:0.78rem; font-weight:600;">RECALL</span><br><b style="font-family:'JetBrains Mono'; font-size:1.35rem; color:#db2777;">{best_row['Recall']:.4f}</b></div>
</div>"""
    else:
        metrics_html = f"""<div style="display: flex; gap: 24px; flex-wrap: wrap; margin-top: 12px;">
<div><span style="color:#64748b; font-size:0.78rem; font-weight:600;">R² SCORE</span><br><b style="font-family:'JetBrains Mono'; font-size:1.35rem; color:#059669;">{best_row['R2 Score']:.4f}</b></div>
<div><span style="color:#64748b; font-size:0.78rem; font-weight:600;">CV R² SCORE</span><br><b style="font-family:'JetBrains Mono'; font-size:1.35rem; color:#2563eb;">{best_row['CV R2']:.4f}</b></div>
<div><span style="color:#64748b; font-size:0.78rem; font-weight:600;">RMSE</span><br><b style="font-family:'JetBrains Mono'; font-size:1.35rem; color:#d97706;">{best_row['RMSE']:.4f}</b></div>
<div><span style="color:#64748b; font-size:0.78rem; font-weight:600;">MAE</span><br><b style="font-family:'JetBrains Mono'; font-size:1.35rem; color:#7c3aed;">{best_row['MAE']:.4f}</b></div>
</div>"""

    html = f"""<div class="winner-spotlight">
<div>
<span style="background:#dcfce7; color:#15803d; border:1px solid #bbf7d0; padding:3px 12px; border-radius:12px; font-size:0.75rem; font-weight:800; letter-spacing:0.04em;">🏆 BEST PERFORMING MODEL</span>
<h2 style="margin: 6px 0 0 0; color: #0f172a !important; font-size: 1.6rem !important; font-weight:800;">{best_name}</h2>
</div>
{metrics_html}
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_podium_cards(df_leaderboard: pd.DataFrame, task_type: str):
    """Renders top-3 leaderboard podium cards in warm/clean light pastel cards without indentation issues."""
    if len(df_leaderboard) < 2:
        return

    top_n = min(3, len(df_leaderboard))
    medals = ["🥇 1st Place", "🥈 2nd Place", "🥉 3rd Place"]
    styles = ["podium-1", "podium-2", "podium-3"]

    cols = st.columns(top_n)

    for i in range(top_n):
        row = df_leaderboard.iloc[i]
        with cols[i]:
            score_label = "Accuracy" if task_type == "classification" else "R² Score"
            score_val = row[score_label]
            f1_label = "F1 Score" if task_type == "classification" else "RMSE"
            f1_val = row[f1_label]

            card_html = f"""<div class="podium-card {styles[i]}">
<span style="font-size:0.8rem; font-weight:700; color:#64748b;">{medals[i]}</span>
<h3 style="margin:4px 0 10px 0; color:#0f172a !important; font-size:1.15rem !important; font-weight:700;">{row['Model']}</h3>
<div style="display:flex; justify-content:space-between; margin-top:8px;">
<div>
<span style="font-size:0.7rem; color:#64748b; font-weight:600;">{score_label.upper()}</span><br>
<span style="font-family:'JetBrains Mono'; font-weight:700; color:#2563eb; font-size:1.15rem;">{score_val:.4f}</span>
</div>
<div>
<span style="font-size:0.7rem; color:#64748b; font-weight:600;">{f1_label.upper()}</span><br>
<span style="font-family:'JetBrains Mono'; font-weight:700; color:#7c3aed; font-size:1.15rem;">{f1_val:.4f}</span>
</div>
</div>
</div>"""
            st.markdown(card_html, unsafe_allow_html=True)
