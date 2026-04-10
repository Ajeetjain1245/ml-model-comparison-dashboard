"""
ML Model Comparison Dashboard  ·  v2.0
=======================================
Fixes applied vs v1:
  • XGBoost use_label_encoder removed (deprecated in XGBoost ≥ 2.0)
  • MODEL_COLORS now includes XGBoost
  • SMOTE import guarded with try/except; imblearn optional
  • predict_proba no longer called twice
  • Multiclass ROC-AUC fixed (ovr + full probability matrix)
  • Confusion matrices / ROC / Feature Importance moved inside run_btn block
  • Classification Reports / Feature Info / Export indentation fixed
    (were accidentally inside the feature-importance for-loop)
  • st.session_state added → results cached; sidebar interactions
    no longer retrigger a full retrain
  • Progress bar per model during training
  • Polished dark UI with teal/blue gradient accents
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix,
)
from sklearn.preprocessing import label_binarize

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# ╔══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ╚══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ML Model Comparison",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ╔══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — clean dark theme with teal/blue accents
# ╚══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base ── */
html, [data-testid="stAppViewContainer"] {
    background: #07090f;
    color: #dce8f5;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1c2a3a;
}
[data-testid="stSidebar"] * { color: #c8dff0 !important; }

/* ── Headings ── */
h1 { font-size: 2rem !important; font-weight: 800 !important;
     background: linear-gradient(90deg, #00d4aa, #4da6ff);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
h2 { font-size: 1.1rem !important; font-weight: 700 !important;
     color: #00d4aa !important; letter-spacing: .02em; }
h3 { color: #4da6ff !important; font-weight: 600 !important; }
h4 { color: #c8dff0 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0d1520;
    border: 1px solid #1c3050;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 2px 14px rgba(0,212,170,.07);
}
[data-testid="stMetricValue"] { color: #00d4aa !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #5a80a0 !important; font-size: .78rem !important; }

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1c2a3a !important;
    border-radius: 10px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg,#00b894,#0078d4) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: .95rem !important; padding: .55rem 1.6rem !important;
    transition: opacity .2s, transform .15s !important;
}
.stButton > button:hover { opacity:.87 !important; transform:translateY(-1px) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #1c2a3a !important;
    border-radius: 10px !important;
    background: #0d1520 !important;
}

/* ── Alerts ── */
.stSuccess { background:#0d2218 !important; border-color:#00b894 !important; color:#00d4aa !important; }
.stInfo    { background:#0d1828 !important; border-color:#0078d4 !important; }
.stWarning { background:#1a1200 !important; border-color:#f0a500 !important; }

/* ── Code ── */
code,.stCode {
    background:#0d1520 !important; color:#7dd3c4 !important;
    border: 1px solid #1c2a3a !important; border-radius:8px !important;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg,#00b894,#0078d4) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button { color:#5a80a0 !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color:#00d4aa !important; border-bottom:2px solid #00d4aa !important;
}

/* ── Divider / sidebar sep ── */
hr { border-color:#1c2a3a !important; }
.sb-sep { border-top:1px solid #1c2a3a; margin:.7rem 0; padding-top:.7rem; }

/* ── Badge pills ── */
.badge {
    display:inline-block; padding:3px 12px;
    border-radius:20px; font-size:.72rem; font-weight:700; margin-right:5px;
}
.b-teal   { background:#003d32; color:#00d4aa; border:1px solid #006652; }
.b-blue   { background:#001e3c; color:#4da6ff; border:1px solid #00449e; }
.b-purple { background:#1a0a2e; color:#c084fc; border:1px solid #5b21b6; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ╚══════════════════════════════════════════════════════════════════════════════
def _build_models():
    pool = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree"      : DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=150, random_state=42),
        "SVM"                : SVC(kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    if HAS_XGBOOST:
        # use_label_encoder was removed in XGBoost 2.0 — do NOT pass it
        pool["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
    return pool

ALL_MODELS = _build_models()

MODEL_COLORS = {
    "Logistic Regression": "#4da6ff",
    "K-Nearest Neighbors": "#00d4aa",
    "Decision Tree"      : "#f0a500",
    "Random Forest"      : "#c084fc",
    "SVM"                : "#f87171",
    "Gradient Boosting"  : "#34d399",
    "XGBoost"            : "#fb923c",   # ← was missing in v1
}


# ╔══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ╚══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame, target: str):
    X_raw = df.drop(columns=[target]).copy()
    y_raw = df[target].copy()

    le = LabelEncoder()
    y  = le.fit_transform(y_raw.astype(str))

    num_cols = X_raw.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_raw.select_dtypes(exclude=np.number).columns.tolist()

    if num_cols:
        X_raw[num_cols] = SimpleImputer(strategy="median").fit_transform(X_raw[num_cols])
    if cat_cols:
        X_raw[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X_raw[cat_cols])
        for col in cat_cols:
            X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    return X, y, list(X_raw.columns), le, scaler


# ╔══════════════════════════════════════════════════════════════════════════════
# TRAINING + EVALUATION
# ╚══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(X, y, active_models: dict, test_size: float,
                       cv_folds: int, use_smote: bool, progress_bar, status_text):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Optional SMOTE oversampling
    if use_smote and HAS_SMOTE:
        try:
            X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
        except Exception:
            pass

    n_classes = len(np.unique(y))
    results   = {}
    n_models  = len(active_models)

    for i, (name, model) in enumerate(active_models.items()):
        status_text.markdown(f"⚙️ Training **{name}** &nbsp;({i+1} / {n_models})…")
        progress_bar.progress(i / n_models)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ── Probabilities — called ONCE ──────────────────────────────────────
        y_prob = None
        roc    = 0.0
        try:
            y_prob_full = model.predict_proba(X_test)       # shape (N, n_classes)
            if n_classes == 2:
                y_prob = y_prob_full[:, 1]                  # binary: 1-D array
                roc    = roc_auc_score(y_test, y_prob)
            else:
                # Multiclass: use full probability matrix with OvR strategy
                roc    = roc_auc_score(
                    y_test, y_prob_full,
                    multi_class="ovr", average="weighted",
                )
                y_prob = y_prob_full                        # keep full for ROC plot
        except Exception:
            pass

        # ── Hold-out metrics ──────────────────────────────────────────────────
        acc       = accuracy_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        # ── Cross-validation ──────────────────────────────────────────────────
        skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean()
        cv_f1  = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted").mean()

        results[name] = {
            "model"     : model,
            "y_test"    : y_test,
            "y_pred"    : y_pred,
            "y_prob"    : y_prob,
            "Accuracy"  : round(acc,       4),
            "F1 Score"  : round(f1,        4),
            "Precision" : round(precision, 4),
            "Recall"    : round(recall,    4),
            "ROC AUC"   : round(roc,       4),
            "CV Acc"    : round(cv_acc,    4),
            "CV F1"     : round(cv_f1,     4),
        }

    progress_bar.progress(1.0)
    status_text.markdown("✅ All models trained!")
    return results


# ╔══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ╚══════════════════════════════════════════════════════════════════════════════
_BG   = "#07090f"
_CARD = "#0d1520"
_GRID = "#141f2b"
_TEXT = "#dce8f5"
_MUTE = "#4a6880"

def _dark_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD)
    ax.spines[:].set_color(_GRID)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_MUTE)
    ax.yaxis.label.set_color(_MUTE)
    return fig, ax


def bar_chart(results: dict, metric: str):
    fig, ax = _dark_fig(8, 4)
    items  = sorted(results.items(), key=lambda x: x[1][metric], reverse=True)
    names  = [n for n, _ in items]
    vals   = [r[metric] for _, r in items]
    colors = [MODEL_COLORS.get(n, "#888") for n in names]
    bars   = ax.barh(names, vals, color=colors, height=0.52, edgecolor="none")
    ax.bar_label(bars, fmt="%.4f", padding=5, color=_TEXT, fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, 1.14)
    ax.set_title(metric, color="#00d4aa", fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", color=_GRID, linewidth=0.5)
    plt.tight_layout()
    return fig


def radar_chart(results: dict):
    cats   = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"]
    N      = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color=_TEXT, fontsize=9)
    ax.yaxis.set_tick_params(labelcolor=_MUTE, labelsize=7)
    ax.spines["polar"].set_color(_GRID)
    ax.grid(color=_GRID, linewidth=0.6)

    for name, r in results.items():
        vals = [r[c] for c in cats] + [r[cats[0]]]
        col  = MODEL_COLORS.get(name, "#888")
        ax.plot(angles, vals, color=col, linewidth=2, label=name)
        ax.fill(angles, vals,  color=col, alpha=0.08)

    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18),
              fontsize=8, framealpha=0, labelcolor=_TEXT)
    ax.set_title("Metric Radar", color="#00d4aa", fontsize=12, fontweight="bold", pad=18)
    plt.tight_layout()
    return fig


def confusion_fig(y_test, y_pred, model_name: str, class_names: list):
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = _dark_fig(5, 4)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.4, linecolor=_GRID,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(model_name, color="#00d4aa", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", color=_MUTE)
    ax.set_ylabel("Actual",    color=_MUTE)
    ax.tick_params(colors=_TEXT, labelsize=8)
    plt.tight_layout()
    return fig


def roc_fig(y_test, y_prob, model_name: str, n_classes: int, class_names: list):
    fig, ax = _dark_fig(5, 4)
    col = MODEL_COLORS.get(model_name, "#00d4aa")

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val     = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"AUC = {auc_val:.3f}")
    else:
        y_bin   = label_binarize(y_test, classes=list(range(n_classes)))
        palette = plt.cm.tab10(np.linspace(0, 1, n_classes))
        for c_idx in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c_idx], y_prob[:, c_idx])
            auc_val     = roc_auc_score(y_bin[:, c_idx], y_prob[:, c_idx])
            lbl = str(class_names[c_idx])[:12]
            ax.plot(fpr, tpr, color=palette[c_idx], lw=1.5, label=f"{lbl} ({auc_val:.2f})")

    ax.plot([0, 1], [0, 1], "--", color=_MUTE, lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("FPR", color=_MUTE); ax.set_ylabel("TPR", color=_MUTE)
    ax.set_title(f"ROC — {model_name}", color="#00d4aa", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, framealpha=.2, labelcolor=_TEXT,
              loc="lower right", facecolor=_CARD, edgecolor=_GRID)
    ax.grid(color=_GRID, linewidth=0.5)
    plt.tight_layout()
    return fig


def feature_importance_fig(model, feat_names: list, model_name: str):
    if not hasattr(model, "feature_importances_"):
        return None
    imp  = model.feature_importances_
    idx  = np.argsort(imp)[-20:]          # top-20 features
    names = [feat_names[i] for i in idx]
    vals  = imp[idx]
    col   = MODEL_COLORS.get(model_name, "#00d4aa")

    fig, ax = _dark_fig(7, max(3, len(names) * 0.36))
    bars = ax.barh(names, vals, color=col, height=0.55, edgecolor="none")
    ax.bar_label(bars, fmt="%.4f", padding=4, color=_TEXT, fontsize=8)
    ax.set_title(f"Feature Importance — {model_name}",
                 color="#00d4aa", fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="x", color=_GRID, linewidth=0.5)
    plt.tight_layout()
    return fig


# ╔══════════════════════════════════════════════════════════════════════════════
# LEADERBOARD DATAFRAME
# ╚══════════════════════════════════════════════════════════════════════════════
def leaderboard_df(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({
            "Model"     : name,
            "Accuracy"  : r["Accuracy"],
            "F1 Score"  : r["F1 Score"],
            "Precision" : r["Precision"],
            "Recall"    : r["Recall"],
            "ROC AUC"   : r["ROC AUC"],
            "CV Acc"    : r["CV Acc"],
            "CV F1"     : r["CV F1"],
        })
    df = (
        pd.DataFrame(rows)
        .sort_values("Accuracy", ascending=False)
        .reset_index(drop=True)
    )
    df.index += 1
    return df


# ╔══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ╚══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ ML Compare")
    st.markdown("<div class='sb-sep'></div>", unsafe_allow_html=True)

    uploaded   = st.file_uploader("📂 Upload CSV dataset", type=["csv"])
    target_col = None
    df_raw     = None

    if uploaded:
        df_raw     = pd.read_csv(uploaded)
        target_col = st.selectbox(
            "🎯 Target column",
            df_raw.columns.tolist(),
            index=len(df_raw.columns) - 1,
        )

    st.markdown("<div class='sb-sep'></div>", unsafe_allow_html=True)
    st.markdown("### 🔧 Training")
    test_size = st.slider("Test split", 0.10, 0.40, 0.20, 0.05,
                          help="Fraction held out for testing")
    cv_folds  = st.slider("CV folds (k)", 2, 10, 5,
                          help="k in stratified k-fold CV")

    st.markdown("<div class='sb-sep'></div>", unsafe_allow_html=True)
    st.markdown("### 🤖 Models")
    selected_models = {name: st.checkbox(name, value=True) for name in ALL_MODELS}

    st.markdown("<div class='sb-sep'></div>", unsafe_allow_html=True)
    if HAS_SMOTE:
        use_smote = st.checkbox("⚖️ Apply SMOTE (imbalanced data)")
    else:
        use_smote = False
        st.caption("imblearn not installed — SMOTE unavailable.\nRun: pip install imbalanced-learn")

    run_btn = st.button("🚀 Run Comparison", use_container_width=True)

    st.markdown("<div class='sb-sep'></div>", unsafe_allow_html=True)
    st.caption("ML Model Comparison v2.0\nBuilt with Streamlit + scikit-learn")


# ╔══════════════════════════════════════════════════════════════════════════════
# HEADER
# ╚══════════════════════════════════════════════════════════════════════════════
st.markdown("# ⚡ ML Model Comparison")
st.markdown(
    '<span class="badge b-teal">scikit-learn</span>'
    '<span class="badge b-blue">Streamlit</span>'
    '<span class="badge b-purple">Auto-Preprocessing</span>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ╔══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE  (no file uploaded yet)
# ╚══════════════════════════════════════════════════════════════════════════════
if df_raw is None:
    c1, c2, c3 = st.columns(3)
    c1.info("**Step 1** — Upload a CSV using the sidebar")
    c2.info("**Step 2** — Choose your target (label) column")
    c3.info("**Step 3** — Hit **Run Comparison** and explore results")

    st.markdown("#### 📌 What you get")
    fc1, fc2 = st.columns(2)
    feats = [
        ("🧹 Auto Preprocessing",  "Imputation · label encoding · standard scaling"),
        ("📊 6+ Classifiers",      "LR · KNN · DT · RF · SVM · GBM · XGBoost"),
        ("📈 7 Metrics per Model", "Accuracy · F1 · Precision · Recall · ROC-AUC · CV Acc · CV F1"),
        ("🔥 Rich Visuals",        "Radar · bar charts · confusion matrices · ROC curves · feature importance"),
    ]
    for i, (t, d) in enumerate(feats):
        (fc1 if i % 2 == 0 else fc2).success(f"**{t}**\n\n{d}")
    st.stop()


# ╔══════════════════════════════════════════════════════════════════════════════
# DATASET OVERVIEW
# ╚══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📋 Dataset Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows",    f"{df_raw.shape[0]:,}")
m2.metric("Columns", f"{df_raw.shape[1]:,}")
m3.metric("Missing", f"{df_raw.isnull().sum().sum():,}")
m4.metric("Target",  target_col)

col_left, col_right = st.columns([1.6, 1])
with col_left:
    with st.expander("🔍 Raw data (first 100 rows)"):
        st.dataframe(df_raw.head(100), use_container_width=True)
    with st.expander("📊 Descriptive statistics"):
        st.dataframe(df_raw.describe(include="all"), use_container_width=True)

with col_right:
    st.markdown("#### Class Distribution")
    class_counts    = df_raw[target_col].value_counts()
    imbalance_ratio = class_counts.min() / class_counts.max()
    st.bar_chart(class_counts)
    if imbalance_ratio < 0.50:
        smote_hint = "SMOTE available in sidebar." if HAS_SMOTE else "Install imblearn to enable SMOTE."
        st.warning(f"⚠️ Imbalanced dataset (ratio {imbalance_ratio:.2f}). {smote_hint}")
    else:
        st.success(f"✅ Balanced dataset (ratio {imbalance_ratio:.2f})")

st.markdown("---")


# ╔══════════════════════════════════════════════════════════════════════════════
# TRAINING  (runs on button click; results cached in session_state)
# ╚══════════════════════════════════════════════════════════════════════════════
if run_btn:
    active_models = {k: ALL_MODELS[k] for k, v in selected_models.items() if v}
    if not active_models:
        st.error("Please select at least one model.")
        st.stop()

    pb   = st.progress(0.0)
    stat = st.empty()

    try:
        X, y, feat_names, le, scaler = preprocess(df_raw, target_col)
        results = train_and_evaluate(
            X, y, active_models, test_size, cv_folds, use_smote, pb, stat
        )
        class_names = le.classes_.tolist()

        # ── Store in session_state so widget changes don't retrain ─────────
        st.session_state["results"]     = results
        st.session_state["feat_names"]  = feat_names
        st.session_state["class_names"] = class_names
        st.session_state["n_classes"]   = len(class_names)
        st.session_state["df_res"]      = leaderboard_df(results)

    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        st.stop()


# ╔══════════════════════════════════════════════════════════════════════════════
# RESULTS  (rendered from session_state — sidebar tweaks don't retrain)
# ╚══════════════════════════════════════════════════════════════════════════════
if "results" not in st.session_state:
    st.info("👈 Configure settings in the sidebar and hit **Run Comparison**.")
    st.stop()

results     = st.session_state["results"]
feat_names  = st.session_state["feat_names"]
class_names = st.session_state["class_names"]
n_classes   = st.session_state["n_classes"]
df_res      = st.session_state["df_res"]

st.success(f"✅ Trained {len(results)} model(s) successfully!")


# ── 1. Leaderboard ────────────────────────────────────────────────────────────
st.markdown("## 🏆 Leaderboard")

metric_cols = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC", "CV Acc", "CV F1"]

def _highlight(s):
    return [
        "background-color:#003830; color:#00d4aa; font-weight:700"
        if v else ""
        for v in (s == s.max())
    ]

styled_df = (
    df_res.style
    .apply(_highlight, subset=metric_cols)
    .format({c: ":.4f" for c in metric_cols})
)
st.dataframe(styled_df, use_container_width=True)

best_row = df_res.iloc[0]
st.markdown(
    f"🥇 **Best model:** `{best_row['Model']}` — "
    f"Accuracy **{best_row['Accuracy']:.4f}** · "
    f"F1 **{best_row['F1 Score']:.4f}** · "
    f"ROC AUC **{best_row['ROC AUC']:.4f}**"
)
st.markdown("---")


# ── 2. Performance Charts ─────────────────────────────────────────────────────
st.markdown("## 📊 Performance Charts")

rc1, rc2 = st.columns([1, 2])
with rc1:
    st.pyplot(radar_chart(results))
with rc2:
    t1, t2, t3, t4 = st.tabs(["Accuracy", "F1 Score", "Precision", "Recall"])
    for tab, metric in zip([t1, t2, t3, t4],
                           ["Accuracy", "F1 Score", "Precision", "Recall"]):
        with tab:
            st.pyplot(bar_chart(results, metric))

st.markdown("---")


# ── 3. Confusion Matrices ─────────────────────────────────────────────────────
st.markdown("## 🔲 Confusion Matrices")
n_cols   = min(len(results), 3)
cm_cols  = st.columns(n_cols)
for i, (name, r) in enumerate(results.items()):
    with cm_cols[i % n_cols]:
        st.pyplot(confusion_fig(r["y_test"], r["y_pred"], name, class_names))

st.markdown("---")


# ── 4. ROC Curves ─────────────────────────────────────────────────────────────
st.markdown("## 📈 ROC Curves")
roc_models = [(n, r) for n, r in results.items() if r["y_prob"] is not None]
if roc_models:
    roc_cols = st.columns(min(len(roc_models), 3))
    for i, (name, r) in enumerate(roc_models):
        with roc_cols[i % len(roc_cols)]:
            st.pyplot(roc_fig(r["y_test"], r["y_prob"], name, n_classes, class_names))
else:
    st.info("No models returned probability estimates for ROC curves.")

st.markdown("---")


# ── 5. Feature Importance ─────────────────────────────────────────────────────
fi_models = [(n, r) for n, r in results.items()
             if hasattr(r["model"], "feature_importances_")]
if fi_models:
    st.markdown("## 🌿 Feature Importance")
    fi_cols = st.columns(min(len(fi_models), 2))
    for i, (name, r) in enumerate(fi_models):
        with fi_cols[i % 2]:
            fig = feature_importance_fig(r["model"], feat_names, name)
            if fig:
                st.pyplot(fig)
    st.markdown("---")


# ── 6. Classification Reports ─────────────────────────────────────────────────
# FIX: was accidentally inside the feature-importance for-loop in v1
st.markdown("## 📝 Classification Reports")
tabs = st.tabs(list(results.keys()))
for tab, name in zip(tabs, results.keys()):
    with tab:
        report = classification_report(
            results[name]["y_test"],
            results[name]["y_pred"],
            target_names=[str(c) for c in class_names],
            zero_division=0,
        )
        st.code(report, language="text")

st.markdown("---")


# ── 7. Feature Information ────────────────────────────────────────────────────
# FIX: was accidentally inside the feature-importance for-loop in v1
st.markdown("## 🔎 Feature Information")
fi_info = pd.DataFrame({
    "Feature"       : feat_names,
    "Type"          : df_raw.drop(columns=[target_col]).dtypes.astype(str).values,
    "Missing (raw)" : df_raw.drop(columns=[target_col]).isnull().sum().values,
    "Unique values" : [df_raw[c].nunique() for c in feat_names],
})
st.dataframe(fi_info, use_container_width=True)

st.markdown("---")


# ── 8. Export ─────────────────────────────────────────────────────────────────
# FIX: was accidentally inside the feature-importance for-loop in v1
st.markdown("## 💾 Export Results")
ec1, ec2 = st.columns(2)
with ec1:
    csv_out = df_res.to_csv(index=True).encode("utf-8")
    st.download_button(
        "⬇️ Download Leaderboard CSV",
        data=csv_out,
        file_name="ml_comparison_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
with ec2:
    report_lines = []
    for name, r in results.items():
        report_lines.append(f"=== {name} ===\n")
        report_lines.append(classification_report(
            r["y_test"], r["y_pred"],
            target_names=[str(c) for c in class_names],
            zero_division=0,
        ))
        report_lines.append("\n")
    st.download_button(
        "⬇️ Download Classification Reports (.txt)",
        data="\n".join(report_lines).encode("utf-8"),
        file_name="classification_reports.txt",
        mime="text/plain",
        use_container_width=True,
    )
