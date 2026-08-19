"""
Debt Collection Decision Support — Streamlit App
==================================================
Implements the three-layer framework from "Intelligent decision support for
debt collection using predictive learning and multi-criteria optimization":

    LAYER 1  Rule Extraction     — K-Means segmentation + decision-tree rules
    LAYER 2  Prediction          — repayment-probability classifier + evaluation
    LAYER 3  Optimization        — TOPSIS / AHP / fuzzy inference / routing

Plus a monthly account-volume trend with a forecast, and an evaluation panel
(with plain-language explanations) for every model in the pipeline.

RUN LOCALLY
-----------
    pip install streamlit scikit-learn pandas numpy scipy plotly
    streamlit run debt_collection_app.py

Replace `generate_synthetic_portfolio()` / `generate_monthly_series()` with
your own data loaders (e.g. `pd.read_csv(...)`) to use this on a real book
of debtors — every downstream layer keys off the same column names.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, brier_score_loss, silhouette_score
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ----------------------------------------------------------------------
# THEME — warm paper background, navy + ochre accents (matches the
# dashboard design used elsewhere in this project)
# ----------------------------------------------------------------------
INK = "#2B2620"
MUTE = "#6B6355"
FAINT = "#B4AC98"
NAVY = "#2F5C6E"
GOLD = "#B8863B"
GREY = "#8A8375"
PAPER = "#F4F0E6"
CARD = "#FFFFFF"
LINE = "#E6E0D2"

st.set_page_config(page_title="Debt Collection Decision Support", layout="wide")

st.markdown(f"""
<style>
    .stApp {{ background-color: {PAPER}; color: {INK}; }}
    section[data-testid="stSidebar"] {{ background-color: #EFE9DA; border-right: 1px solid {LINE}; }}
    h1, h2, h3 {{ color: #211D17; font-family: Georgia, serif; }}
    div[data-testid="stMetric"] {{
        background-color: {CARD}; border: 1px solid {LINE}; border-radius: 10px; padding: 10px 14px;
    }}
    .explain-box {{
        background-color: #FBF7EC; border-left: 3px solid {GOLD}; border-radius: 6px;
        padding: 10px 14px; font-size: 14px; color: {MUTE}; margin-top: 6px; margin-bottom: 14px;
    }}
    .stTabs [data-baseweb="tab"] {{ font-size: 14px; }}
</style>
""", unsafe_allow_html=True)

CHART_TEMPLATE = dict(
    plot_bgcolor=CARD, paper_bgcolor=CARD,
    font=dict(color=INK, family="Georgia, serif"),
    xaxis=dict(gridcolor=LINE, zeroline=False),
    yaxis=dict(gridcolor=LINE, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=40, l=10, r=10, b=10),
)


def explain(text: str):
    st.markdown(f'<div class="explain-box">{text}</div>', unsafe_allow_html=True)


# ========================================================================
# DATA — replace these two functions with real loaders for production use
# ========================================================================
@st.cache_data
def generate_synthetic_portfolio(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "debtor_id": [f"D{i:05d}" for i in range(n)],
        "outstanding_balance": rng.gamma(2.0, 15000, n).round(2),
        "days_past_due": rng.integers(1, 365, n),
        "num_prior_defaults": rng.poisson(1.2, n),
        "num_contacts_attempted": rng.integers(0, 20, n),
        "monthly_income": rng.normal(35000, 12000, n).clip(5000, None).round(2),
        "credit_score": rng.normal(620, 80, n).clip(300, 850).round(0),
        "prior_partial_payment": rng.choice([0, 1], n, p=[0.55, 0.45]),
        "latitude": rng.normal(13.75, 0.12, n),
        "longitude": rng.normal(100.50, 0.12, n),
    })
    df["num_successful_contacts"] = (df["num_contacts_attempted"] * rng.uniform(0.1, 0.7, n)).round().astype(int)

    logit = (
        -0.00002 * df["outstanding_balance"] - 0.01 * df["days_past_due"]
        - 0.35 * df["num_prior_defaults"] + 0.15 * df["num_successful_contacts"]
        + 0.00004 * df["monthly_income"] + 0.004 * df["credit_score"]
        + 0.8 * df["prior_partial_payment"] + rng.normal(0, 0.6, n)
    )
    df["repaid"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


@st.cache_data
def generate_monthly_series() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    months = pd.date_range("2025-01-01", periods=18, freq="MS")
    t = np.arange(18)
    actual = (1200 + np.linspace(0, 260, 18) + 60 * np.sin(np.linspace(0, 3 * np.pi, 18))
              + rng.normal(0, 25, 18)).round().astype(int)
    return pd.DataFrame({"month": months, "accounts": actual})


# ========================================================================
# FORECASTING — linear trend + seasonal component, with a backtest
# evaluation so forecast quality has numbers behind it, not just a chart.
# ========================================================================
def fit_forecast(actual: np.ndarray, horizon: int):
    t = np.arange(len(actual))
    coeffs = np.polyfit(t, actual, 1)
    trend = np.poly1d(coeffs)
    resid_std = np.std(actual - trend(t))

    future_t = np.arange(len(actual), len(actual) + horizon)
    period = 18
    phase = (future_t / period) * 3 * np.pi
    season = 60 * np.sin(phase)
    pred = trend(future_t) + season
    band = resid_std * np.sqrt(1 + (future_t - (len(actual) - 1)) / len(actual))
    return pred.round().astype(int), (pred - 1.28 * band).round().astype(int), (pred + 1.28 * band).round().astype(int)


def backtest_forecast(actual: np.ndarray, holdout: int = 6) -> dict:
    """Hold out the last `holdout` months, forecast them from the rest,
    and score the forecast against what actually happened."""
    train, test = actual[:-holdout], actual[-holdout:]
    pred, _, _ = fit_forecast(train, holdout)
    mae = np.mean(np.abs(test - pred))
    rmse = np.sqrt(np.mean((test - pred) ** 2))
    mape = np.mean(np.abs((test - pred) / test)) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape, "test": test, "pred": pred}


# ========================================================================
# LAYER 1 — RULE EXTRACTION
# ========================================================================
@st.cache_resource
def fit_rule_extraction(df: pd.DataFrame, feature_cols: list, n_clusters: int = 4):
    X = StandardScaler().fit_transform(df[feature_cols])
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    clusters = km.fit_predict(X)
    tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    tree.fit(df[feature_cols], clusters)
    sil = silhouette_score(X, clusters)
    return clusters, tree, sil


# ========================================================================
# LAYER 2 — PREDICTION + EVALUATION
# ========================================================================
@st.cache_resource
def fit_prediction_model(df: pd.DataFrame, feature_cols: list):
    X, y = df[feature_cols], df["repaid"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(n_estimators=300, max_depth=6,
                                    random_state=RANDOM_STATE, class_weight="balanced")
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = model.predict(X_test)

    metrics = {
        "auc": roc_auc_score(y_test, proba_test),
        "accuracy": accuracy_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test),
        "recall": recall_score(y_test, pred_test),
        "f1": f1_score(y_test, pred_test),
        "brier": brier_score_loss(y_test, proba_test),
        "confusion": confusion_matrix(y_test, pred_test),
        "roc": roc_curve(y_test, proba_test),
        "y_test": y_test, "proba_test": proba_test,
    }
    return model, metrics


# ========================================================================
# LAYER 3a — TOPSIS
# ========================================================================
def topsis(matrix: pd.DataFrame, weights: dict, is_benefit: dict) -> pd.Series:
    cols = list(matrix.columns)
    X = matrix[cols].to_numpy(dtype=float)
    norm = X / np.sqrt((X ** 2).sum(axis=0))
    w = np.array([weights[c] for c in cols])
    V = norm * w
    ideal_best = np.array([V[:, j].max() if is_benefit[cols[j]] else V[:, j].min() for j in range(len(cols))])
    ideal_worst = np.array([V[:, j].min() if is_benefit[cols[j]] else V[:, j].max() for j in range(len(cols))])
    d_best = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
    return pd.Series(d_worst / (d_best + d_worst + 1e-12), index=matrix.index, name="topsis_score")


def priority_bucket(score: pd.Series) -> pd.Series:
    lo, hi = score.quantile([0.33, 0.67])
    return pd.cut(score, bins=[-np.inf, lo, hi, np.inf], labels=["Low", "Medium", "High"])


# ========================================================================
# LAYER 3b — AHP
# ========================================================================
_RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp_weights(pairwise: pd.DataFrame):
    A = pairwise.to_numpy(dtype=float)
    n = A.shape[0]
    col_sums = A.sum(axis=0)
    weights = (A / col_sums).mean(axis=1)
    lambda_max = (A @ weights / weights).mean()
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = _RI.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    return pd.Series(weights, index=pairwise.index, name="ahp_weight"), cr


def ahp_rank_alternatives(alt_scores: pd.DataFrame, criteria_weights: pd.Series) -> pd.Series:
    w = criteria_weights.reindex(alt_scores.columns)
    return (alt_scores * w).sum(axis=1).sort_values(ascending=False)


def select_strategy_for_segment(bucket: str, w: pd.Series) -> pd.Series:
    strategies = pd.DataFrame({
        "expected_recovery": [0.9, 0.6, 0.5, 0.8, 0.3],
        "cost_efficiency":   [0.3, 0.8, 0.9, 0.5, 0.6],
        "speed":             [0.6, 0.9, 0.95, 0.4, 0.7],
        "relationship_pres": [0.5, 0.85, 0.9, 0.3, 0.7],
    }, index=["Field Visit", "Call Campaign", "SMS/Email Nudge", "Legal Action", "Settlement Offer"])
    bias = {
        "High":   {"expected_recovery": 1.3, "speed": 1.3, "cost_efficiency": 0.8, "relationship_pres": 0.8},
        "Medium": {"expected_recovery": 1.0, "speed": 1.0, "cost_efficiency": 1.0, "relationship_pres": 1.0},
        "Low":    {"expected_recovery": 0.8, "speed": 0.7, "cost_efficiency": 1.3, "relationship_pres": 1.3},
    }[bucket]
    adj_w = (w * pd.Series(bias)).pipe(lambda s: s / s.sum())
    return ahp_rank_alternatives(strategies, adj_w)


# ========================================================================
# LAYER 3c — Mamdani fuzzy urgency engine
# ========================================================================
def _tri(x, a, b, c):
    return np.clip(np.minimum((x - a) / (b - a + 1e-12), (c - x) / (c - b + 1e-12)), 0, 1)


class FuzzyUrgencyEngine:
    def __init__(self):
        self.out_x = np.linspace(0, 100, 401)

    def infer(self, prob, exposure):
        pl, pm, ph = _tri(prob, -0.1, 0, 0.5), _tri(prob, 0.2, 0.5, 0.8), _tri(prob, 0.5, 1.0, 1.1)
        el, eh = _tri(exposure, -0.1, 0, 0.5), _tri(exposure, 0.5, 1.0, 1.1)
        r1, r2, r3, r4, r5 = min(pl, eh), min(pl, el), min(ph, eh), min(ph, el), pm
        agg_low = np.minimum(r4, _tri(self.out_x, -10, 0, 50))
        agg_med = np.minimum(max(r2, r3, r5), _tri(self.out_x, 20, 50, 80))
        agg_high = np.minimum(r1, _tri(self.out_x, 50, 100, 110))
        agg = np.maximum.reduce([agg_low, agg_med, agg_high])
        return float((self.out_x * agg).sum() / agg.sum()) if agg.sum() else 50.0


# ========================================================================
# LAYER 3d — spatial clustering + nearest-neighbour routing
# ========================================================================
def cluster_and_route(df_field: pd.DataFrame, n_collectors: int = 3) -> pd.DataFrame:
    if len(df_field) == 0:
        return df_field.assign(collector_zone=[], visit_order=[])
    coords = df_field[["latitude", "longitude"]].to_numpy()
    n_zones = min(n_collectors, len(df_field))
    zones = KMeans(n_clusters=n_zones, random_state=RANDOM_STATE, n_init=10).fit_predict(coords)
    df_field = df_field.copy()
    df_field["collector_zone"] = zones
    df_field["visit_order"] = -1
    for z in range(n_zones):
        mask = df_field["collector_zone"] == z
        zc = df_field.loc[mask, ["latitude", "longitude"]].to_numpy()
        idxs = df_field.loc[mask].index.to_list()
        if len(idxs) <= 1:
            if idxs:
                df_field.loc[idxs, "visit_order"] = 1
            continue
        visited, remaining = [0], list(range(1, len(idxs)))
        while remaining:
            last = zc[visited[-1]].reshape(1, -1)
            nxt = remaining[int(np.argmin(cdist(last, zc[remaining]).flatten()))]
            visited.append(nxt)
            remaining.remove(nxt)
        for step, pos in enumerate(visited):
            df_field.loc[idxs[pos], "visit_order"] = step + 1
    return df_field.sort_values(["collector_zone", "visit_order"])


# ========================================================================
# BUILD PIPELINE (cached — recomputes only when inputs change)
# ========================================================================
FEATURE_COLS = ["outstanding_balance", "days_past_due", "num_prior_defaults",
                 "num_contacts_attempted", "num_successful_contacts",
                 "monthly_income", "credit_score", "prior_partial_payment"]

df = generate_synthetic_portfolio(800)
clusters, rule_tree, silhouette = fit_rule_extraction(df, FEATURE_COLS)
df["segment"] = clusters
model, eval_metrics = fit_prediction_model(df, FEATURE_COLS + ["segment"])
df["prob_repay"] = model.predict_proba(df[FEATURE_COLS + ["segment"]])[:, 1]
df["exposure_norm"] = MinMaxScaler().fit_transform(df[["outstanding_balance"]])
df["dpd_norm"] = MinMaxScaler().fit_transform(df[["days_past_due"]])
df["topsis_score"] = topsis(
    df[["prob_repay", "exposure_norm", "dpd_norm"]],
    {"prob_repay": 0.5, "exposure_norm": 0.3, "dpd_norm": 0.2},
    {"prob_repay": True, "exposure_norm": True, "dpd_norm": True},
)
df["priority"] = priority_bucket(df["topsis_score"])

crit = ["expected_recovery", "cost_efficiency", "speed", "relationship_pres"]
pairwise = pd.DataFrame([
    [1, 3, 2, 4], [1/3, 1, 1/2, 2], [1/2, 2, 1, 3], [1/4, 1/2, 1/3, 1],
], index=crit, columns=crit)
ahp_w, ahp_cr = ahp_weights(pairwise)

fuzzy_engine = FuzzyUrgencyEngine()
monthly = generate_monthly_series()


# ========================================================================
# UI
# ========================================================================
st.title("Debt Collection Decision Support")

col_refresh1, col_refresh2 = st.columns([1,3])
with col_refresh1:
    if st.button("🔄 Refresh Portfolio", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
with col_refresh2:
    st.caption("Reload portfolio data and recalculate all models")
st.caption("Rule extraction → prediction → multi-criteria optimization, with evaluation metrics for every model.")

tabs = st.tabs([
    "📈 Monthly Trend", "🧩 Layer 1 · Rule Extraction", "🎯 Layer 2 · Prediction",
    "⚖️ Layer 3 · Optimization", "🔍 Score a Debtor"
])

# ---------------------------------------------------------------- TAB 1
with tabs[0]:
    st.subheader("Accounts in active collection — actual vs. forecast")
    horizon = st.slider("Forecast horizon (months)", 1, 12, 6)
    pred, lower, upper = fit_forecast(monthly["accounts"].to_numpy(), horizon)
    future_months = pd.date_range(monthly["month"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_months, y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_months, y=lower, fill="tonexty", fillcolor="rgba(184,134,59,0.15)",
                              line=dict(width=0), name="Forecast range"))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["accounts"], mode="lines+markers",
                              name="Actual", line=dict(color=NAVY, width=2.5)))
    fig.add_trace(go.Scatter(
        x=[monthly["month"].iloc[-1]] + list(future_months),
        y=[monthly["accounts"].iloc[-1]] + list(pred),
        mode="lines+markers", name="Predicted",
        line=dict(color=GOLD, width=2.5, dash="dash"),
    ))
    fig.update_layout(**CHART_TEMPLATE, height=420, yaxis_title="Accounts", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Forecast evaluation")
    bt = backtest_forecast(monthly["accounts"].to_numpy(), holdout=6)
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{bt['mae']:.0f} accounts")
    c2.metric("RMSE", f"{bt['rmse']:.0f} accounts")
    c3.metric("MAPE", f"{bt['mape']:.1f}%")
    explain(f"""
    <b>How this was evaluated:</b> the last 6 months of actual data were hidden from the model, a
    forecast was generated for those months using only the earlier data, then compared back against
    what actually happened.<br><br>
    <b>MAE</b> (Mean Absolute Error) — on average, the forecast was off by
    <b>{bt['mae']:.0f} accounts</b> per month, in either direction.<br>
    <b>RMSE</b> (Root Mean Squared Error) — similar to MAE but penalizes large misses more heavily;
    a value close to MAE means errors were fairly consistent in size, not dominated by one bad month.<br>
    <b>MAPE</b> (Mean Absolute Percentage Error) — the forecast was off by about
    <b>{bt['mape']:.1f}%</b> on a typical month, which gives a scale-free sense of accuracy you can
    compare across portfolios of different sizes.
    """)

# ---------------------------------------------------------------- TAB 2
with tabs[1]:
    st.subheader("Debtor segmentation (K-Means) + extracted business rules")
    profile = df.groupby("segment")[FEATURE_COLS + ["repaid"]].mean().round(2)
    st.dataframe(profile, use_container_width=True)

    st.markdown("#### Extracted rules (decision tree approximating the clusters)")
    st.code(export_text(rule_tree, feature_names=FEATURE_COLS), language="text")

    st.markdown("#### Clustering evaluation")
    st.metric("Silhouette score", f"{silhouette:.3f}")
    explain(f"""
    <b>Silhouette score</b> measures how well-separated the segments are, from -1 (badly mixed) to +1
    (perfectly separated). A score of <b>{silhouette:.3f}</b> means the clusters are
    {"reasonably distinct" if silhouette > 0.25 else "somewhat overlapping"} — debtors within a segment
    look more like each other than like debtors in other segments, but real-world debtor data is noisy
    so scores in the 0.2–0.4 range are typical and still useful for rule extraction.
    """)

# ---------------------------------------------------------------- TAB 3
with tabs[2]:
    st.subheader("Repayment-probability model (Random Forest)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC", f"{eval_metrics['auc']:.3f}")
    c2.metric("Accuracy", f"{eval_metrics['accuracy']:.1%}")
    c3.metric("Precision", f"{eval_metrics['precision']:.1%}")
    c4.metric("Recall", f"{eval_metrics['recall']:.1%}")
    c5, c6 = st.columns(2)
    c5.metric("F1 score", f"{eval_metrics['f1']:.3f}")
    c6.metric("Brier score", f"{eval_metrics['brier']:.3f}")

    explain(f"""
    <b>AUC-ROC ({eval_metrics['auc']:.3f})</b> — the probability that the model ranks a randomly chosen
    debtor who <i>did</i> repay higher than one who <i>didn't</i>. 0.5 = no better than a coin flip,
    1.0 = perfect ranking. {eval_metrics['auc']:.3f} is a {"strong" if eval_metrics['auc']>0.8 else "reasonable"} result.<br><br>
    <b>Accuracy ({eval_metrics['accuracy']:.1%})</b> — share of held-out debtors correctly classified as
    repaid/defaulted at a 0.5 cutoff. Can be misleading on imbalanced data (most debtors repay here),
    so treat it alongside precision/recall, not alone.<br>
    <b>Precision ({eval_metrics['precision']:.1%})</b> — of the debtors the model flagged as "will repay",
    this share actually did. High precision means fewer wasted low-touch strategies on people who won't pay.<br>
    <b>Recall ({eval_metrics['recall']:.1%})</b> — of the debtors who actually repaid, this share the model
    correctly identified. High recall means fewer likely-payers get mistakenly sent to costly recovery paths.<br>
    <b>F1 ({eval_metrics['f1']:.3f})</b> — the balance between precision and recall in one number.<br>
    <b>Brier score ({eval_metrics['brier']:.3f})</b> — measures how well-calibrated the predicted
    probabilities are (lower is better, 0 = perfect). Unlike accuracy, it rewards the model for saying
    "70%" when things really do happen about 70% of the time, not just for getting the yes/no right.
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ROC curve")
        fpr, tpr, _ = eval_metrics["roc"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name="Model", line=dict(color=NAVY, width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random guess", line=dict(color=FAINT, dash="dash")))
        fig.update_layout(**CHART_TEMPLATE, height=340, xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Confusion matrix (held-out debtors)")
        cm = eval_metrics["confusion"]
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted default", "Predicted repay"], y=["Actual default", "Actual repay"],
            colorscale=[[0, CARD], [1, NAVY]], showscale=False, text=cm, texttemplate="%{text}", textfont_size=16,
        ))
        fig.update_layout(**CHART_TEMPLATE, height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Calibration — predicted vs. actual repayment rate")
    calib_df = pd.DataFrame({"proba": eval_metrics["proba_test"], "actual": eval_metrics["y_test"].values})
    calib_df["decile"] = pd.qcut(calib_df["proba"], 8, labels=False, duplicates="drop")
    calib = calib_df.groupby("decile").agg(mean_pred=("proba", "mean"), actual_rate=("actual", "mean")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", line=dict(color=FAINT, dash="dash")))
    fig.add_trace(go.Scatter(x=calib["mean_pred"], y=calib["actual_rate"], mode="lines+markers",
                              name="Model", line=dict(color=GOLD, width=2.5)))
    fig.update_layout(**CHART_TEMPLATE, height=360, xaxis_title="Mean predicted probability", yaxis_title="Actual repayment rate")
    st.plotly_chart(fig, use_container_width=True)
    explain("""
    Points sitting on the dashed diagonal mean the model is well-calibrated: when it says "80% chance
    of repayment," about 80% of those debtors actually did repay. Points above the line mean the model
    is under-confident there (actual outcomes better than predicted); points below mean it's over-confident.
    """)

    st.markdown("#### Feature importance")
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS + ["segment"]).sort_values()
    fig = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h", marker_color=NAVY))
    fig.update_layout(**CHART_TEMPLATE, height=320)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- TAB 4
with tabs[3]:
    st.subheader("TOPSIS prioritization")
    st.dataframe(
        df[["debtor_id", "prob_repay", "outstanding_balance", "days_past_due", "topsis_score", "priority"]]
        .sort_values("topsis_score", ascending=False).head(15),
        use_container_width=True,
    )
    explain("""
    <b>TOPSIS</b> ranks each debtor by how close they are to an "ideal" profile (high repayment
    probability, high exposure, high days-past-due — i.e. worth prioritizing) and how far from a
    "worst-case" profile. The closeness score (0–1) is bucketed into Low/Medium/High priority by
    terciles of the portfolio.
    """)

    st.markdown("#### AHP — criteria weights for strategy selection")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(ahp_w.round(3).rename("weight"), use_container_width=True)
    with c2:
        st.metric("Consistency ratio (CR)", f"{ahp_cr:.3f}")
        explain(f"""
        CR measures how logically consistent the expert's pairwise comparisons were
        (e.g. if A > B and B > C, does A > C by a compatible margin?). A CR under 0.10 is
        considered acceptable; <b>{ahp_cr:.3f}</b> is {"within" if ahp_cr < 0.1 else "above"} that
        threshold, so these weights are {"trustworthy" if ahp_cr < 0.1 else "worth revisiting"}.
        """)

    st.markdown("#### Recommended strategy per priority segment")
    for bucket in ["High", "Medium", "Low"]:
        ranking = select_strategy_for_segment(bucket, ahp_w)
        st.write(f"**{bucket}** → {ranking.index[0]} (score {ranking.iloc[0]:.3f}) — full order: {', '.join(ranking.index)}")

    st.markdown("#### Field-visit routing (High-priority debtors)")
    n_collectors = st.slider("Number of collectors", 1, 6, 3)
    field_df = df[df["priority"] == "High"].copy()
    routed = cluster_and_route(field_df, n_collectors)
    fig = go.Figure()
    for z in sorted(routed["collector_zone"].unique()):
        zdf = routed[routed["collector_zone"] == z]
        fig.add_trace(go.Scatter(x=zdf["longitude"], y=zdf["latitude"], mode="markers+lines",
                                  name=f"Zone {z}", marker=dict(size=7)))
    fig.update_layout(**CHART_TEMPLATE, height=420, xaxis_title="Longitude", yaxis_title="Latitude")
    st.plotly_chart(fig, use_container_width=True)
    explain("""
    Debtors flagged for a field visit are grouped into geographic zones (K-Means on lat/long), one per
    collector, then routed within each zone with a nearest-neighbour heuristic — a lightweight stand-in
    for a full vehicle-routing solver (e.g. Google OR-Tools) if you need optimal multi-stop routes.
    """)

# ---------------------------------------------------------------- TAB 5
with tabs[4]:
    st.subheader("Score an individual debtor")
    st.write("Adjust the sliders in the sidebar, then read the prediction below.")

    with st.sidebar:
        st.header("Debtor profile")
        balance = st.slider("Outstanding balance (THB)", 1000, 200000, 45000, step=500)
        dpd = st.slider("Days past due", 1, 365, 90)
        defaults = st.slider("Prior defaults", 0, 6, 1)
        attempted = st.slider("Contacts attempted", 0, 25, 8)
        successful = st.slider("Successful contacts", 0, 15, 3)
        income = st.slider("Monthly income (THB)", 5000, 120000, 32000, step=500)
        credit = st.slider("Credit score", 300, 850, 610, step=5)
        partial = st.radio("Prior partial payment?", ["No", "Yes"], horizontal=True)

    segment_input = rule_tree.predict(pd.DataFrame([{
        "outstanding_balance": balance, "days_past_due": dpd, "num_prior_defaults": defaults,
        "num_contacts_attempted": attempted, "num_successful_contacts": successful,
        "monthly_income": income, "credit_score": credit, "prior_partial_payment": 1 if partial == "Yes" else 0,
    }]))[0]

    row = pd.DataFrame([{
        "outstanding_balance": balance, "days_past_due": dpd, "num_prior_defaults": defaults,
        "num_contacts_attempted": attempted, "num_successful_contacts": successful,
        "monthly_income": income, "credit_score": credit,
        "prior_partial_payment": 1 if partial == "Yes" else 0, "segment": segment_input,
    }])
    proba = model.predict_proba(row)[0, 1]
    bucket = "High" if proba >= df["prob_repay"].quantile(0.67) else ("Medium" if proba >= df["prob_repay"].quantile(0.33) else "Low")
    urgency = fuzzy_engine.infer(proba, min(balance / df["outstanding_balance"].max(), 1.0))

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted repayment probability", f"{proba:.1%}")
    c2.metric("Priority bucket", bucket)
    c3.metric("Fuzzy urgency score", f"{urgency:.0f} / 100")

    strat = select_strategy_for_segment(bucket, ahp_w)
    st.write(f"**Recommended strategy:** {strat.index[0]}")
    explain(f"""
    This debtor's probability is compared against the portfolio's own distribution to assign a bucket
    (top third = High, middle third = Medium, bottom third = Low) — the same logic used for TOPSIS
    prioritization above. The fuzzy urgency score blends probability and exposure through the Mamdani
    rules in Layer 3c, giving a second, independently-computed read on how urgently this debtor needs
    attention; when the two roughly agree, that's a good sign the prioritization is robust.
    """)

    st.markdown("#### Where this debtor falls on the portfolio's calibration curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", line=dict(color=FAINT, dash="dash")))
    fig.add_trace(go.Scatter(x=calib["mean_pred"], y=calib["actual_rate"], mode="lines+markers",
                              name="Portfolio", line=dict(color=NAVY, width=2)))
    fig.add_vline(x=proba, line=dict(color=GOLD, width=2, dash="dot"))
    fig.update_layout(**CHART_TEMPLATE, height=340, xaxis_title="Predicted probability", yaxis_title="Actual repayment rate")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("All data, forecasts, and models on this page are synthetic and for demonstration. "
           "Swap in your own portfolio (same column names) to make every layer reflect real numbers.")
