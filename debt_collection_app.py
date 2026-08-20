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
# "NEW INPUT" — monthly payment-cycle forecast
#   Debtors are billed on one of five cycle days: 1, 5, 10, 15, 20.
#     - Due date 1  -> exclusively TDR (restructured) accounts.
#     - Due dates 5/10/15/20 -> Normal and OD accounts, mixed across all four.
#   All accounts start the month already flagged as debt accounts. As each
#   due date passes, a share of the accounts due that day "cure" (pay) and
#   exit debt status; the rest remain debt accounts for the rest of the
#   month. The forecast walks through checkpoints 1 -> 5 -> 10 -> 15 -> 20
#   and predicts how many accounts are still in debt after each one,
#   split by account type (TDR / Normal / OD).
# ========================================================================
CYCLE_DUE_DATES = [1, 5, 10, 15, 20]
CYCLE_CHECKPOINT_LABELS = ["Start", "Day 1", "Day 5", "Day 10", "Day 15", "Day 20 (EOM)"]


@st.cache_data
def generate_cycle_sample(n: int = 4000, tdr_share: float = 0.08, od_share: float = 0.3) -> pd.DataFrame:
    """Synthetic accounts tagged with a payment-cycle due date and an account
    type, with feature distributions nudged per type so the existing Layer 2
    model has something meaningful to score them on."""
    rng = np.random.default_rng(RANDOM_STATE + 11)
    n_tdr = int(n * tdr_share)
    n_rest = n - n_tdr

    due_date = np.concatenate([np.full(n_tdr, 1), rng.choice([5, 10, 15, 20], size=n_rest)])
    account_type = np.concatenate([
        np.full(n_tdr, "TDR", dtype=object),
        rng.choice(["Normal", "OD"], size=n_rest, p=[1 - od_share, od_share]),
    ])
    is_tdr = account_type == "TDR"
    is_od = account_type == "OD"

    balance = np.where(is_tdr, rng.gamma(1.6, 11000, n),
               np.where(is_od, rng.gamma(1.4, 9000, n), rng.gamma(2.0, 15000, n)))
    dpd = np.where(is_tdr, rng.integers(30, 200, n), rng.integers(1, 120, n))
    defaults = np.where(is_tdr, rng.poisson(1.8, n), rng.poisson(0.9, n))
    attempted = rng.integers(0, 20, n)
    successful = (attempted * rng.uniform(0.1, 0.7, n)).round().astype(int)
    income = rng.normal(35000, 12000, n).clip(5000, None).round(2)
    credit = rng.normal(620, 80, n).clip(300, 850).round(0)
    partial = rng.choice([0, 1], n, p=[0.5, 0.5])

    return pd.DataFrame({
        "account_type": account_type, "due_date_cycle": due_date,
        "outstanding_balance": balance.round(2), "days_past_due": dpd,
        "num_prior_defaults": defaults, "num_contacts_attempted": attempted,
        "num_successful_contacts": successful, "monthly_income": income,
        "credit_score": credit, "prior_partial_payment": partial,
    })


def estimate_cure_rates(cycle_sample: pd.DataFrame, _model, _rule_tree, feature_cols: list) -> dict:
    """Scores the cycle sample with the existing Layer 1 segment tree and
    Layer 2 repayment model, then averages predicted probability by account
    type to get a model-implied 'cure rate' assumption for each bucket."""
    scored = cycle_sample.copy()
    scored["segment"] = _rule_tree.predict(scored[feature_cols])
    scored["cure_prob"] = _model.predict_proba(scored[feature_cols + ["segment"]])[:, 1]
    return scored.groupby("account_type")["cure_prob"].mean().to_dict()


def simulate_cycle_depletion(total_accounts: int, tdr_pct: float, od_pct: float,
                              cure_rates: dict, noise: bool = False, rng=None) -> pd.DataFrame:
    """Depletes `total_accounts` debt accounts checkpoint by checkpoint.
    Returns one row per checkpoint (Start, Day1, Day5, Day10, Day15, Day20)
    with remaining debt accounts for TDR / Normal / OD / total."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE + 99)

    n_tdr = round(total_accounts * tdr_pct)
    n_rest = total_accounts - n_tdr
    n_od = round(n_rest * od_pct)
    n_normal = n_rest - n_od

    other_days = [5, 10, 15, 20]
    due_today_count = {("TDR", 1): n_tdr}
    base_normal, base_od = n_normal // len(other_days), n_od // len(other_days)
    for i, d in enumerate(other_days):
        due_today_count[("Normal", d)] = base_normal + (n_normal % len(other_days) if i == 0 else 0)
        due_today_count[("OD", d)] = base_od + (n_od % len(other_days) if i == 0 else 0)

    remaining = {"TDR": n_tdr, "Normal": n_normal, "OD": n_od}
    rows = [{"checkpoint": "Start", "TDR": n_tdr, "Normal": n_normal, "OD": n_od, "total": total_accounts}]

    for day in CYCLE_DUE_DATES:
        for t in ["TDR", "Normal", "OD"]:
            due_today = due_today_count.get((t, day), 0)
            if due_today == 0:
                continue
            rate = float(np.clip(cure_rates.get(t, 0.5), 0, 1))
            if noise:
                realized_rate = float(np.clip(rate + rng.normal(0, 0.04), 0, 1))
                cured = rng.binomial(due_today, realized_rate)
            else:
                cured = due_today * rate
            remaining[t] = max(remaining[t] - cured, 0)
        label = "Day 20 (EOM)" if day == 20 else f"Day {day}"
        rows.append({"checkpoint": label, "TDR": remaining["TDR"], "Normal": remaining["Normal"],
                     "OD": remaining["OD"], "total": remaining["TDR"] + remaining["Normal"] + remaining["OD"]})
    return pd.DataFrame(rows)


def backtest_cycle_forecast(tdr_pct: float, od_pct: float, cure_rates: dict,
                             prior_total: int, n_sims: int = 300) -> dict:
    """Evaluates the forecasting method itself: simulate a 'true' realized
    outcome many times with cure-rate noise (representing month-to-month
    variability actually observed in operations), and compare each realized
    end-of-month total against the single deterministic point forecast."""
    rng = np.random.default_rng(RANDOM_STATE + 7)
    point_forecast = simulate_cycle_depletion(prior_total, tdr_pct, od_pct, cure_rates, noise=False).iloc[-1]["total"]
    realized = np.array([
        simulate_cycle_depletion(prior_total, tdr_pct, od_pct, cure_rates, noise=True, rng=rng).iloc[-1]["total"]
        for _ in range(n_sims)
    ])
    mae = np.mean(np.abs(realized - point_forecast))
    rmse = np.sqrt(np.mean((realized - point_forecast) ** 2))
    mape = np.mean(np.abs((realized - point_forecast) / prior_total)) * 100
    p5, p95 = np.percentile(realized, [5, 95])
    return {"point_forecast": point_forecast, "mae": mae, "rmse": rmse, "mape": mape,
            "p5": p5, "p95": p95, "realized": realized}


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

cycle_sample = generate_cycle_sample()
model_cure_rates = estimate_cure_rates(cycle_sample, model, rule_tree, FEATURE_COLS)


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
    "⚖️ Layer 3 · Optimization", "🔍 Score a Debtor", "🆕 New Input · Cycle Forecast"
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

    st.markdown("#### Actual vs. predicted — table (this year onward)")
    current_year = pd.Timestamp.now().year
    actual_yr = (monthly[monthly["month"].dt.year >= current_year][["month", "accounts"]]
                 .rename(columns={"accounts": "Actual"}))
    pred_yr = pd.DataFrame({"month": future_months, "Predicted": pred})
    table = pd.merge(actual_yr, pred_yr, on="month", how="outer").sort_values("month")
    table["Month"] = table["month"].dt.strftime("%b %Y")
    table["Actual"] = table["Actual"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
    table["Predicted"] = table["Predicted"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
    st.dataframe(table[["Month", "Actual", "Predicted"]].set_index("Month"), use_container_width=True)
    explain(f"""
    Rows with only an <b>Actual</b> value are months from {current_year} that have already
    happened; rows with only a <b>Predicted</b> value are the {horizon} month(s) ahead in the
    forecast horizon set above. There's no overlap between the two columns since the forecast
    always starts the month right after the latest actual data point.
    """)

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

# ---------------------------------------------------------------- TAB 6
with tabs[5]:
    st.subheader("New Input — monthly payment-cycle forecast")
    st.write(
        "Debtors are billed on one of five cycle days: **1, 5, 10, 15, 20**. Due date **1 is "
        "exclusively TDR** (restructured) accounts; due dates **5/10/15/20 are Normal and OD "
        "accounts, mixed across all four**. All accounts start the month as debt accounts — this "
        "predicts how many remain in debt after each due date passes, split by type."
    )

    colL, colR = st.columns(2)
    with colL:
        st.markdown("**This month**")
        this_month_total = st.number_input("Total accounts at start of month", min_value=100,
                                            max_value=1_000_000, value=10000, step=100, key="cyc_total")
        tdr_pct = st.slider("Share of accounts that are TDR (due date = 1st)", 0, 30, 8, step=1,
                             format="%d%%") / 100
        od_pct = st.slider("Share of non-TDR accounts that are OD (rest = Normal)", 0, 100, 30, step=1,
                            format="%d%%") / 100
    with colR:
        st.markdown("**Cure rate assumptions** *(model-estimated from Layer 1 + Layer 2, editable)*")
        tdr_rate = st.slider("TDR cure rate", 0.0, 1.0, float(model_cure_rates.get("TDR", 0.5)), step=0.01)
        normal_rate = st.slider("Normal cure rate", 0.0, 1.0, float(model_cure_rates.get("Normal", 0.5)), step=0.01)
        od_rate = st.slider("OD cure rate", 0.0, 1.0, float(model_cure_rates.get("OD", 0.5)), step=0.01)
        prior_month_total = st.number_input("Last month's starting total (for the actual comparison line)",
                                             min_value=100, max_value=1_000_000,
                                             value=int(monthly["accounts"].iloc[-1]), step=100, key="cyc_prior")

    cure_rates = {"TDR": tdr_rate, "Normal": normal_rate, "OD": od_rate}
    rng_actual = np.random.default_rng(RANDOM_STATE + 5)

    predicted_path = simulate_cycle_depletion(this_month_total, tdr_pct, od_pct, cure_rates, noise=False)
    actual_path = simulate_cycle_depletion(prior_month_total, tdr_pct, od_pct, cure_rates, noise=True, rng=rng_actual)

    # ---- headline metrics ----
    eom_pred = predicted_path.iloc[-1]
    eom_actual = actual_path.iloc[-1]
    cured_total = this_month_total - eom_pred["total"]
    pred_pct_remaining = eom_pred["total"] / this_month_total
    actual_pct_remaining = eom_actual["total"] / prior_month_total

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted remaining at EOM (this month)", f"{int(eom_pred['total']):,}",
              f"{pred_pct_remaining:.1%} of starting volume")
    c2.metric("Actual remaining at EOM (last month)", f"{int(eom_actual['total']):,}",
              f"{actual_pct_remaining:.1%} of starting volume")
    c3.metric("Predicted cured by EOM", f"{int(cured_total):,}", f"{cured_total/this_month_total:.1%}")
    c4, c5 = st.columns(2)
    c4.metric("↳ Normal remaining at EOM (predicted)", f"{int(eom_pred['Normal']):,}")
    c5.metric("↳ OD remaining at EOM (predicted)", f"{int(eom_pred['OD']):,}")
    explain(f"""
    Last month started with <b>{prior_month_total:,}</b> accounts and this month started with
    <b>{this_month_total:,}</b> — since the starting volumes differ, compare the <b>% of starting
    volume remaining</b> shown under each metric above, not the raw counts, for an apples-to-apples
    read. Predicted leaves <b>{pred_pct_remaining:.1%}</b> of accounts still in debt at EOM vs.
    <b>{actual_pct_remaining:.1%}</b> actually realized last month.
    """)

    # ---- chart: actual (last month) vs predicted (this month) ----
    st.markdown("#### Remaining debt accounts by checkpoint — last month (actual) vs. this month (predicted)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_path["checkpoint"], y=actual_path["total"], mode="lines+markers",
                              name="Actual (last month)", line=dict(color=NAVY, width=2.5)))
    fig.add_trace(go.Scatter(x=predicted_path["checkpoint"], y=predicted_path["total"], mode="lines+markers",
                              name="Predicted (this month)", line=dict(color=GOLD, width=2.5, dash="dash")))
    fig.update_layout(**CHART_TEMPLATE, height=380, yaxis_title="Remaining debt accounts", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    explain("""
    <b>Actual (last month)</b> is a simulated realized outcome — cure rates applied with
    month-to-month noise, the way real operations data would look. <b>Predicted (this month)</b>
    applies the same cure-rate assumptions deterministically to this month's starting volume. The
    two lines being similar in shape is a sanity check that the assumptions are reasonable; a
    large gap suggests this month's mix or cure rates should be revisited.
    """)

    st.markdown("#### Predicted remaining accounts by type, per checkpoint")
    fig2 = go.Figure()
    for col, color in [("Normal", NAVY), ("OD", GOLD), ("TDR", GREY)]:
        fig2.add_trace(go.Scatter(x=predicted_path["checkpoint"], y=predicted_path[col], mode="lines+markers",
                                   name=col, line=dict(width=2.5, color=color)))
    fig2.update_layout(**CHART_TEMPLATE, height=360, yaxis_title="Remaining debt accounts")
    st.plotly_chart(fig2, use_container_width=True)
    explain("""
    TDR accounts only have a due date on the 1st, so their line drops once (at Day 1) and then
    stays flat for the rest of the month — they don't get another chance to cure until next
    month's cycle. Normal and OD accounts are spread across the 5th/10th/15th/20th, so they
    deplete in four smaller steps.
    """)

    st.markdown("#### Checkpoint breakdown — actual (last month) vs. predicted (this month)")
    merged = actual_path.merge(predicted_path, on="checkpoint", suffixes=(" · Actual", " · Predicted"))
    merged = merged.rename(columns={"checkpoint": "Checkpoint"})[[
        "Checkpoint", "total · Actual", "total · Predicted",
        "Normal · Actual", "Normal · Predicted",
        "OD · Actual", "OD · Predicted",
        "TDR · Actual", "TDR · Predicted",
    ]].rename(columns={
        "total · Actual": "Total (Actual)", "total · Predicted": "Total (Predicted)",
        "Normal · Actual": "Normal (Actual)", "Normal · Predicted": "Normal (Predicted)",
        "OD · Actual": "OD (Actual)", "OD · Predicted": "OD (Predicted)",
        "TDR · Actual": "TDR (Actual)", "TDR · Predicted": "TDR (Predicted)",
    })
    for c in merged.columns[1:]:
        merged[c] = merged[c].round(0).astype(int)

    def _highlight_eom(row):
        is_eom = row.name == "Day 20 (EOM)"
        return ["background-color: #FBF7EC; font-weight: 600" if is_eom else "" for _ in row]

    st.dataframe(merged.set_index("Checkpoint").style.apply(_highlight_eom, axis=1), use_container_width=True)
    explain("""
    The highlighted <b>Day 20 (EOM)</b> row is the end-of-month result — the number that matters
    most for reporting. Compare its Actual and Predicted columns directly: if Predicted is
    consistently higher than Actual, this month's cure-rate assumptions may be too pessimistic
    (or vice versa), which is a good signal to revisit the sliders above.
    """)

    # ---- evaluation ----
    st.markdown("#### Forecast evaluation")
    bt = backtest_cycle_forecast(tdr_pct, od_pct, cure_rates, prior_month_total, n_sims=300)
    e1, e2, e3 = st.columns(3)
    e1.metric("Expected deviation (MAE-like)", f"{bt['mae']:.0f} accounts")
    e2.metric("RMSE-like", f"{bt['rmse']:.0f} accounts")
    e3.metric("MAPE-like", f"{bt['mape']:.1f}%")
    explain(f"""
    <b>How this was evaluated:</b> this checks the reliability of the <i>method</i>, not this
    month's specific number. Using last month's starting total ({prior_month_total:,} accounts) as
    a baseline, 300 alternate versions of "what could have happened" were simulated with the same
    cure-rate assumptions plus realistic month-to-month noise, then compared against the single
    deterministic point forecast the method would have produced for that baseline
    (<b>{bt['point_forecast']:.0f} accounts</b>).<br><br>
    On average, a realized month-end total differed from the point forecast by about
    <b>±{bt['mae']:.0f} accounts ({bt['mape']:.1f}%)</b>, with 90% of outcomes falling between
    <b>{bt['p5']:.0f}</b> and <b>{bt['p95']:.0f}</b>. Apply that same error margin to this month's
    prediction above (<b>{int(eom_pred['total']):,} accounts</b>) as a rough confidence range,
    rather than treating it as an exact guarantee, when setting collection targets.
    """)

st.markdown("---")
st.caption("All data, forecasts, and models on this page are synthetic and for demonstration. "
           "Swap in your own portfolio (same column names) to make every layer reflect real numbers.")
