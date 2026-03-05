import os
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Insightify Live Dashboard", layout="wide")

DATA_PATH = Path("cleaned_customer_data.csv")
MODEL_PATH = Path("models/model.pkl")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.pkl")
FEATURE_MEDIANS_PATH = Path("models/feature_medians.pkl")
MODEL_METADATA_PATH = Path("models/model_metadata.json")

MEMBERSHIP_LABELS = {
    0: "Basic",
    1: "Silver",
    2: "Gold",
    3: "Platinum",
}


def inject_custom_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
        :root {
            --bg-0: #faf8f4;
            --bg-1: #ffffff;
            --ink-0: #2d2a26;
            --ink-1: #6b6560;
            --olive: #6b7c3e;
            --olive-light: #e8edda;
            --olive-dark: #4a5a28;
            --cream: #f5f0e8;
            --peach: #fce8dc;
            --sage: #dde5cc;
            --warm-white: #fefcf9;
        }
        .stApp {
            background: var(--bg-0);
            color: var(--ink-0);
            font-family: 'DM Sans', sans-serif;
        }
        h1, h2, h3 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            letter-spacing: -0.02em;
            color: var(--ink-0);
        }
        .stMetric {
            background: var(--bg-1);
            border: 1px solid #ece7df;
            border-radius: 16px;
            padding: 10px 12px;
            box-shadow: 0 2px 8px rgba(45, 42, 38, 0.04);
        }
        @keyframes heroSlideUp {
            from { opacity: 0; transform: translateY(18px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes chipFadeIn {
            from { opacity: 0; transform: translateY(10px) scale(0.96); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        .hero-card {
            border-radius: 20px;
            padding: 26px 28px 22px 28px;
            background: linear-gradient(135deg, var(--olive) 0%, var(--olive-dark) 100%);
            color: #fefcf9;
            box-shadow: 0 8px 28px rgba(107, 124, 62, 0.18);
            margin-bottom: 18px;
            animation: heroSlideUp 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
        }
        .hero-card h3 {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.45rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #ffffff;
            margin: 0 0 6px 0;
        }
        .hero-card p {
            font-size: 0.9rem;
            color: rgba(255,255,255,0.8);
            line-height: 1.55;
        }
        .pitch-chip {
            display: inline-block;
            border-radius: 10px;
            padding: 7px 16px;
            margin-right: 10px;
            margin-top: 8px;
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--olive-dark);
            background: linear-gradient(180deg, #f4f7e8 0%, #dce3c6 100%);
            border: 1px solid #c8d1a8;
            box-shadow: 0 3px 0 #b5be96, 0 5px 10px rgba(74, 90, 40, 0.15);
            text-shadow: 0 1px 0 rgba(255,255,255,0.6);
            animation: chipFadeIn 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
            transition: transform 0.1s ease, box-shadow 0.1s ease;
        }
        .pitch-chip:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 0 #b5be96, 0 7px 14px rgba(74, 90, 40, 0.2);
        }
        .pitch-chip:nth-child(1) { animation-delay: 0.15s; }
        .pitch-chip:nth-child(2) { animation-delay: 0.28s; }
        .pitch-chip:nth-child(3) { animation-delay: 0.41s; }
        .hero-subline {
            margin-top: 12px;
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.55);
            animation: chipFadeIn 0.5s cubic-bezier(0.22, 1, 0.36, 1) 0.52s both;
        }
        .hero-subline .dot {
            display: inline-block;
            width: 4px;
            height: 4px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.4);
            margin: 0 10px;
            transform: translateY(-2px);
        }
        .insight-card {
            border: 1px solid #ece7df;
            background: var(--bg-1);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 3px 14px rgba(45, 42, 38, 0.05);
            margin: 6px 0 12px 0;
        }
        .section-highlight {
            display: inline-block;
            background: linear-gradient(135deg, var(--olive) 0%, var(--olive-dark) 100%);
            color: #ffffff;
            padding: 8px 20px 8px 16px;
            border-radius: 10px;
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            margin-bottom: 4px;
            box-shadow: 0 3px 10px rgba(107, 124, 62, 0.2);
        }
        .stDataFrame, .stPlotlyChart {
            border-radius: 16px;
            overflow: hidden;
            background: var(--bg-1);
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid #d6cfbf;
            background: var(--bg-1);
            color: var(--olive-dark);
            font-weight: 600;
            transition: all 0.15s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            background: var(--olive-light);
            border-color: var(--olive);
        }
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stSlider > div {
            border-radius: 12px;
        }
        .stSidebar > div:first-child {
            background: var(--cream);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_block():
    st.markdown(
        """
        <div class="hero-card">
            <h3 style="margin:0;">Upgrade Growth Intelligence</h3>
            <p style="margin:6px 0 10px 0;opacity:.85;font-size:0.88rem;">
                Audience &rarr; Offer &rarr; ROI
            </p>
            <span class="pitch-chip">Propensity</span>
            <span class="pitch-chip">ROI Planning</span>
            <span class="pitch-chip">Action Playbooks</span>
            <div class="hero-subline">
                Acquisition <span class="dot"></span> Retention <span class="dot"></span> Upgrade
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_mistral_api_key() -> str | None:
    # Priority: Streamlit secrets, then environment variable.
    key = None
    if hasattr(st, "secrets"):
        key = st.secrets.get("MISTRAL_API_KEY")
    if not key:
        key = os.getenv("MISTRAL_API_KEY")
    return key


@st.cache_data(ttl=120)
def generate_mistral_summary(prompt_text: str, model_name: str, api_key: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "temperature": 0.2,
        "max_tokens": 450,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a senior marketing analytics assistant. "
                    "Summarize dashboard data with practical campaign actions. "
                    "Do not hallucinate values; use only provided context."
                ),
            },
            {
                "role": "user",
                "content": prompt_text,
            },
        ],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

@st.cache_data(ttl=30)
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    return df


@st.cache_resource(ttl=60)
def load_artifacts():
    required = [MODEL_PATH, FEATURE_COLUMNS_PATH, FEATURE_MEDIANS_PATH, MODEL_METADATA_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts. Expected files not found: " + ", ".join(missing)
        )

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with FEATURE_COLUMNS_PATH.open("rb") as f:
        feature_columns = pickle.load(f)
    with FEATURE_MEDIANS_PATH.open("rb") as f:
        feature_medians = pickle.load(f)
    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, feature_columns, feature_medians, metadata


def kpi_block(df: pd.DataFrame):
    labeled = df["Membership_upgrade"].notna().sum() if "Membership_upgrade" in df.columns else 0
    upgrade_rate = (
        pd.to_numeric(df["Membership_upgrade"], errors="coerce").mean() if "Membership_upgrade" in df.columns else np.nan
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Labeled Rows", f"{int(labeled):,}")
    c4.metric("Upgrade Rate", f"{upgrade_rate:.2%}" if pd.notna(upgrade_rate) else "N/A")


def model_info_block(metadata: dict):
    st.markdown('<div class="section-highlight">Model Info</div>', unsafe_allow_html=True)
    metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", str(metadata.get("model_name", "N/A")))
    c2.metric("Version", str(metadata.get("version", "N/A")))
    c3.metric("Holdout Accuracy", f"{float(metrics.get('accuracy', 0.0)):.3f}")
    c4.metric("PR-AUC", f"{float(metrics.get('pr_auc', 0.0)):.3f}")


def charts_block(df: pd.DataFrame):
    st.markdown('<div class="section-highlight">Behavior Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    if {"Spending_Score", "Membership_upgrade"}.issubset(df.columns):
        fig1 = px.histogram(
            df,
            x="Spending_Score",
            color="Membership_upgrade",
            barmode="overlay",
            nbins=25,
            title="Spending Score vs Upgrade",
        )
        c1.plotly_chart(fig1, width="stretch")

    if {"Purchase_Frequency", "Membership_upgrade"}.issubset(df.columns):
        fig2 = px.box(
            df,
            x="Membership_upgrade",
            y="Purchase_Frequency",
            title="Purchase Frequency by Upgrade",
        )
        c2.plotly_chart(fig2, width="stretch")

    if {"Avg_Order_Value", "Weekend_Order_Ratio", "Membership_upgrade"}.issubset(df.columns):
        fig3 = px.scatter(
            df,
            x="Avg_Order_Value",
            y="Weekend_Order_Ratio",
            color="Membership_upgrade",
            title="Order Value vs Weekend Ratio",
            opacity=0.6,
        )
        st.plotly_chart(fig3, width="stretch")


def _to_float(value, default=0.0):
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _tier_label(value) -> str:
    try:
        if pd.isna(value):
            return "Unknown"
        value_int = int(float(value))
        return MEMBERSHIP_LABELS.get(value_int, f"Tier-{value_int}")
    except Exception:
        return str(value)


def score_customers(df: pd.DataFrame, model, model_cols, feature_medians: dict) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    scored = df.copy()
    model_ready = pd.DataFrame(index=scored.index)

    for col in model_cols:
        if col in scored.columns:
            series = pd.to_numeric(scored[col], errors="coerce")
        else:
            series = pd.Series([np.nan] * len(scored), index=scored.index)
        model_ready[col] = series.fillna(_to_float(feature_medians.get(col, 0.0), 0.0))

    scored["Upgrade_Probability"] = model.predict_proba(model_ready[model_cols])[:, 1]
    return scored


def tier_strategy_block(scored_df: pd.DataFrame):
    st.markdown('<div class="section-highlight">Membership Tier Opportunity</div>', unsafe_allow_html=True)
    view = scored_df.copy()
    if "Membership_Level" in view.columns:
        view["Membership_Tier"] = view["Membership_Level"].apply(_tier_label)
    else:
        view["Membership_Tier"] = "Unknown"

    required = {"Membership_Tier", "Upgrade_Probability", "Avg_Order_Value", "Purchase_Frequency"}
    if not required.issubset(view.columns):
        st.info("Tier strategy needs membership, order value, and purchase frequency columns.")
        return

    summary = (
        view.groupby("Membership_Tier", as_index=False)
        .agg(
            customers=("Membership_Tier", "count"),
            avg_propensity=("Upgrade_Probability", "mean"),
            avg_order_value=("Avg_Order_Value", "mean"),
            avg_purchase_frequency=("Purchase_Frequency", "mean"),
        )
        .sort_values("avg_propensity", ascending=False)
    )

    n_rows = len(summary)
    table_height = 38 + (n_rows * 36)

    c1, c2 = st.columns([1.3, 1.7])
    c1.dataframe(
        summary.style.format(
            {
                "avg_propensity": "{:.2%}",
                "avg_order_value": "{:.2f}",
                "avg_purchase_frequency": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
        height=table_height,
    )

    fig = px.bar(
        summary,
        x="Membership_Tier",
        y="avg_propensity",
        color="customers",
        title="Propensity by Tier",
        text="avg_propensity",
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(yaxis_tickformat=".0%", margin=dict(t=36, b=10, l=10, r=10), height=max(table_height + 40, 250))
    c2.plotly_chart(fig, width="stretch")


def campaign_targeting_block(scored_df: pd.DataFrame):
    st.markdown('<div class="section-highlight">Targeting &amp; Offer Planner</div>', unsafe_allow_html=True)
    st.caption("English action plan for campaign team: target audience, recommended offer, and expected ROI.")

    if scored_df.empty or "Upgrade_Probability" not in scored_df.columns:
        st.info("No scored data available for targeting recommendations.")
        return

    horizon_days = st.slider("Planning horizon (days)", min_value=7, max_value=60, value=30, step=1)
    budget_cap = st.number_input("Campaign budget cap", min_value=0.0, value=20000.0, step=500.0)

    view = scored_df.copy()
    complaints = pd.to_numeric(
        view["Last_Month_Complaints"] if "Last_Month_Complaints" in view.columns else pd.Series([0.0] * len(view)),
        errors="coerce",
    ).fillna(0)
    low_rating = pd.to_numeric(
        view["Low_Rating"] if "Low_Rating" in view.columns else pd.Series([0.0] * len(view)),
        errors="coerce",
    ).fillna(0)
    avg_order = pd.to_numeric(
        view["Avg_Order_Value"] if "Avg_Order_Value" in view.columns else pd.Series([0.0] * len(view)),
        errors="coerce",
    ).fillna(0)
    avg_freq = pd.to_numeric(
        view["Purchase_Frequency"] if "Purchase_Frequency" in view.columns else pd.Series([0.0] * len(view)),
        errors="coerce",
    ).fillna(0)

    scenarios = [
        {
            "segment": "Premium Upsell",
            "mask": (view["Upgrade_Probability"] >= 0.72) & (complaints == 0) & (low_rating == 0),
            "offer": "Priority delivery + 12% upgrade credit",
            "channel": "App push + email",
            "uplift": 0.18,
            "cost_rate": 0.12,
        },
        {
            "segment": "Value Seekers",
            "mask": view["Upgrade_Probability"].between(0.55, 0.719, inclusive="both"),
            "offer": "7-day free delivery + points booster",
            "channel": "WhatsApp + app inbox",
            "uplift": 0.12,
            "cost_rate": 0.09,
        },
        {
            "segment": "Service Recovery",
            "mask": (view["Upgrade_Probability"] >= 0.35) & ((complaints >= 1) | (low_rating >= 1)),
            "offer": "Service apology credit + concierge callback",
            "channel": "Outbound call + SMS",
            "uplift": 0.09,
            "cost_rate": 0.06,
        },
    ]

    rows = []
    horizon_factor = max(horizon_days / 30.0, 0.25)
    for sc in scenarios:
        seg_df = view[sc["mask"]]
        customers = int(len(seg_df))
        if customers == 0:
            continue

        seg_prob = float(seg_df["Upgrade_Probability"].mean())
        seg_order = float(avg_order.loc[seg_df.index].mean())
        seg_freq = float(avg_freq.loc[seg_df.index].mean())

        expected_upgrades = customers * seg_prob * (1.0 + sc["uplift"])
        expected_upgrades = max(expected_upgrades, 0.0)
        revenue_per_upgrade = max(seg_order * max(seg_freq * 0.22, 0.8) * horizon_factor, 1.0)
        projected_revenue = expected_upgrades * revenue_per_upgrade
        campaign_cost = customers * max(seg_order, 1.0) * sc["cost_rate"]
        roi = (projected_revenue - campaign_cost) / campaign_cost if campaign_cost > 0 else np.nan

        rows.append(
            {
                "Segment": sc["segment"],
                "Who to target": f"{customers:,} customers",
                "Recommended offer": sc["offer"],
                "Primary channel": sc["channel"],
                "Expected upgrades": expected_upgrades,
                "Projected revenue lift": projected_revenue,
                "Campaign cost": campaign_cost,
                "Expected ROI": roi,
            }
        )

    if not rows:
        st.warning("No targetable segment found with current data filters.")
        return

    plan_df = pd.DataFrame(rows).sort_values("Expected ROI", ascending=False)
    cumulative_cost = plan_df["Campaign cost"].cumsum()
    plan_df = plan_df[cumulative_cost <= budget_cap] if budget_cap > 0 else plan_df.iloc[0:0]
    if plan_df.empty:
        st.warning("Budget cap is too low for current segment sizes. Increase budget to see recommendations.")
        return

    top_pick = plan_df.iloc[0]
    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Top recommendation:</strong> Focus on <strong>{top_pick['Segment']}</strong> using
            <strong>{top_pick['Recommended offer']}</strong>. Expected ROI is
            <strong>{top_pick['Expected ROI']:.1%}</strong> over the selected planning horizon.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected Segments", f"{len(plan_df)}")
    c2.metric("Total Planned Cost", f"{float(plan_df['Campaign cost'].sum()):,.0f}")
    c3.metric("Total Projected Lift", f"{float(plan_df['Projected revenue lift'].sum()):,.0f}")

    # Marketing-ops friendly export: one row per campaign segment.
    ops_export_df = plan_df.copy()
    ops_export_df["target_customers"] = (
        ops_export_df["Who to target"].astype(str).str.replace(r"[^0-9]", "", regex=True)
    )
    ops_export_df["target_customers"] = pd.to_numeric(ops_export_df["target_customers"], errors="coerce").fillna(0).astype(int)
    ops_export_df["planning_horizon_days"] = int(horizon_days)
    ops_export_df["budget_cap"] = float(budget_cap)
    ops_export_df = ops_export_df.rename(
        columns={
            "Segment": "segment",
            "Recommended offer": "recommended_offer",
            "Primary channel": "primary_channel",
            "Expected upgrades": "expected_upgrades",
            "Projected revenue lift": "projected_revenue_lift",
            "Campaign cost": "campaign_cost",
            "Expected ROI": "expected_roi",
        }
    )
    ops_export_df = ops_export_df[
        [
            "segment",
            "target_customers",
            "recommended_offer",
            "primary_channel",
            "expected_upgrades",
            "projected_revenue_lift",
            "campaign_cost",
            "expected_roi",
            "planning_horizon_days",
            "budget_cap",
        ]
    ]
    st.session_state["last_campaign_plan_df"] = ops_export_df
    st.download_button(
        "Download Targeting Plan DF (.csv)",
        data=ops_export_df.to_csv(index=False).encode("utf-8"),
        file_name="insightify_targeting_plan_df.csv",
        mime="text/csv",
    )

    st.dataframe(
        plan_df.style.format(
            {
                "Expected upgrades": "{:.0f}",
                "Projected revenue lift": "{:.0f}",
                "Campaign cost": "{:.0f}",
                "Expected ROI": "{:.1%}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    fig = px.bar(
        plan_df,
        x="Segment",
        y="Expected ROI",
        color="Segment",
        text="Expected ROI",
        title="Expected ROI by Recommended Segment",
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")


def offer_simulator_block(scored_df: pd.DataFrame):
    st.markdown('<div class="section-highlight">Offer Impact Simulator</div>', unsafe_allow_html=True)
    st.caption(
        "Estimate campaign impact before launch using a practical what-if model."
    )

    if scored_df.empty or "Upgrade_Probability" not in scored_df.columns:
        st.info("No scored data available for simulation.")
        return

    sim_col1, sim_col2, sim_col3, sim_col4 = st.columns(4)
    discount_pct = sim_col1.slider("Discount %", min_value=0, max_value=35, value=12, step=1)
    free_delivery_days = sim_col2.slider("Free Delivery Days", min_value=0, max_value=30, value=7, step=1)
    loyalty_points = sim_col3.slider("Bonus Points", min_value=0, max_value=2000, value=600, step=100)
    priority_support = sim_col4.toggle("Priority Support", value=True)

    min_prob, max_prob = st.slider(
        "Target propensity range",
        min_value=0.0,
        max_value=1.0,
        value=(0.4, 0.8),
        step=0.05,
    )

    target = scored_df[
        (scored_df["Upgrade_Probability"] >= min_prob)
        & (scored_df["Upgrade_Probability"] <= max_prob)
    ].copy()

    if target.empty:
        st.warning("No customers matched selected propensity range.")
        return

    avg_prob = float(target["Upgrade_Probability"].mean())
    order_series = pd.to_numeric(
        target["Avg_Order_Value"] if "Avg_Order_Value" in target.columns else pd.Series([0.0] * len(target)),
        errors="coerce",
    ).fillna(0)
    freq_series = pd.to_numeric(
        target["Purchase_Frequency"] if "Purchase_Frequency" in target.columns else pd.Series([0.0] * len(target)),
        errors="coerce",
    ).fillna(0)
    avg_order = float(order_series.mean())
    avg_freq = float(freq_series.mean())

    uplift_multiplier = (
        1.0
        + (discount_pct * 0.012)
        + (free_delivery_days * 0.01)
        + (loyalty_points / 1000.0 * 0.05)
        + (0.08 if priority_support else 0.0)
    )
    uplift_multiplier = min(uplift_multiplier, 1.9)

    expected_conversions = len(target) * avg_prob * uplift_multiplier
    incremental_orders = expected_conversions * max(avg_freq * 0.25, 1.0)
    projected_incremental_revenue = incremental_orders * max(avg_order, 1.0)

    offer_cost = (
        len(target) * (discount_pct / 100.0) * max(avg_order, 1.0) * 0.7
        + len(target) * free_delivery_days * 0.2
        + len(target) * loyalty_points * 0.003
        + len(target) * (0.6 if priority_support else 0.0)
    )
    estimated_roi = (
        (projected_incremental_revenue - offer_cost) / offer_cost
        if offer_cost > 0
        else np.nan
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target Customers", f"{len(target):,}")
    c2.metric("Expected Upgrades", f"{expected_conversions:,.0f}")
    c3.metric("Projected Revenue Lift", f"{projected_incremental_revenue:,.0f}")
    c4.metric("Estimated ROI", f"{estimated_roi:.1%}" if pd.notna(estimated_roi) else "N/A")

    st.markdown(
        "**Operational tip:** Start with this segment, run a 2-week A/B test, and compare observed upgrades versus forecast." 
    )


def next_best_action_block(scored_df: pd.DataFrame):
    st.markdown('<div class="section-highlight">Next Best Action Engine</div>', unsafe_allow_html=True)
    st.caption("Segment-specific campaign actions with recommended offers for every customer.")

    if scored_df.empty or "Upgrade_Probability" not in scored_df.columns:
        st.info("No scored customers available.")
        return

    view = scored_df.copy()
    complaints = pd.to_numeric(
        view["Last_Month_Complaints"] if "Last_Month_Complaints" in view.columns else pd.Series([0.0] * len(view)),
        errors="coerce",
    ).fillna(0)
    low_rating = pd.to_numeric(
        view["Low_Rating"] if "Low_Rating" in view.columns else pd.Series([0.0] * len(view)),
        errors="coerce",
    ).fillna(0)

    rescue_mask = (view["Upgrade_Probability"] >= 0.35) & ((complaints >= 1) | (low_rating >= 1))
    upsell_mask = (view["Upgrade_Probability"] >= 0.7) & (~rescue_mask)
    nurture_mask = (view["Upgrade_Probability"].between(0.4, 0.699, inclusive="both")) & (~rescue_mask)

    view["Action_Segment"] = np.where(
        rescue_mask,
        "Rescue & Recover",
        np.where(
            upsell_mask,
            "Upsell Now",
            np.where(nurture_mask, "Nurture Campaign", "Low Priority"),
        ),
    )

    offer_map = {
        "Upsell Now": "Gold/Platinum Fast-Track + Priority Delivery",
        "Nurture Campaign": "7-Day Free Delivery + Personalized Bundle Offer",
        "Rescue & Recover": "Service Recovery Credit + Support Callback",
        "Low Priority": "Brand Awareness Content / No discount",
    }
    view["Recommended_Offer"] = view["Action_Segment"].map(offer_map)

    split = (
        view.groupby("Action_Segment", as_index=False)
        .agg(customers=("Action_Segment", "count"), avg_prob=("Upgrade_Probability", "mean"))
        .sort_values("customers", ascending=False)
    )
    split["share"] = split["customers"] / max(len(view), 1)

    c1, c2 = st.columns([1.4, 1.6])
    n_action = len(split)
    action_table_h = 38 + (n_action * 36)
    c1.dataframe(
        split.style.format({"avg_prob": "{:.2%}", "share": "{:.1%}"}),
        width="stretch",
        hide_index=True,
        height=action_table_h,
    )

    fig = px.pie(
        split,
        names="Action_Segment",
        values="customers",
        title="Action Mix for Campaign Team",
        hole=0.45,
    )
    fig.update_layout(margin=dict(t=36, b=10, l=10, r=10), height=max(action_table_h + 40, 250))
    c2.plotly_chart(fig, width="stretch")

    show_cols = [
        c
        for c in ["CustomerID", "Membership_Level", "Upgrade_Probability", "Action_Segment", "Recommended_Offer"]
        if c in view.columns
    ]
    if show_cols:
        preview = view.sort_values("Upgrade_Probability", ascending=False).head(25)[show_cols].copy()
        if "Membership_Level" in preview.columns:
            preview["Membership_Level"] = preview["Membership_Level"].apply(_tier_label)
        st.dataframe(
            preview.style.format({"Upgrade_Probability": "{:.2%}"}),
            width="stretch",
            hide_index=True,
            height=420,
        )


def prediction_block(df: pd.DataFrame, model, model_cols, feature_medians: dict, metadata: dict):
    st.markdown('<div class="section-highlight">Live Upgrade Prediction</div>', unsafe_allow_html=True)
    holdout_acc = float(metadata.get("metrics", {}).get("accuracy", 0.0))
    st.caption(f"Using saved artifact model. Holdout accuracy: {holdout_acc:.3f}")

    latest = df.tail(1).copy()
    if latest.empty:
        st.warning("No rows available for prediction")
        return

    row = latest.iloc[0].to_dict()
    with st.form("predict_form"):
        left, right = st.columns(2)
        spending = left.number_input("Spending_Score", value=float(row.get("Spending_Score", 50.0)))
        freq = left.number_input("Purchase_Frequency", value=float(row.get("Purchase_Frequency", 8.0)))
        order_val = left.number_input("Avg_Order_Value", value=float(row.get("Avg_Order_Value", 60.0)))
        weekend_ratio = right.number_input("Weekend_Order_Ratio", value=float(row.get("Weekend_Order_Ratio", 0.5)))
        membership = right.number_input("Membership_Level", value=float(row.get("Membership_Level", 1.0)))
        discount = right.number_input("Discount_Usage_Freq", value=float(row.get("Discount_Usage_Freq", 1.0)))
        submitted = st.form_submit_button("Predict Upgrade Probability")

    if submitted:
        input_map = {col: _to_float(row.get(col), feature_medians.get(col, 0.0)) for col in model_cols}
        input_map.update(
            {
                "Spending_Score": spending,
                "Purchase_Frequency": freq,
                "Avg_Order_Value": order_val,
                "Weekend_Order_Ratio": weekend_ratio,
                "Membership_Level": membership,
                "Discount_Usage_Freq": discount,
            }
        )
        for col in model_cols:
            input_map[col] = _to_float(input_map.get(col), feature_medians.get(col, 0.0))
        input_df = pd.DataFrame([input_map])[model_cols]
        proba = model.predict_proba(input_df)[0, 1]
        st.success(f"Predicted upgrade probability: {proba:.2%}")
        if proba >= 0.7:
            st.info("Confidence band: High upgrade likelihood")
        elif proba >= 0.4:
            st.info("Confidence band: Medium upgrade likelihood")
        else:
            st.info("Confidence band: Low upgrade likelihood")


def build_prompt(context_text: str, template_name: str, user_query: str = "") -> str:
    base = f"Dashboard context JSON:\n{context_text}\n\n"
    if template_name == "Executive Summary":
        return (
            base
            + "Create an executive marketing summary in markdown with sections: "
            "Key Trends, Risk Flags, 3 Campaign Actions, Persona Snapshot, and Next 7-Day Plan. "
            "Use only values present in context."
        )
    if template_name == "Campaign Plan":
        return (
            base
            + "Create a focused campaign plan in markdown with sections: "
            "Target Segment, Incentive Offer, Channel, Message Angle, Expected Impact. "
            "Give 3 concrete campaign ideas and mention likely upgrade confidence bands."
        )
    if template_name == "Persona Deep Dive":
        return (
            base
            + "Create one detailed persona in markdown with sections: "
            "Profile, Behavior Signals, Upgrade Triggers, Retention Risks, Recommended Offer."
        )
    return (
        base
        + "Answer the user's dashboard question in markdown using only context values. "
        f"User question: {user_query}"
    )


def build_report_markdown(summary_text: str, context: dict, metadata: dict) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        "# Insightify AI Marketing Report\n\n"
        f"Generated at: `{generated_at}`\n\n"
        "## Model Snapshot\n"
        f"- Model: `{metadata.get('model_name', 'N/A')}`\n"
        f"- Version: `{metadata.get('version', 'N/A')}`\n"
        f"- Accuracy: `{float(metadata.get('metrics', {}).get('accuracy', 0.0)):.3f}`\n"
        f"- PR-AUC: `{float(metadata.get('metrics', {}).get('pr_auc', 0.0)):.3f}`\n\n"
        "## Dashboard Context\n"
        "```json\n"
        f"{json.dumps(context, indent=2)}\n"
        "```\n\n"
        "## AI Summary\n\n"
        f"{summary_text}\n"
    )


def build_report_dataframe(summary_text: str, context: dict, metadata: dict) -> pd.DataFrame:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    model_metrics = metadata.get("metrics", {}) if isinstance(metadata, dict) else {}
    report_row = {
        "generated_at_utc": generated_at,
        "model_name": metadata.get("model_name", "N/A"),
        "model_version": metadata.get("version", "N/A"),
        "accuracy": float(model_metrics.get("accuracy", 0.0)),
        "pr_auc": float(model_metrics.get("pr_auc", 0.0)),
        "rows": context.get("rows"),
        "columns": context.get("columns"),
        "labeled_rows": context.get("labeled_rows"),
        "upgrade_rate": context.get("upgrade_rate"),
        "avg_spending_score": context.get("avg_spending_score"),
        "avg_purchase_frequency": context.get("avg_purchase_frequency"),
        "avg_order_value": context.get("avg_order_value"),
        "segment_upgrade_rate_json": json.dumps(context.get("segment_upgrade_rate", [])),
        "ai_summary": summary_text,
    }
    return pd.DataFrame([report_row])


def ai_summary_block(df: pd.DataFrame, metadata: dict, auto_summary: bool = False):
    st.markdown('<div class="section-highlight">AI Marketing Summary (Mistral)</div>', unsafe_allow_html=True)

    api_key = get_mistral_api_key()
    if not api_key:
        st.info(
            "Mistral API key not found. Add `MISTRAL_API_KEY` in environment or Streamlit secrets to enable summaries."
        )
        return

    model_name = st.selectbox(
        "Mistral Model",
        options=["mistral-small-latest", "mistral-large-latest"],
        index=0,
    )
    template_name = st.selectbox(
        "Prompt Template",
        options=["Executive Summary", "Campaign Plan", "Persona Deep Dive", "Custom Q&A"],
        index=0,
    )
    user_query = ""
    if template_name == "Custom Q&A":
        user_query = st.text_input("Ask a dashboard question")

    labeled = int(df["Membership_upgrade"].notna().sum()) if "Membership_upgrade" in df.columns else 0
    upgrade_rate = (
        float(pd.to_numeric(df["Membership_upgrade"], errors="coerce").mean())
        if "Membership_upgrade" in df.columns
        else np.nan
    )

    segment_stats = []
    if {"Membership_Level", "Membership_upgrade"}.issubset(df.columns):
        grp = (
            df.groupby("Membership_Level")["Membership_upgrade"]
            .mean()
            .sort_index()
            .reset_index()
        )
        segment_stats = grp.to_dict(orient="records")

    context = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "labeled_rows": labeled,
        "upgrade_rate": None if pd.isna(upgrade_rate) else round(upgrade_rate, 4),
        "avg_spending_score": float(df["Spending_Score"].mean()) if "Spending_Score" in df.columns else None,
        "avg_purchase_frequency": float(df["Purchase_Frequency"].mean()) if "Purchase_Frequency" in df.columns else None,
        "avg_order_value": float(df["Avg_Order_Value"].mean()) if "Avg_Order_Value" in df.columns else None,
        "segment_upgrade_rate": segment_stats,
        "model": {
            "name": metadata.get("model_name", "N/A"),
            "version": metadata.get("version", "N/A"),
            "accuracy": float(metadata.get("metrics", {}).get("accuracy", 0.0)),
            "pr_auc": float(metadata.get("metrics", {}).get("pr_auc", 0.0)),
        },
    }
    context_text = json.dumps(context, indent=2)

    run_summary = st.button("Generate AI Summary") or auto_summary
    if run_summary:
        if template_name == "Custom Q&A" and not user_query.strip():
            st.warning("Please enter a question for Custom Q&A.")
            return
        with st.spinner("Generating summary from Mistral..."):
            try:
                prompt_text = build_prompt(context_text, template_name, user_query)
                summary = generate_mistral_summary(prompt_text, model_name, api_key)
                st.session_state["last_ai_summary"] = summary
                st.session_state["last_ai_context"] = context
                st.markdown(summary)
            except Exception as exc:
                st.error(f"Failed to generate AI summary: {exc}")

    if st.session_state.get("last_ai_summary"):
        st.markdown('<div class="section-highlight" style="font-size:1rem;padding:6px 16px;">Export AI Report</div>', unsafe_allow_html=True)
        report_md = build_report_markdown(
            st.session_state["last_ai_summary"],
            st.session_state.get("last_ai_context", context),
            metadata,
        )
        report_df = build_report_dataframe(
            st.session_state["last_ai_summary"],
            st.session_state.get("last_ai_context", context),
            metadata,
        )
        csv_bytes = report_df.to_csv(index=False).encode("utf-8")

        dc1, dc2 = st.columns(2)
        dc1.download_button(
            "Download Summary Report (.md)",
            data=report_md,
            file_name="insightify_ai_summary_report.md",
            mime="text/markdown",
        )
        dc2.download_button(
            "Download Summary Report DF (.csv)",
            data=csv_bytes,
            file_name="insightify_ai_summary_report_df.csv",
            mime="text/csv",
        )

        with st.expander("Preview downloadable report dataframe"):
            st.dataframe(report_df, width="stretch", hide_index=True)


def main():
    inject_custom_style()
    st.title("Insightify Growth Center", anchor=False)
    st.caption("Targeting, offers, and ROI — all in one view.")
    hero_block()

    manual = st.sidebar.button("Refresh Now")
    auto_ai_summary = st.sidebar.checkbox("Auto AI Summary", value=False)

    df = load_data(DATA_PATH)
    try:
        model, model_cols, feature_medians, metadata = load_artifacts()
    except Exception as exc:
        st.error(f"Model artifacts could not be loaded: {exc}")
        st.stop()

    membership_filter = st.sidebar.multiselect(
        "Membership Level",
        sorted(df["Membership_Level"].dropna().unique().tolist()) if "Membership_Level" in df.columns else [],
        default=sorted(df["Membership_Level"].dropna().unique().tolist()) if "Membership_Level" in df.columns else [],
    )

    if membership_filter and "Membership_Level" in df.columns:
        df = df[df["Membership_Level"].isin(membership_filter)]

    scored_df = score_customers(df, model, model_cols, feature_medians)

    kpi_block(df)
    charts_block(df)
    model_info_block(metadata)
    prediction_block(df, model, model_cols, feature_medians, metadata)
    st.markdown('<div class="section-highlight" style="font-size:1.45rem;padding:10px 24px;">Business Growth Studio</div>', unsafe_allow_html=True)
    tier_strategy_block(scored_df)
    campaign_targeting_block(scored_df)
    offer_simulator_block(scored_df)
    next_best_action_block(scored_df)
    ai_summary_block(df, metadata, auto_summary=auto_ai_summary)

    st.markdown("---")
    st.caption(
        "Last refresh: "
        + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    if manual:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


if __name__ == "__main__":
    main()
