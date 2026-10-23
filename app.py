"""Streamlit dashboard for the Ad Budget Optimizer."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ad_optimizer.meta_ads import MetaAdsAPIError, fetch_campaign_insights
from ad_optimizer.metrics import normalize_campaign_rows
from ad_optimizer.optimizer import recommend_budget

st.set_page_config(page_title="Ad Budget Optimizer", page_icon="📊", layout="wide")
st.title("Ad Budget Optimizer")
st.caption("Live Meta campaign insights · advisory budget recommendations")

with st.sidebar:
    st.header("Settings")
    date_preset = st.selectbox("Reporting period", ["last_7d", "last_14d", "last_30d", "last_90d"], index=2)
    total_budget = st.number_input("Total budget", min_value=1.0, value=1000.0, step=100.0)
    min_budget = st.number_input("Minimum per campaign", min_value=0.0, value=0.0, step=10.0)
    refresh = st.button("Fetch latest data", type="primary")

if "campaign_rows" not in st.session_state:
    st.session_state.campaign_rows = None
if refresh or st.session_state.campaign_rows is None:
    try:
        with st.spinner("Fetching campaign insights from Meta..."):
            st.session_state.campaign_rows = fetch_campaign_insights(date_preset)
        st.session_state.last_period = date_preset
    except MetaAdsAPIError as exc:
        st.error(str(exc))
        st.info("Configure META_ACCESS_TOKEN and META_AD_ACCOUNT_ID in your local .env file, then restart the app.")
        st.stop()

# Refetch when the selected period changes.
if st.session_state.get("last_period") != date_preset:
    try:
        st.session_state.campaign_rows = fetch_campaign_insights(date_preset)
        st.session_state.last_period = date_preset
    except MetaAdsAPIError as exc:
        st.error(str(exc))
        st.stop()

campaigns = normalize_campaign_rows(st.session_state.campaign_rows)
spend_total = float(campaigns["spend"].sum())
clicks_total = int(campaigns["clicks"].sum())
conversions_total = float(campaigns["conversions"].sum())
cpa_total = spend_total / conversions_total if conversions_total else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Campaigns", len(campaigns))
c2.metric("Spend", f"{spend_total:,.2f}")
c3.metric("Clicks", f"{clicks_total:,}")
c4.metric("Reported conversions", f"{conversions_total:,.0f}")
st.caption(f"Reporting period: {date_preset.replace('_', ' ')}. Currency follows your Meta ad account.")

st.subheader("Campaign performance")
display_columns = [
    col for col in ["campaign_name", "date_start", "date_stop", "impressions", "clicks",
                    "spend", "conversions", "ctr", "cpc", "cpa"] if col in campaigns.columns
]
st.dataframe(campaigns[display_columns], use_container_width=True, hide_index=True)

st.subheader("Recommended budget allocation")
try:
    recommendations = recommend_budget(campaigns, total_budget, min_budget)
    st.plotly_chart(
        px.bar(
            recommendations,
            x="campaign_name",
            y="recommended_budget",
            title="Suggested allocation by campaign",
            labels={"campaign_name": "Campaign", "recommended_budget": "Recommended budget"},
        ),
        use_container_width=True,
    )
    st.dataframe(
        recommendations[[
            col for col in ["campaign_name", "spend", "conversions", "cpa",
                            "efficiency_score", "recommended_budget", "recommended_share_pct"]
            if col in recommendations.columns
        ]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download recommendation CSV",
        recommendations.to_csv(index=False).encode("utf-8"),
        file_name="budget_recommendations.csv",
        mime="text/csv",
    )
except ValueError as exc:
    st.warning(str(exc))

st.warning(
    "Recommendations are estimates based on historical reported performance. "
    "They do not guarantee improved results and do not change live campaign budgets."
)
