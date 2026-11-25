"""Executive overview: headline KPIs, trends and platform mix."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from analytics import summarize
from dashboard.context import Context
from dashboard.theme import (SIMULATION_NOTICE, compact, empty_state, multiple, money, page_header, percent,
                             platform_color_map, style_figure)


def render(ctx: Context) -> None:
    page_header("Executive Overview", "Headline performance across all selected platforms and campaigns.")
    if not ctx.has_data or ctx.campaign_daily.empty:
        empty_state("No data to show", "Generate or select a dataset in **Dataset Management** first.")
        return
    st.caption(SIMULATION_NOTICE)

    totals = summarize(ctx.campaign_daily).iloc[0]
    row1 = st.columns(4)
    row1[0].metric("Ad spend", compact(totals["spend"]), help="Total spend in the selected period and platforms.")
    row1[1].metric("Revenue", compact(totals["revenue"]), help="Revenue attributed to conversions.")
    row1[2].metric("ROAS", multiple(totals["roas"]), help="Return on ad spend = revenue / spend.")
    row1[3].metric("Conversions", compact(totals["conversions"]))
    row2 = st.columns(4)
    row2[0].metric("Impressions", compact(totals["impressions"]))
    row2[1].metric("CTR", percent(totals["ctr_pct"]), help="Click-through rate = clicks / impressions.")
    row2[2].metric("CPA", money(totals["cpa"], 2), help="Cost per acquisition = spend / conversions.")
    row2[3].metric("Conversion rate", percent(totals["conversion_rate_pct"]), help="Conversions / clicks.")

    st.subheader("Spend and revenue over time")
    grain = st.radio("Time grain", ["Daily", "Weekly", "Monthly"], horizontal=True, index=1, key="overview_grain")
    rule = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[grain]
    trend = (ctx.campaign_daily.set_index("date")[["spend", "revenue"]].resample(rule).sum().reset_index())
    fig = px.line(trend, x="date", y=["spend", "revenue"], labels={"value": "Amount", "date": "", "variable": ""})
    st.plotly_chart(style_figure(fig), width="stretch")

    left, right = st.columns(2)
    platforms = summarize(ctx.campaign_daily, "platform")
    colors = platform_color_map(platforms["platform"])
    with left:
        fig = px.bar(platforms.sort_values("roas", ascending=False), x="platform", y="roas", color="platform",
                     color_discrete_map=colors, labels={"roas": "ROAS (x)", "platform": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_figure(fig, "ROAS by platform", 340), width="stretch")
    with right:
        fig = px.pie(platforms, names="platform", values="spend", hole=0.55, color="platform", color_discrete_map=colors)
        st.plotly_chart(style_figure(fig, "Spend distribution by platform", 340), width="stretch")

    left, right = st.columns(2)
    with left:
        fig = px.bar(platforms.sort_values("ctr_pct", ascending=False), x="platform", y="ctr_pct", color="platform",
                     color_discrete_map=colors, labels={"ctr_pct": "CTR (%)", "platform": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_figure(fig, "CTR by platform", 340), width="stretch")
    with right:
        comparison = platforms.melt(id_vars="platform", value_vars=["spend", "revenue"], var_name="measure", value_name="amount")
        fig = px.bar(comparison, x="platform", y="amount", color="measure", barmode="group",
                     labels={"amount": "Amount", "platform": "", "measure": ""})
        st.plotly_chart(style_figure(fig, "Spend vs revenue by platform", 340), width="stretch")

    st.subheader("Best and weakest campaigns by ROAS")
    shown = ["campaign_id", "platform", "objective", "target_audience", "total_spend", "roas", "ctr_pct"]
    names = {"campaign_id": "Campaign", "platform": "Platform", "objective": "Objective", "target_audience": "Audience",
             "total_spend": "Spend", "roas": "ROAS", "ctr_pct": "CTR %"}
    config = {"Spend": st.column_config.NumberColumn(format="%.0f"), "ROAS": st.column_config.NumberColumn(format="%.2f"),
              "CTR %": st.column_config.NumberColumn(format="%.2f")}
    top, bottom = st.columns(2)
    ranked = ctx.catalog.sort_values("roas", ascending=False)
    top.markdown("**Top 5**")
    top.dataframe(ranked.head(5)[shown].rename(columns=names), hide_index=True, width="stretch", column_config=config)
    bottom.markdown("**Bottom 5**")
    bottom.dataframe(ranked.tail(5)[shown].rename(columns=names), hide_index=True, width="stretch", column_config=config)
