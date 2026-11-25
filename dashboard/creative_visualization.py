"""Simulated advertisement previews next to the performance of the campaign they represent."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from analytics import summarize
from creatives import build_creative, render_ad_preview_html
from dashboard.context import Context
from dashboard.theme import (RECOMMENDED_COLOR, empty_state, money, multiple, page_header, percent, style_figure)

NOTICE = ("Creatives on this page are **simulated illustrations** generated from each campaign's objective, "
          "audience and platform. They are not real advertisements and were never published.")


def _metric_with_delta(column, label: str, value: float, baseline: float, formatter, higher_is_better: bool = True) -> None:
    if baseline and baseline == baseline:
        delta = (value / baseline - 1) * 100
        column.metric(label, formatter(value), delta=f"{delta:+.1f}% vs platform", delta_color="normal" if higher_is_better else "inverse")
    else:
        column.metric(label, formatter(value))


def render(ctx: Context) -> None:
    page_header("Advertising Creative Visualization", "Preview each campaign's advertisement alongside its measured performance.")
    if not ctx.has_data or ctx.catalog.empty:
        empty_state("No campaigns available", "Generate or select a dataset in **Dataset Management** first.")
        return
    st.warning(NOTICE)

    catalog = ctx.catalog
    labels = {r.campaign_id: f"{r.campaign_id}  |  {r.campaign_name}" for r in catalog.itertuples()}
    chosen = st.selectbox("Campaign", list(labels), format_func=labels.get, key="creative_campaign")
    row = catalog[catalog["campaign_id"] == chosen].iloc[0]
    creative = build_creative(row["campaign_id"], row["platform"], row["objective"], row["target_audience"])

    preview, details = st.columns([1, 1.2], gap="large")
    with preview:
        st.html(render_ad_preview_html(creative))
    with details:
        st.subheader(creative.headline)
        st.write(creative.description)
        st.markdown(
            f"**Call to action:** {creative.cta}  \n**Platform:** {row['platform']}  \n"
            f"**Objective:** {row['objective']}  \n**Target audience:** {row['target_audience']}  \n"
            f"**Status:** {row['status']}  \n**Budget (simulated):** {money(row['budget'])}  \n"
            f"**Total spend:** {money(row['total_spend'])}")
        platform_totals = summarize(ctx.full_campaign_daily[ctx.full_campaign_daily["platform"] == row["platform"]]).iloc[0]
        m1 = st.columns(3)
        _metric_with_delta(m1[0], "ROAS", row["roas"], platform_totals["roas"], multiple)
        _metric_with_delta(m1[1], "CTR", row["ctr_pct"], platform_totals["ctr_pct"], percent)
        _metric_with_delta(m1[2], "Conversion rate", row["conversion_rate_pct"], platform_totals["conversion_rate_pct"], percent)
        m2 = st.columns(3)
        _metric_with_delta(m2[0], "CPC", row["cpc"], platform_totals["cpc"], lambda v: money(v, 2), higher_is_better=False)
        _metric_with_delta(m2[1], "CPA", row["cpa"], platform_totals["cpa"], lambda v: money(v, 2), higher_is_better=False)
        _metric_with_delta(m2[2], "CPM", row["cpm"], platform_totals["cpm"], lambda v: money(v, 2), higher_is_better=False)
        st.caption("Deltas compare this campaign with the average of all campaigns on the same platform.")

    peers = catalog[catalog["platform"] == row["platform"]].sort_values("roas", ascending=False)
    peers = peers.assign(highlight=peers["campaign_id"].eq(chosen).map({True: "Selected", False: "Other"}))
    fig = px.bar(peers, x="campaign_id", y="roas", color="highlight",
                 color_discrete_map={"Selected": RECOMMENDED_COLOR, "Other": "#D1D5DB"},
                 labels={"roas": "ROAS (x)", "campaign_id": ""})
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_figure(fig, f"ROAS of {row['platform']} campaigns", 320), width="stretch")

    st.subheader("Top creatives by ROAS")
    top = catalog.sort_values("roas", ascending=False).head(3)
    columns = st.columns(len(top))
    for column, (_, item) in zip(columns, top.iterrows()):
        with column:
            st.html(render_ad_preview_html(build_creative(item["campaign_id"], item["platform"], item["objective"],
                                                          item["target_audience"])))
            st.markdown(f"**{item['campaign_id']}**  \nROAS {multiple(item['roas'])} | CTR {percent(item['ctr_pct'])}")
