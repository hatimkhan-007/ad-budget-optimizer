"""Optional sentiment analytics for comment datasets (e.g. output of the YouTube sentiment project)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.context import Context
from dashboard.theme import SENTIMENT_COLORS, empty_state, page_header, percent, style_figure
from sentiment import (CANONICAL_COLUMNS, MIN_COMMENTS_PER_CAMPAIGN, SENTIMENT_LABELS, SentimentDataError,
                       campaign_sentiment_summary, link_to_campaigns, load_sentiment_csv, net_sentiment,
                       sentiment_distribution, sentiment_performance_association, sentiment_trend, validate_sentiment)

TEMPLATE_CSV = ",".join(CANONICAL_COLUMNS) + "\n"
FORMAT_HELP = """
Upload a CSV of labelled comments. Recognised columns (alternative names such as `label`, `comment`, `published_at`
or `like_count` are mapped automatically):

| Column | Required | Notes |
|---|---|---|
| `sentiment` | yes | Positive, Neutral or Negative (also `pos`/`neu`/`neg`) |
| `comment_text` | no | The comment itself |
| `video_id` | no | Needed to use a video-to-campaign mapping |
| `campaign_id` | no | Must equal a campaign id in the selected dataset to link performance |
| `timestamp` | no | Enables sentiment trends |
| `likes` | no | Used for like-weighted summaries |
"""


def _read_upload(uploaded, label: str) -> pd.DataFrame | None:
    try:
        return load_sentiment_csv(uploaded)
    except SentimentDataError as error:
        st.error(f"{label}: {error}")
        return None


def _distribution_section(frame: pd.DataFrame, has_timestamps: bool) -> None:
    counts = sentiment_distribution(frame)
    cards = st.columns(5)
    cards[0].metric("Comments", f"{len(frame):,}")
    for column, label in zip(cards[1:4], SENTIMENT_LABELS):
        share = counts.loc[counts["sentiment"] == label, "share_pct"].iloc[0]
        column.metric(label, percent(share, 1))
    cards[4].metric("Net sentiment", f"{net_sentiment(frame):+.2f}", help="(positive - negative) / total, from -1 to 1.")
    left, right = st.columns(2)
    fig = px.pie(counts, names="sentiment", values="comments", hole=0.55, color="sentiment", color_discrete_map=SENTIMENT_COLORS)
    left.plotly_chart(style_figure(fig, "Sentiment distribution", 340), width="stretch")
    if has_timestamps:
        freq = right.radio("Trend grain", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key="sent_grain")
        trend = sentiment_trend(frame, {"Daily": "D", "Weekly": "W", "Monthly": "M"}[freq])
        fig = px.area(trend, x="period", y="comments", color="sentiment", color_discrete_map=SENTIMENT_COLORS,
                      labels={"period": "", "comments": "Comments"})
        right.plotly_chart(style_figure(fig, "Sentiment over time", 300), width="stretch")
    else:
        right.info("No valid timestamps in this file, so trends over time are not available.")


def _linking_section(frame: pd.DataFrame, ctx: Context, mapping: pd.DataFrame | None) -> None:
    st.subheader("Connection to campaign performance")
    if not ctx.has_data:
        st.info("Select a dataset to check whether comments can be linked to campaigns.")
        return
    try:
        linked, outcome = link_to_campaigns(frame, ctx.full_catalog["campaign_id"], mapping)
    except SentimentDataError as error:
        st.error(f"The video-to-campaign mapping is invalid: {error}")
        return
    if outcome.status != "linked":
        st.info(outcome.message)
        return
    st.success(outcome.message)
    summary = campaign_sentiment_summary(linked, ctx.full_catalog)
    st.dataframe(summary, hide_index=True, width="stretch")
    plot = summary[summary["sufficient_sample"]].dropna(subset=["roas"])
    if plot.empty:
        st.info(f"No campaign has {MIN_COMMENTS_PER_CAMPAIGN}+ linked comments yet, so no comparison chart is drawn.")
    else:
        fig = px.scatter(plot, x="net_sentiment", y="roas", size="comments", hover_name="campaign_id",
                         labels={"net_sentiment": "Net sentiment", "roas": "ROAS (x)"})
        st.plotly_chart(style_figure(fig, "Net sentiment vs ROAS (campaigns with enough comments)"), width="stretch")
        association = sentiment_performance_association(summary)
        if association["available"]:
            st.caption(f"Spearman rank correlation across {association['campaigns']} campaigns: {association['spearman']:+.2f}.")
        else:
            st.caption(association["reason"])
    st.warning("These views show association only. They do not show that sentiment causes better or worse campaign "
               "performance, and linked comments are not necessarily representative of a campaign's audience.")


def render(ctx: Context) -> None:
    page_header("Sentiment Analytics", "Optional: analyse labelled comments and connect them to campaigns when a valid link exists.")
    uploaded = st.file_uploader("Sentiment CSV", type="csv", key="sentiment_file")
    mapping_file = st.file_uploader("Optional video-to-campaign mapping CSV (video_id, campaign_id)", type="csv", key="sentiment_mapping")
    with st.expander("Expected file format"):
        st.markdown(FORMAT_HELP)
        st.download_button("Download CSV template", TEMPLATE_CSV, "sentiment_template.csv", "text/csv")
    if uploaded is None:
        empty_state("No sentiment data loaded",
                    "This module is optional and independent of the advertising dataset. Upload the output of the "
                    "YouTube Comments Sentiment Analysis project (or any labelled comment CSV) to begin.")
        return
    raw = _read_upload(uploaded, "Sentiment file")
    if raw is None:
        return
    try:
        data = validate_sentiment(raw)
    except SentimentDataError as error:
        st.error(f"The sentiment file is not usable: {error}")
        return
    report = data.report
    with st.expander(f"Validation report: {report.rows_valid:,} of {report.rows_in:,} rows kept", expanded=bool(report.warnings)):
        st.write(f"Columns found: {', '.join(report.columns_found) or 'none'}")
        if report.columns_missing:
            st.write(f"Columns not provided: {', '.join(report.columns_missing)}")
        for warning in report.warnings:
            st.warning(warning)
        if report.dropped_duplicates:
            st.write(f"Duplicate comments removed: {report.dropped_duplicates:,}")
    mapping = None
    if mapping_file is not None:
        mapping = _read_upload(mapping_file, "Mapping file")
    _distribution_section(data.frame, data.has_timestamps)
    _linking_section(data.frame, ctx, mapping)
