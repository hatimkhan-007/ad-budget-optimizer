"""Big data analytics: processing summary, audience breakdowns and data quality."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from analytics import summarize
from dashboard.context import Context
from dashboard.theme import compact, empty_state, page_header, style_figure
from data_quality import assess_csv, optimize_dtypes

BREAKDOWNS = {"Device": "device", "Age": "age", "Gender": "gender", "City": "city"}
METRICS = {"ROAS": "roas", "CTR (%)": "ctr_pct", "Conversion rate (%)": "conversion_rate_pct", "CPA": "cpa",
           "CPC": "cpc", "Spend": "spend", "Revenue": "revenue"}
SAMPLE_ROWS = 200_000


def _age_bands(frame: pd.DataFrame) -> pd.DataFrame:
    bins = [17, 24, 34, 44, 54, 64, 200]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    out = frame.copy()
    out["age"] = pd.cut(out["age"], bins=bins, labels=labels)
    return out


def render(ctx: Context) -> None:
    page_header("Big Data Analytics", "How the dataset was processed and how performance varies across audiences.")
    if not ctx.has_data:
        empty_state("No dataset selected", "Generate or select a dataset in **Dataset Management** first.")
        return
    agg = ctx.aggregates
    quality = agg.quality

    st.subheader("Processing summary")
    cards = st.columns(5)
    cards[0].metric("Records read", compact(quality.rows_in))
    cards[1].metric("Valid records", compact(quality.rows_valid), help="Records that passed validation and entered the analytics.")
    cards[2].metric("Rejected", compact(quality.rows_rejected))
    cards[3].metric("Chunks", f"{agg.chunks:,}", help="The file is processed in fixed-size chunks, so memory stays bounded.")
    cards[4].metric("Aggregation time", f"{agg.elapsed_seconds:,.1f}s",
                    delta="loaded from cache" if agg.from_cache else f"{agg.throughput_rows_per_second:,.0f} rows/s", delta_color="off")
    st.caption("The raw rows are reduced to small summary tables (campaign x day and audience breakdowns) in one pass. "
               "Every chart in this dashboard is built from those tables.")

    st.subheader("Audience breakdowns")
    c1, c2 = st.columns(2)
    dimension_name = c1.selectbox("Break down by", list(BREAKDOWNS), key="bd_dimension")
    metric_name = c2.selectbox("Metric", list(METRICS), key="bd_metric")
    dimension = BREAKDOWNS[dimension_name]
    frame = agg.frames[dimension]
    if dimension == "age":
        frame = _age_bands(frame)
    table = summarize(frame, dimension)
    metric = METRICS[metric_name]
    fig = px.bar(table, x=dimension, y=metric, labels={dimension: dimension_name, metric: metric_name},
                 color_discrete_sequence=["#4F46E5"])
    st.plotly_chart(style_figure(fig, f"{metric_name} by {dimension_name.lower()}"), width="stretch")
    st.dataframe(table, hide_index=True, width="stretch")

    st.subheader("Data quality")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Missing key fields", f"{quality.missing_key_rows:,}")
    q2.metric("Missing metrics", f"{quality.missing_numeric_rows:,}")
    q3.metric("Negative values", f"{quality.negative_rows:,}")
    q4.metric("Funnel violations", f"{quality.funnel_violation_rows:,}",
              help="Rows where clicks exceed impressions or conversions exceed clicks.")
    for issue in quality.issues:
        st.warning(issue)
    if not quality.issues:
        st.success("Every record passed validation.")

    left, right = st.columns(2)
    with left:
        if st.button("Run full quality check (includes global duplicate detection)", key="bd_quality"):
            with st.spinner("Scanning the dataset for duplicates..."):
                try:
                    report = assess_csv(ctx.dataset.path)
                    st.write(f"Exact duplicate rows: **{report.duplicate_rows:,}**")
                except (OSError, ValueError) as error:
                    st.error(f"Quality check failed: {error}")
    with right:
        if st.button(f"Measure memory savings from dtype optimisation ({SAMPLE_ROWS:,}-row sample)", key="bd_dtypes"):
            with st.spinner("Loading a sample..."):
                sample = pd.read_csv(ctx.dataset.path, nrows=SAMPLE_ROWS)
                before = sample.memory_usage(deep=True).sum() / 1e6
                after = optimize_dtypes(sample).memory_usage(deep=True).sum() / 1e6
            st.write(f"Default dtypes: **{before:,.1f} MB** -> optimised: **{after:,.1f} MB** "
                     f"({(1 - after / before) * 100:.0f}% smaller)")
