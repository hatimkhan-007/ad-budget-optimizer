"""Digital Advertising Budget Optimizer - Streamlit dashboard entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from dashboard import (about, benchmarking, bigdata, budget_optimization, campaign_management,
                       creative_visualization, datasets, overview, sentiment_page)
from dashboard.context import Context, build_context
from dashboard.data import default_dataset_index, list_datasets, load_aggregates
from dashboard.theme import inject_css

SECTIONS = {
    "Executive Overview": overview,
    "Campaign Management": campaign_management,
    "Advertising Creatives": creative_visualization,
    "Big Data Analytics": bigdata,
    "Framework Benchmarking": benchmarking,
    "Budget Optimization": budget_optimization,
    "Sentiment Analytics": sentiment_page,
    "Dataset Management": datasets,
    "About / Documentation": about,
}
NEEDS_DATASET = {"Executive Overview", "Campaign Management", "Advertising Creatives", "Big Data Analytics",
                 "Budget Optimization", "Sentiment Analytics"}
USES_PLATFORM_FILTER = {"Executive Overview", "Campaign Management", "Advertising Creatives"}


def sidebar() -> tuple[str, Context]:
    st.sidebar.markdown("## Ad Budget Optimizer")
    section = st.sidebar.radio("Navigate", list(SECTIONS), key="nav", label_visibility="collapsed")
    if section not in NEEDS_DATASET:
        return section, Context()

    st.sidebar.divider()
    available = list_datasets()
    if not available:
        st.sidebar.info("No dataset yet. Open **Dataset Management** to generate one.")
        return section, Context()
    dataset = st.sidebar.selectbox("Dataset", available, index=default_dataset_index(available),
                                   format_func=lambda d: d.label, key="dataset")
    try:
        aggregates = load_aggregates(dataset)
    except (OSError, ValueError) as error:
        st.sidebar.error(f"This dataset cannot be analysed: {error}")
        return section, Context()
    platforms = sorted(aggregates.frames["campaign_daily"]["platform"].unique())
    chosen = None
    if section in USES_PLATFORM_FILTER:
        chosen = st.sidebar.multiselect("Platforms", platforms, default=platforms, key="platform_filter")
        if not chosen:
            st.sidebar.warning("Select at least one platform.")
            chosen = platforms
    st.sidebar.caption("Simulated data. Not real campaign results.")
    return section, build_context(dataset, aggregates, chosen)


def main() -> None:
    st.set_page_config(page_title="Digital Advertising Budget Optimizer", layout="wide")
    inject_css()
    section, context = sidebar()
    SECTIONS[section].render(context)


main()
