"""Dataset management: list, generate and rebuild analytics for local datasets."""

from __future__ import annotations

import shutil

import pandas as pd
import streamlit as st

from dashboard import data as store
from dashboard.theme import page_header
from data_generator import DEFAULT_SEED, SCALES, ensure_dataset

BYTES_PER_RECORD = 105
LARGE_THRESHOLD = 5_000_000


def render(ctx=None) -> None:
    page_header("Dataset Management", "Generate reproducible simulated datasets locally and manage their analytics cache.")
    st.caption(f"Datasets are stored in `{store.DATA_DIR}/` and are never committed to Git. Generation is deterministic for a given seed.")
    datasets = store.list_datasets()
    if datasets:
        table = pd.DataFrame([{
            "Records": d.records, "Size (MB)": round(d.size_mb, 1), "Modified": d.modified,
            "Analytics cached": (store.CACHE_DIR / d.path.stem / "meta.json").exists(), "File": str(d.path)} for d in datasets])
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"Records": st.column_config.NumberColumn(format="%d")})
    else:
        st.info("No datasets found yet. Generate one below.")

    st.subheader("Generate a dataset")
    with st.form("generate_form"):
        c1, c2 = st.columns(2)
        records = c1.selectbox("Records", SCALES, index=1, format_func=lambda n: f"{n:,}")
        seed = c2.number_input("Random seed", min_value=0, value=DEFAULT_SEED, step=1)
        replace = st.checkbox("Regenerate if it already exists")
        submitted = st.form_submit_button("Generate")
    estimate = records * BYTES_PER_RECORD / 1e6
    st.caption(f"Estimated file size: about {estimate:,.0f} MB.")
    if records >= LARGE_THRESHOLD:
        st.warning("This is a large dataset. Generation may take several minutes; the first analysis pass also takes time "
                   "(it is cached afterwards). Pandas benchmarking at this size needs several GB of memory.")
    if submitted:
        with st.spinner(f"Generating {records:,} records..."):
            try:
                ensure_dataset(records, store.DATA_DIR, int(seed), regenerate=replace)
            except OSError as error:
                st.error(f"Could not write the dataset (check disk space and permissions): {error}")
                return
        st.success(f"Dataset with {records:,} records is ready.")
        st.rerun()

    if datasets:
        st.subheader("Analytics cache")
        st.caption("Aggregated summaries are cached so reopening a large dataset is instant. Rebuilding only deletes derived files.")
        choice = st.selectbox("Dataset", datasets, format_func=lambda d: d.label, key="cache_dataset")
        if st.button("Rebuild analytics cache for this dataset"):
            shutil.rmtree(store.CACHE_DIR / choice.path.stem, ignore_errors=True)
            st.cache_data.clear()
            st.success("Cache cleared; it will be rebuilt the next time the dataset is opened.")
