"""Framework benchmarking: methodology, measured results and a runner for new measurements."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from benchmark_analysis import ENGINES, engine_available, run_benchmarks
from dashboard import data as store
from dashboard.theme import empty_state, page_header, style_figure

RESULTS_FILE = "benchmark_results.csv"
ENGINE_LABELS = {"pandas": "Pandas", "mapreduce": "MapReduce (simulated)", "pyspark": "PySpark"}
METHODOLOGY = """
**What is measured.** Each engine reads the CSV and aggregates impressions, clicks, spend, conversions and revenue by
platform, campaign and audience. All engines must return identical results; the *results match* column verifies that.

**Isolation.** Every trial runs in a fresh worker process, so one engine's memory or caches cannot influence another.

**Cold vs warm.** The first trial of a configuration is reported separately (*first trial*, which includes cold operating
system file caches). The *warm* time is the median of the remaining trials. *Time* is the median across all trials.

**Spark start-up.** JVM and session start-up is measured separately and excluded from Spark's processing time; it is
reported in *start-up* because it is a real cost for short jobs.

**Data loading.** Pandas and MapReduce time includes CSV parsing, which usually dominates. The *phases* column splits
load vs aggregate (Pandas) and map vs reduce (MapReduce); Spark is lazy so it is reported as one phase.

**Memory.** Peak proportional memory (PSS where available, otherwise RSS) summed over the worker process tree, sampled
during the run. Short runs can be under-sampled, and JVM memory is included for Spark.

**Throughput.** Records divided by the median processing time.

**Interpretation.** Results depend on CPU cores, disk speed and memory. No framework is faster in every situation:
fixed start-up costs favour single-process tools on small data, while parallel or distributed engines pay off only when
the data and available cores are large enough.
"""


def _load_results() -> pd.DataFrame | None:
    path = store.RESULTS_DIR / RESULTS_FILE
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def _observations(ok: pd.DataFrame) -> list[str]:
    notes = []
    for records, group in ok.groupby("records"):
        best = group.loc[group["seconds"].idxmin()]
        slowest = group.loc[group["seconds"].idxmax()]
        if len(group) > 1:
            notes.append(f"At {int(records):,} records the fastest engine was **{ENGINE_LABELS.get(best['engine'], best['engine'])}** "
                         f"({best['seconds']:.2f}s); the slowest was {ENGINE_LABELS.get(slowest['engine'], slowest['engine'])} "
                         f"({slowest['seconds']:.2f}s).")
    winners = ok.loc[ok.groupby("records")["seconds"].idxmin(), "engine"].nunique()
    if ok["records"].nunique() > 1:
        notes.append("The fastest engine **changes** with data size in this run." if winners > 1 else
                     "The same engine was fastest at every size measured here; this can change with more cores or larger data.")
    return notes


def _render_results(results: pd.DataFrame) -> None:
    ok = results[results["status"] == "ok"].copy()
    ok["engine_label"] = ok["engine"].map(ENGINE_LABELS).fillna(ok["engine"])
    if ok.empty:
        empty_state("No successful runs", "Every configuration in this file failed or was skipped; see the table below.")
    else:
        left, right = st.columns(2)
        with left:
            fig = px.line(ok, x="records", y="seconds", color="engine_label", markers=True, log_x=True, log_y=True,
                          labels={"records": "Records", "seconds": "Processing time (s)", "engine_label": ""})
            st.plotly_chart(style_figure(fig, "Processing time vs data size (log-log)"), width="stretch")
        with right:
            if "throughput_rps" in ok:
                fig = px.line(ok, x="records", y="throughput_rps", color="engine_label", markers=True, log_x=True,
                              labels={"records": "Records", "throughput_rps": "Records per second", "engine_label": ""})
                st.plotly_chart(style_figure(fig, "Processing throughput"), width="stretch")
        left, right = st.columns(2)
        with left:
            fig = px.bar(ok, x="records", y="peak_mem_mb", color="engine_label", barmode="group", log_x=False,
                         labels={"records": "Records", "peak_mem_mb": "Peak memory (MB)", "engine_label": ""})
            fig.update_xaxes(type="category")
            st.plotly_chart(style_figure(fig, "Peak memory"), width="stretch")
        with right:
            if "warm_seconds" in ok and ok["warm_seconds"].notna().any():
                long = ok.melt(id_vars=["records", "engine_label"], value_vars=["first_trial_seconds", "warm_seconds"],
                               var_name="run", value_name="duration").dropna()
                long["run"] = long["run"].map({"first_trial_seconds": "First trial (cold)", "warm_seconds": "Warm (median of rest)"})
                fig = px.bar(long, x="engine_label", y="duration", color="run", barmode="group", facet_col="records",
                             labels={"engine_label": "", "duration": "Seconds"})
                fig.update_xaxes(matches=None, showticklabels=True)
                st.plotly_chart(style_figure(fig, "Cold vs warm execution"), width="stretch")
            else:
                st.info("Cold vs warm comparison needs a run with at least two trials.")
        for note in _observations(ok):
            st.markdown(f"- {note}")

    shown = [c for c in ["records", "engine", "status", "seconds", "first_trial_seconds", "warm_seconds", "startup_seconds",
                         "peak_mem_mb", "throughput_rps", "vs_pandas", "consistent", "trials", "time_std_seconds", "phases", "detail"]
             if c in results.columns]
    st.dataframe(results[shown], hide_index=True, width="stretch")
    if "phases" in results:
        phases = results[results["phases"].notna()][["records", "engine", "phases"]]
        phases = phases[phases["phases"].map(lambda v: isinstance(v, str) and v not in ("{}", ""))]
        if not phases.empty:
            with st.expander("Phase breakdown (median seconds)"):
                expanded = phases.assign(**{"phase timings": phases["phases"].map(lambda v: ", ".join(f"{k}={x}" for k, x in json.loads(v).items()))})
                st.dataframe(expanded[["records", "engine", "phase timings"]], hide_index=True, width="stretch")


def _runner() -> None:
    st.subheader("Run a new benchmark")
    datasets = store.list_datasets()
    if not datasets:
        empty_state("No datasets", "Generate a dataset in **Dataset Management** before benchmarking.")
        return
    options = {d.records: d for d in datasets}
    with st.form("benchmark_form"):
        chosen = st.multiselect("Dataset sizes", list(options), default=[r for r in options if r <= 100_000][-2:],
                                format_func=lambda r: options[r].label)
        engines = st.multiselect("Frameworks", list(ENGINES), default=["pandas", "mapreduce"],
                                 format_func=lambda e: ENGINE_LABELS[e])
        trials = st.slider("Trials per configuration", 1, 5, 2, help="The first trial is reported as the cold run.")
        submitted = st.form_submit_button("Run benchmark")
    for engine in engines:
        available, reason = engine_available(engine)
        if not available:
            st.warning(f"{ENGINE_LABELS[engine]} is not available here and will be skipped: {reason}")
    if any(r >= 5_000_000 for r in chosen):
        st.warning("Large datasets can take many minutes and several GB of memory, especially with Pandas.")
    if submitted:
        if not chosen or not engines:
            st.error("Select at least one dataset size and one framework.")
            return
        with st.spinner("Running benchmark (each trial runs in a separate process)..."):
            try:
                results = run_benchmarks({r: options[r].path for r in chosen}, tuple(engines), trials)
            except (ValueError, OSError) as error:
                st.error(f"Benchmark failed: {error}")
                return
        store.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(store.RESULTS_DIR / RESULTS_FILE, index=False)
        st.success("Benchmark finished and saved; results are shown above after refresh.")
        st.rerun()


def render(ctx=None) -> None:
    page_header("Framework Benchmarking", "Measured execution time, memory and throughput of Pandas, MapReduce and PySpark.")
    with st.expander("Benchmark methodology", expanded=False):
        st.markdown(METHODOLOGY)
    results = _load_results()
    uploaded = st.file_uploader("Or inspect another results file (CSV from this tool)", type="csv", key="bench_upload")
    if uploaded is not None:
        try:
            results = pd.read_csv(uploaded)
            if not {"records", "engine", "status", "seconds"} <= set(results.columns):
                st.error("That file is not a benchmark results file (missing records, engine, status or seconds columns).")
                results = _load_results()
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
            st.error(f"Could not read the file: {error}")
            results = _load_results()
    if results is None or results.empty:
        empty_state("No benchmark results yet", "Run a benchmark below, or `python main.py` to produce results/benchmark_results.csv.")
    else:
        st.caption("Measured values only; nothing on this page is estimated or hard-coded.")
        _render_results(results)
    _runner()
