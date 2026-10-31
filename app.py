from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Expected project structure:
# app.py
# results/
#   kpis_by_platform.csv
#   budget_recommendation.csv
#   benchmark_results.csv

st.set_page_config(
    page_title="Ad Budget Optimizer",
    page_icon="📊",
    layout="wide",
)

RESULTS_DIR = Path("results")


@st.cache_data
def load_csv(path: Path, modified: float = 0.0):
    if path.exists():
        try:
            return pd.read_csv(path)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError):
            return pd.DataFrame()
    return pd.DataFrame()


def money(value):
    if pd.isna(value):
        return "—"
    return f"{value:,.2f}"


def number(value):
    if pd.isna(value):
        return "—"
    return f"{value:,.0f}"


def pct(value):
    if pd.isna(value):
        return "—"
    return f"{value:.2f}%"


def show_missing(filename):
    st.warning(
        f"Couldn't find usable data for `{filename}`. "
        f"Place the CSV inside `{RESULTS_DIR.as_posix()}/` and refresh the page."
    )


st.title("📊 Digital Advertising Budget Optimizer")
st.caption(
    "Campaign performance, budget recommendations, and processing-engine benchmarks. "
    "Recommendations are model-based estimates, not guaranteed future results."
)

with st.sidebar:
    st.header("Dashboard settings")
    st.write("Expected CSV files are loaded from the project's `results` folder.")
    st.caption("You can also upload a CSV below to preview it in the relevant section.")
    kpi_upload = st.file_uploader("Upload KPI CSV", type="csv", key="kpi_upload")
    budget_upload = st.file_uploader("Upload budget recommendation CSV", type="csv", key="budget_upload")
    benchmark_upload = st.file_uploader("Upload benchmark CSV", type="csv", key="benchmark_upload")

    refresh_dashboard = st.button(
        "Update dashboard",
        type="primary",
        width="stretch",
        help="Apply the uploaded CSV files and refresh all dashboard sections.",
    )
    if refresh_dashboard:
        st.cache_data.clear()
        st.session_state["dashboard_updated"] = True
        st.rerun()

if st.session_state.get("dashboard_updated", False):
    st.success("Dashboard refreshed. The displayed metrics are based on the currently uploaded or saved CSV files.")


def load_with_upload(upload, filename):
    if upload is not None:
        try:
            return pd.read_csv(upload)
        except (pd.errors.ParserError, UnicodeDecodeError, OSError):
            st.error(f"Could not read {filename}. Check that it is a valid CSV.")
            return pd.DataFrame()
    path = RESULTS_DIR / filename
    return load_csv(path, path.stat().st_mtime if path.exists() else 0.0)


kpis = load_with_upload(kpi_upload, "kpis_by_platform.csv")
budget = load_with_upload(budget_upload, "budget_recommendation.csv")
benchmark = load_with_upload(benchmark_upload, "benchmark_results.csv")
summary = load_with_upload(None, "budget_summary.csv")

tab_overview, tab_platforms, tab_budget, tab_benchmark, tab_data = st.tabs(
    ["Overview", "Platform performance", "Budget optimizer", "Benchmarks", "Data tables"]
)

with tab_overview:
    st.subheader("Performance overview")

    if kpis.empty:
        show_missing("kpis_by_platform.csv")
    else:
        required = {"platform", "spend", "revenue", "clicks", "conversions", "impressions"}
        missing = required - set(kpis.columns)
        if missing:
            st.error(f"KPI CSV is missing required columns: {', '.join(sorted(missing))}")
        else:
            for col in ["spend", "revenue", "clicks", "conversions", "impressions"]:
                kpis[col] = pd.to_numeric(kpis[col], errors="coerce")

            total_spend = kpis["spend"].sum()
            total_revenue = kpis["revenue"].sum()
            total_clicks = kpis["clicks"].sum()
            total_conversions = kpis["conversions"].sum()
            total_impressions = kpis["impressions"].sum()
            overall_roas = total_revenue / total_spend if total_spend else float("nan")
            overall_ctr = total_clicks / total_impressions * 100 if total_impressions else float("nan")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total ad spend", money(total_spend))
            c2.metric("Total revenue", money(total_revenue))
            c3.metric("Overall ROAS", f"{overall_roas:.2f}×" if pd.notna(overall_roas) else "—")
            c4.metric("Conversions", number(total_conversions))

            c5, c6, c7 = st.columns(3)
            c5.metric("Impressions", number(total_impressions))
            c6.metric("Clicks", number(total_clicks))
            c7.metric("Overall CTR", pct(overall_ctr))

            st.info(
                "Currency is not inferred from the CSV. The spend and revenue values are displayed "
                "as provided by your data."
            )

            if "roas" not in kpis.columns:
                kpis["roas"] = kpis["revenue"] / kpis["spend"].replace(0, pd.NA)

            fig = px.bar(
                kpis.sort_values("roas", ascending=False),
                x="platform",
                y="roas",
                color="platform",
                title="ROAS by advertising platform",
                labels={"platform": "Platform", "roas": "ROAS (×)"},
            )
            st.plotly_chart(fig, width="stretch")

            best = kpis.loc[kpis["roas"].idxmax()] if kpis["roas"].notna().any() else None
            if best is not None:
                st.success(f"Highest reported ROAS: {best['platform']} ({best['roas']:.2f}×).")

with tab_platforms:
    st.subheader("Platform performance comparison")

    if kpis.empty:
        show_missing("kpis_by_platform.csv")
    else:
        numeric_cols = [
            c for c in ["impressions", "clicks", "spend", "conversions", "revenue",
                        "ctr_pct", "cpc", "cpa", "conversion_rate_pct", "roas", "cpm"]
            if c in kpis.columns
        ]
        selected_metric = st.selectbox(
            "Metric to compare",
            options=numeric_cols,
            index=numeric_cols.index("roas") if "roas" in numeric_cols else 0,
        )
        plot_df = kpis.copy()
        plot_df[selected_metric] = pd.to_numeric(plot_df[selected_metric], errors="coerce")
        fig = px.bar(
            plot_df.sort_values(selected_metric, ascending=False),
            x="platform",
            y=selected_metric,
            color="platform",
            title=f"{selected_metric.replace('_', ' ').upper()} by platform",
            labels={selected_metric: selected_metric.replace("_", " ").title(), "platform": "Platform"},
        )
        st.plotly_chart(fig, width="stretch")

        display_cols = ["platform"] + [c for c in [
            "ctr_pct", "cpc", "cpa", "conversion_rate_pct", "roas", "cpm"
        ] if c in kpis.columns]
        st.dataframe(kpis[display_cols], width="stretch", hide_index=True)

with tab_budget:
    st.subheader("Recommended budget allocation")

    if budget.empty:
        show_missing("budget_recommendation.csv")
    else:
        needed = {"platform", "current_share_pct", "recommended_share_pct"}
        missing = needed - set(budget.columns)
        if missing:
            st.error(f"Budget CSV is missing required columns: {', '.join(sorted(missing))}")
        else:
            budget = budget.copy()
            for col in ["current_share_pct", "recommended_share_pct", "current_roas",
                        "projected_roas", "marginal_roas", "fit_r2",
                        "current_daily_spend", "recommended_daily_spend",
                        "recommended_campaign_amount"]:
                if col in budget.columns:
                    budget[col] = pd.to_numeric(budget[col], errors="coerce")

            share_sum = budget["recommended_share_pct"].sum()
            st.caption(f"Sum of recommended shares in source file: {share_sum:.2f}%")

            if not summary.empty:
                row = summary.iloc[0]
                m1, m2, m3 = st.columns(3)
                if "uplift_vs_compliant_baseline_pct" in summary.columns:
                    m1.metric("Uplift vs limit-compliant baseline", pct(row["uplift_vs_compliant_baseline_pct"]))
                if "uplift_vs_current_pct" in summary.columns:
                    m2.metric("Uplift vs current allocation", pct(row["uplift_vs_current_pct"]))
                if {"current_modeled_roas", "recommended_modeled_roas"} <= set(summary.columns):
                    m3.metric("Modeled ROAS", f"{row['current_modeled_roas']:.2f}× → {row['recommended_modeled_roas']:.2f}×")
                if "current_within_limits" in summary.columns and not bool(row["current_within_limits"]):
                    st.info(
                        "The current allocation breaks the configured share limits, so uplift versus current "
                        "also reflects the cost of complying with those limits. The limit-compliant baseline "
                        "is the fairer comparison."
                    )

            daily_budget_default = (
                float(budget["current_daily_spend"].sum())
                if "current_daily_spend" in budget.columns
                else 10000.0
            )
            daily_budget_default = max(0.0, daily_budget_default)
            daily_budget = st.number_input(
                "Scenario daily budget (same currency units as your data)",
                min_value=0.0,
                value=daily_budget_default,
                step=1000.0,
            )
            st.caption(
                "This control rescales the existing recommended percentages for a scenario. "
                "It does not retrain the response curves or rerun the optimization algorithm."
            )

            budget["scenario_daily_spend"] = (
                daily_budget * budget["recommended_share_pct"] / 100
            )

            fig = px.bar(
                budget,
                x="platform",
                y=["current_share_pct", "recommended_share_pct"],
                barmode="group",
                title="Current vs recommended allocation",
                labels={"value": "Budget share (%)", "platform": "Platform", "variable": "Allocation"},
            )
            st.plotly_chart(fig, width="stretch")

            budget_display = budget[[
                c for c in ["platform", "current_share_pct", "recommended_share_pct",
                            "recommended_campaign_amount", "scenario_daily_spend", "current_roas",
                            "projected_roas", "marginal_roas", "elasticity", "fit_r2", "n_observations"]
                if c in budget.columns
            ]].copy()
            if "scenario_daily_spend" in budget_display.columns:
                budget_display["scenario_daily_spend"] = budget_display["scenario_daily_spend"].round(2)
            st.dataframe(budget_display, width="stretch", hide_index=True)

            csv_bytes = budget_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scenario recommendations",
                data=csv_bytes,
                file_name="budget_scenario_recommendations.csv",
                mime="text/csv",
            )

            for flag, message in [
                ("extrapolated", "Recommended spend is outside the observed daily range for: {}. Treat as extrapolation."),
                ("elasticity_clipped", "The fitted elasticity was clipped to its allowed range for: {}."),
            ]:
                if flag in budget.columns:
                    flagged = budget[budget[flag].astype(str).str.lower() == "true"]
                    if not flagged.empty:
                        st.warning(message.format(", ".join(flagged["platform"].astype(str))))

            if "fit_r2" in budget.columns:
                weak_fit = budget[budget["fit_r2"] < 0.8]
                if not weak_fit.empty:
                    st.warning(
                        "Some fitted curves have R² below 0.80: "
                        + ", ".join(weak_fit["platform"].astype(str))
                        + ". Treat their projections cautiously. This threshold is a dashboard flag, "
                        "not a universal statistical rule."
                    )

            st.warning(
                "Projected ROAS is a model estimate. Validate recommendations on held-out data "
                "or with a controlled experiment before making real spending decisions."
            )

with tab_benchmark:
    st.subheader("Processing-engine benchmark")

    if benchmark.empty:
        show_missing("benchmark_results.csv")
    else:
        needed = {"records", "engine", "seconds"}
        missing = needed - set(benchmark.columns)
        if missing:
            st.error(f"Benchmark CSV is missing required columns: {', '.join(sorted(missing))}")
        else:
            benchmark = benchmark.copy()
            benchmark["records"] = pd.to_numeric(benchmark["records"], errors="coerce")
            benchmark["seconds"] = pd.to_numeric(benchmark["seconds"], errors="coerce")
            for col in ["startup_seconds", "peak_mem_mb", "vs_pandas"]:
                if col in benchmark.columns:
                    benchmark[col] = pd.to_numeric(benchmark[col], errors="coerce")
            if "status" in benchmark.columns:
                problems = benchmark[benchmark["status"].isin(["skipped", "error"])]
                for item in problems.itertuples():
                    detail = getattr(item, "detail", "")
                    st.warning(f"{item.engine} was {item.status} at {item.records:,.0f} records. {detail if isinstance(detail, str) else ''}")
            if "time_std_seconds" in benchmark.columns:
                benchmark["time_std_seconds"] = pd.to_numeric(
                    benchmark["time_std_seconds"], errors="coerce"
                )

            engines = sorted(benchmark["engine"].dropna().unique().tolist())
            selected_engines = st.multiselect(
                "Engines",
                options=engines,
                default=engines,
            )
            plot_df = benchmark[benchmark["engine"].isin(selected_engines) & benchmark["seconds"].notna()].copy()
            plot_df["records_label"] = plot_df["records"].map(
                lambda x: f"{x:,.0f}" if pd.notna(x) else "Unknown"
            )
            plot_df = plot_df.sort_values("records")

            fig = px.line(
                plot_df,
                x="records_label",
                y="seconds",
                color="engine",
                markers=True,
                title="Runtime by dataset size",
                labels={"records_label": "Records", "seconds": "Runtime (seconds)", "engine": "Engine"},
            )
            st.plotly_chart(fig, width="stretch")

            if "peak_mem_mb" in plot_df.columns and plot_df["peak_mem_mb"].notna().any():
                mem_fig = px.line(
                    plot_df.dropna(subset=["peak_mem_mb"]),
                    x="records_label",
                    y="peak_mem_mb",
                    color="engine",
                    markers=True,
                    title="Peak memory by dataset size",
                    labels={"records_label": "Records", "peak_mem_mb": "Peak memory (MB)", "engine": "Engine"},
                )
                st.plotly_chart(mem_fig, width="stretch")
            if "startup_seconds" in plot_df.columns and (plot_df["startup_seconds"] > 0).any():
                st.caption("Runtime is processing time only. Engine start-up (for example the Spark session) is reported separately in `startup_seconds`.")

            if "consistent" in benchmark.columns:
                st.write("Consistency status")
                st.dataframe(
                    benchmark[[c for c in ["records", "engine", "status", "consistent",
                                           "seconds", "startup_seconds", "vs_pandas", "trials",
                                           "time_std_seconds", "peak_mem_mb"] if c in benchmark.columns]],
                    width="stretch",
                    hide_index=True,
                )
            if "peak_mem_mb" not in benchmark.columns or benchmark["peak_mem_mb"].isna().all():
                st.caption("Peak memory was not measured in this benchmark output.")

with tab_data:
    st.subheader("Source data tables")
    data_choice = st.selectbox(
        "Choose a dataset",
        ["KPI results", "Budget recommendations", "Benchmark results"],
    )
    data_map = {
        "KPI results": kpis,
        "Budget recommendations": budget,
        "Benchmark results": benchmark,
    }
    chosen = data_map[data_choice]
    if chosen.empty:
        st.info("No data loaded for this table.")
    else:
        st.dataframe(chosen, width="stretch", hide_index=True)
        st.download_button(
            f"Download {data_choice.lower().replace(' ', '_')}.csv",
            data=chosen.to_csv(index=False).encode("utf-8"),
            file_name=f"{data_choice.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Digital Advertising Budget Optimizer • Results depend on the quality of the input data "
    "and the assumptions of the underlying models."
)
