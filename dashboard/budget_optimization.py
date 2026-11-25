"""Budget optimisation: platform and campaign level recommendations with clear model caveats."""

from __future__ import annotations

import copy

import pandas as pd
import plotly.express as px
import streamlit as st

from budget_optimizer import AllocationResult, BudgetOptimizer
from campaign_optimizer import CampaignOptimizer
from dashboard.context import Context
from dashboard.data import DatasetInfo
from dashboard.theme import (CURRENT_COLOR, RECOMMENDED_COLOR, empty_state, money, multiple, page_header,
                             platform_color_map, style_figure)

ASSUMPTIONS = """
**Response model.** For each platform, daily revenue is modelled as `a * spend^b` (0 < b < 1), fitted by log-log
regression on daily history. `b` is the spend elasticity: the closer to 0, the faster returns diminish. The curve is
calibrated to reproduce the platform's observed average daily spend and revenue.

**Campaigns.** A single campaign has too little history to estimate its own elasticity, so each campaign inherits its
platform's `b` and gets its own efficiency coefficient `a` from its own results.

**Optimisation.** Revenue is maximised for a fixed total budget by equalising marginal ROAS across campaigns, except
where a limit binds. Limits: platform share range, an optional absolute minimum/maximum per campaign, and a maximum
percentage change from each campaign's current budget. The change limit keeps recommendations near spend levels the
data has actually seen. The solution is exact for these curves (it satisfies the optimality conditions).

**What this is not.** The projections are model estimates built on past simulated data. They assume the observed
relationship between spend and revenue continues, ignore creative fatigue, seasonality shifts, competition and
cross-platform effects, and carry no guarantee of any revenue increase.
"""


@st.cache_data(show_spinner="Fitting response curves...")
def _fit_campaign_model(path: str, size: int, mtime_ns: int, _daily: pd.DataFrame) -> CampaignOptimizer:
    return CampaignOptimizer().fit(_daily)


def _platform_reason(row, low: float, high: float) -> str:
    text = f"Historical ROAS {row.current_roas:.2f}x; modelled marginal ROAS at the recommendation {row.marginal_roas:.2f}x. "
    if row.recommended_share_pct >= high * 100 - 1e-6:
        return text + "Held at the maximum platform share."
    if row.recommended_share_pct <= low * 100 + 1e-6:
        return text + "Held at the minimum platform share."
    if row.change_pp > 0.01:
        return text + "Budget moves here because extra spend still earns more than on average."
    if row.change_pp < -0.01:
        return text + "Budget moves out because returns have saturated compared with other platforms."
    return text + "Already close to optimal."


def _by_platform(table: pd.DataFrame) -> pd.DataFrame:
    work = table.assign(current_revenue=table["current_daily_spend"] * table["current_roas"],
                        recommended_revenue=table["recommended_daily_spend"] * table["projected_roas"])
    return work.groupby("platform", observed=True)[
        ["current_daily_spend", "recommended_daily_spend", "current_revenue", "recommended_revenue"]].sum().reset_index()


def _summary(result: AllocationResult, historical_spend: float, historical_revenue: float, scenario: bool) -> None:
    st.markdown("#### Historical results (observed)")
    h = st.columns(3)
    h[0].metric("Average daily spend", money(historical_spend))
    h[1].metric("Average daily revenue", money(historical_revenue))
    h[2].metric("Historical ROAS", multiple(historical_revenue / historical_spend))
    st.markdown("#### Model-based estimate after reallocation (not guaranteed)")
    e = st.columns(4)
    base = "Current mix at this budget" if scenario else "Current allocation"
    e[0].metric(f"{base}: revenue/day", money(result.current_revenue))
    e[1].metric("Optimised: revenue/day", money(result.recommended_revenue),
                delta=f"{result.uplift_vs_compliant_pct:+.2f}% vs limit-compliant baseline", delta_color="off")
    e[2].metric("Optimised ROAS", multiple(result.recommended_roas), delta=f"{result.recommended_roas - result.current_roas:+.2f}", delta_color="off")
    e[3].metric("Budget per day", money(result.total_daily_budget))
    if not result.current_within_limits:
        st.info("The current allocation breaks the chosen limits, so part of the change is the cost of complying. "
                f"Estimated change versus the raw current mix: {result.uplift_pct:+.2f}%.")


def _charts(table: pd.DataFrame) -> None:
    platforms = _by_platform(table)
    colors = platform_color_map(platforms["platform"])
    long = platforms.melt(id_vars="platform", value_vars=["current_daily_spend", "recommended_daily_spend"],
                          var_name="plan", value_name="daily_spend")
    long["plan"] = long["plan"].map({"current_daily_spend": "Original", "recommended_daily_spend": "Optimised"})
    left, right = st.columns(2)
    with left:
        fig = px.bar(long, x="platform", y="daily_spend", color="plan", barmode="group",
                     color_discrete_map={"Original": CURRENT_COLOR, "Optimised": RECOMMENDED_COLOR},
                     labels={"daily_spend": "Daily budget", "platform": ""})
        st.plotly_chart(style_figure(fig, "Original vs optimised budget by platform"), width="stretch")
    with right:
        rev = platforms.melt(id_vars="platform", value_vars=["current_revenue", "recommended_revenue"],
                             var_name="plan", value_name="revenue")
        rev["plan"] = rev["plan"].map({"current_revenue": "Current (model)", "recommended_revenue": "Optimised (estimate)"})
        fig = px.bar(rev, x="platform", y="revenue", color="plan", barmode="group",
                     color_discrete_map={"Current (model)": CURRENT_COLOR, "Optimised (estimate)": RECOMMENDED_COLOR},
                     labels={"revenue": "Revenue per day", "platform": ""})
        st.plotly_chart(style_figure(fig, "Expected vs current revenue by platform"), width="stretch")
    a, b = st.columns(2)
    for column, name, label in ((a, "current_daily_spend", "Current budget distribution"),
                                (b, "recommended_daily_spend", "Optimised budget distribution")):
        fig = px.pie(platforms, names="platform", values=name, hole=0.55, color="platform", color_discrete_map=colors)
        column.plotly_chart(style_figure(fig, label, 320), width="stretch")


def _campaign_changes(table: pd.DataFrame) -> None:
    work = table.assign(change=table["recommended_daily_spend"] - table["current_daily_spend"])
    shown = pd.concat([work.nlargest(10, "change"), work.nsmallest(10, "change")]).drop_duplicates("campaign_id")
    fig = px.bar(shown.sort_values("change"), x="change", y="campaign_id", orientation="h", color="action",
                 color_discrete_map={"Increase": "#10B981", "Decrease": "#EF4444", "Hold": CURRENT_COLOR},
                 labels={"change": "Change in daily budget", "campaign_id": ""})
    st.plotly_chart(style_figure(fig, "Largest campaign budget changes", 520), width="stretch")


def render(ctx: Context) -> None:
    page_header("Budget Optimization", "Recommend how to redistribute a fixed daily budget, with the assumptions made explicit.")
    if not ctx.has_data:
        empty_state("No dataset selected", "Generate or select a dataset in **Dataset Management** first.")
        return
    st.caption("Uses all platforms and campaigns in the selected dataset (the sidebar platform filter does not apply here).")
    daily = ctx.full_campaign_daily
    dataset: DatasetInfo = ctx.dataset
    stat = dataset.path.stat()
    try:
        fitted = _fit_campaign_model(str(dataset.path), stat.st_size, stat.st_mtime_ns, daily)
    except ValueError as error:
        st.error(f"The optimiser cannot be fitted on this dataset: {error}")
        return
    historical_spend = sum(c.avg_daily_spend for c in fitted.curves.values())
    historical_revenue = sum(c.avg_daily_revenue for c in fitted.curves.values())

    level = st.radio("Optimise at", ["Platform", "Campaign"], horizontal=True, key="opt_level")
    with st.form("optimizer_form"):
        c1, c2 = st.columns(2)
        budget = c1.number_input("Total daily budget", min_value=1.0, value=float(round(historical_spend, 2)), step=100.0,
                                 help="Defaults to the historical average total daily spend.")
        shares = c2.slider("Platform share limits (%)", 0, 100, (5, 40))
        if level == "Campaign":
            d1, d2, d3 = st.columns(3)
            max_change = d1.slider("Max change per campaign (%)", 0, 100, 50,
                                   help="Limits how far a campaign budget may move from its current level.")
            min_budget = d2.number_input("Minimum campaign budget", min_value=0.0, value=0.0, step=10.0)
            max_budget = d3.number_input("Maximum campaign budget (0 = no limit)", min_value=0.0, value=0.0, step=100.0)
        submitted = st.form_submit_button("Optimise")
    if not submitted and "opt_result" not in st.session_state:
        empty_state("Set the constraints and press Optimise", "Defaults reproduce the current total budget with sensible limits.")
        return
    if submitted:
        try:
            low, high = shares[0] / 100, shares[1] / 100
            if level == "Platform":
                platform_daily = daily.groupby(["platform", "date"], observed=True)[["spend", "revenue"]].sum().reset_index()
                result = BudgetOptimizer(low, high).fit(platform_daily).recommend(budget)
                result.table["reason"] = [_platform_reason(r, low, high) for r in result.table.itertuples()]
            else:
                model = copy.copy(fitted)
                model.max_change = max_change / 100
                model.min_budget = float(min_budget)
                model.max_budget = float(max_budget) if max_budget > 0 else None
                model.platform_limits = (low, high)
                result = model.recommend_campaigns(budget, ctx.full_catalog)
        except (ValueError, RuntimeError) as error:
            st.error(f"No recommendation could be produced: {error}")
            st.session_state.pop("opt_result", None)
            return
        st.session_state["opt_result"] = (level, budget, result)
    level_used, budget_used, result = st.session_state["opt_result"]
    if level_used != level:
        st.info(f"Showing the last {level_used.lower()}-level run. Press Optimise to run {level.lower()} level.")

    _summary(result, historical_spend, historical_revenue, abs(budget_used - historical_spend) > 0.5)
    for note in result.notes:
        st.warning(note)
    _charts(result.table)
    if level_used == "Campaign":
        _campaign_changes(result.table)

    st.subheader("Recommendations")
    table = result.table
    if level_used == "Campaign":
        counts = table["performance_tier"].value_counts()
        st.markdown(f"High performers: **{counts.get('High performer', 0)}** | Underperformers: "
                    f"**{counts.get('Underperformer', 0)}** | Average: **{counts.get('Average', 0)}** "
                    "(tiers compare each campaign's historical ROAS with the portfolio ROAS: +/-15%).")
        columns = ["campaign_id", "platform", "objective", "target_audience", "performance_tier", "action", "current_daily_spend",
                   "recommended_daily_spend", "change_pp", "historical_roas", "marginal_roas", "constraint", "reason"]
    else:
        columns = ["platform", "current_share_pct", "recommended_share_pct", "change_pp", "current_daily_spend",
                   "recommended_daily_spend", "current_roas", "projected_roas", "marginal_roas", "elasticity", "fit_r2", "reason"]
    columns = [c for c in columns if c in table.columns]
    st.dataframe(table[columns], hide_index=True, width="stretch", height=420)
    st.download_button("Download recommendations (CSV)", table.to_csv(index=False).encode("utf-8"),
                       f"{level_used.lower()}_budget_recommendation.csv", "text/csv")
    with st.expander("Optimisation assumptions and method"):
        st.markdown(ASSUMPTIONS)
