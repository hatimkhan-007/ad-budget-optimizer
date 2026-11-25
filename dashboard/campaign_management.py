"""Campaign catalog with filters and a campaign detail view."""

from __future__ import annotations

import streamlit as st
from plotly.subplots import make_subplots

from dashboard.context import Context
from dashboard.theme import (SIMULATION_NOTICE, empty_state, money, multiple, page_header, percent, style_figure,
                             whole)

STATUS_HELP = ("Status is derived from delivery: Active = delivered within the last 30 days of the data period, "
               "Paused = within 90 days, Completed = earlier. Budgets are simulated planning figures.")


def campaign_label(row) -> str:
    return f"{row.campaign_id}  |  {row.campaign_name}"


def render(ctx: Context) -> None:
    page_header("Campaign Management", "Browse campaigns, filter by objective or status and inspect individual results.")
    if not ctx.has_data or ctx.catalog.empty:
        empty_state("No campaigns available", "Generate or select a dataset in **Dataset Management** first.")
        return
    st.caption(SIMULATION_NOTICE)

    catalog = ctx.catalog
    f1, f2, f3 = st.columns(3)
    statuses = f1.multiselect("Status", sorted(catalog["status"].unique()), default=sorted(catalog["status"].unique()),
                              help=STATUS_HELP)
    objectives = f2.multiselect("Objective", sorted(catalog["objective"].unique()), default=sorted(catalog["objective"].unique()))
    audiences = f3.multiselect("Target audience", sorted(catalog["target_audience"].unique()),
                               default=sorted(catalog["target_audience"].unique()))
    view = catalog[catalog["status"].isin(statuses) & catalog["objective"].isin(objectives)
                   & catalog["target_audience"].isin(audiences)]
    if view.empty:
        empty_state("No campaigns match the filters", "Widen the status, objective or audience selection.")
        return

    st.markdown(f"**{len(view):,} campaigns** | budget {money(view['budget'].sum())} | spend {money(view['total_spend'].sum())}")
    columns = ["campaign_id", "campaign_name", "platform", "objective", "target_audience", "status", "start_date",
               "end_date", "budget", "total_spend", "budget_utilization_pct", "impressions", "clicks", "ctr_pct", "cpc",
               "conversions", "conversion_rate_pct", "cpa", "revenue", "roas", "cpm"]
    st.dataframe(
        view[columns], hide_index=True, width="stretch", height=380,
        column_config={
            "campaign_id": "ID", "campaign_name": "Campaign", "platform": "Platform", "objective": "Objective",
            "target_audience": "Target audience", "status": "Status",
            "start_date": st.column_config.DateColumn("Start"), "end_date": st.column_config.DateColumn("End"),
            "budget": st.column_config.NumberColumn("Budget (simulated)", format="%.0f"),
            "total_spend": st.column_config.NumberColumn("Spend", format="%.0f"),
            "budget_utilization_pct": st.column_config.ProgressColumn("Budget used", format="%.0f%%", min_value=0, max_value=100),
            "impressions": st.column_config.NumberColumn("Impressions", format="%d"),
            "clicks": st.column_config.NumberColumn("Clicks", format="%d"),
            "ctr_pct": st.column_config.NumberColumn("CTR %", format="%.2f"),
            "cpc": st.column_config.NumberColumn("CPC", format="%.2f"),
            "conversions": st.column_config.NumberColumn("Conversions", format="%d"),
            "conversion_rate_pct": st.column_config.NumberColumn("CVR %", format="%.2f"),
            "cpa": st.column_config.NumberColumn("CPA", format="%.2f"),
            "revenue": st.column_config.NumberColumn("Revenue", format="%.0f"),
            "roas": st.column_config.NumberColumn("ROAS", format="%.2f"),
            "cpm": st.column_config.NumberColumn("CPM", format="%.2f"),
        })
    st.download_button("Download filtered campaigns (CSV)", view.to_csv(index=False).encode("utf-8"),
                       "campaigns.csv", "text/csv")

    st.subheader("Campaign detail")
    labels = {r.campaign_id: campaign_label(r) for r in view.itertuples()}
    chosen = st.selectbox("Campaign", list(labels), format_func=labels.get, key="campaign_detail")
    row = view[view["campaign_id"] == chosen].iloc[0]
    info = st.columns(4)
    info[0].metric("Platform", row["platform"])
    info[1].metric("Objective", row["objective"])
    info[2].metric("Target audience", row["target_audience"])
    info[3].metric("Status", row["status"], help=STATUS_HELP)
    kpis = st.columns(5)
    kpis[0].metric("Budget (simulated)", whole(row["budget"]))
    kpis[1].metric("Spend", whole(row["total_spend"]), delta=f"{row['budget_utilization_pct']:.0f}% of budget", delta_color="off")
    kpis[2].metric("Revenue", whole(row["revenue"]))
    kpis[3].metric("ROAS", multiple(row["roas"]))
    kpis[4].metric("Conversions", whole(row["conversions"]))
    more = st.columns(5)
    more[0].metric("Impressions", whole(row["impressions"]))
    more[1].metric("Clicks", whole(row["clicks"]))
    more[2].metric("CTR", percent(row["ctr_pct"]))
    more[3].metric("CPC", money(row["cpc"], 2))
    more[4].metric("CPA", money(row["cpa"], 2))
    st.caption(f"Delivery period: {row['start_date']:%Y-%m-%d} to {row['end_date']:%Y-%m-%d}.")

    history = ctx.campaign_daily[ctx.campaign_daily["campaign_id"] == chosen]
    weekly = history.set_index("date")[["spend", "revenue", "conversions"]].resample("W").sum().reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=weekly["date"], y=weekly["spend"], name="Spend", marker_color="#9CA3AF", secondary_y=False)
    fig.add_scatter(x=weekly["date"], y=weekly["revenue"], name="Revenue", mode="lines", line=dict(color="#4F46E5"), secondary_y=False)
    fig.add_scatter(x=weekly["date"], y=weekly["conversions"], name="Conversions", mode="lines",
                    line=dict(color="#10B981", dash="dot"), secondary_y=True)
    fig.update_yaxes(title_text="Spend / revenue", secondary_y=False)
    fig.update_yaxes(title_text="Conversions", secondary_y=True, showgrid=False)
    st.plotly_chart(style_figure(fig, "Weekly spend, revenue and conversions"), width="stretch")
