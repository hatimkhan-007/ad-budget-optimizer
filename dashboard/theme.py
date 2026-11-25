"""Shared visual style, formatting helpers and reusable UI fragments."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PLATFORM_COLORS = {
    "Google": "#4285F4", "Facebook": "#1877F2", "Instagram": "#E1306C", "YouTube": "#FF0000", "TikTok": "#111111",
}
SERIES_COLORS = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#0EA5E9", "#8B5CF6", "#14B8A6"]
SENTIMENT_COLORS = {"Positive": "#10B981", "Neutral": "#9CA3AF", "Negative": "#EF4444"}
CURRENT_COLOR = "#9CA3AF"
RECOMMENDED_COLOR = "#4F46E5"
SIMULATION_NOTICE = ("All advertising data in this application is **simulated** for demonstration. "
                     "It does not describe real campaigns, audiences or results.")

CSS = """
<style>
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.06); border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px; padding: 14px 16px;
}
div[data-testid="stMetricLabel"] p { font-size: 0.82rem; opacity: 0.75; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.page-caption { opacity: 0.7; margin-top: -0.6rem; margin-bottom: 1rem; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, caption: str) -> None:
    st.title(title)
    st.markdown(f'<div class="page-caption">{caption}</div>', unsafe_allow_html=True)


def empty_state(title: str, message: str) -> None:
    st.info(f"**{title}**\n\n{message}")


def style_figure(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, title=dict(text=title, x=0, font=dict(size=16)) if title else None,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10), colorway=SERIES_COLORS,
        font=dict(family="Inter, Segoe UI, Roboto, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


def platform_color_map(platforms) -> dict[str, str]:
    return {p: PLATFORM_COLORS.get(p, "#6B7280") for p in platforms}


def money(value: float, decimals: int = 0) -> str:
    return "-" if pd.isna(value) else f"{value:,.{decimals}f}"


def whole(value: float) -> str:
    return "-" if pd.isna(value) else f"{value:,.0f}"


def percent(value: float, decimals: int = 2) -> str:
    return "-" if pd.isna(value) else f"{value:.{decimals}f}%"


def multiple(value: float) -> str:
    return "-" if pd.isna(value) else f"{value:.2f}x"


def compact(value: float) -> str:
    if pd.isna(value):
        return "-"
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= limit:
            return f"{value / limit:.2f}{suffix}"
    return f"{value:,.0f}"
