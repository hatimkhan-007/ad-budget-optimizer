from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

PLATFORM_COLORS = {
    "Google": "#4285F4",
    "Facebook": "#1877F2",
    "Instagram": "#E1306C",
    "YouTube": "#FF0000",
    "TikTok": "#111111",
}
FALLBACK_COLOR = "#6B7280"
CURRENT_COLOR = "#9CA3AF"
RECOMMENDED_COLOR = "#16A34A"
ENGINE_COLORS = {"pandas": "#2563EB", "mapreduce": "#F59E0B", "pyspark": "#DC2626"}
DPI = 150


def _color(platform: str) -> str:
    return PLATFORM_COLORS.get(platform, FALLBACK_COLOR)


def _money(value: float, _pos=None) -> str:
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= limit:
            return f"{value / limit:g}{suffix}"
    return f"{value:g}"


def _style(ax, title: str, subtitle: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=10, color="#4B5563", va="bottom")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def chart_roas(kpis: pd.DataFrame, path: Path) -> Path:
    data = kpis.sort_values("roas")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(data["platform"], data["roas"], color=[_color(p) for p in data["platform"]])
    ax.bar_label(bars, labels=[f"{v:.2f}x" for v in data["roas"]], padding=4, fontsize=11, fontweight="bold")
    ax.axvline(1, color="#DC2626", linestyle="--", linewidth=1.2)
    ax.text(1.02, -0.45, "Break-even (1.0x)", color="#DC2626", fontsize=9)
    ax.set_xlim(0, data["roas"].max() * 1.18)
    _style(ax, "Return on Ad Spend (ROAS) by Platform",
           "Revenue earned for every 1 unit spent. Higher is better.", xlabel="ROAS (revenue / spend)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, path)


def chart_spend_vs_revenue(kpis: pd.DataFrame, path: Path) -> Path:
    data = kpis.sort_values("revenue", ascending=False).reset_index(drop=True)
    x = np.arange(len(data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    spend_bars = ax.bar(x - width / 2, data["spend"], width, label="Spend", color="#F59E0B")
    revenue_bars = ax.bar(x + width / 2, data["revenue"], width, label="Revenue", color="#16A34A")
    ax.bar_label(spend_bars, labels=[_money(v) for v in data["spend"]], padding=3, fontsize=9)
    ax.bar_label(revenue_bars, labels=[_money(v) for v in data["revenue"]], padding=3, fontsize=9)
    ax.set_xticks(x, data["platform"])
    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    ax.legend(frameon=False, loc="upper right")
    _style(ax, "Total Spend vs Total Revenue by Platform",
           "A platform earns more than it costs when the green bar is taller than the orange bar.",
           ylabel="Amount (K = thousand, M = million)")
    return _save(fig, path)


def chart_budget_shares(table: pd.DataFrame, min_share: float, max_share: float, path: Path) -> Path:
    data = table.sort_values("recommended_share_pct", ascending=False).reset_index(drop=True)
    x = np.arange(len(data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cur = ax.bar(x - width / 2, data["current_share_pct"], width, label="Current share", color=CURRENT_COLOR)
    rec = ax.bar(x + width / 2, data["recommended_share_pct"], width, label="Recommended share", color=RECOMMENDED_COLOR)
    ax.bar_label(cur, labels=[f"{v:.1f}%" for v in data["current_share_pct"]], padding=3, fontsize=9)
    ax.bar_label(rec, labels=[f"{v:.1f}%" for v in data["recommended_share_pct"]], padding=3, fontsize=9, fontweight="bold")
    ax.axhline(max_share * 100, color="#DC2626", linestyle="--", linewidth=1, label=f"Max share limit ({max_share:.0%})")
    ax.axhline(min_share * 100, color="#DC2626", linestyle=":", linewidth=1, label=f"Min share limit ({min_share:.0%})")
    ax.set_xticks(x, data["platform"])
    ax.set_ylim(0, max(data[["current_share_pct", "recommended_share_pct"]].max().max(), max_share * 100) * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}%"))
    ax.legend(frameon=False, loc="upper right")
    _style(ax, "Budget Share: Current vs Recommended",
           "How the optimizer moves money between platforms (red lines = allowed limits).",
           ylabel="Share of total budget")
    return _save(fig, path)


def chart_response_curves(optimizer, table: pd.DataFrame, daily: pd.DataFrame, path: Path) -> Path:
    platforms = list(table["platform"])
    cols = 3
    rows = int(np.ceil(len(platforms) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.4 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), platforms):
        curve = optimizer.curves[name]
        row = table.loc[table["platform"] == name].iloc[0]
        sample = daily.loc[daily["platform"] == name]
        ax.scatter(sample["spend"], sample["revenue"], s=10, alpha=0.35, color=_color(name), label="Daily results")
        top = max(sample["spend"].max(), row["recommended_daily_spend"]) * 1.1
        grid = np.linspace(sample["spend"].min() * 0.6, top, 200)
        ax.plot(grid, [curve.revenue(s) for s in grid], color="black", linewidth=2, label="Fitted curve")
        ax.scatter([row["current_daily_spend"]], [curve.revenue(row["current_daily_spend"])], s=90,
                   color=CURRENT_COLOR, edgecolor="black", zorder=5, label="Current spend")
        ax.scatter([row["recommended_daily_spend"]], [curve.revenue(row["recommended_daily_spend"])], s=110,
                   marker="*", color=RECOMMENDED_COLOR, edgecolor="black", zorder=6, label="Recommended spend")
        ax.xaxis.set_major_formatter(FuncFormatter(_money))
        ax.yaxis.set_major_formatter(FuncFormatter(_money))
        ax.set_title(f"{name}  (elasticity {curve.b:.2f})", fontsize=11, fontweight="bold", loc="left")
        ax.set_xlabel("Daily spend", fontsize=9)
        ax.set_ylabel("Daily revenue", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(platforms):]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Diminishing Returns: Daily Revenue vs Daily Spend", fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.925, "Each curve flattens as spend grows, so the next unit of spend earns less. Dots are real daily results.",
             fontsize=10, color="#4B5563")
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def chart_monthly_revenue(daily: pd.DataFrame, path: Path) -> Path:
    frame = daily.copy()
    frame["month"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
    monthly = frame.pivot_table(index="month", columns="platform", values="revenue", aggfunc="sum").sort_index()
    order = monthly.sum().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.stackplot(monthly.index, [monthly[p] for p in order], labels=list(order), colors=[_color(p) for p in order], alpha=0.9)
    ax.yaxis.set_major_formatter(FuncFormatter(_money))
    ax.legend(frameon=False, loc="upper left", ncol=len(order))
    ax.margins(x=0)
    fig.autofmt_xdate()
    _style(ax, "Monthly Revenue by Platform",
           "Total revenue per month, stacked by platform. Shows seasonality and each platform's contribution.",
           ylabel="Revenue per month")
    return _save(fig, path)


def chart_benchmark(benchmark: pd.DataFrame, path: Path) -> Path | None:
    ok = benchmark[benchmark["status"] == "ok"]
    if ok.empty:
        return None
    engines = [e for e in ENGINE_COLORS if e in set(ok["engine"])] or sorted(set(ok["engine"]))
    sizes = sorted(ok["records"].unique())
    x = np.arange(len(sizes))
    width = 0.8 / len(engines)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, column, title, unit in (
        (axes[0], "seconds", "Processing Time", "Seconds (log scale, lower is better)"),
        (axes[1], "peak_mem_mb", "Peak Memory", "MB (lower is better)"),
    ):
        for i, engine in enumerate(engines):
            values = [ok[(ok["engine"] == engine) & (ok["records"] == n)][column].mean() for n in sizes]
            bars = ax.bar(x + (i - (len(engines) - 1) / 2) * width, values, width,
                          label=engine, color=ENGINE_COLORS.get(engine, FALLBACK_COLOR))
            ax.bar_label(bars, labels=["" if pd.isna(v) else (f"{v:.2f}" if column == "seconds" else f"{v:.0f}") for v in values],
                         padding=2, fontsize=8)
        ax.set_xticks(x, [f"{n:,}" for n in sizes])
        ax.set_xlabel("Number of records", fontsize=10)
        ax.set_ylabel(unit, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, title="Engine")
    fig.suptitle("Benchmark: " + " vs ".join(e.capitalize() if e == "pandas" else {"mapreduce": "MapReduce", "pyspark": "PySpark"}.get(e, e) for e in engines), fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def generate_charts(
    charts_dir: str | Path,
    kpis_by_platform: pd.DataFrame,
    daily: pd.DataFrame,
    optimizer,
    table: pd.DataFrame,
    min_share: float,
    max_share: float,
    benchmark: pd.DataFrame | None = None,
) -> list[Path]:
    out = Path(charts_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("[0-9][0-9]_*.png"):
        old.unlink()
    made = [
        chart_roas(kpis_by_platform, out / "01_roas_by_platform.png"),
        chart_spend_vs_revenue(kpis_by_platform, out / "02_spend_vs_revenue.png"),
        chart_budget_shares(table, min_share, max_share, out / "03_budget_share_current_vs_recommended.png"),
        chart_response_curves(optimizer, table, daily, out / "04_response_curves.png"),
        chart_monthly_revenue(daily, out / "05_monthly_revenue_by_platform.png"),
    ]
    if benchmark is not None and not benchmark.empty:
        extra = chart_benchmark(benchmark, out / "06_benchmark_time_and_memory.png")
        if extra:
            made.append(extra)
    return made
