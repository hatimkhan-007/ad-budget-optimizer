"""Digital marketing KPI calculations.

KPIs are always derived from summed totals (never averaged row-level ratios),
which keeps results correct at any aggregation level.

    CTR   = clicks / impressions * 100
    CPC   = spend / clicks
    CPA   = spend / conversions
    CVR   = conversions / clicks * 100
    ROAS  = revenue / spend
    CPM   = spend / impressions * 1000
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

SUM_COLUMNS = ["impressions", "clicks", "spend", "conversions", "revenue"]
KPI_COLUMNS = ["ctr_pct", "cpc", "cpa", "conversion_rate_pct", "roas", "cpm"]


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN instead of inf where the denominator is zero."""
    return numerator / denominator.where(denominator != 0)


def add_kpis(totals: pd.DataFrame) -> pd.DataFrame:
    """Append KPI columns to a frame that holds summed metric columns."""
    out = totals.copy()
    out["ctr_pct"] = _ratio(out["clicks"], out["impressions"]) * 100
    out["cpc"] = _ratio(out["spend"], out["clicks"])
    out["cpa"] = _ratio(out["spend"], out["conversions"])
    out["conversion_rate_pct"] = _ratio(out["conversions"], out["clicks"]) * 100
    out["roas"] = _ratio(out["revenue"], out["spend"])
    out["cpm"] = _ratio(out["spend"], out["impressions"]) * 1000
    return out


def compute_kpis(df: pd.DataFrame, by: str | Sequence[str] | None = None) -> pd.DataFrame:
    """Aggregate raw rows and compute KPIs, overall or grouped by column(s)."""
    if by is None:
        totals = df[SUM_COLUMNS].sum().to_frame().T
    else:
        keys = [by] if isinstance(by, str) else list(by)
        totals = df.groupby(keys, observed=True)[SUM_COLUMNS].sum().reset_index()
    return add_kpis(totals)


def aggregate_csv(path: str | Path, by: str | Sequence[str],
                  chunksize: int = 1_000_000) -> pd.DataFrame:
    """Memory-safe grouped sums over a CSV of any size (chunked read)."""
    keys = [by] if isinstance(by, str) else list(by)
    dtypes = {
        "impressions": "int64", "clicks": "int64", "spend": "float64",
        "conversions": "int64", "revenue": "float64",
    }
    partials = []
    reader = pd.read_csv(path, usecols=keys + SUM_COLUMNS, dtype=dtypes, chunksize=chunksize)
    for chunk in reader:
        partials.append(chunk.groupby(keys)[SUM_COLUMNS].sum())
    combined = pd.concat(partials).groupby(level=list(range(len(keys)))).sum()
    combined.index.names = keys
    return combined.reset_index()


def kpis_from_csv(path: str | Path, by: str | Sequence[str] | None = None,
                  chunksize: int = 1_000_000) -> pd.DataFrame:
    """Chunked aggregation followed by KPI calculation."""
    if by is None:
        frame = pd.read_csv(path, usecols=SUM_COLUMNS, chunksize=chunksize)
        totals = sum(chunk.sum() for chunk in frame).to_frame().T
        return add_kpis(totals)
    return add_kpis(aggregate_csv(path, by, chunksize))


def format_kpi_table(kpis: pd.DataFrame, key_columns: Sequence[str]) -> str:
    """Readable console table for a KPI frame."""
    view = kpis[list(key_columns) + ["impressions", "clicks", "spend", "conversions",
                                     "revenue"] + KPI_COLUMNS].copy()
    for col in ["impressions", "clicks", "conversions"]:
        view[col] = view[col].map("{:,.0f}".format)
    for col in ["spend", "revenue"]:
        view[col] = view[col].map("${:,.0f}".format)
    view["ctr_pct"] = view["ctr_pct"].map("{:.2f}%".format)
    view["conversion_rate_pct"] = view["conversion_rate_pct"].map("{:.2f}%".format)
    for col in ["cpc", "cpa", "cpm"]:
        view[col] = view[col].map("${:,.2f}".format)
    view["roas"] = view["roas"].map("{:.2f}x".format)
    view = view.rename(columns={"ctr_pct": "CTR", "cpc": "CPC", "cpa": "CPA",
                                "conversion_rate_pct": "CVR", "roas": "ROAS", "cpm": "CPM"})
    return view.to_string(index=False)