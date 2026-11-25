"""Campaign catalog built from processed advertising data.

The generated dataset stores one row per delivery record with the columns
``platform``, ``campaign`` (the campaign *objective*) and ``audience``. A
campaign is therefore defined as one (platform, objective, audience) line item,
and receives a stable identifier derived from those three fields.

Everything that is observable in the data (dates, spend, delivery status) is
derived from it. The planned ``budget`` is an input that the synthetic dataset
does not contain, so it is simulated deterministically and flagged as such.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import pandas as pd

from metrics_calculator import NUMERIC_COLUMNS, add_ratios

CAMPAIGN_KEYS = ["platform", "campaign", "audience"]
PLATFORM_CODES = {"Facebook": "FB", "Google": "GG", "Instagram": "IG", "TikTok": "TT", "YouTube": "YT"}
OBJECTIVE_CODES = {
    "Brand Awareness": "BRAND", "Conversions": "CONV", "Retargeting": "RTGT",
    "Lead Generation": "LEAD", "Product Launch": "LAUNCH",
}
AUDIENCE_CODES = {"New Customers": "NEW", "Returning Customers": "RET", "Lookalike": "LAL", "Interest Based": "INT"}

ACTIVE_WINDOW_DAYS = 30
COMPLETED_WINDOW_DAYS = 90
BUDGET_HEADROOM_RANGE = (1.02, 1.20)


def _code(value: str, table: dict[str, str], length: int) -> str:
    if value in table:
        return table[value]
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
    return cleaned[:length] or "NA"


def make_campaign_id(platform: str, objective: str, audience: str) -> str:
    """Stable identifier such as ``FB-CONV-NEW``."""
    return "-".join([
        _code(platform, PLATFORM_CODES, 2),
        _code(objective, OBJECTIVE_CODES, 5),
        _code(audience, AUDIENCE_CODES, 3),
    ])


def add_campaign_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``campaign_id`` and ``campaign_name`` columns."""
    missing = set(CAMPAIGN_KEYS) - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot build campaign ids; missing columns: {', '.join(sorted(missing))}")
    out = frame.copy()
    if out.empty:
        out["campaign_id"] = pd.Series(dtype="object")
        out["campaign_name"] = pd.Series(dtype="object")
        return out
    keys = out[CAMPAIGN_KEYS].astype(str).drop_duplicates()
    lookup = {
        tuple(row): (make_campaign_id(*row), f"{row[0]} | {row[1]} | {row[2]}")
        for row in keys.itertuples(index=False, name=None)
    }
    pairs = [lookup[tuple(row)] for row in out[CAMPAIGN_KEYS].astype(str).itertuples(index=False, name=None)]
    out["campaign_id"] = [p[0] for p in pairs]
    out["campaign_name"] = [p[1] for p in pairs]
    return out


def _delivery_status(last_delivery: pd.Timestamp, data_end: pd.Timestamp) -> str:
    idle = (data_end - last_delivery).days
    if idle <= ACTIVE_WINDOW_DAYS:
        return "Active"
    if idle <= COMPLETED_WINDOW_DAYS:
        return "Paused"
    return "Completed"


def _simulated_budget(campaign_id: str, total_spend: float, seed: int) -> float:
    rng = np.random.default_rng([seed, zlib.crc32(campaign_id.encode("utf-8"))])
    factor = rng.uniform(*BUDGET_HEADROOM_RANGE)
    return float(np.ceil(total_spend * factor / 100.0) * 100.0)


def build_campaign_catalog(campaign_daily: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """One row per campaign with metadata, delivery dates, status, budget and performance.

    ``status`` is derived from delivery: Active if the campaign delivered within the
    last 30 days of the dataset period, Paused within 90 days, otherwise Completed.
    ``budget`` is simulated (see module docstring).
    """
    required = set(CAMPAIGN_KEYS) | {"date"} | set(NUMERIC_COLUMNS)
    missing = required - set(campaign_daily.columns)
    if missing:
        raise ValueError(f"Campaign data is missing columns: {', '.join(sorted(missing))}")
    if campaign_daily.empty:
        return pd.DataFrame(columns=[
            "campaign_id", "campaign_name", "platform", "objective", "target_audience", "status",
            "start_date", "end_date", "budget", "total_spend", "budget_utilization_pct",
            *NUMERIC_COLUMNS, "ctr_pct", "cpc", "cpa", "conversion_rate_pct", "roas", "cpm",
        ])

    data = add_campaign_id(campaign_daily)
    data["date"] = pd.to_datetime(data["date"])
    data_end = data["date"].max()
    grouped = data.groupby(["campaign_id", "campaign_name", *CAMPAIGN_KEYS], observed=True)
    totals = grouped[NUMERIC_COLUMNS].sum()
    dates = grouped["date"].agg(start_date="min", end_date="max")
    catalog = totals.join(dates).reset_index()
    catalog = catalog.rename(columns={"campaign": "objective", "audience": "target_audience"})
    catalog["status"] = [_delivery_status(d, data_end) for d in catalog["end_date"]]
    catalog["total_spend"] = catalog["spend"]
    catalog["budget"] = [_simulated_budget(cid, spend, seed)
                         for cid, spend in zip(catalog["campaign_id"], catalog["total_spend"])]
    catalog["budget_utilization_pct"] = catalog["total_spend"] / catalog["budget"] * 100
    catalog = add_ratios(catalog)
    front = ["campaign_id", "campaign_name", "platform", "objective", "target_audience", "status",
             "start_date", "end_date", "budget", "total_spend", "budget_utilization_pct"]
    rest = [c for c in catalog.columns if c not in front]
    return catalog[front + rest].sort_values("campaign_id").reset_index(drop=True)
