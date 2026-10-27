from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"platform", "campaign", "date", "impressions", "clicks", "spend", "conversions", "revenue"}
NUMERIC_COLUMNS = ["impressions", "clicks", "spend", "conversions", "revenue"]

def validate_columns(columns: Iterable[str]) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.div(denominator.where(denominator != 0))
    return result.replace([np.inf, -np.inf], np.nan)

def compute_kpis(data: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    validate_columns(data.columns)
    keys = [by] if isinstance(by, str) else list(by or [])
    if keys:
        missing = set(keys) - set(data.columns)
        if missing:
            raise ValueError(f"Grouping columns missing: {', '.join(sorted(missing))}")
        grouped = data.groupby(keys, dropna=False, observed=True)[NUMERIC_COLUMNS].sum().reset_index()
    else:
        grouped = data[NUMERIC_COLUMNS].sum().to_frame().T
    grouped["ctr_pct"] = _ratio(grouped["clicks"], grouped["impressions"]) * 100
    grouped["cpc"] = _ratio(grouped["spend"], grouped["clicks"])
    grouped["cpa"] = _ratio(grouped["spend"], grouped["conversions"])
    grouped["conversion_rate_pct"] = _ratio(grouped["conversions"], grouped["clicks"]) * 100
    grouped["roas"] = _ratio(grouped["revenue"], grouped["spend"])
    grouped["cpm"] = _ratio(grouped["spend"], grouped["impressions"]) * 1000
    return grouped

def kpis_from_csv(path: str | Path, by: str | list[str] | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    partials = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        validate_columns(chunk.columns)
        partials.append(compute_kpis(chunk, by))
    if not partials:
        # Header-only CSV: preserve grouping shape and return a well-defined empty result.
        empty = pd.read_csv(path, nrows=0)
        validate_columns(empty.columns)
        keys = [by] if isinstance(by, str) else list(by or [])
        return pd.DataFrame(columns=keys + NUMERIC_COLUMNS + [
            "ctr_pct", "cpc", "cpa", "conversion_rate_pct", "roas", "cpm"
        ])
    keys = [by] if isinstance(by, str) else list(by or [])
    if not keys:
        totals = pd.concat(partials, ignore_index=True)[NUMERIC_COLUMNS].sum().to_frame().T
        return _add_ratios(totals)
    combined = pd.concat(partials, ignore_index=True)
    totals = combined.groupby(keys, dropna=False, observed=True)[NUMERIC_COLUMNS].sum().reset_index()
    return _add_ratios(totals)

def _add_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["ctr_pct"] = _ratio(frame["clicks"], frame["impressions"]) * 100
    frame["cpc"] = _ratio(frame["spend"], frame["clicks"])
    frame["cpa"] = _ratio(frame["spend"], frame["conversions"])
    frame["conversion_rate_pct"] = _ratio(frame["conversions"], frame["clicks"]) * 100
    frame["roas"] = _ratio(frame["revenue"], frame["spend"])
    frame["cpm"] = _ratio(frame["spend"], frame["impressions"]) * 1000
    return frame

def aggregate_csv(path: str | Path, group_by: list[str], chunksize: int = 500_000) -> pd.DataFrame:
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    partials = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        validate_columns(chunk.columns)
        missing = set(group_by) - set(chunk.columns)
        if missing:
            raise ValueError(f"Grouping columns missing: {', '.join(sorted(missing))}")
        partials.append(chunk.groupby(group_by, dropna=False, observed=True)[NUMERIC_COLUMNS].sum().reset_index())
    if not partials:
        empty = pd.read_csv(path, nrows=0)
        validate_columns(empty.columns)
        return pd.DataFrame(columns=group_by + NUMERIC_COLUMNS)
    combined = pd.concat(partials, ignore_index=True)
    return combined.groupby(group_by, dropna=False, observed=True)[NUMERIC_COLUMNS].sum().reset_index()

def format_kpi_table(frame: pd.DataFrame, index_cols: list[str]) -> str:
    display = frame.copy()
    for col in display.select_dtypes(include="number").columns:
        display[col] = display[col].map(lambda x: "—" if pd.isna(x) else f"{x:,.2f}")
    return display.to_string(index=False)
