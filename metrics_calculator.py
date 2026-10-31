from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data_generator import COLUMN_DTYPES

REQUIRED_COLUMNS = {"platform", "campaign", "date", "impressions", "clicks", "spend", "conversions", "revenue"}
NUMERIC_COLUMNS = ["impressions", "clicks", "spend", "conversions", "revenue"]
COUNT_COLUMNS = ["impressions", "clicks", "conversions"]
RATIO_COLUMNS = ["ctr_pct", "cpc", "cpa", "conversion_rate_pct", "roas", "cpm"]


def validate_columns(columns: Iterable[str]) -> None:
    missing = REQUIRED_COLUMNS - set(columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")


def check_values(data: pd.DataFrame) -> dict[str, int]:
    """Reject negative values and warn about funnel violations (clicks > impressions, conversions > clicks)."""
    for column in NUMERIC_COLUMNS:
        if (data[column] < 0).any():
            raise ValueError(f"Column '{column}' contains negative values.")
    report = {
        "clicks_exceed_impressions": int((data["clicks"] > data["impressions"]).sum()),
        "conversions_exceed_clicks": int((data["conversions"] > data["clicks"]).sum()),
    }
    if any(report.values()):
        warnings.warn(f"Funnel inconsistencies detected: {report}", UserWarning, stacklevel=2)
    return report


def validate_data(data: pd.DataFrame) -> dict[str, int]:
    validate_columns(data.columns)
    return check_values(data)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.div(denominator.where(denominator != 0))
    return result.replace([np.inf, -np.inf], np.nan)


def _add_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["ctr_pct"] = _ratio(frame["clicks"], frame["impressions"]) * 100
    frame["cpc"] = _ratio(frame["spend"], frame["clicks"])
    frame["cpa"] = _ratio(frame["spend"], frame["conversions"])
    frame["conversion_rate_pct"] = _ratio(frame["conversions"], frame["clicks"]) * 100
    frame["roas"] = _ratio(frame["revenue"], frame["spend"])
    frame["cpm"] = _ratio(frame["spend"], frame["impressions"]) * 1000
    return frame


def _keys(by: str | list[str] | None) -> list[str]:
    return [by] if isinstance(by, str) else list(by or [])


def _sum_by(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if keys:
        return data.groupby(keys, dropna=False, observed=True)[NUMERIC_COLUMNS].sum().reset_index()
    return data[NUMERIC_COLUMNS].sum().to_frame().T


def compute_kpis(data: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    validate_columns(data.columns)
    keys = _keys(by)
    missing = set(keys) - set(data.columns)
    if missing:
        raise ValueError(f"Grouping columns missing: {', '.join(sorted(missing))}")
    return _add_ratios(_sum_by(data, keys))


def _read_chunks(path: Path, columns: list[str], chunksize: int):
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0)
    validate_columns(header.columns)
    missing = set(columns) - set(header.columns)
    if missing:
        raise ValueError(f"Grouping columns missing: {', '.join(sorted(missing))}")
    dtypes = {c: COLUMN_DTYPES[c] for c in columns if c in COLUMN_DTYPES and COLUMN_DTYPES[c] != "string"}
    return pd.read_csv(path, usecols=columns, dtype=dtypes, chunksize=chunksize)


def aggregate_csv(path: str | Path, group_by: list[str], chunksize: int = 500_000) -> pd.DataFrame:
    path = Path(path)
    columns = list(dict.fromkeys(list(group_by) + NUMERIC_COLUMNS))
    partials = []
    for chunk in _read_chunks(path, columns, chunksize):
        check_values(chunk)
        partials.append(_sum_by(chunk, list(group_by)))
    if not partials:
        return pd.DataFrame(columns=list(group_by) + NUMERIC_COLUMNS)
    return _sum_by(pd.concat(partials, ignore_index=True), list(group_by))


def kpis_from_csv(path: str | Path, by: str | list[str] | None = None, chunksize: int = 500_000) -> pd.DataFrame:
    keys = _keys(by)
    totals = aggregate_csv(path, keys, chunksize) if keys else _total_from_csv(Path(path), chunksize)
    if totals.empty:
        return pd.DataFrame(columns=keys + NUMERIC_COLUMNS + RATIO_COLUMNS)
    return _add_ratios(totals)


def _total_from_csv(path: Path, chunksize: int) -> pd.DataFrame:
    partials = []
    for chunk in _read_chunks(path, NUMERIC_COLUMNS, chunksize):
        check_values(chunk)
        partials.append(chunk.sum().to_frame().T)
    if not partials:
        return pd.DataFrame(columns=NUMERIC_COLUMNS)
    return pd.concat(partials, ignore_index=True).sum().to_frame().T


def format_kpi_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        decimals = 0 if column in COUNT_COLUMNS else 2
        display[column] = display[column].map(lambda x, d=decimals: "—" if pd.isna(x) else f"{x:,.{d}f}")
    return display.to_string(index=False)
