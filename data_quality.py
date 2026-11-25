"""Data validation, cleaning and memory optimisation for advertising records."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from metrics_calculator import NUMERIC_COLUMNS, validate_columns

logger = logging.getLogger(__name__)

KEY_COLUMNS = ["date", "platform", "campaign"]
DIMENSION_COLUMNS = ["audience", "device", "age", "gender", "city"]
UNKNOWN_LABEL = "Unknown"


@dataclass
class QualityReport:
    """Counts of problems found while validating a dataset."""

    rows_in: int = 0
    rows_valid: int = 0
    missing_key_rows: int = 0
    missing_numeric_rows: int = 0
    negative_rows: int = 0
    funnel_violation_rows: int = 0
    duplicate_rows: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def rows_rejected(self) -> int:
        return self.rows_in - self.rows_valid

    def merge(self, other: "QualityReport") -> "QualityReport":
        for name in ("rows_in", "rows_valid", "missing_key_rows", "missing_numeric_rows",
                     "negative_rows", "funnel_violation_rows", "duplicate_rows"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.issues.extend(i for i in other.issues if i not in self.issues)
        return self

    def to_dict(self) -> dict:
        data = asdict(self)
        data["rows_rejected"] = self.rows_rejected
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "QualityReport":
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in names})


def sanitize_chunk(frame: pd.DataFrame, numeric_missing: str = "drop") -> tuple[pd.DataFrame, QualityReport]:
    """Return rows that pass validation, plus a report of what was rejected.

    Rules: rows missing a key field (date, platform, campaign) are dropped; missing
    numeric metrics are dropped (``numeric_missing="drop"``) or treated as zero
    (``"zero"``); negative metrics and funnel violations (clicks > impressions,
    conversions > clicks) are dropped; missing descriptive dimensions become "Unknown".
    """
    if numeric_missing not in {"drop", "zero"}:
        raise ValueError("numeric_missing must be 'drop' or 'zero'")
    validate_columns(frame.columns)
    report = QualityReport(rows_in=len(frame))
    if frame.empty:
        return frame.copy(), report

    data = frame.copy()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data.loc[~np.isfinite(data[column].astype("float64")), column] = np.nan

    missing_key = data[KEY_COLUMNS].isna().any(axis=1)
    missing_numeric = data[NUMERIC_COLUMNS].isna().any(axis=1)
    if numeric_missing == "zero":
        data[NUMERIC_COLUMNS] = data[NUMERIC_COLUMNS].fillna(0)
        missing_drop = pd.Series(False, index=data.index)
    else:
        missing_drop = missing_numeric
    negative = (data[NUMERIC_COLUMNS] < 0).any(axis=1)
    funnel = (data["clicks"] > data["impressions"]) | (data["conversions"] > data["clicks"])

    keep = ~(missing_key | missing_drop | negative | funnel)
    report.missing_key_rows = int(missing_key.sum())
    report.missing_numeric_rows = int(missing_numeric.sum())
    report.negative_rows = int(negative.sum())
    report.funnel_violation_rows = int(funnel.sum())
    report.rows_valid = int(keep.sum())

    data = data.loc[keep].copy()
    for column in DIMENSION_COLUMNS:
        if column in data:
            data[column] = data[column].astype("object").where(data[column].notna(), UNKNOWN_LABEL)
    for column in ("impressions", "clicks", "conversions"):
        data[column] = data[column].astype("int64")
    if report.rows_rejected:
        report.issues.append(f"{report.rows_rejected:,} of {report.rows_in:,} rows failed validation.")
    return data, report


def clean_dataframe(frame: pd.DataFrame, drop_duplicates: bool = True,
                    numeric_missing: str = "drop") -> tuple[pd.DataFrame, QualityReport]:
    """Validate, optionally de-duplicate and return a cleaned in-memory frame."""
    cleaned, report = sanitize_chunk(frame, numeric_missing)
    if drop_duplicates and not cleaned.empty:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        report.duplicate_rows = before - len(cleaned)
        report.rows_valid = len(cleaned)
        if report.duplicate_rows:
            report.issues.append(f"{report.duplicate_rows:,} exact duplicate rows removed.")
    return cleaned, report


def optimize_dtypes(frame: pd.DataFrame, categorical_threshold: float = 0.5) -> pd.DataFrame:
    """Reduce memory: low-cardinality text becomes categorical, integers are downcast."""
    out = frame.copy()
    for column in out.columns:
        series = out[column]
        if pd.api.types.is_integer_dtype(series):
            out[column] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            out[column] = pd.to_numeric(series, downcast="float") if column not in NUMERIC_COLUMNS else series
        elif (series.dtype == object or pd.api.types.is_string_dtype(series)) and len(series):
            if series.nunique(dropna=True) / len(series) <= categorical_threshold:
                out[column] = series.astype("category")
    return out


def assess_csv(path: str | Path, chunksize: int = 500_000, numeric_missing: str = "drop") -> QualityReport:
    """Chunked quality assessment of a CSV, including global exact-duplicate detection.

    Row hashes (8 bytes each) are collected so duplicates are found across chunks
    without holding the data itself in memory.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    total = QualityReport()
    hashes: list[np.ndarray] = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        _, report = sanitize_chunk(chunk, numeric_missing)
        total.merge(report)
        hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
    if hashes:
        combined = np.concatenate(hashes)
        total.duplicate_rows = int(len(combined) - len(np.unique(combined)))
        if total.duplicate_rows:
            total.issues.append(f"{total.duplicate_rows:,} rows are exact duplicates of another row.")
    logger.info("Quality assessment of %s: %s", path.name, total.to_dict())
    return total
