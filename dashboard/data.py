"""Dataset discovery and cached loading for the dashboard."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics import Aggregates, load_or_build_aggregates

DATA_DIR = Path(os.environ.get("AD_OPT_DATA_DIR", "data"))
CACHE_DIR = Path(os.environ.get("AD_OPT_CACHE_DIR", "cache"))
RESULTS_DIR = Path(os.environ.get("AD_OPT_RESULTS_DIR", "results"))
PREFERRED_DEFAULT_RECORDS = 100_000


@dataclass(frozen=True)
class DatasetInfo:
    records: int
    path: Path
    size_mb: float
    modified: datetime

    @property
    def label(self) -> str:
        return f"{self.records:,} records ({self.size_mb:,.1f} MB)"


def list_datasets(data_dir: Path | None = None) -> list[DatasetInfo]:
    """Datasets named ``ads_<records>.csv`` in the data directory, smallest first."""
    folder = Path(data_dir) if data_dir is not None else DATA_DIR
    found = []
    for path in folder.glob("ads_*.csv"):
        suffix = path.stem.removeprefix("ads_")
        if not suffix.isdigit():
            continue
        stat = path.stat()
        found.append(DatasetInfo(int(suffix), path, stat.st_size / 1e6, datetime.fromtimestamp(stat.st_mtime)))
    return sorted(found, key=lambda d: d.records)


def default_dataset_index(datasets: list[DatasetInfo]) -> int:
    for i, dataset in enumerate(datasets):
        if dataset.records == PREFERRED_DEFAULT_RECORDS:
            return i
    below = [i for i, d in enumerate(datasets) if d.records <= 1_000_000]
    return below[-1] if below else 0


@st.cache_data(show_spinner="Processing the dataset in chunks (first load only)...")
def cached_aggregates(path: str, size: int, mtime_ns: int) -> Aggregates:
    """Streamlit-cached wrapper; ``size`` and ``mtime_ns`` invalidate the cache when the file changes."""
    return load_or_build_aggregates(path, CACHE_DIR)


def load_aggregates(dataset: DatasetInfo) -> Aggregates:
    stat = dataset.path.stat()
    return cached_aggregates(str(dataset.path), stat.st_size, stat.st_mtime_ns)


def with_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out
