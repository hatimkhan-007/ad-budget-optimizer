"""Single-pass, memory-bounded aggregation of advertising datasets.

A dataset of any size is read in fixed-size chunks and reduced to a handful of
small summary tables (campaign x day, and breakdowns by device, age group,
gender and city). Every analytic view in the dashboard is built from these
tables, so the raw rows are never held in memory all at once.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from campaigns import add_campaign_id
from data_quality import QualityReport, sanitize_chunk
from metrics_calculator import NUMERIC_COLUMNS, add_ratios, validate_columns

logger = logging.getLogger(__name__)

AGGREGATES_VERSION = 1
DEFAULT_CHUNKSIZE = 500_000
GROUP_SETS: dict[str, list[str]] = {
    "campaign_daily": ["platform", "campaign", "audience", "date"],
    "device": ["device"],
    "age": ["age"],
    "gender": ["gender"],
    "city": ["city"],
}
COUNT_COLUMNS = ["impressions", "clicks", "conversions"]


@dataclass
class Aggregates:
    """Summary tables produced from one dataset plus processing metadata."""

    frames: dict[str, pd.DataFrame]
    quality: QualityReport
    source: str
    elapsed_seconds: float
    chunks: int = 0
    from_cache: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def rows_processed(self) -> int:
        return self.quality.rows_valid

    @property
    def throughput_rows_per_second(self) -> float:
        return self.quality.rows_in / self.elapsed_seconds if self.elapsed_seconds > 0 else float("nan")


def summarize(frame: pd.DataFrame, by: str | Iterable[str] | None = None) -> pd.DataFrame:
    """Sum metrics (optionally grouped) and append CTR, CPC, CPA, CVR, ROAS and CPM."""
    missing = set(NUMERIC_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {', '.join(sorted(missing))}")
    if by is None:
        totals = frame[NUMERIC_COLUMNS].sum().to_frame().T
    else:
        keys = [by] if isinstance(by, str) else list(by)
        totals = frame.groupby(keys, observed=True)[NUMERIC_COLUMNS].sum().reset_index()
    return add_ratios(totals)


def _reduce(partials: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    if not partials:
        return pd.DataFrame(columns=keys + NUMERIC_COLUMNS)
    merged = pd.concat(partials, ignore_index=True).groupby(keys, observed=True)[NUMERIC_COLUMNS].sum().reset_index()
    merged[COUNT_COLUMNS] = merged[COUNT_COLUMNS].round().astype("int64")
    return merged


def build_aggregates(path: str | Path, chunksize: int = DEFAULT_CHUNKSIZE, numeric_missing: str = "drop",
                     progress: Callable[[int, int], None] | None = None) -> Aggregates:
    """Read a CSV in chunks, validate each chunk and reduce it to the summary tables."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    header = pd.read_csv(path, nrows=0)
    validate_columns(header.columns)
    absent = {c for keys in GROUP_SETS.values() for c in keys} - set(header.columns)
    if absent:
        raise ValueError(f"Dataset is missing columns required for analytics: {', '.join(sorted(absent))}")

    started = time.perf_counter()
    partials: dict[str, list[pd.DataFrame]] = {name: [] for name in GROUP_SETS}
    quality = QualityReport()
    total_bytes = path.stat().st_size
    chunks = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        clean, report = sanitize_chunk(chunk, numeric_missing)
        quality.merge(report)
        for name, keys in GROUP_SETS.items():
            partials[name].append(clean.groupby(keys, observed=True)[NUMERIC_COLUMNS].sum().reset_index())
        chunks += 1
        if progress is not None:
            progress(chunks, quality.rows_in)
    frames = {name: _reduce(parts, GROUP_SETS[name]) for name, parts in partials.items()}
    frames["campaign_daily"] = add_campaign_id(frames["campaign_daily"])
    elapsed = time.perf_counter() - started
    logger.info("Aggregated %s rows from %s in %.2fs (%d chunks)", f"{quality.rows_in:,}", path.name, elapsed, chunks)
    return Aggregates(frames, quality, str(path), elapsed, chunks, extras={"file_bytes": total_bytes})


def _fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"version": AGGREGATES_VERSION, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_or_build_aggregates(path: str | Path, cache_dir: str | Path = "cache",
                             chunksize: int = DEFAULT_CHUNKSIZE,
                             progress: Callable[[int, int], None] | None = None) -> Aggregates:
    """Return cached aggregates when the dataset is unchanged, otherwise rebuild and cache them."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    folder = Path(cache_dir) / path.stem
    meta_path = folder / "meta.json"
    fingerprint = _fingerprint(path)
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            files = {name: folder / f"{name}.csv" for name in GROUP_SETS}
            if meta.get("fingerprint") == fingerprint and all(f.exists() for f in files.values()):
                frames = {name: pd.read_csv(file) for name, file in files.items()}
                return Aggregates(frames, QualityReport.from_dict(meta["quality"]), str(path),
                                  float(meta["elapsed_seconds"]), int(meta.get("chunks", 0)), from_cache=True,
                                  extras=meta.get("extras", {}))
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        logger.warning("Ignoring unreadable aggregate cache for %s: %s", path.name, error)

    result = build_aggregates(path, chunksize, progress=progress)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in result.frames.items():
            frame.to_csv(folder / f"{name}.csv", index=False)
        meta_path.write_text(json.dumps({
            "fingerprint": fingerprint, "quality": result.quality.to_dict(),
            "elapsed_seconds": result.elapsed_seconds, "chunks": result.chunks, "extras": result.extras,
        }), encoding="utf-8")
    except OSError as error:
        logger.warning("Could not write aggregate cache for %s: %s", path.name, error)
    return result
