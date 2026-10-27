from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

DEFAULT_SEED = 42
DEFAULT_CHUNK_SIZE = 500_000
DAYS = 365
SCALES = [10_000, 100_000, 1_000_000, 10_000_000]
PLATFORMS = {
    "Facebook": {"cpm": 8.6, "ctr": .012, "cvr": .031, "aov": 62, "base": 1.0},
    "Google": {"cpm": 22.2, "ctr": .030, "cvr": .039, "aov": 73, "base": 1.1},
    "Instagram": {"cpm": 9.9, "ctr": .010, "cvr": .026, "aov": 69, "base": .98},
    "TikTok": {"cpm": 5.5, "ctr": .0145, "cvr": .014, "aov": 46, "base": .95},
    "YouTube": {"cpm": 9.1, "ctr": .008, "cvr": .032, "aov": 73, "base": .9},
}
CAMPAIGNS = ["Brand Awareness", "Conversions", "Retargeting", "Lead Generation", "Product Launch"]
DEVICES = ["Mobile", "Desktop", "Tablet"]
AUDIENCES = ["New Customers", "Returning Customers", "Lookalike", "Interest Based"]

COLUMNS = ["date", "platform", "campaign", "audience", "device", "impressions",
           "clicks", "spend", "conversions", "revenue"]

def _validate_n(n_records: int, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    if not isinstance(n_records, int) or isinstance(n_records, bool) or n_records < 0:
        raise ValueError("n_records must be a non-negative integer")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

def generate_dataframe(n_records: int, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    _validate_n(n_records)
    rng = np.random.default_rng(seed)
    if n_records == 0:
        return pd.DataFrame(columns=COLUMNS)
    return _generate_chunk(n_records, rng)

def _generate_chunk(n: int, rng: np.random.Generator) -> pd.DataFrame:
    platform_names = np.array(list(PLATFORMS))
    platform = rng.choice(platform_names, n, p=[.25, .25, .20, .15, .15])
    campaign = rng.choice(CAMPAIGNS, n)
    audience = rng.choice(AUDIENCES, n)
    device = rng.choice(DEVICES, n, p=[.72, .23, .05])
    day_index = rng.integers(0, DAYS, n)
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(day_index, unit="D")
    seasonality = 1 + .18 * np.sin(2 * np.pi * day_index / DAYS)
    demand = np.exp(rng.normal(0, .20, n)) * seasonality
    platform_base = np.array([PLATFORMS[p]["base"] for p in platform])
    impressions = np.maximum(1, (rng.lognormal(7.6, .85, n) * demand * platform_base)).astype(np.int64)
    cpm = np.array([PLATFORMS[p]["cpm"] for p in platform])
    spend = np.maximum(.01, impressions / 1000 * cpm * rng.lognormal(0, .08, n))
    ctr = np.array([PLATFORMS[p]["ctr"] for p in platform])
    clicks = rng.binomial(impressions.clip(max=2_000_000), np.clip(ctr * rng.lognormal(0, .08, n), .0001, .5)).astype(np.int64)
    cvr = np.array([PLATFORMS[p]["cvr"] for p in platform])
    saturation = np.clip(demand ** -.12, .65, 1.25)
    conversions = rng.binomial(clicks, np.clip(cvr * saturation, .0001, .8)).astype(np.int64)
    aov = np.array([PLATFORMS[p]["aov"] for p in platform])
    revenue = conversions * aov * rng.lognormal(0, .12, n)
    return pd.DataFrame({"date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
        "platform": platform, "campaign": campaign, "audience": audience, "device": device,
        "impressions": impressions, "clicks": clicks, "spend": np.round(spend, 2),
        "conversions": conversions, "revenue": np.round(revenue, 2)})

def generate_csv(n_records: int, path: str | Path, seed: int = DEFAULT_SEED,
                 chunk_size: int = DEFAULT_CHUNK_SIZE) -> Path:
    _validate_n(n_records, chunk_size)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    remaining = n_records
    first = True
    if n_records == 0:
        pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
    while remaining:
        count = min(chunk_size, remaining)
        chunk = _generate_chunk(count, rng)
        chunk.to_csv(path, index=False, mode="w" if first else "a", header=first)
        first = False
        remaining -= count
    return path

def ensure_dataset(n_records: int, data_dir: str | Path = "data", seed: int = DEFAULT_SEED,
                   regenerate: bool = False) -> Path:
    _validate_n(n_records)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"ads_{n_records}.csv"
    metadata_path = data_dir / f"ads_{n_records}.metadata.json"
    expected = {"n_records": n_records, "seed": seed, "generator_version": 2}
    valid = False
    if path.exists() and metadata_path.exists() and not regenerate:
        try:
            valid = json.loads(metadata_path.read_text()) == expected
        except (OSError, json.JSONDecodeError):
            valid = False
    if regenerate or not valid:
        generate_csv(n_records, path, seed)
        metadata_path.write_text(json.dumps(expected, indent=2))
    return path
