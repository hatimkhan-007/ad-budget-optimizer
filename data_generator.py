"""Synthetic digital advertising data generator.

Produces campaign-level records (10K to 10M+ rows) with realistic funnel
behaviour: impressions -> clicks -> conversions -> revenue. Each platform has
its own CPM, CTR, conversion rate, order value and a saturation exponent that
makes returns diminish as daily spend rises. That structure is what the budget
optimizer later recovers from the data.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "date", "campaign", "platform", "impressions", "clicks", "spend",
    "conversions", "revenue", "age", "gender", "city", "device",
]

DTYPES = {
    "date": "category",
    "campaign": "category",
    "platform": "category",
    "impressions": "int64",
    "clicks": "int64",
    "spend": "float64",
    "conversions": "int64",
    "revenue": "float64",
    "age": "int16",
    "gender": "category",
    "city": "category",
    "device": "category",
}

SCALES = [10_000, 100_000, 1_000_000, 10_000_000]
START_DATE = "2025-01-01"
DAYS = 365
DEFAULT_SEED = 42
DEFAULT_CHUNK_SIZE = 500_000


@dataclass(frozen=True)
class PlatformProfile:
    name: str
    row_share: float
    cpm: float
    ctr: float
    cvr: float
    aov: float
    saturation: float
    mobile_share: float


PLATFORMS = [
    PlatformProfile("Facebook", 0.30, 8.5, 0.011, 0.026, 62.0, 0.25, 0.80),
    PlatformProfile("Instagram", 0.26, 9.8, 0.009, 0.022, 68.0, 0.20, 0.92),
    PlatformProfile("Google", 0.14, 22.0, 0.028, 0.032, 72.0, 0.40, 0.55),
    PlatformProfile("TikTok", 0.18, 5.5, 0.013, 0.012, 45.0, 0.12, 0.97),
    PlatformProfile("YouTube", 0.12, 9.0, 0.0075, 0.025, 72.0, 0.15, 0.60),
]

CAMPAIGNS = {
    "Brand_Awareness": (1.10, 0.70),
    "Retargeting": (1.30, 1.60),
    "Spring_Sale": (1.05, 1.10),
    "Summer_Promo": (1.00, 1.00),
    "Back_To_School": (0.95, 1.05),
    "Black_Friday": (1.20, 1.25),
    "Holiday_Season": (1.10, 1.15),
    "Product_Launch": (1.15, 0.90),
}

CITIES = [
    "New York", "London", "Dubai", "Lahore", "Karachi",
    "Toronto", "Sydney", "Singapore", "Berlin", "Mumbai",
]
GENDERS = ["Male", "Female", "Other"]
GENDER_WEIGHTS = [0.48, 0.48, 0.04]
DEVICES = ["Mobile", "Desktop", "Tablet"]
DEVICE_CTR = np.array([1.00, 0.90, 0.95])
DEVICE_CVR = np.array([1.00, 1.25, 0.90])


def _date_labels() -> np.ndarray:
    dates = pd.date_range(START_DATE, periods=DAYS)
    return dates.strftime("%Y-%m-%d").to_numpy()


def _demand_matrix(seed: int) -> np.ndarray:
    """Platform x day demand multiplier (weekly, seasonal and random shocks)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=DAYS)
    weekly = np.where(dates.dayofweek >= 5, 1.10, 1.0)
    month = dates.month.to_numpy()
    seasonal = np.where(np.isin(month, [11, 12]), 1.30,
                        np.where(np.isin(month, [6, 7]), 0.92, 1.0))
    sigma = 0.35
    shocks = rng.lognormal(-0.5 * sigma**2, sigma, size=(len(PLATFORMS), DAYS))
    return shocks * (weekly * seasonal)[None, :]


def _generate_chunk(n: int, rng: np.random.Generator, demand: np.ndarray,
                    labels: np.ndarray) -> pd.DataFrame:
    n_platforms = len(PLATFORMS)
    campaign_names = list(CAMPAIGNS)

    cpm = np.array([p.cpm for p in PLATFORMS])
    ctr = np.array([p.ctr for p in PLATFORMS])
    cvr = np.array([p.cvr for p in PLATFORMS])
    aov = np.array([p.aov for p in PLATFORMS])
    sat = np.array([p.saturation for p in PLATFORMS])
    mobile = np.array([p.mobile_share for p in PLATFORMS])
    weights = np.array([p.row_share for p in PLATFORMS])
    camp_ctr = np.array([CAMPAIGNS[c][0] for c in campaign_names])
    camp_cvr = np.array([CAMPAIGNS[c][1] for c in campaign_names])

    p_idx = rng.choice(n_platforms, size=n, p=weights / weights.sum())
    d_idx = rng.integers(0, DAYS, size=n)
    c_idx = rng.integers(0, len(campaign_names), size=n)
    g_idx = rng.choice(len(GENDERS), size=n, p=GENDER_WEIGHTS)
    city_idx = rng.integers(0, len(CITIES), size=n)
    dev_idx = np.where(
        rng.random(n) < mobile[p_idx], 0, np.where(rng.random(n) < 0.8, 1, 2)
    )
    age = np.clip(np.rint(rng.normal(32, 10, size=n)), 18, 65).astype(np.int16)

    demand_factor = demand[p_idx, d_idx]
    base_impressions = rng.lognormal(np.log(4000), 0.9, size=n)
    impressions = np.maximum(1, np.rint(base_impressions * demand_factor)).astype(np.int64)
    spend = np.round(
        impressions * cpm[p_idx] / 1000.0 * rng.lognormal(0, 0.12, size=n), 2
    )

    ctr_eff = np.clip(
        ctr[p_idx] * camp_ctr[c_idx] * DEVICE_CTR[dev_idx]
        * rng.lognormal(0, 0.15, size=n), 0, 1)
    clicks = rng.binomial(impressions, ctr_eff).astype(np.int64)

    age_mult = np.where((age >= 25) & (age <= 44), 1.10, 1.0)
    cvr_eff = np.clip(
        cvr[p_idx] * camp_cvr[c_idx] * DEVICE_CVR[dev_idx] * age_mult
        * np.power(demand_factor, -sat[p_idx]) * rng.lognormal(0, 0.15, size=n), 0, 1)
    conversions = rng.binomial(clicks, cvr_eff).astype(np.int64)
    revenue = np.round(conversions * aov[p_idx] * rng.lognormal(0, 0.15, size=n), 2)

    platform_names = [p.name for p in PLATFORMS]
    frame = pd.DataFrame({
        "date": pd.Categorical.from_codes(d_idx, categories=labels),
        "campaign": pd.Categorical.from_codes(c_idx, categories=campaign_names),
        "platform": pd.Categorical.from_codes(p_idx, categories=platform_names),
        "impressions": impressions,
        "clicks": clicks,
        "spend": spend,
        "conversions": conversions,
        "revenue": revenue,
        "age": age,
        "gender": pd.Categorical.from_codes(g_idx, categories=GENDERS),
        "city": pd.Categorical.from_codes(city_idx, categories=CITIES),
        "device": pd.Categorical.from_codes(dev_idx, categories=DEVICES),
    })
    return frame[COLUMNS]


def _chunk_rng(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index + 1,)))


def generate_dataframe(n_records: int, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Generate an in-memory dataset (suitable up to a few million rows)."""
    return _generate_chunk(n_records, _chunk_rng(seed, 0), _demand_matrix(seed), _date_labels())


def generate_csv(n_records: int, path: str | Path, seed: int = DEFAULT_SEED,
                 chunk_size: int = DEFAULT_CHUNK_SIZE, verbose: bool = True) -> Path:
    """Stream a dataset of any size to CSV in fixed-size chunks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    demand = _demand_matrix(seed)
    labels = _date_labels()
    start = time.perf_counter()
    written = 0
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        index = 0
        while written < n_records:
            size = min(chunk_size, n_records - written)
            chunk = _generate_chunk(size, _chunk_rng(seed, index), demand, labels)
            chunk.to_csv(handle, index=False, header=(index == 0), lineterminator="\n")
            written += size
            index += 1
            if verbose and n_records >= 5 * chunk_size:
                print(f"  {written:>12,} / {n_records:,} rows", flush=True)
    os.replace(tmp_path, path)
    if verbose:
        mb = path.stat().st_size / 1e6
        print(f"Generated {n_records:,} rows -> {path} ({mb:,.1f} MB) in "
              f"{time.perf_counter() - start:.1f}s")
    return path


def dataset_path(data_dir: str | Path, n_records: int) -> Path:
    return Path(data_dir) / f"marketing_{n_records}.csv"


def ensure_dataset(n_records: int, data_dir: str | Path = "data", seed: int = DEFAULT_SEED,
                   regenerate: bool = False, verbose: bool = True) -> Path:
    """Return the CSV for a given scale, generating it if it does not exist."""
    path = dataset_path(data_dir, n_records)
    if path.exists() and not regenerate:
        if verbose:
            print(f"Using existing dataset {path}")
        return path
    return generate_csv(n_records, path, seed=seed, verbose=verbose)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ad campaign data.")
    parser.add_argument("--records", type=int, nargs="+", default=SCALES[:3],
                        help="Row counts to generate (default: 10K, 100K, 1M).")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    for n in args.records:
        ensure_dataset(n, args.output_dir, args.seed, args.regenerate)


if __name__ == "__main__":
    main()