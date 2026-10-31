from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SEED = 42
DEFAULT_CHUNK_SIZE = 500_000
GENERATOR_VERSION = 3
DAYS = 365
START_DATE = pd.Timestamp("2025-01-01")
SCALES = [10_000, 100_000, 1_000_000, 10_000_000]

PLATFORMS = {
    "Facebook": {"cpm": 8.6, "ctr": 0.012, "cvr": 0.031, "aov": 62, "base": 1.00, "sat": 0.14},
    "Google": {"cpm": 22.2, "ctr": 0.030, "cvr": 0.039, "aov": 73, "base": 1.10, "sat": 0.10},
    "Instagram": {"cpm": 9.9, "ctr": 0.010, "cvr": 0.026, "aov": 69, "base": 0.98, "sat": 0.18},
    "TikTok": {"cpm": 5.5, "ctr": 0.0145, "cvr": 0.014, "aov": 46, "base": 0.95, "sat": 0.24},
    "YouTube": {"cpm": 9.1, "ctr": 0.008, "cvr": 0.032, "aov": 73, "base": 0.90, "sat": 0.20},
}
PLATFORM_WEIGHTS = [0.25, 0.25, 0.20, 0.15, 0.15]

CAMPAIGNS = ["Brand Awareness", "Conversions", "Retargeting", "Lead Generation", "Product Launch"]
AUDIENCES = ["New Customers", "Returning Customers", "Lookalike", "Interest Based"]
DEVICES = ["Mobile", "Desktop", "Tablet"]
DEVICE_WEIGHTS = [0.72, 0.23, 0.05]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_WEIGHTS = [0.22, 0.34, 0.22, 0.13, 0.09]
GENDERS = ["Female", "Male", "Unspecified"]
GENDER_WEIGHTS = [0.48, 0.49, 0.03]
CITIES = ["Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar"]
CITY_WEIGHTS = [0.28, 0.26, 0.12, 0.10, 0.09, 0.08, 0.07]

WEEKDAY_FACTOR = np.array([1.00, 1.03, 1.04, 1.05, 1.02, 0.90, 0.86])
MONTH_FACTOR = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.08, 1.28, 1.22])
SEASONAL_AMPLITUDE = 0.12
DAILY_DEMAND_SIGMA = 0.25
ROW_DEMAND_SIGMA = 0.10

COLUMN_DTYPES = {
    "date": "string",
    "campaign": "string",
    "platform": "string",
    "audience": "string",
    "device": "string",
    "age": "string",
    "gender": "string",
    "city": "string",
    "impressions": "int64",
    "clicks": "int64",
    "spend": "float64",
    "conversions": "int64",
    "revenue": "float64",
}
COLUMNS = list(COLUMN_DTYPES)


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
    return _generate_chunk(n_records, rng, _demand_table(rng))


def _demand_table(rng: np.random.Generator) -> np.ndarray:
    """Daily demand level per (day, platform): weekday, month and seasonal effects plus a shared daily shock.

    Every record on the same day and platform shares this level, so daily platform totals vary with demand
    and conversion efficiency falls as demand rises (the saturation exponent), producing diminishing returns.
    """
    days = START_DATE + pd.to_timedelta(np.arange(DAYS), unit="D")
    weekday = WEEKDAY_FACTOR[days.dayofweek.to_numpy()]
    monthly = MONTH_FACTOR[days.month.to_numpy() - 1]
    seasonal = 1 + SEASONAL_AMPLITUDE * np.sin(2 * np.pi * np.arange(DAYS) / DAYS)
    shocks = np.exp(rng.normal(0, DAILY_DEMAND_SIGMA, (DAYS, len(PLATFORMS))))
    return (weekday * monthly * seasonal)[:, None] * shocks


def _generate_chunk(n: int, rng: np.random.Generator, table: np.ndarray) -> pd.DataFrame:
    platform_names = list(PLATFORMS)
    platform_idx = rng.choice(len(platform_names), n, p=PLATFORM_WEIGHTS)
    platform = np.array(platform_names)[platform_idx]
    campaign = rng.choice(CAMPAIGNS, n)
    audience = rng.choice(AUDIENCES, n)
    device = rng.choice(DEVICES, n, p=DEVICE_WEIGHTS)
    age = rng.choice(AGE_GROUPS, n, p=AGE_WEIGHTS)
    gender = rng.choice(GENDERS, n, p=GENDER_WEIGHTS)
    city = rng.choice(CITIES, n, p=CITY_WEIGHTS)

    day_index = rng.integers(0, DAYS, n)
    dates = START_DATE + pd.to_timedelta(day_index, unit="D")
    demand = table[day_index, platform_idx] * np.exp(rng.normal(0, ROW_DEMAND_SIGMA, n))

    base = np.array([PLATFORMS[p]["base"] for p in platform])
    cpm = np.array([PLATFORMS[p]["cpm"] for p in platform])
    ctr = np.array([PLATFORMS[p]["ctr"] for p in platform])
    cvr = np.array([PLATFORMS[p]["cvr"] for p in platform])
    aov = np.array([PLATFORMS[p]["aov"] for p in platform])
    sat = np.array([PLATFORMS[p]["sat"] for p in platform])

    impressions = np.maximum(1, rng.lognormal(7.6, 0.85, n) * demand * base).astype(np.int64)
    spend = np.maximum(0.01, impressions / 1000 * cpm * rng.lognormal(0, 0.08, n))
    click_rate = np.clip(ctr * rng.lognormal(0, 0.08, n), 0.0001, 0.5)
    clicks = rng.binomial(impressions, click_rate).astype(np.int64)
    efficiency = np.clip(demand ** -sat, 0.65, 1.25)
    conversions = rng.binomial(clicks, np.clip(cvr * efficiency, 0.0001, 0.8)).astype(np.int64)
    revenue = conversions * aov * rng.lognormal(0, 0.12, n)

    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "campaign": campaign,
        "platform": platform,
        "audience": audience,
        "device": device,
        "age": age,
        "gender": gender,
        "city": city,
        "impressions": impressions,
        "clicks": clicks,
        "spend": np.round(spend, 2),
        "conversions": conversions,
        "revenue": np.round(revenue, 2),
    })


def generate_csv(n_records: int, path: str | Path, seed: int = DEFAULT_SEED,
                 chunk_size: int = DEFAULT_CHUNK_SIZE) -> Path:
    _validate_n(n_records, chunk_size)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if n_records == 0:
        pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)
        return path
    table = _demand_table(rng)
    remaining, first = n_records, True
    while remaining:
        count = min(chunk_size, remaining)
        _generate_chunk(count, rng, table).to_csv(path, index=False, mode="w" if first else "a", header=first)
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
    expected = {"n_records": n_records, "seed": seed, "generator_version": GENERATOR_VERSION}
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic advertising campaign dataset.")
    parser.add_argument("--records", type=int, default=SCALES[0], help="Number of records to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--regenerate", action="store_true", help="Rebuild the file even if a valid one exists.")
    args = parser.parse_args(argv)
    if args.records < 0:
        parser.error("--records must be non-negative")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    path = ensure_dataset(args.records, args.data_dir, args.seed, args.regenerate)
    print(f"Dataset ready: {path} ({args.records:,} records, seed {args.seed})")


if __name__ == "__main__":
    main()
