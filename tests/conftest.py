import pandas as pd
import pytest

from data_generator import generate_csv, generate_dataframe


@pytest.fixture(scope="session")
def small_csv(tmp_path_factory):
    return generate_csv(5_000, tmp_path_factory.mktemp("data") / "ads_5000.csv", seed=11, chunk_size=1_700)


@pytest.fixture(scope="session")
def large_frame():
    return generate_dataframe(150_000, seed=5)


@pytest.fixture
def daily_frame():
    dates = pd.date_range("2025-01-01", periods=40)
    rows = []
    for platform, multiplier, exponent in [("A", 1.2, 0.8), ("B", 0.9, 0.7), ("C", 0.7, 0.9)]:
        for i, date in enumerate(dates):
            spend = 100 + 3 * i
            rows.append({"platform": platform, "date": date, "spend": spend, "revenue": multiplier * spend ** exponent})
    return pd.DataFrame(rows)
