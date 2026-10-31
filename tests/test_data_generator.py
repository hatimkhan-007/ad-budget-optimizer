import json

import pandas as pd
import pytest

import data_generator as dg


def test_generation_is_reproducible():
    pd.testing.assert_frame_equal(dg.generate_dataframe(500, 7), dg.generate_dataframe(500, 7))


def test_different_seeds_differ():
    assert not dg.generate_dataframe(500, 1).equals(dg.generate_dataframe(500, 2))


def test_schema_and_demographics(large_frame):
    assert list(large_frame.columns) == dg.COLUMNS
    assert {"age", "gender", "city"} <= set(large_frame.columns)
    assert set(large_frame["platform"]) == set(dg.PLATFORMS)
    assert set(large_frame["age"]) <= set(dg.AGE_GROUPS)
    assert set(large_frame["city"]) <= set(dg.CITIES)


def test_funnel_invariant_and_non_negative(large_frame):
    assert (large_frame.conversions <= large_frame.clicks).all()
    assert (large_frame.clicks <= large_frame.impressions).all()
    assert (large_frame[["impressions", "clicks", "spend", "conversions", "revenue"]] >= 0).all().all()


def test_weekend_demand_is_lower(large_frame):
    weekday = pd.to_datetime(large_frame["date"]).dt.dayofweek
    volume = large_frame.groupby(weekday)["impressions"].sum() / large_frame.groupby(weekday).size()
    assert volume.loc[[5, 6]].mean() < volume.loc[[0, 1, 2, 3, 4]].mean()


def test_fourth_quarter_peak(large_frame):
    month = pd.to_datetime(large_frame["date"]).dt.month
    per_row = large_frame.groupby(month)["impressions"].sum() / large_frame.groupby(month).size()
    assert per_row.loc[[11, 12]].mean() > per_row.loc[[1, 2, 3]].mean()


def test_saturation_lowers_conversion_efficiency_on_busy_days(large_frame):
    google = large_frame[large_frame.platform == "Google"].copy()
    google["rate"] = google.conversions / google.clicks.where(google.clicks > 0)
    google["busy"] = google.impressions > google.impressions.median()
    assert google.groupby("busy")["rate"].mean().loc[True] < google.groupby("busy")["rate"].mean().loc[False]


def test_zero_records_has_schema(tmp_path):
    path = dg.generate_csv(0, tmp_path / "empty.csv")
    assert list(pd.read_csv(path).columns) == dg.COLUMNS


def test_chunked_csv_matches_record_count(small_csv):
    assert len(pd.read_csv(small_csv)) == 5_000


def test_ensure_dataset_reuses_and_invalidates(tmp_path):
    path = dg.ensure_dataset(300, tmp_path, seed=3)
    stamp = path.stat().st_mtime_ns
    assert dg.ensure_dataset(300, tmp_path, seed=3).stat().st_mtime_ns == stamp
    meta = tmp_path / "ads_300.metadata.json"
    meta.write_text(json.dumps({"n_records": 300, "seed": 3, "generator_version": 0}))
    dg.ensure_dataset(300, tmp_path, seed=3)
    assert json.loads(meta.read_text())["generator_version"] == dg.GENERATOR_VERSION


@pytest.mark.parametrize("bad", [-1, 1.5, True, "10"])
def test_invalid_record_counts(bad):
    with pytest.raises(ValueError):
        dg.generate_dataframe(bad)


def test_cli_creates_dataset(tmp_path):
    dg.main(["--records", "250", "--data-dir", str(tmp_path)])
    assert (tmp_path / "ads_250.csv").exists()
