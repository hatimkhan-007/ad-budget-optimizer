import numpy as np
import pandas as pd
import pytest

import metrics_calculator as mc


def test_kpi_formulas(large_frame):
    out = mc.compute_kpis(large_frame).iloc[0]
    totals = large_frame[mc.NUMERIC_COLUMNS].sum()
    assert np.isclose(out["ctr_pct"], totals.clicks / totals.impressions * 100)
    assert np.isclose(out["cpc"], totals.spend / totals.clicks)
    assert np.isclose(out["cpa"], totals.spend / totals.conversions)
    assert np.isclose(out["conversion_rate_pct"], totals.conversions / totals.clicks * 100)
    assert np.isclose(out["roas"], totals.revenue / totals.spend)
    assert np.isclose(out["cpm"], totals.spend / totals.impressions * 1000)


def test_kpis_use_totals_not_row_averages(large_frame):
    out = mc.compute_kpis(large_frame).iloc[0]
    row_average = (large_frame.revenue / large_frame.spend).mean()
    assert not np.isclose(out["roas"], row_average)


def test_zero_denominators_become_nan():
    frame = pd.DataFrame([{"platform": "A", "campaign": "c", "date": "2025-01-01", "impressions": 0,
                           "clicks": 0, "spend": 0.0, "conversions": 0, "revenue": 0.0}])
    out = mc.compute_kpis(frame, "platform").iloc[0]
    assert out[mc.RATIO_COLUMNS].isna().all()


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        mc.compute_kpis(pd.DataFrame({"platform": ["A"]}))


def test_unknown_grouping_column_raises(large_frame):
    with pytest.raises(ValueError, match="Grouping columns missing"):
        mc.compute_kpis(large_frame, "nope")


def test_negative_values_rejected(large_frame):
    broken = large_frame.head(10).copy()
    broken.loc[broken.index[0], "spend"] = -1
    with pytest.raises(ValueError, match="negative"):
        mc.validate_data(broken)


def test_funnel_violation_warns(large_frame):
    broken = large_frame.head(10).copy()
    broken.loc[broken.index[0], "clicks"] = broken.loc[broken.index[0], "impressions"] + 5
    with pytest.warns(UserWarning, match="Funnel"):
        report = mc.validate_data(broken)
    assert report["clicks_exceed_impressions"] == 1


def test_chunked_csv_matches_in_memory(small_csv):
    in_memory = mc.compute_kpis(pd.read_csv(small_csv), "platform").sort_values("platform").reset_index(drop=True)
    chunked = mc.kpis_from_csv(small_csv, "platform", chunksize=900).sort_values("platform").reset_index(drop=True)
    pd.testing.assert_frame_equal(in_memory, chunked, check_dtype=False)


def test_overall_kpis_from_csv(small_csv):
    expected = mc.compute_kpis(pd.read_csv(small_csv)).iloc[0]["roas"]
    assert np.isclose(mc.kpis_from_csv(small_csv, chunksize=900).iloc[0]["roas"], expected)


def test_multi_key_aggregation(small_csv):
    out = mc.aggregate_csv(small_csv, ["platform", "date"], chunksize=1000)
    assert out["impressions"].sum() == pd.read_csv(small_csv)["impressions"].sum()


def test_empty_csv(tmp_path):
    from data_generator import generate_csv
    path = generate_csv(0, tmp_path / "empty.csv")
    assert mc.aggregate_csv(path, ["platform", "date"]).empty
    assert mc.kpis_from_csv(path, "platform").empty


def test_bad_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        mc.kpis_from_csv(tmp_path / "missing.csv")
    with pytest.raises(ValueError):
        mc.aggregate_csv(tmp_path / "missing.csv", ["platform"], chunksize=0)


def test_format_table_shows_counts_without_decimals(large_frame):
    text = mc.format_kpi_table(mc.compute_kpis(large_frame, "platform"))
    assert ".00" not in text.splitlines()[1].split()[1]
