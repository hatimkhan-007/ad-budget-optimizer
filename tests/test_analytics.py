import pandas as pd
import pytest

from analytics import build_aggregates, load_or_build_aggregates, summarize
from metrics_calculator import NUMERIC_COLUMNS


def test_aggregates_match_full_load_totals(campaign_csv, aggregates):
    full = pd.read_csv(campaign_csv)
    for name, frame in aggregates.frames.items():
        assert frame["spend"].sum() == pytest.approx(full["spend"].sum()), name
        assert int(frame["impressions"].sum()) == int(full["impressions"].sum()), name
        assert int(frame["conversions"].sum()) == int(full["conversions"].sum()), name
    assert aggregates.quality.rows_in == len(full)
    assert aggregates.chunks == 3


def test_chunk_size_does_not_change_results(campaign_csv, aggregates):
    other = build_aggregates(campaign_csv, chunksize=7_001)
    for name in aggregates.frames:
        left = aggregates.frames[name].sort_values(list(aggregates.frames[name].columns[:-5])).reset_index(drop=True)
        right = other.frames[name].sort_values(list(other.frames[name].columns[:-5])).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_invalid_rows_are_excluded_and_reported(tmp_path, small_csv):
    frame = pd.read_csv(small_csv)
    frame.loc[0, "spend"] = -1
    frame.loc[1, "clicks"] = frame.loc[1, "impressions"] + 5
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    result = build_aggregates(path, chunksize=2_000)
    assert result.quality.rows_rejected == 2
    assert result.frames["device"]["impressions"].sum() == frame["impressions"].sum() - frame.loc[:1, "impressions"].sum()


def test_cache_is_reused_and_invalidated_on_change(tmp_path, small_csv):
    cache = tmp_path / "cache"
    first = load_or_build_aggregates(small_csv, cache)
    second = load_or_build_aggregates(small_csv, cache)
    assert not first.from_cache and second.from_cache
    pd.testing.assert_frame_equal(first.frames["device"], second.frames["device"], check_dtype=False)

    copy = tmp_path / "ads_copy.csv"
    copy.write_bytes(small_csv.read_bytes())
    load_or_build_aggregates(copy, cache)
    copy.write_text(copy.read_text().replace("Mobile", "Desktop", 1))
    assert not load_or_build_aggregates(copy, cache).from_cache


def test_corrupt_cache_is_rebuilt(tmp_path, small_csv):
    cache = tmp_path / "cache"
    load_or_build_aggregates(small_csv, cache)
    (cache / small_csv.stem / "device.csv").write_text("")
    assert not load_or_build_aggregates(small_csv, cache).from_cache


def test_input_errors(tmp_path, small_csv):
    with pytest.raises(FileNotFoundError):
        build_aggregates(tmp_path / "nope.csv")
    with pytest.raises(ValueError, match="chunksize"):
        build_aggregates(small_csv, chunksize=0)
    pd.read_csv(small_csv).drop(columns=["device"]).to_csv(tmp_path / "nodevice.csv", index=False)
    with pytest.raises(ValueError, match="device"):
        build_aggregates(tmp_path / "nodevice.csv")


def test_header_only_file_gives_empty_tables(tmp_path, small_csv):
    path = tmp_path / "empty.csv"
    pd.read_csv(small_csv, nrows=0).to_csv(path, index=False)
    result = build_aggregates(path)
    assert result.quality.rows_in == 0 and all(frame.empty for frame in result.frames.values())


def test_summarize_computes_ratios_from_totals():
    frame = pd.DataFrame({"device": ["a", "a", "b"], "impressions": [1000, 1000, 500], "clicks": [10, 30, 5],
                          "spend": [10.0, 30.0, 5.0], "conversions": [1, 3, 0], "revenue": [50.0, 150.0, 0.0]})
    out = summarize(frame, "device").set_index("device")
    assert out.loc["a", "ctr_pct"] == pytest.approx(2.0)
    assert out.loc["a", "roas"] == pytest.approx(5.0)
    assert pd.isna(out.loc["b", "cpa"])
    assert summarize(frame).iloc[0]["spend"] == 45.0


def test_summarize_requires_metric_columns():
    with pytest.raises(ValueError, match="Missing metric columns"):
        summarize(pd.DataFrame({"spend": [1.0]}))
    assert set(NUMERIC_COLUMNS) <= set(summarize(pd.DataFrame({c: [1] for c in NUMERIC_COLUMNS})).columns)
