import numpy as np
import pandas as pd
import pytest

import benchmark_analysis as ba
import processing_engines as pe

SPARK_OK, SPARK_REASON = pe.spark_available()


def test_mapreduce_matches_pandas_with_many_splits(small_csv):
    expected = pe.run_pandas(small_csv)
    assert len(pe.byte_splits(small_csv, split_bytes=20_000)) > 5
    actual = pe.run_mapreduce(small_csv, workers=2, split_bytes=20_000)
    assert ba.results_match(expected, actual)


def test_byte_splits_cover_data_exactly_once(small_csv):
    splits = pe.byte_splits(small_csv, split_bytes=15_000, min_parts=3)
    with open(small_csv, "rb") as handle:
        header_end = len(handle.readline())
        data = handle.read()
    assert splits[0][0] == header_end
    assert all(a[1] == b[0] for a, b in zip(splits, splits[1:]))
    assert splits[-1][1] == header_end + len(data)
    with open(small_csv, "rb") as handle:
        for start, end in splits[1:]:
            handle.seek(start - 1)
            assert handle.read(1) == b"\n"


def test_byte_splits_validation(small_csv):
    with pytest.raises(ValueError):
        pe.byte_splits(small_csv, split_bytes=0)


def test_row_counts_add_up(small_csv):
    assert pe.run_pandas(small_csv)["rows"].sum() == 5_000


def test_mapreduce_handles_header_only_file(tmp_path):
    from data_generator import generate_csv
    path = generate_csv(0, tmp_path / "empty.csv")
    assert pe.run_mapreduce(path, workers=1).empty


@pytest.mark.skipif(not SPARK_OK, reason=SPARK_REASON or "PySpark unavailable")
def test_pyspark_matches_pandas_and_reports_startup(small_csv):
    result, startup = pe.run_pyspark(small_csv)
    assert startup > 0
    assert ba.results_match(pe.run_pandas(small_csv), result)


def test_results_match_detects_differences(small_csv):
    reference = pe.run_pandas(small_csv)
    assert ba.results_match(reference, reference.copy())
    changed = reference.copy()
    changed.loc[0, "spend"] += 1.0
    assert not ba.results_match(reference, changed)
    changed = reference.copy()
    changed.loc[0, "rows"] += 1
    assert not ba.results_match(reference, changed)
    assert not ba.results_match(reference, reference.iloc[1:])


def test_benchmark_compares_engines_and_measures_memory(small_csv):
    frame = ba.run_benchmarks({5_000: small_csv}, ("pandas", "mapreduce"), trials=2)
    assert list(frame["status"]) == ["ok", "ok"]
    assert (frame["consistent"] == "yes").all()
    assert (frame["rows_counted"] == 5_000).all()
    assert frame["peak_mem_mb"].notna().all() and (frame["peak_mem_mb"] > 0).all()
    assert frame.loc[frame.engine == "pandas", "vs_pandas"].iloc[0] == pytest.approx(1.0)
    assert (frame["startup_seconds"] == 0).all()


def test_missing_engine_is_skipped_not_fatal(small_csv, monkeypatch):
    monkeypatch.setattr(ba, "engine_available", lambda e: (False, "Java was not found") if e == "pyspark" else (True, ""))
    frame = ba.run_benchmarks({5_000: small_csv}, ("pandas", "pyspark"), trials=1)
    skipped = frame[frame.engine == "pyspark"].iloc[0]
    assert skipped.status == "skipped" and "Java" in skipped.detail
    assert frame[frame.engine == "pandas"].iloc[0].status == "ok"


def test_failed_engine_reports_error(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("platform,campaign\nA,B\n")
    frame = ba.run_benchmarks({1: bad}, ("pandas",), trials=1)
    assert frame.iloc[0].status == "error" and frame.iloc[0].detail


def test_benchmark_input_validation(small_csv):
    with pytest.raises(ValueError):
        ba.run_benchmarks({1: small_csv}, ("pandas",), trials=0)
    with pytest.raises(ValueError):
        ba.run_benchmarks({1: small_csv}, ("spark-sql",))


def test_formatted_report_lists_notes(small_csv, monkeypatch):
    monkeypatch.setattr(ba, "engine_available", lambda e: (False, "Java was not found") if e == "pyspark" else (True, ""))
    text = ba.format_results(ba.run_benchmarks({5_000: small_csv}, ("pandas", "pyspark"), trials=1))
    assert "Notes:" in text and "Java was not found" in text
