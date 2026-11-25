import numpy as np
import pandas as pd
import pytest

from data_generator import generate_dataframe
from data_quality import assess_csv, clean_dataframe, optimize_dtypes, sanitize_chunk


@pytest.fixture
def good_frame():
    return generate_dataframe(500, seed=3)


def test_clean_data_passes_untouched(good_frame):
    cleaned, report = sanitize_chunk(good_frame)
    assert len(cleaned) == len(good_frame)
    assert report.rows_rejected == 0 and not report.issues


def test_missing_key_fields_are_dropped(good_frame):
    frame = good_frame.copy().astype({"platform": "object"})
    frame.loc[0, "platform"] = None
    frame.loc[1, "date"] = None
    cleaned, report = sanitize_chunk(frame)
    assert report.missing_key_rows == 2 and len(cleaned) == len(frame) - 2


def test_missing_numeric_values_drop_or_zero(good_frame):
    frame = good_frame.copy()
    frame["revenue"] = frame["revenue"].astype(float)
    frame.loc[[0, 1, 2], "revenue"] = np.nan
    dropped, report = sanitize_chunk(frame, numeric_missing="drop")
    assert len(dropped) == len(frame) - 3 and report.missing_numeric_rows == 3
    zeroed, _ = sanitize_chunk(frame, numeric_missing="zero")
    assert len(zeroed) == len(frame) and (zeroed.loc[[0, 1, 2], "revenue"] == 0).all()


def test_text_in_numeric_column_is_treated_as_missing(good_frame):
    frame = good_frame.copy().astype({"spend": "object"})
    frame.loc[0, "spend"] = "n/a"
    cleaned, report = sanitize_chunk(frame)
    assert report.missing_numeric_rows == 1 and len(cleaned) == len(frame) - 1


def test_negative_and_infinite_values_are_rejected(good_frame):
    frame = good_frame.copy().astype({"spend": "float64"})
    frame.loc[0, "spend"] = -5.0
    frame.loc[1, "spend"] = np.inf
    cleaned, report = sanitize_chunk(frame)
    assert report.negative_rows == 1 and len(cleaned) == len(frame) - 2


def test_funnel_violations_are_rejected(good_frame):
    frame = good_frame.copy()
    frame.loc[0, "clicks"] = frame.loc[0, "impressions"] + 1
    frame.loc[1, "conversions"] = frame.loc[1, "clicks"] + 1
    cleaned, report = sanitize_chunk(frame)
    assert report.funnel_violation_rows == 2 and len(cleaned) == len(frame) - 2
    assert (cleaned["clicks"] <= cleaned["impressions"]).all()


def test_missing_dimensions_become_unknown(good_frame):
    frame = good_frame.copy().astype({"city": "object"})
    frame.loc[0, "city"] = None
    cleaned, _ = sanitize_chunk(frame)
    assert (cleaned["city"] == "Unknown").sum() == 1


def test_empty_frame_and_bad_arguments(good_frame):
    cleaned, report = sanitize_chunk(good_frame.iloc[0:0])
    assert cleaned.empty and report.rows_in == 0
    with pytest.raises(ValueError, match="numeric_missing"):
        sanitize_chunk(good_frame, numeric_missing="impute")
    with pytest.raises(ValueError, match="Missing required columns|missing"):
        sanitize_chunk(good_frame.drop(columns=["spend"]))


def test_clean_dataframe_removes_exact_duplicates(good_frame):
    doubled = pd.concat([good_frame, good_frame.iloc[:25]], ignore_index=True)
    cleaned, report = clean_dataframe(doubled)
    assert report.duplicate_rows == 25 and len(cleaned) == len(good_frame)
    kept, _ = clean_dataframe(doubled, drop_duplicates=False)
    assert len(kept) == len(doubled)


def test_optimize_dtypes_saves_memory_without_changing_values(good_frame):
    optimized = optimize_dtypes(good_frame)
    assert optimized.memory_usage(deep=True).sum() < good_frame.memory_usage(deep=True).sum()
    pd.testing.assert_series_equal(optimized["spend"].astype(float), good_frame["spend"].astype(float), check_dtype=False)
    assert (optimized["impressions"].to_numpy() == good_frame["impressions"].to_numpy()).all()


def test_assess_csv_finds_duplicates_across_chunks(tmp_path, good_frame):
    frame = pd.concat([good_frame, good_frame.iloc[:10]], ignore_index=True)
    path = tmp_path / "dups.csv"
    frame.to_csv(path, index=False)
    report = assess_csv(path, chunksize=100)
    assert report.duplicate_rows == 10 and report.rows_in == len(frame)


def test_assess_csv_input_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        assess_csv(tmp_path / "missing.csv")
    path = tmp_path / "x.csv"
    generate_dataframe(10).to_csv(path, index=False)
    with pytest.raises(ValueError, match="chunksize"):
        assess_csv(path, chunksize=0)
