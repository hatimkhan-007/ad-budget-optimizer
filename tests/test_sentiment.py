import io

import numpy as np
import pandas as pd
import pytest

from sentiment import (SentimentDataError, campaign_sentiment_summary, link_to_campaigns, load_sentiment_csv, net_sentiment,
                       sentiment_distribution, sentiment_performance_association, sentiment_trend, validate_sentiment,
                       validate_video_mapping)

CSV = """videoId,comment,label,publishedAt,like_count,campaign_id
v1,great ad,Positive,2025-01-03T10:00:00Z,5,FB-CONV-NEW
v1,awful,negative,2025-01-04,x,FB-CONV-NEW
v2,fine,NEU,not a date,2,
v2,,,2025-01-05,1,
v3,spam,weird,2025-01-05,1,
v1,great ad,Positive,2025-01-03T10:00:00Z,5,FB-CONV-NEW
"""


def parse(text):
    return validate_sentiment(load_sentiment_csv(io.StringIO(text)))


def test_aliases_labels_and_invalid_rows_are_handled():
    data = parse(CSV)
    frame, report = data.frame, data.report
    assert list(frame["sentiment"]) == ["Positive", "Negative", "Neutral"]
    assert report.rows_in == 6 and report.rows_valid == 3
    assert report.dropped_missing_sentiment == 1 and report.dropped_unrecognised_label == 1
    assert report.dropped_duplicates == 1


def test_mixed_timestamp_formats_and_bad_values():
    data = parse(CSV)
    stamps = data.frame["timestamp"]
    assert stamps.iloc[0] == pd.Timestamp("2025-01-03 10:00") and stamps.iloc[1] == pd.Timestamp("2025-01-04")
    assert pd.isna(stamps.iloc[2]) and data.report.invalid_timestamps == 1
    assert list(data.frame["likes"]) == [5, 0, 2] and data.report.invalid_likes == 1
    assert data.has_timestamps and data.has_campaign_ids


def test_optional_columns_may_be_absent():
    data = parse("sentiment\nPositive\nNegative\n")
    assert data.report.columns_missing and not data.has_timestamps and not data.has_campaign_ids
    assert sentiment_trend(data.frame).empty


@pytest.mark.parametrize("text,message", [
    ("", "empty"),
    ("a,b\n1,2\n", "No sentiment column"),
    ("sentiment\nweird\nodd\n", "recognised"),
    ("sentiment\n", "no rows"),
])
def test_unusable_files_raise_clear_errors(text, message):
    with pytest.raises(SentimentDataError, match=message):
        parse(text)


def test_missing_file_is_a_sentiment_error(tmp_path):
    with pytest.raises(SentimentDataError, match="not found"):
        load_sentiment_csv(tmp_path / "missing.csv")


def test_distribution_and_net_sentiment():
    frame = parse("sentiment,video_id\nPositive,a\nPositive,a\nNegative,b\nNeutral,b\n").frame
    dist = sentiment_distribution(frame)
    assert dist["share_pct"].sum() == pytest.approx(100.0)
    assert dist.set_index("sentiment").loc["Positive", "comments"] == 2
    assert net_sentiment(frame) == pytest.approx(0.25)
    by_video = sentiment_distribution(frame, by="video_id")
    assert by_video.groupby("video_id")["share_pct"].sum().tolist() == pytest.approx([100.0, 100.0])
    assert sentiment_distribution(frame.iloc[0:0]).empty and np.isnan(net_sentiment(frame.iloc[0:0]))


def test_trend_shares_sum_to_100_per_period():
    frame = parse("sentiment,timestamp\nPositive,2025-01-01\nNegative,2025-01-02\nPositive,2025-01-09\n").frame
    trend = sentiment_trend(frame, "W")
    assert trend.groupby("period")["share_pct"].sum().tolist() == pytest.approx([100.0, 100.0])
    assert trend["comments"].sum() == 3


def test_without_any_identifier_nothing_is_linked():
    frame = parse("sentiment,video_id\nPositive,v1\n").frame
    linked, outcome = link_to_campaigns(frame, ["FB-CONV-NEW"])
    assert linked.empty and outcome.status == "no_identifier" and outcome.method is None


def test_campaign_id_column_links_case_insensitively():
    frame = parse("sentiment,campaign_id\nPositive,fb-conv-new\nNegative,GG-LEAD-NEW\nNeutral,\n").frame
    linked, outcome = link_to_campaigns(frame, ["FB-CONV-NEW", "GG-LEAD-NEW"])
    assert outcome.status == "linked" and outcome.matched_rows == 2 and outcome.unmatched_rows == 1
    assert set(linked["campaign_id"]) == {"FB-CONV-NEW", "GG-LEAD-NEW"}


def test_identifiers_that_match_nothing_are_not_forced():
    frame = parse("sentiment,campaign_id\nPositive,SOMETHING-ELSE\n").frame
    linked, outcome = link_to_campaigns(frame, ["FB-CONV-NEW"])
    assert linked.empty and outcome.status == "no_matches" and "SOMETHING-ELSE" in outcome.message


def test_video_mapping_links_only_mapped_videos():
    frame = parse("sentiment,video_id\nPositive,v1\nNegative,v2\n").frame
    mapping = pd.DataFrame({"video_id": ["v1"], "campaign_id": ["yt-brand-lal"]})
    linked, outcome = link_to_campaigns(frame, ["YT-BRAND-LAL"], mapping)
    assert outcome.status == "linked" and outcome.method == "video_mapping"
    assert list(linked["video_id"]) == ["v1"] and outcome.unmatched_rows == 1


def test_conflicting_or_malformed_mappings_are_rejected():
    with pytest.raises(SentimentDataError, match="more than one campaign"):
        validate_video_mapping(pd.DataFrame({"video_id": ["v1", "v1"], "campaign_id": ["A", "B"]}))
    with pytest.raises(SentimentDataError, match="needs video_id"):
        validate_video_mapping(pd.DataFrame({"video": ["v1"]}))
    with pytest.raises(SentimentDataError, match="no usable"):
        validate_video_mapping(pd.DataFrame({"video_id": [None], "campaign_id": [None]}))


def _linked(n_campaigns, comments_each=40, correlated=True):
    rng = np.random.default_rng(0)
    rows, kpis = [], []
    for i in range(n_campaigns):
        positive_share = (i + 1) / (n_campaigns + 1)
        labels = rng.choice(["Positive", "Negative"], size=comments_each, p=[positive_share, 1 - positive_share])
        rows += [{"campaign_id": f"C{i}", "sentiment": s, "likes": 1} for s in labels]
        kpis.append({"campaign_id": f"C{i}", "roas": float(i + 1) if correlated else 2.0})
    return pd.DataFrame(rows), pd.DataFrame(kpis)


def test_campaign_summary_joins_kpis_and_flags_small_samples():
    linked, kpis = _linked(3, comments_each=10)
    summary = campaign_sentiment_summary(linked, kpis, min_comments=30)
    assert len(summary) == 3 and not summary["sufficient_sample"].any()
    assert summary[["positive_pct", "neutral_pct", "negative_pct"]].sum(axis=1).tolist() == pytest.approx([100.0] * 3)
    assert campaign_sentiment_summary(linked.iloc[0:0], kpis).empty


def test_association_requires_enough_campaigns_and_reports_no_causal_claim():
    few = campaign_sentiment_summary(*_linked(4))
    assert not sentiment_performance_association(few)["available"]
    many = campaign_sentiment_summary(*_linked(12))
    result = sentiment_performance_association(many)
    assert result["available"] and result["spearman"] > 0.5 and "cause" not in str(result).lower()
    flat = campaign_sentiment_summary(*_linked(12, correlated=False))
    assert not sentiment_performance_association(flat)["available"]
