"""Optional sentiment-data ingestion and analytics (integration point for the YouTube project).

The Digital Budget Optimizer does not depend on this module. When a labelled
comment dataset (for example the output of the YouTube Comments Sentiment
Analysis project) is supplied, it can be validated, summarised and - only when
a genuine shared identifier exists - joined to campaign performance.

Design rules
------------
* A relationship between comments and campaigns is never inferred. A join is made
  only through an explicit ``campaign_id`` column or an explicit video-to-campaign
  mapping supplied by the user.
* Combined views report association, not causation.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]
CANONICAL_COLUMNS = ["video_id", "campaign_id", "comment_text", "sentiment", "timestamp", "likes"]
COLUMN_ALIASES = {
    "video_id": ["video_id", "videoid", "video", "video id"],
    "campaign_id": ["campaign_id", "campaignid", "campaign id"],
    "comment_text": ["comment_text", "comment", "text", "comment text", "content"],
    "sentiment": ["sentiment", "label", "predicted_sentiment", "sentiment_label"],
    "timestamp": ["timestamp", "published_at", "publishedat", "date", "created_at", "time"],
    "likes": ["likes", "like_count", "likecount", "votes"],
}
LABEL_ALIASES = {
    "positive": "Positive", "pos": "Positive", "negative": "Negative", "neg": "Negative",
    "neutral": "Neutral", "neu": "Neutral",
}
MIN_COMMENTS_PER_CAMPAIGN = 30
MIN_CAMPAIGNS_FOR_CORRELATION = 8


class SentimentDataError(ValueError):
    """Raised when a sentiment dataset cannot be used at all."""


@dataclass
class SentimentReport:
    rows_in: int = 0
    rows_valid: int = 0
    dropped_missing_sentiment: int = 0
    dropped_unrecognised_label: int = 0
    dropped_duplicates: int = 0
    invalid_timestamps: int = 0
    invalid_likes: int = 0
    columns_found: list[str] = field(default_factory=list)
    columns_missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_valid


@dataclass
class SentimentData:
    frame: pd.DataFrame
    report: SentimentReport

    @property
    def has_timestamps(self) -> bool:
        return bool(self.frame["timestamp"].notna().any())

    @property
    def has_campaign_ids(self) -> bool:
        return bool(self.frame["campaign_id"].notna().any())


@dataclass
class LinkResult:
    status: str
    method: str | None
    matched_rows: int
    unmatched_rows: int
    matched_campaigns: int
    message: str


def load_sentiment_csv(source: str | Path | BinaryIO | io.StringIO) -> pd.DataFrame:
    """Read a sentiment CSV, converting parser failures into ``SentimentDataError``."""
    try:
        frame = pd.read_csv(source)
    except pd.errors.EmptyDataError as error:
        raise SentimentDataError("The sentiment file is empty.") from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise SentimentDataError(f"The sentiment file could not be parsed as CSV: {error}") from error
    except FileNotFoundError as error:
        raise SentimentDataError(f"Sentiment file not found: {source}") from error
    return frame


def _rename_to_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    normalised = {str(c).strip().lower(): c for c in frame.columns}
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalised and normalised[alias] not in rename:
                rename[normalised[alias]] = canonical
                break
    return frame.rename(columns=rename)


def _clean_ids(series: pd.Series) -> pd.Series:
    cleaned = series.astype("object").where(series.notna(), None)
    cleaned = cleaned.map(lambda v: str(v).strip() if v is not None else None)
    return cleaned.replace({"": None, "nan": None, "None": None})


def validate_sentiment(frame: pd.DataFrame, drop_duplicates: bool = True) -> SentimentData:
    """Validate and normalise a sentiment dataset.

    A ``sentiment`` column is mandatory. Duplicates are only detected for rows that carry comment text,
    because label-only rows cannot be told apart. Rows with a missing or unrecognised label are
    dropped; unparseable timestamps and like counts are reported and neutralised
    (timestamp -> missing, likes -> 0) rather than discarding the comment.
    """
    if frame is None or frame.empty:
        raise SentimentDataError("The sentiment dataset has no rows.")
    data = _rename_to_canonical(frame)
    report = SentimentReport(rows_in=len(data))
    if "sentiment" not in data.columns:
        raise SentimentDataError(
            "No sentiment column found. Expected one of: " + ", ".join(COLUMN_ALIASES["sentiment"]) + ".")
    for column in CANONICAL_COLUMNS:
        if column not in data.columns:
            data[column] = pd.Series([None] * len(data), dtype="object")
    report.columns_found = [c for c in CANONICAL_COLUMNS if c in _rename_to_canonical(frame).columns]
    report.columns_missing = [c for c in CANONICAL_COLUMNS if c not in report.columns_found]
    data = data[CANONICAL_COLUMNS].copy()

    missing_label = data["sentiment"].isna() | (data["sentiment"].astype(str).str.strip() == "")
    report.dropped_missing_sentiment = int(missing_label.sum())
    data = data.loc[~missing_label].copy()
    normalised = data["sentiment"].astype(str).str.strip().str.lower().map(LABEL_ALIASES)
    report.dropped_unrecognised_label = int(normalised.isna().sum())
    data = data.loc[normalised.notna()].copy()
    data["sentiment"] = normalised.loc[data.index]

    data["video_id"] = _clean_ids(data["video_id"])
    data["campaign_id"] = _clean_ids(data["campaign_id"])
    data["comment_text"] = data["comment_text"].astype("object").where(data["comment_text"].notna(), None)

    parsed_time = pd.to_datetime(data["timestamp"], errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    report.invalid_timestamps = int((parsed_time.isna() & data["timestamp"].notna()).sum())
    data["timestamp"] = parsed_time

    likes = pd.to_numeric(data["likes"], errors="coerce")
    report.invalid_likes = int((likes.isna() & data["likes"].notna()).sum() + (likes < 0).sum())
    data["likes"] = likes.where(likes >= 0).fillna(0).astype("int64")

    if drop_duplicates:
        before = len(data)
        repeated = data.duplicated(subset=["video_id", "comment_text", "timestamp", "sentiment"]) & data["comment_text"].notna()
        data = data.loc[~repeated].copy()
        report.dropped_duplicates = before - len(data)

    data = data.reset_index(drop=True)
    report.rows_valid = len(data)
    if data.empty:
        raise SentimentDataError("No rows with a recognised sentiment label (Positive, Neutral or Negative).")
    if report.rows_dropped:
        report.warnings.append(f"{report.rows_dropped:,} of {report.rows_in:,} rows were dropped during validation.")
    if report.invalid_timestamps:
        report.warnings.append(f"{report.invalid_timestamps:,} timestamps could not be parsed and were left empty.")
    if report.invalid_likes:
        report.warnings.append(f"{report.invalid_likes:,} invalid like counts were set to 0.")
    return SentimentData(data, report)


def sentiment_distribution(frame: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Counts and shares for each sentiment label, overall or per ``by`` value."""
    if frame.empty:
        columns = ([by] if by else []) + ["sentiment", "comments", "share_pct"]
        return pd.DataFrame(columns=columns)
    if by is None:
        counts = frame["sentiment"].value_counts().reindex(SENTIMENT_LABELS, fill_value=0)
        out = counts.rename("comments").rename_axis("sentiment").reset_index()
        out["share_pct"] = out["comments"] / out["comments"].sum() * 100
        return out
    grouped = frame.groupby([by, "sentiment"]).size().unstack(fill_value=0).reindex(columns=SENTIMENT_LABELS, fill_value=0)
    long = grouped.stack().rename("comments").reset_index()
    long["share_pct"] = long["comments"] / long.groupby(by)["comments"].transform("sum") * 100
    return long


def net_sentiment(frame: pd.DataFrame) -> float:
    """(positive - negative) / total, in the range -1 to 1; NaN for an empty frame."""
    if frame.empty:
        return float("nan")
    counts = frame["sentiment"].value_counts()
    return float((counts.get("Positive", 0) - counts.get("Negative", 0)) / len(frame))


def sentiment_trend(frame: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Comment counts and shares per period and label. Empty when no timestamps exist."""
    dated = frame.dropna(subset=["timestamp"])
    if dated.empty:
        return pd.DataFrame(columns=["period", "sentiment", "comments", "share_pct"])
    period = dated["timestamp"].dt.to_period(freq).dt.start_time
    counts = (dated.assign(period=period).groupby(["period", "sentiment"]).size()
              .unstack(fill_value=0).reindex(columns=SENTIMENT_LABELS, fill_value=0))
    long = counts.stack().rename("comments").reset_index()
    long["share_pct"] = long["comments"] / long.groupby("period")["comments"].transform("sum") * 100
    return long


def _normalise_id(series: pd.Series) -> pd.Series:
    return series.astype("object").map(lambda v: str(v).strip().upper() if v is not None and pd.notna(v) else None)


def validate_video_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    """Validate a user-supplied video_id -> campaign_id mapping."""
    renamed = _rename_to_canonical(mapping)
    if not {"video_id", "campaign_id"} <= set(renamed.columns):
        raise SentimentDataError("The mapping file needs video_id and campaign_id columns.")
    pairs = renamed[["video_id", "campaign_id"]].dropna().astype(str).apply(lambda s: s.str.strip())
    pairs = pairs[(pairs["video_id"] != "") & (pairs["campaign_id"] != "")].drop_duplicates()
    if pairs.empty:
        raise SentimentDataError("The mapping file contains no usable video_id/campaign_id pairs.")
    conflicts = pairs.groupby("video_id")["campaign_id"].nunique()
    if (conflicts > 1).any():
        raise SentimentDataError(
            "Some videos map to more than one campaign (" + ", ".join(conflicts[conflicts > 1].index[:5]) +
            "). Each video must map to exactly one campaign.")
    return pairs.reset_index(drop=True)


def link_to_campaigns(frame: pd.DataFrame, known_campaign_ids: Iterable[str],
                      video_mapping: pd.DataFrame | None = None) -> tuple[pd.DataFrame, LinkResult]:
    """Attach comments to campaigns using explicit identifiers only.

    Returns the comments that could be linked (with a normalised ``campaign_id``) and
    a ``LinkResult`` explaining the outcome. Nothing is inferred from text or titles.
    """
    known = {str(c).strip().upper(): str(c) for c in known_campaign_ids}
    linked = frame.copy()
    linked["campaign_id"] = _normalise_id(linked["campaign_id"])
    methods = []
    if linked["campaign_id"].notna().any():
        methods.append("campaign_id")
    if video_mapping is not None:
        pairs = validate_video_mapping(video_mapping)
        lookup = dict(zip(_normalise_id(pairs["video_id"]), _normalise_id(pairs["campaign_id"])))
        from_mapping = _normalise_id(linked["video_id"]).map(lookup)
        filled = linked["campaign_id"].isna() & from_mapping.notna()
        if filled.any():
            linked.loc[filled, "campaign_id"] = from_mapping[filled]
            methods.append("video_mapping")

    if not methods:
        return linked.iloc[0:0], LinkResult(
            "no_identifier", None, 0, len(frame), 0,
            "No campaign_id column or video-to-campaign mapping was provided, so sentiment is shown "
            "independently of campaign performance.")
    candidates = linked[linked["campaign_id"].notna()]
    matched = candidates[candidates["campaign_id"].isin(known)].copy()
    matched["campaign_id"] = matched["campaign_id"].map(known)
    method = "+".join(methods)
    if matched.empty:
        example = ", ".join(sorted(candidates["campaign_id"].unique())[:3])
        return matched, LinkResult(
            "no_matches", method, 0, len(frame), 0,
            f"Identifiers were provided (for example {example}) but none match a campaign in the selected dataset.")
    return matched, LinkResult(
        "linked", method, len(matched), len(frame) - len(matched), int(matched["campaign_id"].nunique()),
        f"{len(matched):,} of {len(frame):,} comments were linked to {matched['campaign_id'].nunique()} campaign(s) "
        f"through {method.replace('+', ' and ').replace('_', ' ')}.")


def campaign_sentiment_summary(linked: pd.DataFrame, campaign_kpis: pd.DataFrame,
                               min_comments: int = MIN_COMMENTS_PER_CAMPAIGN) -> pd.DataFrame:
    """Per-campaign sentiment mix joined to campaign KPIs (one row per linked campaign)."""
    if linked.empty:
        return pd.DataFrame()
    shares = linked.groupby(["campaign_id", "sentiment"]).size().unstack(fill_value=0).reindex(
        columns=SENTIMENT_LABELS, fill_value=0)
    summary = pd.DataFrame({"comments": shares.sum(axis=1)})
    for label in SENTIMENT_LABELS:
        summary[f"{label.lower()}_pct"] = shares[label] / summary["comments"] * 100
    summary["net_sentiment"] = (shares["Positive"] - shares["Negative"]) / summary["comments"]
    summary["avg_likes"] = linked.groupby("campaign_id")["likes"].mean()
    summary["sufficient_sample"] = summary["comments"] >= min_comments
    kpi_columns = [c for c in ["campaign_id", "campaign_name", "platform", "spend", "revenue", "ctr_pct",
                               "conversion_rate_pct", "roas"] if c in campaign_kpis.columns]
    return summary.reset_index().merge(campaign_kpis[kpi_columns], on="campaign_id", how="left")


def sentiment_performance_association(summary: pd.DataFrame, metric: str = "roas") -> dict:
    """Descriptive Spearman correlation between net sentiment and a KPI across campaigns.

    This is an association across a small number of campaigns, never evidence that
    sentiment influences performance. Only campaigns with a sufficient sample are used.
    """
    usable = summary[summary["sufficient_sample"]].dropna(subset=["net_sentiment", metric]) if not summary.empty else summary
    if len(usable) < MIN_CAMPAIGNS_FOR_CORRELATION:
        return {"available": False, "campaigns": int(len(usable)),
                "reason": f"At least {MIN_CAMPAIGNS_FOR_CORRELATION} campaigns with {MIN_COMMENTS_PER_CAMPAIGN}+ "
                          "comments are needed for a meaningful comparison."}
    if usable["net_sentiment"].nunique() < 2 or usable[metric].nunique() < 2:
        return {"available": False, "campaigns": int(len(usable)), "reason": "One of the series has no variation."}
    value = float(usable["net_sentiment"].corr(usable[metric], method="spearman"))
    return {"available": True, "campaigns": int(len(usable)), "spearman": value if np.isfinite(value) else float("nan"),
            "metric": metric}
