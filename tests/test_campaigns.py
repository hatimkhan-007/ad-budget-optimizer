import pandas as pd
import pytest

from campaigns import CAMPAIGN_KEYS, add_campaign_id, build_campaign_catalog, make_campaign_id
from metrics_calculator import NUMERIC_COLUMNS


def test_campaign_id_is_stable_and_readable():
    assert make_campaign_id("Facebook", "Conversions", "New Customers") == "FB-CONV-NEW"
    assert make_campaign_id("Facebook", "Conversions", "New Customers") == make_campaign_id("Facebook", "Conversions", "New Customers")


def test_campaign_id_falls_back_for_unknown_values():
    assert make_campaign_id("Pinterest", "Spring Sale!", "Moms") == "PI-SPRIN-MOM"
    assert make_campaign_id("", "", "") == "NA-NA-NA"


def test_each_platform_objective_audience_combination_gets_one_id(aggregates):
    frame = add_campaign_id(aggregates.frames["campaign_daily"])
    combos = frame[CAMPAIGN_KEYS].drop_duplicates()
    assert frame["campaign_id"].nunique() == len(combos)
    assert frame.groupby("campaign_id")[CAMPAIGN_KEYS].nunique().max().max() == 1


def test_add_campaign_id_requires_key_columns():
    with pytest.raises(ValueError, match="missing columns"):
        add_campaign_id(pd.DataFrame({"platform": ["Google"]}))


def test_add_campaign_id_handles_empty_frame():
    empty = pd.DataFrame(columns=CAMPAIGN_KEYS)
    out = add_campaign_id(empty)
    assert out.empty and {"campaign_id", "campaign_name"} <= set(out.columns)


def test_catalog_totals_match_source(aggregates):
    daily = aggregates.frames["campaign_daily"]
    catalog = build_campaign_catalog(daily)
    assert catalog["campaign_id"].is_unique
    assert catalog["total_spend"].sum() == pytest.approx(daily["spend"].sum())
    assert catalog["revenue"].sum() == pytest.approx(daily["revenue"].sum())
    assert (catalog["start_date"] <= catalog["end_date"]).all()
    assert set(catalog["status"]) <= {"Active", "Paused", "Completed"}


def test_catalog_budget_is_simulated_deterministic_and_covers_spend(aggregates):
    daily = aggregates.frames["campaign_daily"]
    first = build_campaign_catalog(daily, seed=1)
    again = build_campaign_catalog(daily, seed=1)
    other = build_campaign_catalog(daily, seed=2)
    pd.testing.assert_series_equal(first["budget"], again["budget"])
    assert not first["budget"].equals(other["budget"])
    assert (first["budget"] >= first["total_spend"]).all()
    assert first["budget_utilization_pct"].between(0, 100).all()


def test_status_is_derived_from_delivery_dates():
    rows = []
    for audience, last_day in [("New Customers", "2025-12-31"), ("Lookalike", "2025-11-01"), ("Interest Based", "2025-06-01")]:
        for date in ["2025-01-01", last_day]:
            rows.append({"platform": "Google", "campaign": "Conversions", "audience": audience, "date": date,
                         **{c: 10 for c in NUMERIC_COLUMNS}})
    catalog = build_campaign_catalog(pd.DataFrame(rows))
    status = dict(zip(catalog["target_audience"], catalog["status"]))
    assert status == {"New Customers": "Active", "Lookalike": "Paused", "Interest Based": "Completed"}


def test_catalog_kpis_use_summed_totals(aggregates):
    row = build_campaign_catalog(aggregates.frames["campaign_daily"]).iloc[0]
    assert row["roas"] == pytest.approx(row["revenue"] / row["spend"])
    assert row["ctr_pct"] == pytest.approx(row["clicks"] / row["impressions"] * 100)


def test_catalog_empty_and_invalid_inputs():
    empty = build_campaign_catalog(pd.DataFrame(columns=CAMPAIGN_KEYS + ["date"] + NUMERIC_COLUMNS))
    assert empty.empty and "campaign_id" in empty.columns
    with pytest.raises(ValueError, match="missing columns"):
        build_campaign_catalog(pd.DataFrame({"platform": ["Google"]}))
