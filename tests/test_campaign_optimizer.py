import numpy as np
import pandas as pd
import pytest

from budget_optimizer import ResponseCurve
from campaign_optimizer import CampaignOptimizer, allocate_with_bounds


def curve(name, a, b, spend=100.0):
    return ResponseCurve(name, a, b, b, 0.95, 60, spend, a * spend ** b, spend * 0.5, spend * 1.5)


def random_curves(n, seed=0):
    rng = np.random.default_rng(seed)
    return [curve(f"c{i}", rng.uniform(1.0, 6.0), rng.uniform(0.4, 0.9)) for i in range(n)]


def revenue(curves, spend):
    return sum(c.revenue(s) for c, s in zip(curves, spend))


def test_allocation_uses_full_budget_and_respects_bounds():
    curves = random_curves(12)
    lower, upper = np.full(12, 20.0), np.full(12, 300.0)
    spend = allocate_with_bounds(curves, 1000.0, lower, upper)
    assert spend.sum() == pytest.approx(1000.0, rel=1e-9)
    assert (spend >= lower - 1e-9).all() and (spend <= upper + 1e-9).all()


def test_free_campaigns_have_equal_marginal_revenue():
    curves = random_curves(10, seed=3)
    lower, upper = np.full(10, 5.0), np.full(10, 500.0)
    spend = allocate_with_bounds(curves, 900.0, lower, upper)
    free = [c.marginal_revenue(s) for c, s, lo, hi in zip(curves, spend, lower, upper) if lo + 1e-6 < s < hi - 1e-6]
    assert len(free) >= 2
    assert max(free) - min(free) < 1e-6 * max(free)


def test_allocation_beats_equal_split_and_random_feasible_plans():
    curves = random_curves(8, seed=5)
    lower, upper = np.full(8, 10.0), np.full(8, 400.0)
    best = revenue(curves, allocate_with_bounds(curves, 800.0, lower, upper))
    assert best >= revenue(curves, np.full(8, 100.0)) - 1e-9
    rng = np.random.default_rng(1)
    for _ in range(200):
        weights = rng.dirichlet(np.ones(8))
        plan = np.clip(weights * 800.0, lower, upper)
        plan = plan * 800.0 / plan.sum()
        if (plan <= upper + 1e-9).all() and (plan >= lower - 1e-9).all():
            assert best >= revenue(curves, plan) - 1e-9


def test_group_limits_are_respected_and_optimal_within_groups():
    curves = random_curves(9, seed=8)
    groups = np.repeat([0, 1, 2], 3)
    lower, upper = np.zeros(9), np.full(9, 1000.0)
    spend = allocate_with_bounds(curves, 900.0, lower, upper, groups, np.full(3, 100.0), np.full(3, 400.0))
    totals = np.array([spend[groups == g].sum() for g in range(3)])
    assert (totals >= 100.0 - 1e-6).all() and (totals <= 400.0 + 1e-6).all()
    assert spend.sum() == pytest.approx(900.0)
    for g in range(3):
        marginals = [curves[i].marginal_revenue(spend[i]) for i in range(9) if groups[i] == g and spend[i] > 1e-6]
        assert max(marginals) - min(marginals) < 1e-6 * max(marginals)


def test_infeasible_limits_raise_clear_errors():
    curves = random_curves(4)
    with pytest.raises(ValueError, match="infeasible"):
        allocate_with_bounds(curves, 1000.0, np.full(4, 10.0), np.full(4, 100.0))
    with pytest.raises(ValueError, match="infeasible"):
        allocate_with_bounds(curves, 10.0, np.full(4, 10.0), np.full(4, 100.0))
    with pytest.raises(ValueError, match="lower bound"):
        allocate_with_bounds(curves, 100.0, np.full(4, 50.0), np.full(4, 10.0))
    with pytest.raises(ValueError, match="same length"):
        allocate_with_bounds(curves, 100.0, np.zeros(3), np.ones(4))
    with pytest.raises(ValueError, match="conflict"):
        allocate_with_bounds(curves, 400.0, np.full(4, 90.0), np.full(4, 200.0), np.zeros(4, dtype=int),
                             np.array([0.0]), np.array([100.0]))


@pytest.fixture(scope="module")
def synthetic_daily():
    rng = np.random.default_rng(0)
    rows = []
    specs = {"Alpha": (0.8, 3.0), "Beta": (0.6, 4.0), "Gamma": (0.9, 2.0)}
    for platform, (b, a_base) in specs.items():
        for k in range(4):
            a = a_base * (0.7 + 0.2 * k)
            for day in pd.date_range("2025-01-01", periods=60):
                spend = float(rng.uniform(50, 150) * (1 + 0.5 * k))
                rows.append({"campaign_id": f"{platform[:2].upper()}-{k}", "platform": platform, "date": day,
                             "spend": spend, "revenue": a * spend ** b * rng.lognormal(0, 0.05)})
    return pd.DataFrame(rows)


def test_fit_recovers_platform_elasticity_and_shares_it_with_campaigns(synthetic_daily):
    optimizer = CampaignOptimizer().fit(synthetic_daily)
    assert len(optimizer.curves) == 12
    for campaign, curve_ in optimizer.curves.items():
        parent = optimizer.platform_curves[optimizer.campaign_platform[campaign]]
        assert curve_.b == parent.b
    assert 0.5 < optimizer.platform_curves["Beta"].b < 0.7


def test_recommendation_respects_all_constraints(synthetic_daily):
    optimizer = CampaignOptimizer(max_change=0.3, min_budget=50.0, max_budget=1500.0, platform_limits=(0.2, 0.5)).fit(synthetic_daily)
    result = optimizer.recommend_campaigns()
    table = result.table
    assert table["recommended_daily_spend"].sum() == pytest.approx(result.total_daily_budget)
    change = table["recommended_daily_spend"] / table["current_daily_spend"] - 1
    assert (change.abs() <= 0.3 + 1e-6).all()
    assert (table["recommended_daily_spend"] >= 50.0 - 1e-6).all() and (table["recommended_daily_spend"] <= 1500.0 + 1e-6).all()
    share = table.groupby("platform")["recommended_share_pct"].sum()
    assert (share >= 20.0 - 1e-6).all() and (share <= 50.0 + 1e-6).all()
    assert result.recommended_revenue >= result.compliant_baseline_revenue - 1e-9


def test_campaign_tiers_actions_and_reasons(synthetic_daily):
    result = CampaignOptimizer(platform_limits=None).fit(synthetic_daily).recommend_campaigns()
    table = result.table
    assert set(table["performance_tier"]) <= {"High performer", "Average", "Underperformer"}
    assert set(table["action"]) <= {"Increase", "Decrease", "Hold"}
    assert table["reason"].str.len().gt(20).all()
    best = table.loc[table["historical_roas"].idxmax()]
    assert best["action"] == "Increase"
    worst = table.loc[table["historical_roas"].idxmin()]
    assert worst["action"] == "Decrease"
    assert not table["campaign_id"].duplicated().any()


def test_scenario_budget_scales_the_current_mix(synthetic_daily):
    optimizer = CampaignOptimizer(platform_limits=None, max_change=0.5).fit(synthetic_daily)
    base = optimizer.recommend_campaigns()
    scaled = optimizer.recommend_campaigns(base.total_daily_budget * 1.2)
    assert scaled.table["current_daily_spend"].sum() == pytest.approx(base.total_daily_budget * 1.2)
    assert scaled.table["recommended_daily_spend"].sum() == pytest.approx(base.total_daily_budget * 1.2)


def test_absolute_limits_that_cannot_hold_the_budget_are_reported(synthetic_daily):
    optimizer = CampaignOptimizer(platform_limits=None, max_budget=1.0).fit(synthetic_daily)
    with pytest.raises(ValueError, match="infeasible"):
        optimizer.recommend_campaigns()
    tight = CampaignOptimizer(platform_limits=None, min_budget=10_000.0).fit(synthetic_daily)
    with pytest.raises(ValueError, match="infeasible"):
        tight.recommend_campaigns()


def test_campaigns_with_too_little_history_are_excluded(synthetic_daily):
    sparse = pd.concat([synthetic_daily, synthetic_daily[synthetic_daily["campaign_id"] == "AL-0"].head(2).assign(campaign_id="AL-NEW")])
    optimizer = CampaignOptimizer().fit(sparse)
    assert "AL-NEW" in optimizer.excluded and "AL-NEW" not in optimizer.curves
    assert any("excluded" in note for note in optimizer.recommend_campaigns().notes)


def test_invalid_usage_raises(synthetic_daily):
    with pytest.raises(ValueError, match="empty"):
        CampaignOptimizer().fit(synthetic_daily.iloc[0:0])
    with pytest.raises(ValueError, match="Missing required columns"):
        CampaignOptimizer().fit(synthetic_daily.drop(columns=["revenue"]))
    with pytest.raises(RuntimeError, match="Fit"):
        CampaignOptimizer().recommend_campaigns()
    with pytest.raises(ValueError, match="max_change"):
        CampaignOptimizer(max_change=-0.1)
    with pytest.raises(ValueError, match="max_budget"):
        CampaignOptimizer(min_budget=10, max_budget=5)
    with pytest.raises(ValueError, match="platform_limits"):
        CampaignOptimizer(platform_limits=(0.6, 0.4))
    with pytest.raises(ValueError, match="greater than zero"):
        CampaignOptimizer().fit(synthetic_daily).recommend_campaigns(0)


def test_works_on_generated_dataset(aggregates):
    optimizer = CampaignOptimizer().fit(aggregates.frames["campaign_daily"])
    result = optimizer.recommend_campaigns()
    shares = result.table.groupby("platform")["recommended_share_pct"].sum()
    assert shares.max() <= 40.0 + 1e-6 and shares.min() >= 5.0 - 1e-6
    assert result.table["recommended_daily_spend"].sum() == pytest.approx(result.total_daily_budget)
    summary = CampaignOptimizer.platform_summary(result.table)
    assert summary["recommended_share_pct"].sum() == pytest.approx(100.0)
