import numpy as np
import pandas as pd
import pytest

from budget_optimizer import (BudgetOptimizer, ResponseCurve, _project_to_limits, add_campaign_amounts, format_recommendation)


def random_feasible(rng, n, lo, hi):
    return _project_to_limits(list(rng.uniform(lo, hi, n)), 1.0, lo, hi)


def test_allocation_respects_budget_and_limits(daily_frame):
    result = BudgetOptimizer(0.05, 0.8).fit(daily_frame).recommend(1000)
    spend = result.table.recommended_daily_spend
    assert np.isclose(spend.sum(), 1000, atol=1e-6)
    assert (result.table.recommended_share_pct >= 5 - 1e-6).all()
    assert (result.table.recommended_share_pct <= 80 + 1e-6).all()


def test_solution_beats_random_feasible_allocations(daily_frame):
    optimizer = BudgetOptimizer(0.1, 0.6).fit(daily_frame)
    result = optimizer.recommend(1000)
    rng = np.random.default_rng(0)
    names = list(optimizer.curves)
    for _ in range(2000):
        shares = random_feasible(rng, len(names), 0.1, 0.6)
        revenue = sum(optimizer.curves[n].revenue(s * 1000) for n, s in zip(names, shares))
        assert revenue <= result.recommended_revenue + 1e-6


def test_kkt_conditions(daily_frame):
    table = BudgetOptimizer(0.01, 0.98).fit(daily_frame).recommend(1000).table
    free = table[(table.recommended_share_pct > 1 + 1e-6) & (table.recommended_share_pct < 98 - 1e-6)]
    assert len(free) >= 2
    assert free.marginal_roas.max() - free.marginal_roas.min() < 1e-6
    level = free.marginal_roas.mean()
    at_floor = table[table.recommended_share_pct <= 1 + 1e-6]
    assert (at_floor.marginal_roas <= level + 1e-9).all()


def test_historical_default_budget(daily_frame):
    optimizer = BudgetOptimizer(0.05, 0.8).fit(daily_frame)
    expected = sum(c.avg_daily_spend for c in optimizer.curves.values())
    assert np.isclose(optimizer.recommend().total_daily_budget, expected)


def test_optimum_never_worse_than_compliant_baseline(daily_frame):
    result = BudgetOptimizer(0.05, 0.5).fit(daily_frame).recommend(1000)
    assert result.recommended_revenue >= result.compliant_baseline_revenue - 1e-9
    assert result.uplift_vs_compliant_pct >= -1e-9


def test_current_allocation_outside_limits_is_flagged(daily_frame):
    skewed = daily_frame.assign(spend=daily_frame.spend.where(daily_frame.platform != "A", daily_frame.spend * 3))
    result = BudgetOptimizer(0.10, 0.50).fit(skewed).recommend(1000)
    assert not result.current_within_limits
    assert any("breaks the share limits" in note for note in result.notes)


def test_tight_equal_limits_force_equal_split(daily_frame):
    result = BudgetOptimizer(1 / 3, 1 / 3).fit(daily_frame).recommend(900)
    assert np.allclose(result.table.recommended_daily_spend, 300)


def test_extrapolation_flag(daily_frame):
    optimizer = BudgetOptimizer(0.05, 0.9).fit(daily_frame)
    assert optimizer.recommend(1000).table.extrapolated.any()
    historical = sum(c.avg_daily_spend for c in optimizer.curves.values())
    assert not optimizer.recommend(historical).table.extrapolated.all()


def test_elasticity_clipping_is_reported():
    dates = pd.date_range("2025-01-01", periods=30)
    rows = [{"platform": p, "date": d, "spend": 100 + 3 * i, "revenue": (100 + 3 * i) * 2.5}
            for p in "AB" for i, d in enumerate(dates)]
    optimizer = BudgetOptimizer(0.1, 0.9).fit(pd.DataFrame(rows))
    assert all(c.elasticity_clipped for c in optimizer.curves.values())
    assert optimizer.recommend(500).table.elasticity_clipped.all()


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_bad_budgets_rejected(daily_frame, bad):
    with pytest.raises(ValueError):
        BudgetOptimizer(0.05, 0.8).fit(daily_frame).recommend(bad)


@pytest.mark.parametrize("low,high", [(-0.1, 0.5), (0.6, 0.5), (0.1, 1.2), (np.nan, 0.5)])
def test_bad_limits_rejected(low, high):
    with pytest.raises(ValueError):
        BudgetOptimizer(low, high)


def test_infeasible_limits_rejected(daily_frame):
    with pytest.raises(ValueError, match="infeasible"):
        BudgetOptimizer(0.05, 0.2).fit(daily_frame)


def test_fit_validation(daily_frame):
    with pytest.raises(ValueError, match="Missing required columns"):
        BudgetOptimizer().fit(daily_frame.drop(columns="revenue"))
    with pytest.raises(ValueError, match="empty"):
        BudgetOptimizer().fit(daily_frame.iloc[0:0])
    with pytest.raises(ValueError, match="at least"):
        BudgetOptimizer(0.05, 0.8).fit(daily_frame.groupby("platform").head(5))
    flat = daily_frame.assign(spend=100.0)
    with pytest.raises(ValueError, match="does not vary"):
        BudgetOptimizer(0.05, 0.8).fit(flat)
    with pytest.raises(RuntimeError):
        BudgetOptimizer().recommend()


def test_response_curve_inverse_marginal():
    curve = ResponseCurve("X", 3.0, 0.7, 0.7, 0.9, 30, 100.0, 200.0, 50.0, 150.0)
    assert np.isclose(curve.spend_at_marginal(curve.marginal_revenue(123.0)), 123.0)


def test_campaign_amounts_sum_to_budget(daily_frame):
    table = add_campaign_amounts(BudgetOptimizer(0.05, 0.8).fit(daily_frame).recommend(1000).table, 1_000_000)
    assert np.isclose(table.recommended_campaign_amount.sum(), 1_000_000)
    with pytest.raises(ValueError):
        add_campaign_amounts(table, 0)


def test_report_text(daily_frame):
    text = format_recommendation(BudgetOptimizer(0.05, 0.8).fit(daily_frame).recommend(1000))
    assert "Uplift vs current" in text and "limit-compliant baseline" in text
