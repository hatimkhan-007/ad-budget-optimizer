"""Campaign-level budget optimisation with explainable recommendations.

Model
-----
Platform-level response curves ``revenue = a * spend**b`` are fitted exactly as in
``budget_optimizer.BudgetOptimizer``. Individual campaigns have too little daily
history to estimate their own saturation reliably, so each campaign *inherits its
platform's elasticity* ``b`` (partial pooling) and receives its own efficiency
coefficient ``a``, calibrated so the curve reproduces that campaign's average
daily spend and revenue.

Allocation
----------
For concave curves the revenue-maximising split of a fixed budget equalises
marginal revenue across campaigns, except where a campaign is held at a bound.
Bounds per campaign are

* an optional absolute minimum and maximum daily budget, and
* a maximum relative change from the current daily budget (default 50%).

The change cap is a trust region: curves are only reliable near the spend levels
that were actually observed, and a nearly linear campaign would otherwise absorb
the entire budget. The bounded problem is solved exactly by bisection on the
marginal-revenue level.

All projected figures are model estimates, not guarantees.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from budget_optimizer import AllocationResult, BudgetOptimizer, ResponseCurve, _project_to_limits

MIN_CAMPAIGN_OBSERVATIONS = 5
DEFAULT_MAX_CHANGE = 0.5
HIGH_PERFORMER_RATIO = 1.15
LOW_PERFORMER_RATIO = 0.85
CHANGE_THRESHOLD_PP = 0.01
BISECTION_ITERATIONS = 200
BOUND_TOLERANCE = 1e-9
MARGINAL_SEARCH_RANGE = (1e-6, 1e6)
DEFAULT_PLATFORM_LIMITS = (0.05, 0.40)


def _spend_at(a: np.ndarray, b: np.ndarray, marginal: float, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Spend at which each curve's marginal revenue equals ``marginal``, clipped to its bounds."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        raw = np.exp(np.log(a * b / marginal) / (1 - b))
    return np.clip(np.nan_to_num(raw, nan=0.0, posinf=float(np.max(upper))), lower, upper)


def _bisect_marginal(total_at: "callable", target: float) -> float:
    """Find the marginal-revenue level (log scale) at which ``total_at`` equals ``target``."""
    lo_log, hi_log = math.log(MARGINAL_SEARCH_RANGE[0]), math.log(MARGINAL_SEARCH_RANGE[1])
    for _ in range(BISECTION_ITERATIONS):
        mid = (lo_log + hi_log) / 2
        if total_at(math.exp(mid)) > target:
            lo_log = mid
        else:
            hi_log = mid
    return math.exp(hi_log)


def allocate_with_bounds(curves: list[ResponseCurve], budget: float, lower: np.ndarray, upper: np.ndarray,
                         groups: np.ndarray | None = None, group_lower: np.ndarray | None = None,
                         group_upper: np.ndarray | None = None) -> np.ndarray:
    """Exact revenue-maximising allocation of ``budget`` across power-law curves.

    Each curve has its own ``lower``/``upper`` bound. Optionally, curves belong to
    ``groups`` (integer ids 0..G-1, for example platforms) whose total spend must lie
    within ``group_lower``/``group_upper``. The solution satisfies the KKT conditions:
    marginal revenue is equal across curves inside a group that is not at a group
    limit, and curves at their own bounds are held there.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    n = len(curves)
    if len(lower) != n or len(upper) != n:
        raise ValueError("curves, lower and upper must have the same length.")
    if np.any(lower > upper + 1e-12):
        raise ValueError("Every campaign needs lower bound <= upper bound.")
    a = np.array([c.a for c in curves])
    b = np.array([c.b for c in curves])
    tolerance = 1e-9 * max(1.0, budget)

    if lower.sum() > budget + tolerance or upper.sum() < budget - tolerance:
        raise ValueError(
            f"Budget limits are infeasible: the limits allow a total between {lower.sum():,.2f} and "
            f"{upper.sum():,.2f}, but the budget is {budget:,.2f}. Widen the limits or change the budget.")

    if groups is None:
        groups = np.zeros(n, dtype=int)
        group_lower = np.array([0.0])
        group_upper = np.array([budget])
    groups = np.asarray(groups, dtype=int)
    group_ids = np.unique(groups)
    masks = [groups == g for g in group_ids]
    g_lo = np.array([max(float(group_lower[g]), lower[m].sum()) for g, m in zip(group_ids, masks)])
    g_hi = np.array([min(float(group_upper[g]), upper[m].sum()) for g, m in zip(group_ids, masks)])
    if np.any(g_lo > g_hi + tolerance):
        raise ValueError("Platform and campaign limits conflict: a platform cannot satisfy both. "
                         "Widen the campaign or platform limits.")
    if g_lo.sum() > budget + tolerance or g_hi.sum() < budget - tolerance:
        raise ValueError(
            f"Budget limits are infeasible: the limits allow a total between {g_lo.sum():,.2f} and "
            f"{g_hi.sum():,.2f}, but the budget is {budget:,.2f}. Widen the limits or change the budget.")

    def group_totals(marginal: float) -> np.ndarray:
        spend = _spend_at(a, b, marginal, lower, upper)
        return np.array([spend[m].sum() for m in masks])

    def total_at(marginal: float) -> float:
        return float(np.clip(group_totals(marginal), g_lo, g_hi).sum())

    level = _bisect_marginal(total_at, budget)
    values = _spend_at(a, b, level, lower, upper)
    sums = group_totals(level)
    for k, mask in enumerate(masks):
        target = float(np.clip(sums[k], g_lo[k], g_hi[k]))
        if abs(target - sums[k]) > 1e-12 * max(1.0, budget):
            sub_a, sub_b, sub_lo, sub_hi = a[mask], b[mask], lower[mask], upper[mask]
            group_level = _bisect_marginal(
                lambda m: float(_spend_at(sub_a, sub_b, m, sub_lo, sub_hi).sum()), target)
            values[mask] = _spend_at(sub_a, sub_b, group_level, sub_lo, sub_hi)

    shortfall = budget - values.sum()
    if abs(shortfall) > 0:
        movable = [i for i in range(n) if lower[i] + 1e-12 < values[i] < upper[i] - 1e-12] or list(range(n))
        best = max(movable, key=lambda i: curves[i].marginal_revenue(max(values[i], 1e-12)))
        values[best] = min(max(values[best] + shortfall, lower[best]), upper[best])
    if abs(budget - values.sum()) > 1e-6 * max(1.0, budget):
        raise RuntimeError("Could not allocate the full budget within the limits.")
    return values


class CampaignOptimizer:
    """Allocate a daily budget across individual campaigns."""

    def __init__(self, max_change: float | None = DEFAULT_MAX_CHANGE, min_budget: float = 0.0,
                 max_budget: float | None = None,
                 platform_limits: tuple[float, float] | None = DEFAULT_PLATFORM_LIMITS):
        if max_change is not None and not (math.isfinite(max_change) and max_change >= 0):
            raise ValueError("max_change must be a non-negative number or None.")
        if not math.isfinite(min_budget) or min_budget < 0:
            raise ValueError("min_budget must be a non-negative number.")
        if max_budget is not None and (not math.isfinite(max_budget) or max_budget < min_budget):
            raise ValueError("max_budget must be at least min_budget.")
        if platform_limits is not None:
            low, high = platform_limits
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high <= 1):
                raise ValueError("platform_limits must satisfy 0 <= min <= max <= 1.")
        self.platform_limits = platform_limits
        self.max_change = max_change
        self.min_budget = float(min_budget)
        self.max_budget = None if max_budget is None else float(max_budget)
        self.platform_curves: dict[str, ResponseCurve] = {}
        self.curves: dict[str, ResponseCurve] = {}
        self.campaign_platform: dict[str, str] = {}
        self.excluded: dict[str, str] = {}

    def fit(self, campaign_daily: pd.DataFrame) -> "CampaignOptimizer":
        """Fit platform curves, then campaign curves that inherit platform elasticity."""
        required = {"campaign_id", "platform", "date", "spend", "revenue"}
        missing = required - set(campaign_daily.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        if campaign_daily.empty:
            raise ValueError("Cannot fit optimizer on an empty dataset.")

        platform_daily = (campaign_daily.groupby(["platform", "date"], observed=True)[["spend", "revenue"]]
                          .sum().reset_index())
        self.platform_curves = BudgetOptimizer(0.0, 1.0).fit(platform_daily).curves

        per_campaign = (campaign_daily.groupby(["campaign_id", "platform", "date"], observed=True)[["spend", "revenue"]]
                        .sum().reset_index())
        self.curves, self.campaign_platform, self.excluded = {}, {}, {}
        for (campaign_id, platform), group in per_campaign.groupby(["campaign_id", "platform"], observed=True):
            usable = group[(group["spend"] > 0) & (group["revenue"] > 0)]
            if len(usable) < MIN_CAMPAIGN_OBSERVATIONS:
                self.excluded[str(campaign_id)] = (
                    f"only {len(usable)} day(s) with positive spend and revenue (minimum {MIN_CAMPAIGN_OBSERVATIONS})")
                continue
            parent = self.platform_curves[str(platform)]
            avg_spend, avg_revenue = float(usable["spend"].mean()), float(usable["revenue"].mean())
            self.curves[str(campaign_id)] = ResponseCurve(
                str(campaign_id), avg_revenue / avg_spend ** parent.b, parent.b, parent.raw_b, parent.r2,
                len(usable), avg_spend, avg_revenue, float(usable["spend"].min()), float(usable["spend"].max()))
            self.campaign_platform[str(campaign_id)] = str(platform)
        if not self.curves:
            raise ValueError("No campaign has enough history to be optimised.")
        return self

    def _bounds(self, current: np.ndarray, budget: float) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(len(current), self.min_budget)
        upper = np.full(len(current), budget if self.max_budget is None else min(self.max_budget, budget))
        if self.max_change is not None:
            lower = np.maximum(lower, current * (1 - self.max_change))
            upper = np.minimum(upper, current * (1 + self.max_change))
        upper = np.maximum(upper, lower)
        return lower, upper

    def recommend_campaigns(self, total_daily_budget: float | None = None,
                            catalog: pd.DataFrame | None = None) -> AllocationResult:
        """Optimise the campaign split and return an annotated table with reasons.

        ``total_daily_budget`` defaults to the historical total. For a different budget the current mix is
        scaled proportionally and the change limit applies around that scaled mix.
        """
        if not self.curves:
            raise RuntimeError("Fit the optimizer before requesting a recommendation.")
        names = list(self.curves)
        curve_list = [self.curves[n] for n in names]
        historical = np.array([c.avg_daily_spend for c in curve_list])
        budget = float(historical.sum()) if total_daily_budget is None else float(total_daily_budget)
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("total_daily_budget must be finite and greater than zero.")

        current = budget * historical / historical.sum()
        lower, upper = self._bounds(current, budget)
        platforms = sorted(set(self.campaign_platform.values()))
        group_index = np.array([platforms.index(self.campaign_platform[n]) for n in names])
        if self.platform_limits is None:
            group_lower = np.zeros(len(platforms))
            group_upper = np.full(len(platforms), budget)
        else:
            group_lower = np.full(len(platforms), self.platform_limits[0] * budget)
            group_upper = np.full(len(platforms), self.platform_limits[1] * budget)
        recommended = allocate_with_bounds(curve_list, budget, lower, upper, group_index, group_lower, group_upper)

        abs_upper = budget if self.max_budget is None else min(self.max_budget, budget)
        abs_ok = bool(np.all(current >= self.min_budget - 1e-9) and np.all(current <= abs_upper + 1e-9))
        current_platform = np.array([current[group_index == g].sum() for g in range(len(platforms))])
        platform_ok = bool(np.all(current_platform >= group_lower - 1e-9) and np.all(current_platform <= group_upper + 1e-9))
        within = abs_ok and platform_ok
        if not platform_ok:
            projected = np.array(_project_to_limits(current_platform.tolist(), budget, group_lower[0], group_upper[0]))
            compliant = current * (projected / current_platform)[group_index]
        elif not abs_ok:
            compliant = np.array(_project_to_limits(current.tolist(), budget, self.min_budget, abs_upper))
        else:
            compliant = current

        def revenue(values: np.ndarray) -> float:
            return float(sum(c.revenue(v) for c, v in zip(curve_list, values)))

        rows = []
        for i, name in enumerate(names):
            curve, cur, rec = curve_list[i], float(current[i]), float(recommended[i])
            rows.append({
                "campaign_id": name, "platform": self.campaign_platform[name],
                "current_share_pct": cur / budget * 100, "recommended_share_pct": rec / budget * 100,
                "change_pp": (rec - cur) / budget * 100, "current_daily_spend": cur, "recommended_daily_spend": rec,
                "min_daily_budget": float(lower[i]), "max_daily_budget": float(upper[i]),
                "current_roas": curve.revenue(cur) / cur, "projected_roas": curve.revenue(rec) / rec,
                "marginal_roas": curve.marginal_revenue(rec), "elasticity": curve.b, "fit_r2": curve.r2,
                "n_observations": curve.n_obs,
                "extrapolated": not (curve.min_daily_spend <= rec <= curve.max_daily_spend),
            })
        table = pd.DataFrame(rows)
        rec_platform = {p: float(recommended[group_index == i].sum()) for i, p in enumerate(platforms)}
        limit_flags = {}
        for i, p in enumerate(platforms):
            if self.platform_limits is None:
                limit_flags[p] = ""
            elif rec_platform[p] >= group_upper[i] * (1 - BOUND_TOLERANCE) - 1e-9:
                limit_flags[p] = "At platform maximum"
            elif rec_platform[p] <= group_lower[i] * (1 + BOUND_TOLERANCE) + 1e-9:
                limit_flags[p] = "At platform minimum"
            else:
                limit_flags[p] = ""
        table["platform_limit"] = table["platform"].map(limit_flags)
        result = AllocationResult(table, budget, revenue(current), revenue(recommended), revenue(compliant), within, [])
        table = self._annotate(table, result)
        if catalog is not None and not catalog.empty:
            table = table.merge(catalog[["campaign_id", "campaign_name", "objective", "target_audience"]],
                                on="campaign_id", how="left")
        table["reason"] = [self._explain(row, result) for row in table.itertuples()]
        result.table = table.sort_values("recommended_share_pct", ascending=False).reset_index(drop=True)
        result.notes = self._notes(result)
        return result

    def _annotate(self, table: pd.DataFrame, result: AllocationResult) -> pd.DataFrame:
        table = table.copy()
        portfolio_roas = result.current_revenue / result.total_daily_budget
        ratio = table["current_roas"] / portfolio_roas
        table["historical_roas"] = table["current_roas"]
        table["performance_tier"] = np.select(
            [ratio >= HIGH_PERFORMER_RATIO, ratio <= LOW_PERFORMER_RATIO],
            ["High performer", "Underperformer"], default="Average")
        table["action"] = np.select(
            [table["change_pp"] > CHANGE_THRESHOLD_PP, table["change_pp"] < -CHANGE_THRESHOLD_PP],
            ["Increase", "Decrease"], default="Hold")
        at_min = table["recommended_daily_spend"] <= table["min_daily_budget"] * (1 + BOUND_TOLERANCE) + 1e-9
        at_max = table["recommended_daily_spend"] >= table["max_daily_budget"] * (1 - BOUND_TOLERANCE) - 1e-9
        table["constraint"] = np.select([at_min, at_max], ["At minimum", "At maximum"], default="Free")
        table["expected_revenue_change"] = (
            table["recommended_daily_spend"] * table["projected_roas"]
            - table["current_daily_spend"] * table["current_roas"])
        return table

    def _explain(self, row, result: AllocationResult) -> str:
        portfolio = result.current_revenue / result.total_daily_budget
        text = (f"Historical ROAS {row.historical_roas:.2f}x vs portfolio {portfolio:.2f}x "
                f"({row.performance_tier.lower()}). Modelled marginal ROAS at the recommended spend: "
                f"{row.marginal_roas:.2f}x.")
        if row.platform_limit:
            text += f" Its platform is held {row.platform_limit.lower().replace('at platform ', 'at its ')} share limit."
        if row.constraint == "At maximum":
            return text + " Held at its maximum limit; the model would add more but the limit applies."
        if row.constraint == "At minimum":
            return text + " Held at its minimum limit; the model would cut further but the limit applies."
        if row.action == "Increase":
            return text + " Budget moves here because extra spend still earns more than in the average campaign."
        if row.action == "Decrease":
            return text + " Budget moves out because returns are lower than elsewhere in the portfolio."
        return text + " The current budget is already close to the optimum."

    def _notes(self, result: AllocationResult) -> list[str]:
        table = result.table
        notes = []
        if not result.current_within_limits:
            notes.append("The current allocation breaks the configured budget or platform share limits, so uplift versus current "
                         "mixes optimisation gains with the cost of complying; compare against the limit-compliant "
                         "baseline as well.")
        if table["extrapolated"].any():
            notes.append(f"{int(table['extrapolated'].sum())} campaign(s) are recommended spend levels outside "
                         "their observed daily range. Treat those projections as extrapolation.")
        weak = table["fit_r2"] < 0.8
        if weak.any():
            platforms = sorted(table.loc[weak, "platform"].unique())
            notes.append(f"The pooled platform curve fit has R2 below 0.80 for: {', '.join(platforms)}. "
                         "Treat their campaign projections cautiously.")
        if self.excluded:
            notes.append(f"{len(self.excluded)} campaign(s) were excluded for insufficient history.")
        return notes

    @staticmethod
    def platform_summary(table: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a campaign recommendation table to platform level."""
        summary = table.groupby("platform", observed=True)[["current_daily_spend", "recommended_daily_spend"]].sum().reset_index()
        summary["current_share_pct"] = summary["current_daily_spend"] / summary["current_daily_spend"].sum() * 100
        summary["recommended_share_pct"] = summary["recommended_daily_spend"] / summary["recommended_daily_spend"].sum() * 100
        return summary
