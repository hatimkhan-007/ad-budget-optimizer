"""Budget allocation recommendation engine.

Approach
--------
1. Aggregate history to platform-day totals (spend, revenue).
2. Fit a diminishing-returns response curve per platform:

       revenue = a * spend ^ b        (0 < b < 1)

   estimated by log-log least squares, then calibrated so the curve reproduces
   each platform's average historical daily revenue at its average daily spend.
3. Allocate the total daily budget in small increments, always giving the next
   slice to the platform with the highest marginal revenue. For concave curves
   this greedy procedure is optimal.
4. Minimum and maximum platform shares keep the plan realistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_ELASTICITY = 0.05
MAX_ELASTICITY = 0.98
MIN_OBSERVATIONS = 10


@dataclass
class ResponseCurve:
    platform: str
    a: float
    b: float
    r2: float
    n_obs: int
    avg_daily_spend: float
    avg_daily_revenue: float

    def revenue(self, spend: float) -> float:
        return self.a * max(spend, 1e-9) ** self.b

    def marginal_revenue(self, spend: float) -> float:
        return self.a * self.b * max(spend, 1e-9) ** (self.b - 1)


@dataclass
class AllocationResult:
    table: pd.DataFrame
    total_daily_budget: float
    current_revenue: float
    recommended_revenue: float

    @property
    def uplift_pct(self) -> float:
        return (self.recommended_revenue / self.current_revenue - 1) * 100

    @property
    def current_roas(self) -> float:
        return self.current_revenue / self.total_daily_budget

    @property
    def recommended_roas(self) -> float:
        return self.recommended_revenue / self.total_daily_budget


class BudgetOptimizer:
    def __init__(self, min_share: float = 0.05, max_share: float = 0.40,
                 steps: int = 2000) -> None:
        if not 0 <= min_share <= max_share <= 1:
            raise ValueError("Require 0 <= min_share <= max_share <= 1")
        self.min_share = min_share
        self.max_share = max_share
        self.steps = steps
        self.curves: dict[str, ResponseCurve] = {}

    def fit(self, daily: pd.DataFrame) -> "BudgetOptimizer":
        """Fit response curves from platform-day rows (platform, date, spend, revenue)."""
        self.curves = {}
        for platform, group in daily.groupby("platform", observed=True):
            usable = group[(group["spend"] > 0) & (group["revenue"] > 0)]
            if len(usable) < MIN_OBSERVATIONS:
                raise ValueError(
                    f"Not enough history for {platform}: {len(usable)} usable days")
            x = np.log(usable["spend"].to_numpy(dtype=float))
            y = np.log(usable["revenue"].to_numpy(dtype=float))
            slope, intercept = np.polyfit(x, y, 1)
            predicted = intercept + slope * x
            ss_res = float(np.sum((y - predicted) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            b = float(np.clip(slope, MIN_ELASTICITY, MAX_ELASTICITY))
            avg_spend = float(usable["spend"].mean())
            avg_revenue = float(usable["revenue"].mean())
            a = avg_revenue / avg_spend ** b
            self.curves[str(platform)] = ResponseCurve(
                str(platform), a, b, r2, len(usable), avg_spend, avg_revenue)
        return self

    def _check_feasible(self) -> None:
        n = len(self.curves)
        if not self.curves:
            raise RuntimeError("Call fit() before recommend()")
        if n * self.min_share > 1 + 1e-9 or n * self.max_share < 1 - 1e-9:
            raise ValueError("Share limits are infeasible for the number of platforms")

    def _allocate(self, budget: float) -> dict[str, float]:
        floor = self.min_share * budget
        cap = self.max_share * budget
        spend = {name: floor for name in self.curves}
        step = (budget - floor * len(spend)) / self.steps
        for _ in range(self.steps):
            best_name, best_gain = None, -np.inf
            for name, curve in self.curves.items():
                if spend[name] + step > cap + 1e-9 * budget:
                    continue
                gain = curve.revenue(spend[name] + step) - curve.revenue(spend[name])
                if gain > best_gain:
                    best_name, best_gain = name, gain
            if best_name is None:
                break
            spend[best_name] += step
        return spend

    def recommend(self, total_daily_budget: float | None = None) -> AllocationResult:
        """Return the revenue-maximising allocation of a total daily budget."""
        self._check_feasible()
        historical = {n: c.avg_daily_spend for n, c in self.curves.items()}
        historical_total = sum(historical.values())
        budget = float(total_daily_budget) if total_daily_budget else historical_total

        current = {n: budget * s / historical_total for n, s in historical.items()}
        recommended = self._allocate(budget)

        rows = []
        for name, curve in self.curves.items():
            cur_rev = curve.revenue(current[name])
            rec_rev = curve.revenue(recommended[name])
            rows.append({
                "platform": name,
                "current_share_pct": current[name] / budget * 100,
                "recommended_share_pct": recommended[name] / budget * 100,
                "change_pp": (recommended[name] - current[name]) / budget * 100,
                "current_daily_spend": current[name],
                "recommended_daily_spend": recommended[name],
                "current_roas": cur_rev / current[name],
                "projected_roas": rec_rev / recommended[name],
                "marginal_roas": curve.marginal_revenue(recommended[name]),
                "elasticity": curve.b,
                "fit_r2": curve.r2,
            })
        table = pd.DataFrame(rows).sort_values(
            "recommended_share_pct", ascending=False).reset_index(drop=True)

        current_revenue = sum(self.curves[n].revenue(current[n]) for n in self.curves)
        recommended_revenue = sum(self.curves[n].revenue(recommended[n]) for n in self.curves)
        return AllocationResult(table, budget, current_revenue, recommended_revenue)


def format_recommendation(result: AllocationResult) -> str:
    t = result.table
    view = pd.DataFrame({
        "Platform": t["platform"],
        "Current %": t["current_share_pct"].map("{:.1f}".format),
        "Recommended %": t["recommended_share_pct"].map("{:.1f}".format),
        "Change (pp)": t["change_pp"].map("{:+.1f}".format),
        "Rec. daily $": t["recommended_daily_spend"].map("{:,.0f}".format),
        "ROAS now": t["current_roas"].map("{:.2f}x".format),
        "ROAS proj.": t["projected_roas"].map("{:.2f}x".format),
        "Marginal ROAS": t["marginal_roas"].map("{:.2f}x".format),
        "Elasticity": t["elasticity"].map("{:.2f}".format),
        "Fit R2": t["fit_r2"].map("{:.2f}".format),
    })
    summary = (
        f"\nTotal daily budget: ${result.total_daily_budget:,.0f}"
        f"\nProjected daily revenue: ${result.current_revenue:,.0f} -> "
        f"${result.recommended_revenue:,.0f} ({result.uplift_pct:+.1f}%)"
        f"\nBlended ROAS: {result.current_roas:.2f}x -> {result.recommended_roas:.2f}x"
    )
    return view.to_string(index=False) + summary