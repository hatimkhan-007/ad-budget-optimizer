from __future__ import annotations
from dataclasses import dataclass
import math
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
        return self.a * max(float(spend), 1e-9) ** self.b
    def marginal_revenue(self, spend: float) -> float:
        return self.a * self.b * max(float(spend), 1e-9) ** (self.b - 1)

@dataclass
class AllocationResult:
    table: pd.DataFrame
    total_daily_budget: float
    current_revenue: float
    recommended_revenue: float
    @property
    def uplift_pct(self) -> float:
        return (self.recommended_revenue / self.current_revenue - 1) * 100 if self.current_revenue > 0 else float("nan")
    @property
    def current_roas(self) -> float:
        return self.current_revenue / self.total_daily_budget
    @property
    def recommended_roas(self) -> float:
        return self.recommended_revenue / self.total_daily_budget

class BudgetOptimizer:
    def __init__(self, min_share: float = .05, max_share: float = .40, steps: int = 2000):
        if not (math.isfinite(min_share) and math.isfinite(max_share) and 0 <= min_share <= max_share <= 1):
            raise ValueError("Shares must satisfy 0 <= min_share <= max_share <= 1.")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 100:
            raise ValueError("steps must be an integer >= 100.")
        self.min_share, self.max_share, self.steps = float(min_share), float(max_share), steps
        self.curves: dict[str, ResponseCurve] = {}

    def fit(self, daily: pd.DataFrame) -> "BudgetOptimizer":
        required = {"platform", "spend", "revenue"}
        missing = required - set(daily.columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        if daily.empty:
            raise ValueError("Cannot fit optimizer on an empty dataset.")
        self.curves = {}
        for platform, group in daily.groupby("platform", observed=True):
            usable = group.loc[
                np.isfinite(group["spend"]) & np.isfinite(group["revenue"]) &
                (group["spend"] > 0) & (group["revenue"] > 0), ["spend", "revenue"]
            ]
            if len(usable) < MIN_OBSERVATIONS:
                raise ValueError(f"{platform}: requires at least {MIN_OBSERVATIONS} positive spend/revenue observations.")
            x = np.log(usable["spend"].to_numpy(dtype=float))
            y = np.log(usable["revenue"].to_numpy(dtype=float))
            slope, intercept = np.polyfit(x, y, 1)
            slope = float(slope)
            fitted = intercept + slope * x
            ss_res = float(np.sum((y - fitted) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            raw_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            b = float(np.clip(slope, MIN_ELASTICITY, MAX_ELASTICITY))
            avg_spend, avg_revenue = float(usable["spend"].mean()), float(usable["revenue"].mean())
            a = avg_revenue / avg_spend ** b
            self.curves[str(platform)] = ResponseCurve(str(platform), a, b, raw_r2, len(usable), avg_spend, avg_revenue)
        self._check_feasible()
        return self

    def _check_feasible(self) -> None:
        n = len(self.curves)
        if not n:
            raise RuntimeError("Fit the optimizer before requesting a recommendation.")
        if n * self.min_share > 1 + 1e-12 or n * self.max_share < 1 - 1e-12:
            raise ValueError(f"Share limits infeasible for {n} platforms: need n*min_share <= 1 <= n*max_share.")

    def _allocate(self, budget: float) -> dict[str, float]:
        names = list(self.curves)
        lower, upper = self.min_share * budget, self.max_share * budget
        allocation = {name: lower for name in names}
        remaining = budget - lower * len(names)
        # Discrete marginal-gain allocation. Use equal budget increments and place any residual exactly.
        step = remaining / self.steps if remaining > 0 else 0
        for _ in range(self.steps):
            if remaining <= 1e-10 * max(1.0, budget):
                break
            candidates = [n for n in names if allocation[n] + min(step, remaining) <= upper + 1e-9 * max(1.0, budget)]
            if not candidates:
                candidates = [n for n in names if allocation[n] < upper - 1e-9 * max(1.0, budget)]
            if not candidates:
                break
            amount = min(step, remaining)
            best = max(candidates, key=lambda n: self.curves[n].revenue(allocation[n] + amount) - self.curves[n].revenue(allocation[n]))
            allocation[best] += amount
            remaining -= amount
        if remaining > 1e-7 * max(1.0, budget):
            for name in sorted(names, key=lambda n: self.curves[n].marginal_revenue(allocation[n]), reverse=True):
                amount = min(remaining, upper - allocation[name])
                if amount > 0:
                    allocation[name] += amount
                    remaining -= amount
                if remaining <= 1e-7 * max(1.0, budget):
                    break
        if remaining > 1e-6 * max(1.0, budget):
            raise RuntimeError("Could not allocate the full budget within share constraints.")
        return allocation

    def recommend(self, total_daily_budget: float | None = None) -> AllocationResult:
        self._check_feasible()
        historical = {n: c.avg_daily_spend for n, c in self.curves.items()}
        historical_total = sum(historical.values())
        if total_daily_budget is None:
            budget = historical_total
        else:
            budget = float(total_daily_budget)
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("total_daily_budget must be finite and greater than zero.")
        current = {n: budget * s / historical_total for n, s in historical.items()}
        recommended = self._allocate(budget)
        rows = []
        for name, curve in self.curves.items():
            cur, rec = current[name], recommended[name]
            rows.append({"platform": name, "current_share_pct": cur / budget * 100,
                "recommended_share_pct": rec / budget * 100, "change_pp": (rec-cur)/budget*100,
                "current_daily_spend": cur, "recommended_daily_spend": rec,
                "current_roas": curve.revenue(cur)/cur, "projected_roas": curve.revenue(rec)/rec,
                "marginal_roas": curve.marginal_revenue(rec), "elasticity": curve.b,
                "fit_r2": curve.r2, "n_observations": curve.n_obs})
        table = pd.DataFrame(rows).sort_values("recommended_share_pct", ascending=False).reset_index(drop=True)
        current_revenue = sum(self.curves[n].revenue(current[n]) for n in self.curves)
        recommended_revenue = sum(self.curves[n].revenue(recommended[n]) for n in self.curves)
        return AllocationResult(table, budget, current_revenue, recommended_revenue)

def format_recommendation(result: AllocationResult) -> str:
    return (result.table.to_string(index=False, float_format=lambda x: f"{x:,.3f}") +
        f"\n\nProjected current revenue/day: {result.current_revenue:,.2f}" +
        f"\nProjected recommended revenue/day: {result.recommended_revenue:,.2f}" +
        f"\nProjected uplift: {result.uplift_pct:,.2f}%" +
        f"\nCurrent modeled ROAS: {result.current_roas:,.3f}" +
        f"\nRecommended modeled ROAS: {result.recommended_roas:,.3f}")
