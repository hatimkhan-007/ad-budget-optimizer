from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_ELASTICITY = 0.05
MAX_ELASTICITY = 0.98
MIN_OBSERVATIONS = 10
BISECTION_ITERATIONS = 200


@dataclass
class ResponseCurve:
    platform: str
    a: float
    b: float
    raw_b: float
    r2: float
    n_obs: int
    avg_daily_spend: float
    avg_daily_revenue: float
    min_daily_spend: float
    max_daily_spend: float

    @property
    def elasticity_clipped(self) -> bool:
        return not math.isclose(self.b, self.raw_b, rel_tol=1e-12, abs_tol=1e-12)

    def revenue(self, spend: float) -> float:
        return self.a * max(float(spend), 1e-9) ** self.b

    def marginal_revenue(self, spend: float) -> float:
        return self.a * self.b * max(float(spend), 1e-9) ** (self.b - 1)

    def spend_at_marginal(self, marginal: float) -> float:
        """Spend at which marginal revenue equals ``marginal`` (inverse of the derivative)."""
        with np.errstate(over="ignore", divide="ignore"):
            return float(np.exp(np.log(self.a * self.b / marginal) / (1 - self.b)))


@dataclass
class AllocationResult:
    table: pd.DataFrame
    total_daily_budget: float
    current_revenue: float
    recommended_revenue: float
    compliant_baseline_revenue: float
    current_within_limits: bool
    notes: list[str] = field(default_factory=list)

    @property
    def uplift_pct(self) -> float:
        return _pct_change(self.recommended_revenue, self.current_revenue)

    @property
    def uplift_vs_compliant_pct(self) -> float:
        return _pct_change(self.recommended_revenue, self.compliant_baseline_revenue)

    @property
    def current_roas(self) -> float:
        return self.current_revenue / self.total_daily_budget

    @property
    def recommended_roas(self) -> float:
        return self.recommended_revenue / self.total_daily_budget


def _pct_change(new: float, old: float) -> float:
    return (new / old - 1) * 100 if old > 0 else float("nan")


def _project_to_limits(weights: list[float], budget: float, lower: float, upper: float) -> list[float]:
    """Scale ``weights`` to sum to ``budget`` while clipping each value to [lower, upper]."""
    w = np.asarray(weights, dtype=float)
    lo_k, hi_k = 0.0, max(upper / max(w.min(), 1e-12), 1.0)
    while np.clip(w * hi_k, lower, upper).sum() < budget and hi_k < 1e18:
        hi_k *= 2
    for _ in range(BISECTION_ITERATIONS):
        mid = (lo_k + hi_k) / 2
        if np.clip(w * mid, lower, upper).sum() < budget:
            lo_k = mid
        else:
            hi_k = mid
    return np.clip(w * hi_k, lower, upper).tolist()


class BudgetOptimizer:
    def __init__(self, min_share: float = 0.05, max_share: float = 0.40):
        if not (math.isfinite(min_share) and math.isfinite(max_share) and 0 <= min_share <= max_share <= 1):
            raise ValueError("Shares must satisfy 0 <= min_share <= max_share <= 1.")
        self.min_share, self.max_share = float(min_share), float(max_share)
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
                np.isfinite(group["spend"]) & np.isfinite(group["revenue"])
                & (group["spend"] > 0) & (group["revenue"] > 0), ["spend", "revenue"]
            ]
            if len(usable) < MIN_OBSERVATIONS:
                raise ValueError(f"{platform}: requires at least {MIN_OBSERVATIONS} positive spend/revenue observations.")
            x = np.log(usable["spend"].to_numpy(dtype=float))
            y = np.log(usable["revenue"].to_numpy(dtype=float))
            if np.ptp(x) == 0:
                raise ValueError(f"{platform}: daily spend does not vary, so a response curve cannot be fitted.")
            slope, intercept = np.polyfit(x, y, 1)
            slope = float(slope)
            ss_res = float(np.sum((y - (intercept + slope * x)) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            b = float(np.clip(slope, MIN_ELASTICITY, MAX_ELASTICITY))
            avg_spend, avg_revenue = float(usable["spend"].mean()), float(usable["revenue"].mean())
            a = avg_revenue / avg_spend ** b
            self.curves[str(platform)] = ResponseCurve(
                str(platform), a, b, slope, r2, len(usable), avg_spend, avg_revenue,
                float(usable["spend"].min()), float(usable["spend"].max()),
            )
        self._check_feasible()
        return self

    def _check_feasible(self) -> None:
        n = len(self.curves)
        if not n:
            raise RuntimeError("Fit the optimizer before requesting a recommendation.")
        if n * self.min_share > 1 + 1e-12 or n * self.max_share < 1 - 1e-12:
            raise ValueError(f"Share limits infeasible for {n} platforms: need n*min_share <= 1 <= n*max_share.")

    def _allocate(self, budget: float) -> dict[str, float]:
        """Exact optimum for concave power curves: equalise marginal revenue across unconstrained platforms."""
        names = list(self.curves)
        lower, upper = self.min_share * budget, self.max_share * budget
        floor = max(lower, 1e-12 * budget)

        def spends(marginal: float) -> np.ndarray:
            return np.array([np.clip(self.curves[n].spend_at_marginal(marginal), lower, upper) for n in names])

        low = min(self.curves[n].marginal_revenue(upper) for n in names) / 2
        high = max(self.curves[n].marginal_revenue(floor) for n in names) * 2
        lo_log, hi_log = math.log(low), math.log(high)
        for _ in range(BISECTION_ITERATIONS):
            mid = (lo_log + hi_log) / 2
            if spends(math.exp(mid)).sum() > budget:
                lo_log = mid
            else:
                hi_log = mid
        values = spends(math.exp(hi_log))
        shortfall = budget - values.sum()
        if abs(shortfall) > 0:
            free = [i for i, v in enumerate(values) if lower + 1e-12 < v < upper - 1e-12] or list(range(len(names)))
            best = max(free, key=lambda i: self.curves[names[i]].marginal_revenue(values[i]))
            values[best] = min(max(values[best] + shortfall, lower), upper)
        if abs(budget - values.sum()) > 1e-6 * max(1.0, budget):
            raise RuntimeError("Could not allocate the full budget within share constraints.")
        return dict(zip(names, values.tolist()))

    def recommend(self, total_daily_budget: float | None = None) -> AllocationResult:
        self._check_feasible()
        historical = {n: c.avg_daily_spend for n, c in self.curves.items()}
        historical_total = sum(historical.values())
        budget = historical_total if total_daily_budget is None else float(total_daily_budget)
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("total_daily_budget must be finite and greater than zero.")

        names = list(self.curves)
        current = {n: budget * s / historical_total for n, s in historical.items()}
        lower, upper = self.min_share * budget, self.max_share * budget
        within_limits = all(lower - 1e-9 <= v <= upper + 1e-9 for v in current.values())
        compliant = dict(zip(names, _project_to_limits([current[n] for n in names], budget, lower, upper)))
        recommended = self._allocate(budget)

        rows = []
        for name, curve in self.curves.items():
            cur, rec = current[name], recommended[name]
            rows.append({
                "platform": name,
                "current_share_pct": cur / budget * 100,
                "recommended_share_pct": rec / budget * 100,
                "change_pp": (rec - cur) / budget * 100,
                "current_daily_spend": cur,
                "recommended_daily_spend": rec,
                "current_roas": curve.revenue(cur) / cur,
                "projected_roas": curve.revenue(rec) / rec,
                "marginal_roas": curve.marginal_revenue(rec),
                "elasticity": curve.b,
                "elasticity_clipped": curve.elasticity_clipped,
                "fit_r2": curve.r2,
                "n_observations": curve.n_obs,
                "extrapolated": not (curve.min_daily_spend <= rec <= curve.max_daily_spend),
            })
        table = pd.DataFrame(rows).sort_values("recommended_share_pct", ascending=False).reset_index(drop=True)

        notes = []
        if not within_limits:
            notes.append("The current allocation breaks the share limits, so uplift versus current mixes "
                         "optimization gains with the cost of complying with the limits; compare against the "
                         "limit-compliant baseline as well.")
        if table["extrapolated"].any():
            flagged = ", ".join(table.loc[table["extrapolated"], "platform"])
            notes.append(f"Recommended spend lies outside the observed daily range for: {flagged}. "
                         "Treat those projections as extrapolation.")
        if table["elasticity_clipped"].any():
            flagged = ", ".join(table.loc[table["elasticity_clipped"], "platform"])
            notes.append(f"The fitted elasticity was clipped to [{MIN_ELASTICITY}, {MAX_ELASTICITY}] for: {flagged}.")
        weak = table.loc[table["fit_r2"] < 0.8, "platform"].tolist()
        if weak:
            notes.append(f"Curve fit R² is below 0.80 for: {', '.join(weak)}. Treat those projections cautiously.")

        current_revenue = sum(self.curves[n].revenue(current[n]) for n in names)
        recommended_revenue = sum(self.curves[n].revenue(recommended[n]) for n in names)
        compliant_revenue = sum(self.curves[n].revenue(compliant[n]) for n in names)
        return AllocationResult(table, budget, current_revenue, recommended_revenue,
                                compliant_revenue, within_limits, notes)


def add_campaign_amounts(table: pd.DataFrame, campaign_budget: float) -> pd.DataFrame:
    if not math.isfinite(campaign_budget) or campaign_budget <= 0:
        raise ValueError("campaign_budget must be finite and greater than zero.")
    out = table.copy()
    out["recommended_campaign_amount"] = out["recommended_share_pct"] / 100 * campaign_budget
    return out


def format_recommendation(result: AllocationResult) -> str:
    columns = ["platform", "current_share_pct", "recommended_share_pct", "change_pp",
               "current_roas", "projected_roas", "marginal_roas", "elasticity", "fit_r2"]
    shown = [c for c in columns + ["recommended_campaign_amount"] if c in result.table.columns]
    lines = [
        result.table[shown].to_string(index=False, float_format=lambda x: f"{x:,.2f}"),
        "",
        f"Projected revenue/day, current allocation:        {result.current_revenue:,.2f}",
        f"Projected revenue/day, limit-compliant baseline:  {result.compliant_baseline_revenue:,.2f}",
        f"Projected revenue/day, recommended allocation:    {result.recommended_revenue:,.2f}",
        f"Uplift vs current: {result.uplift_pct:,.2f}%   |   vs limit-compliant baseline: {result.uplift_vs_compliant_pct:,.2f}%",
        f"Modeled ROAS: {result.current_roas:,.3f} -> {result.recommended_roas:,.3f}",
    ]
    lines += [f"Note: {note}" for note in result.notes]
    return "\n".join(lines)
