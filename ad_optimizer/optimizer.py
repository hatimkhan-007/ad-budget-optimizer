"""Advisory budget allocation using observed campaign performance."""
from __future__ import annotations

import pandas as pd


def recommend_budget(
    campaigns: pd.DataFrame,
    total_budget: float,
    min_budget: float = 0.0,
) -> pd.DataFrame:
    """Allocate total_budget across campaigns based on smoothed conversions per spend.

    This is a transparent heuristic, not a causal or predictive ML model.
    `total_budget` and `min_budget` must use the same currency as the source account.
    """
    if total_budget <= 0:
        raise ValueError("total_budget must be greater than zero.")
    if min_budget < 0:
        raise ValueError("min_budget cannot be negative.")
    if campaigns.empty:
        raise ValueError("No campaigns available to optimize.")

    result = campaigns.copy()
    if "spend" not in result or "conversions" not in result:
        raise ValueError("Campaign data must contain 'spend' and 'conversions'.")

    result["spend"] = pd.to_numeric(result["spend"], errors="coerce").fillna(0).clip(lower=0)
    result["conversions"] = pd.to_numeric(
        result["conversions"], errors="coerce"
    ).fillna(0).clip(lower=0)

    n = len(result)
    if min_budget * n > total_budget:
        raise ValueError("Minimum budget per campaign exceeds the total budget.")

    # Small smoothing term avoids assigning zero weight to campaigns with no observed conversions.
    result["efficiency_score"] = (result["conversions"] + 0.5) / (result["spend"] + 1.0)
    weights = result["efficiency_score"] / result["efficiency_score"].sum()
    remaining = total_budget - min_budget * n
    result["recommended_budget"] = min_budget + weights * remaining

    # Correct floating-point drift so allocations add up to the requested total.
    if len(result):
        result.loc[result.index[-1], "recommended_budget"] += (
            total_budget - result["recommended_budget"].sum()
        )
    result["recommended_share_pct"] = result["recommended_budget"] / total_budget * 100
    return result.sort_values("recommended_budget", ascending=False).reset_index(drop=True)
