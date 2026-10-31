"""Convert raw Meta campaign rows into a consistent metrics table."""
from __future__ import annotations

import pandas as pd


def count_conversions(actions: list[dict] | None) -> float:
    """Sum common conversion action types, avoiding click/view duplicates."""
    if not actions:
        return 0.0

    preferred = {
        "purchase", "omni_purchase", "lead", "onsite_conversion.lead_grouped",
        "complete_registration", "omni_complete_registration", "submit_application",
    }
    values = []
    for action in actions:
        action_type = str(action.get("action_type", "")).lower()
        if action_type in preferred:
            try:
                values.append(float(action.get("value", 0)))
            except (TypeError, ValueError):
                continue

    # Prefer purchases if present, otherwise sum the selected lead/registration types.
    purchases = [
        float(a.get("value", 0))
        for a in actions
        if str(a.get("action_type", "")).lower() in {"purchase", "omni_purchase"}
        and str(a.get("value", "0")).replace(".", "", 1).isdigit()
    ]
    if purchases:
        return max(purchases)
    return sum(values)


def normalize_campaign_rows(rows: list[dict]) -> pd.DataFrame:
    """Return one row per campaign/date with numeric metrics and derived rates."""
    if not rows:
        return pd.DataFrame(columns=[
            "campaign_id", "campaign_name", "date_start", "date_stop",
            "impressions", "clicks", "spend", "conversions", "ctr", "cpc", "cpa"
        ])

    frame = pd.DataFrame(rows)
    for column in ("impressions", "clicks", "spend", "ctr", "cpc"):
        if column not in frame:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    if "actions" not in frame:
        frame["actions"] = [[] for _ in range(len(frame))]
    frame["conversions"] = frame["actions"].apply(count_conversions)
    frame["cpa"] = frame.apply(
        lambda row: row["spend"] / row["conversions"] if row["conversions"] > 0 else float("inf"),
        axis=1,
    )
    frame["ctr"] = frame.apply(
        lambda row: row["clicks"] / row["impressions"] * 100 if row["impressions"] > 0 else 0,
        axis=1,
    )
    frame["cpc"] = frame.apply(
        lambda row: row["spend"] / row["clicks"] if row["clicks"] > 0 else 0,
        axis=1,
    )
    return frame
