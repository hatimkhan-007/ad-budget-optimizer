"""Read-only client for Meta Marketing API campaign insights."""
from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


class MetaAdsAPIError(RuntimeError):
    """Raised when Meta credentials or API requests fail."""


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise MetaAdsAPIError(
            f"Missing {name}. Copy .env.example to .env and configure your Meta API credentials."
        )
    return value


def fetch_campaign_insights(date_preset: str = "last_30d") -> list[dict[str, Any]]:
    """Fetch campaign-level insights, following Meta's pagination cursors."""
    token = _required_env("META_ACCESS_TOKEN")
    account_id = _required_env("META_AD_ACCOUNT_ID")
    version = os.getenv("META_API_VERSION", "v26.0").strip()
    if not account_id.startswith("act_"):
        account_id = f"act_{account_id}"

    url = f"https://graph.facebook.com/{version}/{account_id}/insights"
    params: dict[str, Any] = {
        "access_token": token,
        "level": "campaign",
        "date_preset": date_preset,
        "time_increment": 1,
        "fields": (
            "campaign_id,campaign_name,date_start,date_stop,"
            "impressions,clicks,spend,ctr,cpc,actions"
        ),
        "limit": 500,
    }

    rows: list[dict[str, Any]] = []
    session = requests.Session()
    while url:
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise MetaAdsAPIError(f"Could not retrieve Meta campaign data: {exc}") from exc
        except ValueError as exc:
            raise MetaAdsAPIError("Meta returned a response that was not valid JSON.") from exc

        if "error" in payload:
            message = payload["error"].get("message", "Unknown Meta API error")
            raise MetaAdsAPIError(f"Meta API error: {message}")

        rows.extend(payload.get("data", []))
        url = payload.get("paging", {}).get("next")
        # The next URL already contains its paging parameters.
        params = {}

    if not rows:
        raise MetaAdsAPIError(
            "The API returned no campaign rows for this period. Check the account, date range, "
            "permissions and whether campaigns have reportable activity."
        )
    return rows
