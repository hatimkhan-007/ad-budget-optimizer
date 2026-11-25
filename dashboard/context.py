"""Per-run dashboard context: the selected dataset and its derived tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analytics import Aggregates
from campaigns import build_campaign_catalog
from dashboard.data import DatasetInfo, with_datetime
from data_generator import DEFAULT_SEED


@dataclass
class Context:
    dataset: DatasetInfo | None = None
    aggregates: Aggregates | None = None
    full_campaign_daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    campaign_daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    full_catalog: pd.DataFrame = field(default_factory=pd.DataFrame)
    catalog: pd.DataFrame = field(default_factory=pd.DataFrame)
    platforms: list[str] = field(default_factory=list)
    selected_platforms: list[str] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return self.aggregates is not None and not self.full_campaign_daily.empty


def build_context(dataset: DatasetInfo | None, aggregates: Aggregates | None,
                  selected_platforms: list[str] | None = None) -> Context:
    if dataset is None or aggregates is None:
        return Context()
    full = with_datetime(aggregates.frames["campaign_daily"])
    platforms = sorted(full["platform"].unique())
    chosen = [p for p in (selected_platforms or platforms) if p in platforms] or platforms
    filtered = full[full["platform"].isin(chosen)]
    full_catalog = build_campaign_catalog(full, DEFAULT_SEED)
    catalog = full_catalog[full_catalog["platform"].isin(chosen)].reset_index(drop=True)
    return Context(dataset, aggregates, full, filtered, full_catalog, catalog, platforms, chosen)
