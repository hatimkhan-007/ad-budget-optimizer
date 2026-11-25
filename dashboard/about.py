"""Project documentation inside the app."""

from __future__ import annotations

import streamlit as st

from dashboard.theme import SIMULATION_NOTICE, page_header

OVERVIEW = """
### What this application does
A big data analytics project for digital advertising: it simulates campaign data at scales from 10 thousand to 10 million
records, computes marketing KPIs, compares Pandas, simulated MapReduce and PySpark processing, and recommends how to
reallocate an ad budget across platforms and campaigns.

### Pipeline
1. **Generate**: reproducible simulated records (`data_generator.py`).
2. **Validate and aggregate**: chunked reading, row validation and reduction to small summary tables (`data_quality.py`, `analytics.py`).
3. **Measure**: KPIs from summed totals (`metrics_calculator.py`); campaign catalog (`campaigns.py`).
4. **Benchmark**: isolated trials per engine (`processing_engines.py`, `benchmark_analysis.py`).
5. **Optimise**: response curves and exact constrained allocation (`budget_optimizer.py`, `campaign_optimizer.py`).
6. **Optional sentiment**: validated comment data linked to campaigns only via explicit identifiers (`sentiment.py`).

### KPI definitions
| KPI | Formula |
|---|---|
| CTR | clicks / impressions |
| CPC | spend / clicks |
| CPA | spend / conversions |
| Conversion rate | conversions / clicks |
| ROAS | revenue / spend |
| CPM | spend / impressions x 1000 |

All ratios are computed from summed totals, never by averaging row-level ratios.

### Limitations
- Data is simulated; relationships between variables reflect the generator, not real markets.
- Campaign budgets and creatives are simulated; creatives are illustrative, not real advertisements.
- Optimisation results are model estimates with no guarantee of real-world improvement.
- Benchmark results depend on the machine; they describe this hardware, not frameworks in general.
"""


def render(ctx=None) -> None:
    page_header("About and Documentation", "Purpose, methodology and limitations of the Digital Advertising Budget Optimizer.")
    st.warning(SIMULATION_NOTICE)
    st.markdown(OVERVIEW)
