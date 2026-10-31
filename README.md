# Digital Advertising Budget Optimizer

A big data analytics project that simulates digital marketing campaigns at scale, benchmarks three data-processing approaches, calculates marketing KPIs, and recommends how to split an advertising budget across platforms.

It answers the management question *"Where should we spend the next Rs. 1 million?"* using datasets from **10K to 10M records**, and compares Pandas, a simulated MapReduce implementation, and PySpark.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Architecture](#architecture)
- [Dashboard UI](#dashboard-ui)
- [Budget Optimization](#budget-optimization)
- [Benchmark Results](#benchmark-results)
- [Platform Performance](#platform-performance)
- [Limitations](#limitations)

## Features

- **Scalable synthetic data generation:** Reproducible campaign data from 10K to 10M records, written in chunks, with platform, campaign, audience, device, age, gender and city fields.
- **Marketing KPI analytics:** CTR, CPC, CPA, conversion rate, ROAS and CPM, calculated from aggregated totals.
- **Processing-engine benchmarks:** Pandas, a multiprocessing MapReduce simulation, and PySpark, each run in an isolated process.
- **Runtime and memory measurement:** Processing time, engine start-up time and peak memory (including child processes).
- **Cross-engine correctness checks:** Every engine's output is compared with the Pandas reference (groups, row counts and totals).
- **Budget optimization:** Diminishing-returns response curves, solved exactly under per-platform share limits.
- **Campaign budget split:** The recommended shares are converted into amounts for a total campaign budget (default Rs. 1,000,000).
- **Risk flags:** Warnings for extrapolation, clipped elasticity, weak curve fits and a current allocation that already breaks the limits.
- **Interactive dashboard:** KPI results, budget recommendations and benchmark outputs.
- **Automated tests and CI:** Unit, integration and dashboard tests run on every push.

## Project Structure

```text
Digital-Advertising-Budget-Optimizer/
├── app.py                    # Streamlit dashboard
├── main.py                   # End-to-end pipeline
├── data_generator.py         # Synthetic campaign data
├── metrics_calculator.py     # KPI calculation and validation
├── processing_engines.py     # Pandas, MapReduce and PySpark engines
├── benchmark_analysis.py     # Benchmark runner (time, memory, consistency)
├── budget_optimizer.py       # Response curves and allocation solver
├── requirements.txt
├── pytest.ini
├── .github/workflows/ci.yml
├── tests/
├── data/                     # Generated datasets (git-ignored)
├── results/
│   ├── kpis_by_platform.csv
│   ├── benchmark_results.csv
│   ├── budget_recommendation.csv
│   └── budget_summary.csv
└── assets/
    ├── budget-optimizer.jpg
    └── UI-of-ad-budget-optimizer.jpg
```

## Tech Stack

- **Python** — core implementation
- **Pandas / NumPy** — aggregation, numerical operations and data generation
- **PySpark** — distributed-processing framework in local mode
- **psutil** — process-tree memory measurement
- **Streamlit / Plotly** — interactive dashboard and charts
- **pytest** — automated testing

## Setup

Requirements:

- Python 3.10 or newer
- Java 17 or newer for PySpark
- Dependencies listed in `requirements.txt`

```bash
pip install -r requirements.txt
```

If Java or PySpark is unavailable, the benchmark marks PySpark as `skipped` (with the reason in the `detail` column) and continues with the other engines.

> PySpark may print a warning that it does not yet fully support pandas 3. The results in this README were produced with pandas 3.0.2 and PySpark 4.3.0 and passed every consistency check; if you see problems, use pandas 2.x.

## Running the Project

Run the standard pipeline (10K, 100K and 1M records):

```bash
python main.py
```

Include the 10M-record scale:

```bash
python main.py --full
```

Run KPI calculations and budget optimization without benchmarking:

```bash
python main.py --skip-benchmark
```

Select processing engines and trial count:

```bash
python main.py --engines pandas mapreduce --trials 3
```

Customize the budget and constraints:

```bash
python main.py --campaign-budget 1000000 --currency "Rs." --min-share 0.10 --max-share 0.35
```

`--budget` sets the **daily** budget used by the optimizer (it defaults to the historical daily spend), while `--campaign-budget` is the total amount split across platforms in the output.

Run individual modules:

```bash
python data_generator.py --records 1000000
python benchmark_analysis.py --scales 100000 --engines pandas mapreduce pyspark
```

Launch the dashboard:

```bash
streamlit run app.py
```

The dashboard reads the CSVs in `results/` (or files uploaded through its sidebar). **Update dashboard** reloads those files; it does not rerun the analytics pipeline. Rerun `main.py` to regenerate results.

Generated datasets are stored in `data/`; analytical results are written to `results/`.

## Testing

```bash
python -m pytest -q
```

The suite covers data generation (reproducibility, funnel invariants, weekday and Q4 effects), KPI formulas and edge cases, optimizer correctness (budget, limits, optimality and risk flags), engine agreement, benchmark behavior (skipped and failed engines, memory), the CLI pipeline and dashboard rendering. PySpark tests are skipped automatically when Java or PySpark is unavailable.

## Architecture

```text
              data_generator.py
                      |
                      v
             Campaign CSV datasets
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
     KPI Analysis  Benchmark   Budget Optimizer
                   Analysis         |
          |           |             |
          v           v             v
     Platform KPIs  Runtime &   Response Curves
                    Memory          |
          |           |             |
          +-----------+-------------+
                      |
                      v
                    main.py
                      |
                      v
             Results in CSV files
                      |
                      v
                 Streamlit UI
```

### Data Generation

The generator simulates campaigns across Facebook, Google, YouTube, Instagram and TikTok. Each platform has its own CPM, click-through rate, conversion rate, average order value, volume weight and **saturation exponent**. Each record also carries a campaign, audience, device, age group, gender and city.

Demand is modeled per day and platform. It combines weekday effects (weaker weekends), seasonality, a fourth-quarter peak (October to December) and a shared daily shock. Conversion efficiency falls as demand rises, at a rate set by the platform's saturation exponent, which produces diminishing returns in the daily totals.

Counts are generated so that:

$$
\text{Conversions} \leq \text{Clicks} \leq \text{Impressions}
$$

A fixed random seed makes generation reproducible. Datasets are cached with a metadata file that records the record count, seed and generator version; a mismatch triggers regeneration.

### KPI Calculation

KPIs are calculated from aggregated totals rather than by averaging row-level ratios.

| KPI | Formula |
|---|---|
| CTR | $\dfrac{\text{Clicks}}{\text{Impressions}} \times 100$ |
| CPC | $\dfrac{\text{Spend}}{\text{Clicks}}$ |
| CPA | $\dfrac{\text{Spend}}{\text{Conversions}}$ |
| Conversion Rate | $\dfrac{\text{Conversions}}{\text{Clicks}} \times 100$ |
| ROAS | $\dfrac{\text{Revenue}}{\text{Spend}}$ |
| CPM | $\dfrac{\text{Spend}}{\text{Impressions}} \times 1000$ |

Ratios with a zero denominator return `NaN`. Negative values raise an error, and funnel violations (clicks above impressions, or conversions above clicks) raise a warning. Large files are processed in chunks.

### Benchmark Methodology

Each engine aggregates the same task: group by platform and campaign, and sum rows, impressions, clicks, spend, conversions and revenue.

- **Pandas:** Reads only the needed columns with explicit dtypes and aggregates with `groupby`.
- **Simulated MapReduce:** Splits the file into line-aligned byte ranges. Worker processes (spawned, so behavior is identical on Linux, macOS and Windows) *map* each split, *combine* locally, *partition* keys by CRC32 hash, and reducers *reduce* the partials. Mappers parse each split with vectorized `read_csv` for speed, so this simulates the MapReduce structure rather than a Hadoop cluster.
- **PySpark:** Local mode with an explicit schema built from the CSV header and `groupBy().agg()`.

Each engine runs in a fresh subprocess, for the requested number of trials. The reported figures are:

- `seconds`: median **processing** time. PySpark session start-up is excluded.
- `startup_seconds`: median engine start-up time, reported separately.
- `peak_mem_mb`: peak memory of the whole process tree, sampled every 20 ms using proportional set size (PSS), so memory shared between worker processes is not counted twice. The JVM used by Spark is included.
- `consistent`: `yes` only if all trials agree, the output matches the Pandas reference (same groups, identical row counts, totals equal within floating-point tolerance), and the row count equals the dataset size.
- `vs_pandas`: speed relative to Pandas (above 1 is faster).

## Dashboard UI

The Streamlit dashboard provides an interface for exploring platform performance, budget recommendations, benchmark timings and memory, and the generated result tables.

![Digital Advertising Budget Optimizer Dashboard](assets/UI-of-ad-budget-optimizer.jpg)

## Budget Optimization

For each platform, historical campaign data is aggregated into daily spend and revenue. A power-law response curve is fitted by log-log regression:

$$
\text{Revenue} = a \cdot \text{Spend}^{b}
$$

Here $b$ is the spend elasticity; a lower value means returns diminish faster. The slope is clipped to $[0.05, 0.98]$ so every curve stays concave, and the curve is calibrated to pass through the platform's average daily spend and revenue.

Because every curve is concave, the best split is found exactly rather than by trial and error. The solver equalizes marginal revenue across all platforms that are not held at their minimum or maximum share (a bisection on the marginal-revenue level, then an exact budget correction). A test compares the result against thousands of random feasible allocations.

The model reports uplift two ways:

- **Versus current:** the recommendation compared with today's split.
- **Versus a limit-compliant baseline:** today's split adjusted just enough to satisfy the share limits. When the current split already breaks a limit, this is the fair comparison, because the limit itself costs revenue.

This approach assumes concave response curves and the simulator's assumptions; it does not guarantee optimal performance in real advertising campaigns.

### Example Budget Recommendation

![Digital Advertising Budget Optimizer - Budget Recommendation](assets/budget-optimizer.jpg)

Output from the 10M-record run with default limits (5% to 40% per platform):

| Platform | Current Share | Recommended Share | Marginal ROAS | Elasticity | Split of Rs. 1,000,000 |
|---|---:|---:|---:|---:|---:|
| Facebook | 17.5% | 40.0% | 2.05× | 0.86 | Rs. 400,000 |
| Google | 50.2% | 40.0% | 3.52× | 0.90 | Rs. 400,000 |
| YouTube | 9.9% | 7.9% | 1.70× | 0.80 | Rs. 78,965 |
| Instagram | 16.0% | 7.1% | 1.70× | 0.82 | Rs. 71,035 |
| TikTok | 6.4% | 5.0% | 1.35× | 0.76 | Rs. 50,000 |

Facebook and Google reach the 40% maximum and TikTok reaches the 5% minimum. Google currently takes more than 50% of spend, so the 40% cap forces money away from the platform with the highest ROAS. That is why the projected uplift is **+4.1% versus the limit-compliant baseline** but **−2.1% versus the current split**, with modeled ROAS moving from 2.99× to 2.93×. Raise `--max-share` if the cap is not a business requirement.

Facebook's recommended daily spend lies above its observed daily range, so its projection is flagged as extrapolation. Treat all figures as simulation-specific, not guaranteed real-world outcomes.

## Benchmark Results

Measured on a 1 vCPU, 4 GB Linux machine with Python 3.12.3, pandas 3.0.2, PySpark 4.3.0 and Java 21, using the median of 3 trials. Results vary with hardware, software versions and workload.

| Records | Pandas | MapReduce | PySpark | Peak Memory: Pandas / MapReduce / Spark |
|---:|---:|---:|---:|---:|
| 10K | 0.03 s | 0.54 s | 8.33 s | 86 / 145 / 499 MB |
| 100K | 0.13 s | 0.57 s | 9.19 s | 96 / 168 / 526 MB |
| 1M | 1.23 s | 1.87 s | 11.33 s | 217 / 284 / 524 MB |
| 10M | 12.04 s | 12.58 s | 29.32 s | 1,310 / 331 / 560 MB |

Processing times exclude PySpark session start-up, which was about 11 s in every run (`startup_seconds` in `benchmark_results.csv`). All engines produced identical results at every scale.

### Key Observations

- **Pandas is fastest at every tested scale** on this machine; its low start-up cost dominates up to 1M records, and at 10M it is only about 4% faster than MapReduce.
- **Memory scales very differently:** Pandas peaks at about 1.3 GB at 10M records, while the MapReduce implementation, which processes bounded byte splits, stays near 330 MB.
- **PySpark carries a large fixed cost** (start-up plus job scheduling) that a single core cannot recover at these sizes. Its memory is roughly constant because of the JVM.
- **MapReduce needs more cores to pay off:** on one core it adds process overhead without parallel speedup.
- **Measurements are environment-specific** and should not be read as universal rankings of the frameworks. Re-run `python main.py --full` on your own hardware to refresh these figures.

## Platform Performance

KPI results from the 10M-record simulation:

| Platform | CTR | CPC | CPA | Conversion Rate | ROAS |
|---|---:|---:|---:|---:|---:|
| Google | 3.01% | $0.74 | $19.17 | 3.86% | 3.83× |
| Facebook | 1.20% | $0.72 | $23.40 | 3.06% | 2.67× |
| YouTube | 0.80% | $1.14 | $36.10 | 3.15% | 2.04× |
| Instagram | 1.00% | $0.99 | $38.77 | 2.55% | 1.79× |
| TikTok | 1.45% | $0.38 | $27.66 | 1.37% | 1.67× |

Google has the highest average ROAS. TikTok has the lowest CPC but also the lowest ROAS. These differences show why budget decisions should consider marginal returns and constraints rather than a single KPI.

## Limitations

- **Synthetic data:** Simulated campaigns do not capture real-world attribution, creative quality, audience overlap or platform reporting differences.
- **Simplified response curves:** Spend is the only driver of revenue; cross-platform interactions are not modeled.
- **Limited historical coverage:** Recommendations are unreliable when projected spend falls far outside the observed range. The optimizer flags these cases.
- **Small samples bias toward linear returns:** At 10K records each platform has only a few rows per day, so row-count noise dominates the daily totals and curve fits are weaker (the pipeline flags R² below 0.80).
- **Benchmark environment:** One-core results do not represent a multi-core workstation or cluster, and MapReduce here is a local simulation, not Hadoop.
- **Memory sampling:** Peak memory is sampled every 20 ms, so very short runs may miss brief spikes.
- **Model validation:** A good fit on synthetic data does not establish predictive accuracy on real campaigns. Real deployments need held-out data, controlled experiments and ongoing monitoring.

## Future Improvements

- Validate recommendations against held-out campaign data.
- Add confidence intervals and sensitivity analysis for budget allocations.
- Compare serial and parallel processing on machines with different core counts.
- Support exporting reports directly from the dashboard.
- Evaluate the optimizer on real, appropriately attributed campaign data before using it for business decisions.

---

**Note:** The benchmark, KPI and budget tables show one particular run. Re-run the pipeline and update these figures whenever the code, data-generation settings, dependencies or benchmark environment change.