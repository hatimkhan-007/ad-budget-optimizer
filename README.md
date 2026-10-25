# Digital Advertising Budget Optimizer

A big data analytics project that simulates digital marketing campaigns at scale (10K to 10M records), benchmarks three data processing approaches, and recommends how to split an ad budget across platforms.

## Project Structure

| File | Purpose |
|---|---|
| `data_generator.py` | Streams synthetic campaign data (date, campaign, platform, impressions, clicks, spend, conversions, revenue, age, gender, city, device) to CSV in chunks |
| `metrics_calculator.py` | CTR, CPC, CPA, conversion rate, ROAS (plus CPM) from summed totals, with chunked CSV aggregation |
| `benchmark_analysis.py` | Times and measures memory for Pandas, simulated MapReduce and PySpark on the same workload |
| `budget_optimizer.py` | Fits diminishing-returns curves per platform and computes the optimal budget split |
| `main.py` | Runs the full pipeline and prints the recommendation |

## Setup

Requires Python 3.10+. PySpark also needs Java 17 or newer (the benchmark skips PySpark with a note if it is unavailable).

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py                      # 10K, 100K, 1M rows, all engines
python main.py --full               # adds the 10M scale (about 800 MB CSV)
python main.py --skip-benchmark     # KPIs and recommendation only
python main.py --engines pandas mapreduce
python main.py --budget 50000 --min-share 0.10 --max-share 0.35
```

Each module also runs standalone: `python data_generator.py --records 1000000`, `python benchmark_analysis.py --scales 100000`.

Outputs are written to `data/` (CSVs, reused on later runs) and `results/` (`kpis_by_platform.csv`, `benchmark_results.csv`, `budget_recommendation.csv`).

## Architecture

```
data_generator -> CSV files (per scale)
                     |
      +--------------+----------------+
      v              v                v
metrics_calculator  benchmark_analysis  budget_optimizer
 (KPIs by platform)  (Pandas / MapReduce  (platform-day aggregates
                      / PySpark)           -> response curves -> allocation)
      \_______________|________________/
                      v
                   main.py
```

**Data generation.** Each platform has its own CPM, CTR, conversion rate, average order value and a saturation exponent. Daily demand is shaped by weekday, seasonality (Q4 peak) and random shocks. Conversion efficiency falls as daily demand rises, so returns diminish with spend. Counts follow binomial draws (conversions <= clicks <= impressions always holds). Output is deterministic for a given seed.

**KPIs.** All ratios are computed from summed totals, never averaged row ratios. Zero denominators return NaN.

**Benchmark.** Workload: read the CSV and aggregate all metrics by platform and campaign. Every engine runs in a fresh process; peak memory is the highest resident memory sampled across the process and its children (worker processes, Spark JVM).
- *Pandas*: single-process `read_csv` plus `groupby`.
- *MapReduce (simulated)*: the file is split into byte ranges aligned to line boundaries; worker processes run the map phase (parse split, locally combine, emit key/value pairs), a shuffle hash-partitions keys, and a reduce phase sums each bucket in parallel.
- *PySpark*: local mode, explicit schema, `groupBy().agg()`. Reported time excludes JVM and session startup, which is shown separately.

All engines are checked for matching group counts and total spend.

**Optimizer.** For each platform, history is aggregated to daily spend and revenue and fitted as `revenue = a * spend^b` by log-log regression (`b` is the spend elasticity; lower means faster saturation). The curve is calibrated to reproduce average historical daily revenue. The budget is then allocated in 2,000 small increments, each going to the platform with the highest marginal revenue, which is optimal for concave curves. Share limits (default 5% to 40% per platform) keep the plan realistic.

## Findings

Results from one run on a 1 vCPU, 3 GB machine (Python 3.12, pandas 3.0, PySpark 4.2). Absolute numbers vary by hardware; the trends are the point.

### Benchmark

| Records | Pandas | MapReduce | PySpark | Peak memory (Pandas / MR / Spark) |
|---|---|---|---|---|
| 10K | 0.02 s | 0.40 s | 7.51 s | 73 / 145 / 512 MB |
| 100K | 0.12 s | 0.54 s | 8.88 s | 95 / 170 / 527 MB |
| 1M | 1.21 s | 1.63 s | 10.57 s | 163 / 233 / 528 MB |
| 10M | 10.74 s | 11.84 s | 28.02 s | 1,379 / 1,152 / 563 MB |

PySpark also spent about 11 s starting the JVM and session on every run. All engines produced identical aggregates.

- **Pandas wins at small and medium scale.** Process spawn and JVM overhead dominate until the data is large.
- **Pandas memory grows linearly with data** (73 MB to 1.4 GB), while Spark stays near 0.5 GB because it streams partitions. Pandas is the first to run out of RAM on larger files.
- **Spark's scaling is the flattest**: 3.7x slower from 1M to 10M rows versus about 9x for Pandas. On this single-core machine it never overtakes Pandas; its advantage needs more cores or a cluster.
- **MapReduce needs parallel hardware.** With one core the workers cannot run in parallel, so it only adds overhead here. On a multi-core machine the map phase scales with core count.

### Platform performance (10M records)

| Platform | CTR | CPC | CPA | CVR | ROAS |
|---|---|---|---|---|---|
| Google | 3.01% | $0.74 | $18.73 | 3.94% | 3.89x |
| Facebook | 1.21% | $0.71 | $22.67 | 3.13% | 2.77x |
| YouTube | 0.81% | $1.12 | $35.56 | 3.15% | 2.05x |
| Instagram | 1.00% | $0.99 | $37.77 | 2.61% | 1.82x |
| TikTok | 1.45% | $0.38 | $26.97 | 1.42% | 1.69x |

### Budget recommendation

| Platform | Current | Recommended | Marginal ROAS | Elasticity |
|---|---|---|---|---|
| Facebook | 25.0% | 40.0% | 1.85x | 0.75 |
| Google | 30.6% | 40.0% | 2.10x | 0.60 |
| Instagram | 24.5% | 7.7% | 1.83x | 0.80 |
| YouTube | 10.1% | 7.3% | 1.83x | 0.85 |
| TikTok | 9.9% | 5.0% | 1.61x | 0.88 |

- Projected daily revenue rises about **4.0%** at the same total spend (blended ROAS 2.70x to 2.81x).
- Google has the best average ROAS but saturates fastest (elasticity 0.60), so the marginal return of extra spend is much lower than the average. Ranking platforms by raw ROAS alone would overweight it.
- Instagram and YouTube end with equal marginal ROAS (about 1.83x), the signature of an optimal allocation. Google and Facebook sit at the 40% cap, and TikTok at the 5% floor.
- The fitted elasticities match the values used by the generator (Google 0.60, Facebook 0.75, Instagram 0.80, YouTube 0.85, TikTok 0.88), confirming the model recovers the true response curves.
- Recommendations are stable across data scales: the 10K-row run already points the same way, and the larger runs mainly tighten the estimates (R2 rises to about 1.00 at 10M).

## Limitations

- Data is synthetic. Real campaigns need attribution handling, creative effects and audience overlap.
- The power-law curve assumes spend is the main driver of revenue and ignores cross-platform interaction.
- Platform-day history only reflects spend levels actually tried; extrapolating far outside them (for example, 5x a platform's usual budget) is unreliable. Share limits mitigate this.
- Memory is sampled RSS, so very short runs are approximate, and shared pages in worker processes are counted more than once.
- Windows users running PySpark may need Hadoop `winutils` configured (`HADOOP_HOME`).