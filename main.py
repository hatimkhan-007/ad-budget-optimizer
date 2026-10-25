"""Digital Advertising Budget Optimizer - pipeline entry point.

Steps: generate data -> compute KPIs -> benchmark engines -> recommend budget.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from benchmark_analysis import ENGINES, format_results, run_benchmarks
from budget_optimizer import BudgetOptimizer, format_recommendation
from data_generator import DEFAULT_SEED, SCALES, ensure_dataset
from metrics_calculator import aggregate_csv, format_kpi_table, kpis_from_csv


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digital Advertising Budget Optimizer")
    parser.add_argument("--scales", type=int, nargs="+", default=SCALES[:3],
                        help="Dataset sizes (default: 10K 100K 1M).")
    parser.add_argument("--full", action="store_true",
                        help="Also include the 10M-row scale (needs ~8 GB RAM for Pandas).")
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--budget", type=float, default=None,
                        help="Total daily budget to allocate (default: historical average).")
    parser.add_argument("--min-share", type=float, default=0.05)
    parser.add_argument("--max-share", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    scales = sorted(set(args.scales + ([10_000_000] if args.full else [])))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    banner("1. DATA GENERATION")
    paths = {n: ensure_dataset(n, args.data_dir, args.seed, args.regenerate) for n in scales}
    largest = max(scales)

    banner(f"2. MARKETING KPIs ({largest:,} records)")
    overall = kpis_from_csv(paths[largest])
    print("Overall:")
    print(format_kpi_table(overall.assign(scope="All"), ["scope"]))
    by_platform = kpis_from_csv(paths[largest], by="platform")
    print("\nBy platform:")
    print(format_kpi_table(by_platform, ["platform"]))
    by_platform.to_csv(results_dir / "kpis_by_platform.csv", index=False)

    if not args.skip_benchmark:
        banner("3. BENCHMARK: Pandas vs MapReduce vs PySpark")
        benchmark = run_benchmarks(paths, tuple(args.engines))
        benchmark.to_csv(results_dir / "benchmark_results.csv", index=False)
        print()
        print(format_results(benchmark))

    banner("4. BUDGET ALLOCATION RECOMMENDATION")
    daily_by_scale = {n: aggregate_csv(p, ["platform", "date"]) for n, p in paths.items()}
    optimizer = BudgetOptimizer(args.min_share, args.max_share).fit(daily_by_scale[largest])
    result = optimizer.recommend(args.budget)
    print(f"Based on {largest:,} historical records "
          f"(share limits {args.min_share:.0%}-{args.max_share:.0%} per platform)\n")
    print(format_recommendation(result))
    result.table.to_csv(results_dir / "budget_recommendation.csv", index=False)

    if len(scales) > 1:
        print("\nRecommended share (%) by amount of history used:")
        stability = {}
        for n, daily in daily_by_scale.items():
            table = BudgetOptimizer(args.min_share, args.max_share).fit(daily).recommend(
                args.budget).table
            stability[f"{n:,}"] = table.set_index("platform")["recommended_share_pct"]
        print(pd.DataFrame(stability).round(1).to_string())

    print(f"\nDone in {time.perf_counter() - started:.1f}s. Outputs saved to {results_dir}/")


if __name__ == "__main__":
    main()