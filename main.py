from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from benchmark_analysis import ENGINES, format_results, run_benchmarks
from budget_optimizer import BudgetOptimizer, add_campaign_amounts, format_recommendation
from data_generator import DEFAULT_SEED, SCALES, ensure_dataset
from metrics_calculator import aggregate_csv, format_kpi_table, kpis_from_csv

DEFAULT_CAMPAIGN_BUDGET = 1_000_000.0


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digital Advertising Budget Optimizer")
    parser.add_argument("--scales", type=int, nargs="+", default=SCALES[:3], help="Dataset sizes to generate and benchmark.")
    parser.add_argument("--full", action="store_true", help="Include the 10M-record dataset.")
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--budget", type=float, default=None,
                        help="Total DAILY budget used for the optimization (defaults to the historical daily spend).")
    parser.add_argument("--campaign-budget", type=float, default=DEFAULT_CAMPAIGN_BUDGET,
                        help="Total campaign budget to split across platforms (default: 1,000,000).")
    parser.add_argument("--currency", default="Rs.", help="Currency label for the campaign budget.")
    parser.add_argument("--min-share", type=float, default=0.05)
    parser.add_argument("--max-share", type=float, default=0.40)
    args = parser.parse_args(argv)
    if not args.scales or any(n <= 0 for n in args.scales):
        parser.error("--scales must contain positive record counts")
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    if args.budget is not None and args.budget <= 0:
        parser.error("--budget must be greater than zero")
    if args.campaign_budget <= 0:
        parser.error("--campaign-budget must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    scales = sorted(set(args.scales + ([10_000_000] if args.full else [])))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    banner("1. DATA GENERATION")
    paths = {n: ensure_dataset(n, args.data_dir, args.seed, args.regenerate) for n in scales}
    for n, path in paths.items():
        print(f"{n:>12,} records -> {path}")
    largest = max(scales)

    banner(f"2. MARKETING KPIs ({largest:,} records)")
    print("Overall:")
    print(format_kpi_table(kpis_from_csv(paths[largest]).assign(scope="All")))
    by_platform = kpis_from_csv(paths[largest], by="platform")
    print("\nBy platform:")
    print(format_kpi_table(by_platform))
    by_platform.to_csv(results_dir / "kpis_by_platform.csv", index=False)

    if args.skip_benchmark:
        banner("3. BENCHMARK (skipped)")
    else:
        banner("3. BENCHMARK: Pandas vs simulated MapReduce vs PySpark")
        benchmark = run_benchmarks(paths, tuple(args.engines), args.trials)
        benchmark.to_csv(results_dir / "benchmark_results.csv", index=False)
        print(format_results(benchmark))

    banner("4. BUDGET ALLOCATION RECOMMENDATION")
    daily = aggregate_csv(paths[largest], ["platform", "date"])
    optimizer = BudgetOptimizer(args.min_share, args.max_share).fit(daily)
    result = optimizer.recommend(args.budget)
    table = add_campaign_amounts(result.table, args.campaign_budget)
    result.table = table
    print(f"Based on {largest:,} historical records (share limits {args.min_share:.0%}-{args.max_share:.0%} per platform).")
    print("Projection is model-based and is not a causal guarantee.\n")
    print(format_recommendation(result))
    print(f"\nRecommended split of {args.currency} {args.campaign_budget:,.0f}:")
    for row in table.itertuples():
        print(f"  {row.platform:<10} {row.recommended_share_pct:5.1f}%   {args.currency} {row.recommended_campaign_amount:>12,.0f}")
    table.to_csv(results_dir / "budget_recommendation.csv", index=False)
    pd.DataFrame([{
        "daily_budget": result.total_daily_budget,
        "campaign_budget": args.campaign_budget,
        "current_projected_revenue": result.current_revenue,
        "compliant_baseline_revenue": result.compliant_baseline_revenue,
        "recommended_projected_revenue": result.recommended_revenue,
        "uplift_vs_current_pct": result.uplift_pct,
        "uplift_vs_compliant_baseline_pct": result.uplift_vs_compliant_pct,
        "current_modeled_roas": result.current_roas,
        "recommended_modeled_roas": result.recommended_roas,
        "current_within_limits": result.current_within_limits,
    }]).to_csv(results_dir / "budget_summary.csv", index=False)
    print(f"\nDone in {time.perf_counter() - started:.1f}s. Outputs saved to {results_dir}/")


if __name__ == "__main__":
    main()
