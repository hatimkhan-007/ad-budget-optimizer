from __future__ import annotations
import argparse, time
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
    parser.add_argument("--scales", type=int, nargs="+", default=SCALES[:3])
    parser.add_argument("--full", action="store_true", help="Include the 10M-row dataset.")
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--min-share", type=float, default=.05)
    parser.add_argument("--max-share", type=float, default=.40)
    args = parser.parse_args()
    if not args.scales or any(n <= 0 for n in args.scales):
        parser.error("--scales must contain positive record counts")
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    return args

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
        banner("3. BENCHMARK: Pandas vs MapReduce-style chunks vs PySpark")
        benchmark = run_benchmarks(paths, tuple(args.engines), args.trials)
        benchmark.to_csv(results_dir / "benchmark_results.csv", index=False)
        print(format_results(benchmark))
    banner("4. BUDGET ALLOCATION RECOMMENDATION")
    daily_by_scale = {n: aggregate_csv(p, ["platform", "date"]) for n, p in paths.items()}
    optimizer = BudgetOptimizer(args.min_share, args.max_share).fit(daily_by_scale[largest])
    result = optimizer.recommend(args.budget)
    print(f"Based on {largest:,} historical records (share limits "
          f"{args.min_share:.0%}-{args.max_share:.0%} per platform).")
    print("Projection is model-based and is not a causal guarantee.\n")
    print(format_recommendation(result))
    result.table.to_csv(results_dir / "budget_recommendation.csv", index=False)
    summary = pd.DataFrame([{"budget": result.total_daily_budget,
        "current_projected_revenue": result.current_revenue,
        "recommended_projected_revenue": result.recommended_revenue,
        "projected_uplift_pct": result.uplift_pct,
        "current_modeled_roas": result.current_roas,
        "recommended_modeled_roas": result.recommended_roas}])
    summary.to_csv(results_dir / "budget_summary.csv", index=False)
    print(f"\nDone in {time.perf_counter()-started:.1f}s. Outputs saved to {results_dir}/")
if __name__ == "__main__":
    main()
