"""Benchmark Pandas, simulated MapReduce and PySpark on the same workload.

Workload: read the campaign CSV and aggregate impressions, clicks, spend,
conversions and revenue by (platform, campaign).

Each engine runs in its own fresh process so memory is measured in isolation.
Peak memory is the highest resident set size sampled across the worker process
and all of its children (multiprocessing workers, the Spark JVM).
"""

from __future__ import annotations

import argparse
import io
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from data_generator import COLUMNS, DTYPES, SCALES, ensure_dataset

KEYS = ["platform", "campaign"]
AGG_COLUMNS = ["impressions", "clicks", "spend", "conversions", "revenue"]
SPEND_INDEX = AGG_COLUMNS.index("spend")
REVENUE_INDEX = AGG_COLUMNS.index("revenue")
ENGINES = ("pandas", "mapreduce", "pyspark")


class PeakMemorySampler(threading.Thread):
    """Samples RSS of this process tree and keeps the peak."""

    def __init__(self, interval: float = 0.02) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_bytes = 0
        self._halt = threading.Event()
        self._proc = psutil.Process()

    def _sample(self) -> None:
        try:
            total = self._proc.memory_info().rss
            for child in self._proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.peak_bytes = max(self.peak_bytes, total)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def run(self) -> None:
        while not self._halt.is_set():
            self._sample()
            self._halt.wait(self.interval)

    def stop(self) -> float:
        self._halt.set()
        self.join()
        self._sample()
        return self.peak_bytes / 1e6


def _summarize(groups: int, total_spend: float, total_revenue: float) -> dict:
    return {"groups": groups, "total_spend": total_spend, "total_revenue": total_revenue}


def run_pandas(path: str) -> dict:
    df = pd.read_csv(path, dtype=DTYPES)
    out = df.groupby(KEYS, observed=True)[AGG_COLUMNS].sum()
    return _summarize(len(out), float(out["spend"].sum()), float(out["revenue"].sum()))


def _split_ranges(size: int, parts: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, size, parts + 1, dtype=np.int64)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def _read_partition(path: str, start: int, end: int) -> bytes:
    """Read the whole lines whose first byte falls inside [start, end)."""
    with open(path, "rb") as handle:
        if start == 0:
            position = 0
        else:
            handle.seek(start - 1)
            handle.readline()
            position = handle.tell()
        if position >= end:
            return b""
        handle.seek(position)
        data = handle.read(end - position)
        if not data.endswith(b"\n"):
            data += handle.readline()
    return data


def _map_partition(task: tuple[str, int, int]) -> list[tuple[tuple, tuple]]:
    """Map phase: parse one file split and emit locally combined (key, values) pairs."""
    path, start, end = task
    data = _read_partition(path, start, end)
    if not data:
        return []
    frame = pd.read_csv(io.BytesIO(data), names=COLUMNS,
                        header=0 if start == 0 else None, dtype=DTYPES)
    combined = frame.groupby(KEYS, observed=True)[AGG_COLUMNS].sum()
    return [(key, tuple(values)) for key, values in
            zip(combined.index.tolist(), combined.to_numpy().tolist())]


def _reduce_bucket(bucket: dict) -> dict:
    """Reduce phase: sum all partial values for each key in a bucket."""
    return {key: tuple(np.sum(np.array(parts, dtype=float), axis=0).tolist())
            for key, parts in bucket.items()}


def run_mapreduce(path: str) -> dict:
    import multiprocessing as mp

    workers = max(1, min(os.cpu_count() or 1, 8))
    splits = _split_ranges(os.path.getsize(path), workers * 2)
    tasks = [(path, start, end) for start, end in splits]
    context = mp.get_context("spawn")

    with context.Pool(workers) as pool:
        mapped = pool.map(_map_partition, tasks)

        buckets: list[dict] = [dict() for _ in range(workers)]
        for pairs in mapped:
            for key, values in pairs:
                buckets[hash(key) % workers].setdefault(key, []).append(values)

        reduced: dict = {}
        for part in pool.map(_reduce_bucket, buckets):
            reduced.update(part)

    spend = sum(v[SPEND_INDEX] for v in reduced.values())
    revenue = sum(v[REVENUE_INDEX] for v in reduced.values())
    return _summarize(len(reduced), float(spend), float(revenue))


def run_pyspark(path: str) -> dict:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    import sys
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    started = time.perf_counter()
    spark = (SparkSession.builder.master("local[*]").appName("ad-budget-benchmark")
             .config("spark.ui.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false")
             .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "2g"))
             .config("spark.sql.shuffle.partitions", "16")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    startup = time.perf_counter() - started

    schema = T.StructType([
        T.StructField("date", T.StringType()),
        T.StructField("campaign", T.StringType()),
        T.StructField("platform", T.StringType()),
        T.StructField("impressions", T.LongType()),
        T.StructField("clicks", T.LongType()),
        T.StructField("spend", T.DoubleType()),
        T.StructField("conversions", T.LongType()),
        T.StructField("revenue", T.DoubleType()),
        T.StructField("age", T.IntegerType()),
        T.StructField("gender", T.StringType()),
        T.StructField("city", T.StringType()),
        T.StructField("device", T.StringType()),
    ])

    work_started = time.perf_counter()
    rows = (spark.read.csv(path, header=True, schema=schema)
            .groupBy(*KEYS)
            .agg(*[F.sum(c).alias(c) for c in AGG_COLUMNS])
            .collect())
    work_seconds = time.perf_counter() - work_started
    spark.stop()

    result = _summarize(len(rows), float(sum(r["spend"] for r in rows)),
                        float(sum(r["revenue"] for r in rows)))
    result["startup_seconds"] = startup
    result["work_seconds"] = work_seconds
    return result


_RUNNERS = {"pandas": run_pandas, "mapreduce": run_mapreduce, "pyspark": run_pyspark}


def _silence_stderr() -> None:
    """Route JVM/Spark log noise to the null device (inherited by the JVM)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)


def _child_entry(engine: str, path: str, results: "queue.Queue") -> None:
    if engine == "pyspark":
        _silence_stderr()
    sampler = PeakMemorySampler()
    sampler.start()
    try:
        started = time.perf_counter()
        summary = _RUNNERS[engine](path)
        elapsed = time.perf_counter() - started
        startup = summary.pop("startup_seconds", 0.0)
        work = summary.pop("work_seconds", None)
        payload = {
            "status": "ok",
            "seconds": work if work is not None else elapsed,
            "startup_seconds": startup,
            "peak_mem_mb": sampler.stop(),
            **summary,
        }
    except ModuleNotFoundError as error:
        sampler.stop()
        payload = {"status": "skipped", "error": f"missing dependency: {error.name}"}
    except Exception as error:
        sampler.stop()
        message = str(error).strip().splitlines()[0][:160] if str(error).strip() else ""
        payload = {"status": "error", "error": f"{type(error).__name__}: {message}"}
    results.put(payload)


def run_isolated(engine: str, path: str | Path) -> dict:
    """Run one engine in a fresh process and return timing and memory stats."""
    import multiprocessing as mp

    context = mp.get_context("spawn")
    results = context.Queue()
    process = context.Process(target=_child_entry, args=(engine, str(path), results))
    process.start()
    payload = None
    while payload is None:
        try:
            payload = results.get(timeout=1)
        except queue.Empty:
            if not process.is_alive():
                try:
                    payload = results.get(timeout=1)
                except queue.Empty:
                    payload = {"status": "error",
                               "error": f"worker exited with code {process.exitcode}"}
    process.join()
    return payload


def run_benchmarks(paths: dict[int, Path], engines: tuple[str, ...] = ENGINES,
                   verbose: bool = True) -> pd.DataFrame:
    """Benchmark every engine at every scale and return a results frame."""
    records = []
    for scale in sorted(paths):
        for engine in engines:
            if verbose:
                print(f"  benchmarking {engine:<10} @ {scale:>10,} rows ...", flush=True)
            outcome = run_isolated(engine, paths[scale])
            records.append({"records": scale, "engine": engine, **outcome})
    results = pd.DataFrame(records)

    for column in ["seconds", "startup_seconds", "peak_mem_mb", "groups", "total_spend"]:
        if column not in results:
            results[column] = np.nan

    results["consistent"] = ""
    for scale, block in results[results["status"] == "ok"].groupby("records"):
        reference = block.iloc[0]
        for index, row in block.iterrows():
            same = (row["groups"] == reference["groups"] and
                    np.isclose(row["total_spend"], reference["total_spend"], rtol=1e-6))
            results.loc[index, "consistent"] = "yes" if same else "NO"

    baseline = results[results["engine"] == "pandas"].set_index("records")["seconds"]
    results["vs_pandas"] = results.apply(
        lambda r: baseline.get(r["records"], np.nan) / r["seconds"]
        if r["status"] == "ok" and r["seconds"] else np.nan, axis=1)
    return results


def format_results(results: pd.DataFrame) -> str:
    view = pd.DataFrame({
        "records": results["records"].map("{:,}".format),
        "engine": results["engine"],
        "status": results["status"],
        "seconds": results["seconds"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}"),
        "startup_s": results["startup_seconds"].map(
            lambda v: "-" if pd.isna(v) or v == 0 else f"{v:.2f}"),
        "peak_mem_mb": results["peak_mem_mb"].map(lambda v: "-" if pd.isna(v) else f"{v:,.0f}"),
        "speedup_vs_pandas": results["vs_pandas"].map(
            lambda v: "-" if pd.isna(v) else f"{v:.2f}x"),
        "results_match": results["consistent"],
    })
    text = view.to_string(index=False)
    failures = results[results["status"] != "ok"]
    notes = [f"  {r.engine} @ {r.records:,}: {r.status} ({r.error})" for r in failures.itertuples()]
    return text + ("\n\nNotes:\n" + "\n".join(notes) if notes else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Pandas vs MapReduce vs PySpark.")
    parser.add_argument("--scales", type=int, nargs="+", default=SCALES[:3])
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="results/benchmark_results.csv")
    args = parser.parse_args()

    paths = {n: ensure_dataset(n, args.data_dir) for n in args.scales}
    results = run_benchmarks(paths, tuple(args.engines))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(format_results(results))


if __name__ == "__main__":
    main()