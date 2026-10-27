from __future__ import annotations
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
from metrics_calculator import NUMERIC_COLUMNS, REQUIRED_COLUMNS, validate_columns

ENGINES = ("pandas", "mapreduce", "pyspark")
DTYPES = {"platform":"category", "campaign":"category", "impressions":"int64", "clicks":"int64",
          "spend":"float64", "conversions":"int64", "revenue":"float64"}

def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["platform","campaign"], observed=True)[NUMERIC_COLUMNS].sum().reset_index()

def run_pandas(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=DTYPES)
    validate_columns(frame.columns)
    return _aggregate(frame)

def run_mapreduce(path: str | Path, workers: int | None = None, partitions: int = 4) -> pd.DataFrame:
    # CSV chunking is a local MapReduce-style implementation: map each chunk to grouped partial sums, then reduce.
    if partitions < 1:
        raise ValueError("partitions must be >= 1")
    partials = []
    for chunk in pd.read_csv(path, chunksize=250_000, dtype=DTYPES):
        validate_columns(chunk.columns)
        partials.append(_aggregate(chunk))
    if not partials:
        return pd.DataFrame(columns=["platform","campaign"] + NUMERIC_COLUMNS)
    combined = pd.concat(partials, ignore_index=True)
    return combined.groupby(["platform","campaign"], observed=True)[NUMERIC_COLUMNS].sum().reset_index()

def run_pyspark(path: str | Path) -> pd.DataFrame:
    try:
        from pyspark.sql import SparkSession, functions as F
    except ImportError as exc:
        raise RuntimeError("PySpark is not installed. Install requirements or exclude pyspark.") from exc
    spark = (SparkSession.builder.master("local[*]").appName("AdvertisingBenchmark")
             .config("spark.ui.enabled", "false").getOrCreate())
    try:
        df = spark.read.option("header", True).option("inferSchema", True).csv(str(path))
        grouped = df.groupBy("platform", "campaign").agg(
            *[F.sum(F.col(c)).alias(c) for c in NUMERIC_COLUMNS]
        )
        return grouped.toPandas()
    finally:
        spark.stop()

def _worker(engine: str, path: str, output_path: str) -> None:
    started = time.perf_counter()
    startup = 0.0
    if engine == "pandas":
        result = run_pandas(path)
    elif engine == "mapreduce":
        result = run_mapreduce(path)
    elif engine == "pyspark":
        before = time.perf_counter()
        from pyspark.sql import SparkSession
        startup = time.perf_counter() - before
        result = run_pyspark(path)
    else:
        raise ValueError(engine)
    elapsed = time.perf_counter() - started
    result.to_csv(output_path, index=False)
    Path(output_path + ".meta.json").write_text(json.dumps({"seconds": elapsed, "startup_seconds": startup}))

def _same_results(reference: pd.DataFrame, candidate: pd.DataFrame) -> bool:
    keys = ["platform", "campaign"]
    if set(map(tuple, reference[keys].astype(str).to_numpy())) != set(map(tuple, candidate[keys].astype(str).to_numpy())):
        return False
    a = reference.sort_values(keys).reset_index(drop=True)
    b = candidate.sort_values(keys).reset_index(drop=True)
    for col in NUMERIC_COLUMNS:
        if not np.allclose(pd.to_numeric(a[col]), pd.to_numeric(b[col]), rtol=1e-6, atol=1e-5, equal_nan=True):
            return False
    return True

def run_benchmarks(paths: dict[int, str | Path], engines: tuple[str, ...] = ENGINES,
                   trials: int = 3) -> pd.DataFrame:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    unknown = set(engines) - set(ENGINES)
    if unknown:
        raise ValueError(f"Unknown engines: {sorted(unknown)}")
    rows = []
    with tempfile.TemporaryDirectory(prefix="adbench_") as tmp:
        for records, path in sorted(paths.items()):
            for engine in engines:
                trial_times, trial_startups, trial_mem = [], [], []
                statuses, outputs = [], []
                for trial in range(trials):
                    output = str(Path(tmp) / f"{records}_{engine}_{trial}.csv")
                    command = [sys.executable, "-c",
                        "import sys; from benchmark_analysis import _worker; _worker(sys.argv[1],sys.argv[2],sys.argv[3])",
                        engine, str(path), output]
                    t0 = time.perf_counter()
                    completed = subprocess.run(command, capture_output=True, text=True)
                    wall = time.perf_counter() - t0
                    if completed.returncode != 0:
                        statuses.append("error")
                        continue
                    meta = json.loads(Path(output + ".meta.json").read_text())
                    trial_times.append(wall)  # end-to-end, includes process and Spark startup overhead
                    trial_startups.append(float(meta.get("startup_seconds", 0)))
                    outputs.append(pd.read_csv(output))
                    statuses.append("ok")
                if not outputs:
                    rows.append({"records": records, "engine": engine, "status": "error",
                                 "seconds": np.nan, "startup_seconds": np.nan, "groups": 0,
                                 "total_spend": np.nan, "total_revenue": np.nan,
                                 "consistent": "no", "vs_pandas": np.nan, "trials": trials})
                    continue
                first = outputs[0]
                consistent = all(_same_results(first, out) for out in outputs[1:])
                rows.append({"records": records, "engine": engine,
                    "status": "ok" if len(outputs) == trials else "partial",
                    "seconds": float(np.median(trial_times)),
                    "startup_seconds": float(np.median(trial_startups)),
                    "peak_mem_mb": np.nan, "groups": len(first),
                    "total_spend": float(first["spend"].sum()),
                    "total_revenue": float(first["revenue"].sum()),
                    "consistent": "yes" if consistent else "no",
                    "vs_pandas": np.nan, "trials": len(outputs),
                    "time_std_seconds": float(np.std(trial_times, ddof=1)) if len(trial_times)>1 else 0.0})
    frame = pd.DataFrame(rows)
    for records, group in frame.groupby("records"):
        pandas_rows = group[group.engine == "pandas"]
        if not pandas_rows.empty and pandas_rows.iloc[0].seconds > 0:
            frame.loc[group.index, "vs_pandas"] = pandas_rows.iloc[0].seconds / group.seconds
    return frame

def format_results(frame: pd.DataFrame) -> str:
    cols = [c for c in ["records","engine","status","seconds","startup_seconds","groups",
        "total_spend","total_revenue","consistent","vs_pandas","trials","time_std_seconds"] if c in frame]
    return frame[cols].to_string(index=False, float_format=lambda x: f"{x:,.3f}")
