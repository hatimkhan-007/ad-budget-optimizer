from __future__ import annotations

import io
import json
import math
import os
import shutil
import sys
import time
import zlib
import multiprocessing
from pathlib import Path

import pandas as pd

GROUP_KEYS = ["platform", "campaign"]
VALUE_COLUMNS = ["impressions", "clicks", "spend", "conversions", "revenue"]
RESULT_COLUMNS = GROUP_KEYS + ["rows"] + VALUE_COLUMNS
LONG_COLUMNS = {"impressions", "clicks", "conversions"}
DOUBLE_COLUMNS = {"spend", "revenue"}
DEFAULT_SPLIT_BYTES = 64 * 1024 * 1024
SPARK_DRIVER_MEMORY = os.environ.get("AD_BENCH_SPARK_DRIVER_MEMORY", "2g")


def _header(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.readline().rstrip("\r\n").split(",")


def _dtypes() -> dict[str, str]:
    types = {c: "int64" for c in LONG_COLUMNS}
    types.update({c: "float64" for c in DOUBLE_COLUMNS})
    return types


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[RESULT_COLUMNS].copy()
    frame[GROUP_KEYS] = frame[GROUP_KEYS].astype(str)
    return frame.sort_values(GROUP_KEYS).reset_index(drop=True)


# ------------------------------------------------------------------ Pandas
def run_pandas(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=GROUP_KEYS + VALUE_COLUMNS, dtype=_dtypes())
    grouped = frame.groupby(GROUP_KEYS, observed=True)
    result = grouped[VALUE_COLUMNS].sum()
    result.insert(0, "rows", grouped.size())
    return _finalise(result.reset_index())


# ------------------------------------------------------------------ MapReduce (local, multiprocessing)
def byte_splits(path: str | Path, split_bytes: int = DEFAULT_SPLIT_BYTES, min_parts: int = 1) -> list[tuple[int, int]]:
    """Split the data region of a CSV into line-aligned byte ranges."""
    if split_bytes <= 0 or min_parts < 1:
        raise ValueError("split_bytes and min_parts must be positive")
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        handle.readline()
        start = handle.tell()
        if start >= size:
            return []
        parts = max(min_parts, math.ceil((size - start) / split_bytes))
        step = max((size - start) // parts, 1)
        bounds = [start]
        for i in range(1, parts):
            handle.seek(start + i * step)
            handle.readline()
            position = handle.tell()
            if bounds[-1] < position < size:
                bounds.append(position)
    bounds.append(size)
    return list(zip(bounds[:-1], bounds[1:]))


def _partition(key: tuple[str, ...], reducers: int) -> int:
    return zlib.crc32("|".join(key).encode("utf-8")) % reducers


def map_task(args: tuple) -> list[dict]:
    """MAP: parse one split. COMBINE: aggregate locally. PARTITION: route each key to a reducer."""
    path, start, end, columns, reducers = args
    with open(path, "rb") as handle:
        handle.seek(start)
        payload = handle.read(end - start)
    parsed = pd.read_csv(io.BytesIO(payload), header=None, names=columns,
                         usecols=GROUP_KEYS + VALUE_COLUMNS, dtype=_dtypes())
    grouped = parsed.groupby(GROUP_KEYS, observed=True)
    combined = grouped[VALUE_COLUMNS].sum()
    combined.insert(0, "rows", grouped.size())
    partitions: list[dict] = [{} for _ in range(reducers)]
    for key, values in zip(combined.index, combined.to_numpy().tolist()):
        partitions[_partition(tuple(key), reducers)][tuple(key)] = values
    return partitions


def reduce_task(partials: list[dict]) -> dict:
    """REDUCE: sum every partial value that belongs to this reducer's keys."""
    out: dict[tuple, list[float]] = {}
    for partial in partials:
        for key, values in partial.items():
            if key in out:
                out[key] = [a + b for a, b in zip(out[key], values)]
            else:
                out[key] = list(values)
    return out


def run_mapreduce(path: str | Path, workers: int | None = None, split_bytes: int = DEFAULT_SPLIT_BYTES) -> pd.DataFrame:
    workers = workers or os.cpu_count() or 1
    reducers = max(workers, 2)
    columns = _header(path)
    splits = byte_splits(path, split_bytes, min_parts=workers)
    if not splits:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    tasks = [(str(path), start, end, columns, reducers) for start, end in splits]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        mapped = pool.map(map_task, tasks)
        reduced = pool.map(reduce_task, [[m[r] for m in mapped] for r in range(reducers)])
    merged: dict[tuple, list[float]] = {}
    for part in reduced:
        merged.update(part)
    rows = [list(key) + values for key, values in merged.items()]
    frame = pd.DataFrame(rows, columns=GROUP_KEYS + ["rows"] + VALUE_COLUMNS)
    frame[["rows"] + sorted(LONG_COLUMNS)] = frame[["rows"] + sorted(LONG_COLUMNS)].astype("int64")
    return _finalise(frame)


# ------------------------------------------------------------------ PySpark
def spark_available() -> tuple[bool, str]:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False, "PySpark is not installed (pip install pyspark)."
    if not (shutil.which("java") or os.environ.get("JAVA_HOME")):
        return False, "Java was not found; PySpark needs Java 17 or newer."
    return True, ""


def _spark_schema(path: str | Path):
    from pyspark.sql import types as T
    fields = []
    for name in _header(path):
        if name in LONG_COLUMNS:
            kind = T.LongType()
        elif name in DOUBLE_COLUMNS:
            kind = T.DoubleType()
        else:
            kind = T.StringType()
        fields.append(T.StructField(name, kind, True))
    return T.StructType(fields)


def run_pyspark(path: str | Path) -> tuple[pd.DataFrame, float]:
    """Return the aggregate and the session start-up time (reported separately from processing)."""
    from pyspark.sql import SparkSession, functions as F
    started = time.perf_counter()
    spark = (SparkSession.builder.master("local[*]").appName("AdvertisingBenchmark")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    startup = time.perf_counter() - started
    try:
        df = spark.read.csv(str(path), header=True, schema=_spark_schema(path))
        aggregations = [F.count(F.lit(1)).alias("rows")] + [F.sum(c).alias(c) for c in VALUE_COLUMNS]
        collected = df.groupBy(*GROUP_KEYS).agg(*aggregations).collect()
    finally:
        spark.stop()
    frame = pd.DataFrame([r.asDict() for r in collected], columns=RESULT_COLUMNS)
    return _finalise(frame), startup


# ------------------------------------------------------------------ Worker entry point
def worker(engine: str, path: str, output: str) -> None:
    startup = 0.0
    started = time.perf_counter()
    if engine == "pandas":
        result = run_pandas(path)
    elif engine == "mapreduce":
        result = run_mapreduce(path)
    elif engine == "pyspark":
        result, startup = run_pyspark(path)
    else:
        raise SystemExit(f"Unknown engine: {engine}")
    elapsed = time.perf_counter() - started
    result.to_csv(output, index=False)
    Path(output + ".meta.json").write_text(json.dumps({
        "processing_seconds": elapsed - startup,
        "startup_seconds": startup,
    }))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python -m processing_engines <pandas|mapreduce|pyspark> <input.csv> <output.csv>")
    worker(sys.argv[1], sys.argv[2], sys.argv[3])
