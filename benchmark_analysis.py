from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from data_generator import DEFAULT_SEED, SCALES, ensure_dataset
from processing_engines import GROUP_KEYS, VALUE_COLUMNS, spark_available

ENGINES = ("pandas", "mapreduce", "pyspark")
PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_INTERVAL_SECONDS = 0.02
RESULT_COLUMNS = ["records", "engine", "status", "seconds", "startup_seconds", "wall_seconds", "peak_mem_mb",
                  "groups", "rows_counted", "total_spend", "total_revenue", "consistent", "vs_pandas",
                  "trials", "time_std_seconds", "detail"]


def engine_available(engine: str) -> tuple[bool, str]:
    if engine == "pyspark":
        return spark_available()
    return True, ""


def _process_memory_mb(process) -> float:
    """Proportional set size (PSS) where available, so memory shared between workers is not counted twice."""
    try:
        info = process.memory_full_info()
        value = getattr(info, "pss", None) or getattr(info, "uss", None) or info.rss
    except Exception:
        try:
            value = process.memory_info().rss
        except Exception:
            return 0.0
    return value / (1024 * 1024)


def _tree_memory_mb(psutil, pid: int) -> float:
    try:
        root = psutil.Process(pid)
        processes = [root] + root.children(recursive=True)
    except psutil.Error:
        return 0.0
    return sum(_process_memory_mb(p) for p in processes)


def run_engine(engine: str, path: str | Path, workdir: Path, trial: int) -> dict:
    """Run one engine in a fresh subprocess; measure processing time, start-up time and peak memory."""
    try:
        import psutil
    except ImportError:
        psutil = None
        warnings.warn("psutil is not installed; peak memory will not be measured.", RuntimeWarning, stacklevel=2)
    output = workdir / f"{Path(path).stem}_{engine}_{trial}.csv"
    command = [sys.executable, "-m", "processing_engines", engine, str(path), str(output)]
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=PROJECT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    peak = 0.0
    if psutil is not None:
        while process.poll() is None:
            peak = max(peak, _tree_memory_mb(psutil, process.pid))
            time.sleep(SAMPLE_INTERVAL_SECONDS)
    _, stderr = process.communicate()
    wall = time.perf_counter() - started
    if process.returncode != 0:
        lines = [line for line in (stderr or "").strip().splitlines() if line.strip()]
        return {"ok": False, "detail": lines[-1] if lines else f"Exited with code {process.returncode}"}
    meta = json.loads(Path(str(output) + ".meta.json").read_text())
    return {"ok": True, "result": pd.read_csv(output), "wall": wall, "peak": peak if psutil and peak > 0 else np.nan,
            "seconds": float(meta["processing_seconds"]), "startup": float(meta["startup_seconds"])}


def results_match(reference: pd.DataFrame, candidate: pd.DataFrame) -> bool:
    """Same groups, identical row counts, and numerically equal totals."""
    if list(reference.columns) != list(candidate.columns) or len(reference) != len(candidate):
        return False
    a = reference.sort_values(GROUP_KEYS).reset_index(drop=True)
    b = candidate.sort_values(GROUP_KEYS).reset_index(drop=True)
    if not (a[GROUP_KEYS].astype(str).values == b[GROUP_KEYS].astype(str).values).all():
        return False
    if not (a["rows"].to_numpy() == b["rows"].to_numpy()).all():
        return False
    return all(np.allclose(a[c].astype(float), b[c].astype(float), rtol=1e-9, atol=1e-6) for c in VALUE_COLUMNS)


def _failure_row(records: int, engine: str, status: str, detail: str, trials: int) -> dict:
    row = {c: np.nan for c in RESULT_COLUMNS}
    row.update({"records": records, "engine": engine, "status": status, "detail": detail, "trials": trials,
                "groups": 0, "consistent": "n/a"})
    return row


def run_benchmarks(paths: dict[int, str | Path], engines: tuple[str, ...] = ENGINES, trials: int = 3) -> pd.DataFrame:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    unknown = set(engines) - set(ENGINES)
    if unknown:
        raise ValueError(f"Unknown engines: {sorted(unknown)}")
    rows = []
    with tempfile.TemporaryDirectory(prefix="adbench_") as tmp:
        workdir = Path(tmp)
        for records, path in sorted(paths.items()):
            outcomes: dict[str, list[dict]] = {}
            for engine in engines:
                available, reason = engine_available(engine)
                if not available:
                    rows.append(_failure_row(records, engine, "skipped", reason, trials))
                    continue
                runs = [run_engine(engine, path, workdir, t) for t in range(trials)]
                good = [r for r in runs if r["ok"]]
                if not good:
                    rows.append(_failure_row(records, engine, "error", runs[0]["detail"], trials))
                    continue
                outcomes[engine] = good
            reference_engine = "pandas" if "pandas" in outcomes else next(iter(outcomes), None)
            reference = outcomes[reference_engine][0]["result"] if reference_engine else None
            for engine, good in outcomes.items():
                first = good[0]["result"]
                consistent = (all(results_match(first, r["result"]) for r in good[1:])
                              and results_match(reference, first) and int(first["rows"].sum()) == records)
                seconds = [r["seconds"] for r in good]
                rows.append({
                    "records": records, "engine": engine,
                    "status": "ok" if len(good) == trials else "partial",
                    "seconds": float(np.median(seconds)),
                    "startup_seconds": float(np.median([r["startup"] for r in good])),
                    "wall_seconds": float(np.median([r["wall"] for r in good])),
                    "peak_mem_mb": float(np.nanmedian([r["peak"] for r in good])) if not all(np.isnan([r["peak"] for r in good])) else np.nan,
                    "groups": len(first), "rows_counted": int(first["rows"].sum()),
                    "total_spend": float(first["spend"].sum()), "total_revenue": float(first["revenue"].sum()),
                    "consistent": "yes" if consistent else "no", "vs_pandas": np.nan, "trials": len(good),
                    "time_std_seconds": float(np.std(seconds, ddof=1)) if len(seconds) > 1 else 0.0,
                    "detail": "" if len(good) == trials else f"{trials - len(good)} trial(s) failed",
                })
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    for records, group in frame.groupby("records"):
        base = group[(group["engine"] == "pandas") & (group["status"] == "ok")]
        if not base.empty and base.iloc[0]["seconds"] > 0:
            usable = group.index[group["status"].isin(["ok", "partial"])]
            frame.loc[usable, "vs_pandas"] = base.iloc[0]["seconds"] / frame.loc[usable, "seconds"]
    return frame


def format_results(frame: pd.DataFrame) -> str:
    columns = ["records", "engine", "status", "seconds", "startup_seconds", "peak_mem_mb", "groups",
               "consistent", "vs_pandas", "trials", "time_std_seconds"]
    display = frame[[c for c in columns if c in frame]].copy()
    text = display.to_string(index=False, na_rep="—", float_format=lambda x: f"{x:,.3f}")
    notes = [f"  {r.records:,} / {r.engine}: {r.detail}" for r in frame.itertuples() if isinstance(r.detail, str) and r.detail]
    return text + ("\n\nNotes:\n" + "\n".join(notes) if notes else "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Pandas, simulated MapReduce and PySpark.")
    parser.add_argument("--scales", type=int, nargs="+", default=SCALES[:3])
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args(argv)
    if not args.scales or any(n <= 0 for n in args.scales):
        parser.error("--scales must contain positive record counts")
    if args.trials < 1:
        parser.error("--trials must be >= 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paths = {n: ensure_dataset(n, args.data_dir, args.seed, args.regenerate) for n in sorted(set(args.scales))}
    results = run_benchmarks(paths, tuple(args.engines), args.trials)
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "benchmark_results.csv", index=False)
    print(format_results(results))


if __name__ == "__main__":
    main()
