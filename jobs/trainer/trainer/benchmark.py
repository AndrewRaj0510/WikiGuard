"""Measure a served model the way serving will use it: one edit at a time, on a limited CPU.

Needs only the serving install, so it runs in a small container with Render's limits. Saves numbers
for whichever machine it ran on, because int8 speed differs between amd64 and arm64.

    python -m trainer.benchmark data/runs/mlp/model --rows snapshots/NAME --save-sample sample.json
    python -m trainer.benchmark data/runs/mlp/model --rows sample.json --out result.json
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from wikiguard_core.artifacts import sha256_file
from wikiguard_core.inference import ServedModel
from wikiguard_core.snapshot import ROWS_FILE, TEST

SAMPLE_ROWS = 200
WARMUP_ROWS = 5
# Long edits are the slow case, so they're measured deliberately and not left to chance.
LONGEST_ROWS = 20


def peak_rss_mb() -> float:
    """Peak memory of this process. Linux containers report it through resource, Windows psutil."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        import psutil

        return float(psutil.Process().memory_info().peak_wset) / (1024 * 1024)


def benchmark_rows(source: Path, count: int = SAMPLE_ROWS, seed: int = 0) -> list[dict[str, Any]]:
    """Rows from a saved sample, or drawn from a snapshot folder.

    Reading a whole snapshot costs about 400 MB, which peak memory would count as the model's. A
    run under Render's 512 MB uses a sample saved earlier with --save-sample.
    """
    if source.is_file():
        rows: list[dict[str, Any]] = json.loads(source.read_text("utf-8"))
        return rows[:count]
    table = pq.read_table(source / ROWS_FILE)
    rows = [row for row in table.to_pylist() if row["split"] == TEST]
    order = np.random.default_rng(seed).permutation(len(rows))[:count]
    return [rows[index] for index in order.tolist()]


def latency_ms(model: ServedModel, rows: list[dict[str, Any]]) -> list[float]:
    times = []
    for row in rows:
        started = time.perf_counter()
        model.probabilities([row])
        times.append((time.perf_counter() - started) * 1000)
    return times


def summarize(times: list[float]) -> dict[str, float]:
    values = np.array(times)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


def measure(folder: Path, rows: list[dict[str, Any]], threads: int) -> dict[str, Any]:
    started = time.perf_counter()
    model = ServedModel(folder, threads=threads)
    load_seconds = time.perf_counter() - started
    memory_after_load = peak_rss_mb()
    latency_ms(model, rows[:WARMUP_ROWS])
    typical = latency_ms(model, rows)
    longest = sorted(
        rows, key=lambda row: len(row["added_text"]) + len(row["removed_text"]), reverse=True
    )[:LONGEST_ROWS]
    slow = latency_ms(model, longest)
    return {
        "model": model.config["name"],
        "threads": threads,
        "rows": len(rows),
        "machine": platform.machine(),
        "system": platform.system(),
        "load_seconds": load_seconds,
        "model_file_bytes": (folder / "model.onnx").stat().st_size,
        "folder_bytes": sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()),
        "checksums_sha256": sha256_file(folder / "checksums.json"),
        "peak_rss_after_load_mb": memory_after_load,
        "peak_rss_mb": peak_rss_mb(),
        "single_edit": summarize(typical),
        "longest_edits": summarize(slow),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--rows", type=Path, required=True, help="snapshot folder or saved sample")
    parser.add_argument("--save-sample", type=Path, help="write the rows to this file and stop")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--count", type=int, default=SAMPLE_ROWS)
    parser.add_argument("--label", default="laptop")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = benchmark_rows(args.rows, args.count)
    if args.save_sample:
        args.save_sample.parent.mkdir(parents=True, exist_ok=True)
        args.save_sample.write_text(json.dumps(rows, default=str), "utf-8")
        return 0
    result = measure(args.folder, rows, args.threads)
    result["label"] = args.label
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, "utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
