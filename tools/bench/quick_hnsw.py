"""hnswlib comparator for the same fixture quick_cagra.py uses.

Settings are FROZEN and never weakened: M=16, ef_construction=200, seed 7,
20 threads requested. Note hnswlib's own binding collapses to a single thread
when rows <= num_threads*4 (bindings.cpp:631), so batch 32 and batch 1 run
single-threaded; the reported `threads_effective` makes that explicit.

    python tools/bench/quick_hnsw.py --scale 100k \
        --module out/comparators/hnswlib-v0.8.0-msvc-avx2-final-a/module
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quick_cagra import SCALES, load_fixture, recall_at_10  # noqa: E402

EF_POINTS = [32, 64, 128, 256]
BATCHES = [32, 256, 1024, 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=tuple(SCALES), default="100k")
    ap.add_argument("--module", type=Path, default=None, help="dir holding hnswlib .pyd (AVX2 build)")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ef", type=str, default=None, help="comma-separated ef values")
    ap.add_argument("--batches", type=str, default=None, help="comma-separated batch sizes")
    args = ap.parse_args()

    if args.module:
        sys.path.insert(0, str(args.module.resolve()))
    import hnswlib

    base, queries, truth = load_fixture(args.scale, args.queries, Path("out/quick"))
    n, dim = base.shape

    index = hnswlib.Index(space="l2", dim=dim)
    index.init_index(max_elements=n, ef_construction=200, M=16, random_seed=7)
    index.set_num_threads(args.threads)
    started = time.perf_counter()
    index.add_items(base, np.arange(n), num_threads=args.threads)
    build_seconds = time.perf_counter() - started

    ef_points = [int(v) for v in args.ef.split(",")] if args.ef else EF_POINTS
    batches = [int(v) for v in args.batches.split(",")] if args.batches else BATCHES

    results = []
    for ef in ef_points:
        index.set_ef(ef)
        for batch in batches:
            chunks = [queries[i : i + batch] for i in range(0, args.queries, batch)]
            for _ in range(args.warmup):
                for c in chunks:
                    index.knn_query(c, k=10, num_threads=args.threads)
            pass_qps, batch_ms, ids_last = [], [], None
            for _ in range(args.passes):
                rows = []
                t_start = time.perf_counter()
                for c in chunks:
                    t0 = time.perf_counter()
                    ids, _ = index.knn_query(c, k=10, num_threads=args.threads)
                    batch_ms.append((time.perf_counter() - t0) * 1000.0)
                    rows.append(np.asarray(ids, dtype=np.int64))
                pass_qps.append(args.queries / (time.perf_counter() - t_start))
                ids_last = np.concatenate(rows, axis=0)
            batch_ms.sort()

            def pct(p):
                return batch_ms[min(len(batch_ms) - 1, int(round(p * (len(batch_ms) - 1))))]

            row = {
                "ef": ef,
                "query_batch_size": batch,
                # hnswlib's binding: rows <= num_threads*4 -> 1 thread
                "threads_effective": 1 if batch <= args.threads * 4 else args.threads,
                "recall_at_10": round(recall_at_10(ids_last, truth), 4),
                "qps_median": round(statistics.median(pass_qps), 1),
                "batch_p50_ms": round(pct(0.50), 4),
                "batch_p99_ms": round(pct(0.99), 4),
            }
            results.append(row)
            print(
                f"ef={ef:<5} batch={batch:<5} threads={row['threads_effective']:<3} "
                f"recall={row['recall_at_10']:.4f}  qps={row['qps_median']:>9.1f}  "
                f"p50={row['batch_p50_ms']:>8.3f}ms  p99={row['batch_p99_ms']:>8.3f}ms",
                flush=True,
            )

    payload = {
        "scale": args.scale,
        "module": str(args.module) if args.module else "ambient",
        "hnswlib_version": getattr(hnswlib, "__version__", "unknown"),
        "n": int(n),
        "dim": int(dim),
        "M": 16,
        "ef_construction": 200,
        "random_seed": 7,
        "threads_requested": args.threads,
        "build_seconds": round(build_seconds, 3),
        "points": results,
    }
    print(json.dumps({k: payload[k] for k in ("scale", "n", "build_seconds")}, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
