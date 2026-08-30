"""Single-process, interleaved ovVS-vs-hnswlib comparison for gate G1.

Both lanes drift badly on this machine -- hnswlib alone was measured between 35,300
and 61,287 QPS at one identical setting -- and that spread is larger than the gap we
are trying to resolve. Running them in separate processes made it worse: each hnswlib
process rebuilt its index with 20 threads first, so the search always ran on a package
that had just been heated differently.

So: build both indexes ONCE, then alternate timed rounds inside one process and report
the median of per-round ratios. Adjacent rounds share thermal state, which is the only
thing that makes a ratio on this box mean anything.

    python tools/bench/ab_g1.py --scale 100k --itopk 64 --width 4 --patience 2 \
        --ef 80 --batch 1024 --rounds 5 \
        --module out/comparators/hnswlib-v0.8.0-msvc-avx2-final-a/module

hnswlib settings stay frozen: M=16, ef_construction=200, seed 7, 20 threads. Only `ef`
moves, and only so its recall can be matched to ours -- never to weaken it.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quick_cagra import (  # noqa: E402
    POLICY_AUTO,
    POLICY_FORCE_GPU,
    SCALES,
    bind_serialization,
    load_fixture,
    recall_at_10,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import ovvs  # noqa: E402


def timed_round(search, chunks, nq):
    started = time.perf_counter()
    rows = [search(chunk) for chunk in chunks]
    elapsed = time.perf_counter() - started
    return nq / elapsed, np.concatenate(rows, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=tuple(SCALES), default="100k")
    ap.add_argument("--graph", type=Path, default=None)
    ap.add_argument("--module", type=Path, default=None)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--itopk", type=int, default=64)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--ef", type=int, default=80)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # Read before either library starts: the walk reads it once per call.
    os.environ["OVVS_CAGRA_PATIENCE"] = str(args.patience)

    if args.module:
        sys.path.insert(0, str(args.module.resolve()))
    import hnswlib

    if args.graph is None:
        args.graph = Path(f"out/quick/sift{args.scale}-base.ovvs")
    bind_serialization()
    base, queries, truth = load_fixture(args.scale, args.queries, args.graph.parent)
    n, dim = base.shape
    nq = queries.shape[0]
    chunks = [queries[i : i + args.batch] for i in range(0, nq, args.batch)]

    resource = ovvs.Resources()
    handle = ctypes.c_void_p()
    rc = ovvs._lib.ovvsCagraDeserialize(resource._h, os.fsencode(args.graph), ctypes.byref(handle))
    if rc != 0:
        raise RuntimeError(f"CAGRA deserialize failed with status {rc}")
    index = ovvs.CagraIndex(
        resource, handle, dim, own_res=False, destroy=ovvs._lib.ovvsCagraDestroy, n=n
    )
    # AUTO permits both engines, which is what the hybrid split needs; without
    # OVVS_HYBRID_WALK set it still runs the GPU path exactly as FORCE_GPU did.
    resource.set_policy(POLICY_AUTO if os.environ.get("OVVS_HYBRID_WALK") else POLICY_FORCE_GPU)

    hnsw = hnswlib.Index(space="l2", dim=dim)
    hnsw.init_index(max_elements=n, ef_construction=200, M=16, random_seed=7)
    hnsw.set_num_threads(args.threads)
    hnsw_build_started = time.perf_counter()
    hnsw.add_items(base, np.arange(n), num_threads=args.threads)
    hnsw_build_seconds = time.perf_counter() - hnsw_build_started
    hnsw.set_ef(args.ef)

    def ovvs_search(chunk):
        ids, _ = index.search(chunk, k=10, itopk_size=args.itopk, search_width=args.width)
        return np.asarray(ids, dtype=np.int64).reshape(chunk.shape[0], 10)

    def hnsw_search(chunk):
        labels, _ = hnsw.knn_query(chunk, k=10, num_threads=args.threads)
        return np.asarray(labels, dtype=np.int64).reshape(chunk.shape[0], 10)

    # One untimed warmup each: SYCL is JIT, so the first ovVS launch pays kernel compilation.
    timed_round(ovvs_search, chunks, nq)
    timed_round(hnsw_search, chunks, nq)

    ovvs_qps, hnsw_qps, ratios = [], [], []
    ovvs_recall = hnsw_recall = 0.0
    for _ in range(args.rounds):
        a, ovvs_ids = timed_round(ovvs_search, chunks, nq)
        b, hnsw_ids = timed_round(hnsw_search, chunks, nq)
        ovvs_qps.append(a)
        hnsw_qps.append(b)
        ratios.append(b / a)
        ovvs_recall = recall_at_10(ovvs_ids, truth)
        hnsw_recall = recall_at_10(hnsw_ids, truth)

    report = {
        "scale": args.scale,
        "n": int(n),
        "batch": args.batch,
        "rounds": args.rounds,
        "ovvs": {
            "itopk": args.itopk,
            "width": args.width,
            "patience": args.patience,
            "recall_at_10": round(ovvs_recall, 4),
            "qps_median": round(statistics.median(ovvs_qps), 1),
            "qps_rounds": [round(v, 1) for v in ovvs_qps],
        },
        "hnswlib": {
            "ef": args.ef,
            "threads_requested": args.threads,
            "threads_effective": 1 if args.batch <= args.threads * 4 else args.threads,
            "build_seconds": round(hnsw_build_seconds, 3),
            "recall_at_10": round(hnsw_recall, 4),
            "qps_median": round(statistics.median(hnsw_qps), 1),
            "qps_rounds": [round(v, 1) for v in hnsw_qps],
        },
        "hnswlib_ahead_by": round(statistics.median(ratios), 3),
        "ratio_rounds": [round(v, 3) for v in ratios],
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
