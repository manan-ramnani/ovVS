"""Clean-run ovVS-vs-hnswlib matrix at one scale: query throughput + latency
percentiles at three load points, insert/update/delete throughput + latency,
recall, and (in dedicated single-engine processes) resident memory.

Both engines run in ONE process with alternating timed rounds -- the ab_g1
discipline -- because absolute QPS on this box drifts more than the gaps being
measured; adjacent rounds share thermal state. CRUD phases alternate engines
over IDENTICAL mutation streams. Memory runs as `mem` in its own process per
engine, since RSS cannot be attributed once both engines have allocated.

    set OVVS_LIBRARY=C:/Personal/ioVS/build-icpx/bin/ovvs.dll
    python tools/bench/matrix.py perf --scale 10k --out out/matrix/perf-10k.json
    python tools/bench/matrix.py mem --scale 10k --engine ovvs --ef 64

hnswlib stays frozen (M=16, ef_construction=200, seed 7, 20 threads). Only ef
moves, auto-calibrated per scale to land its recall as close as possible to the
recall ovVS measured -- never to weaken it. ovVS runs the shipped HETERO
policy: threaded CPU walk below the nq crossover, CPU+GPU split above it.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quick_cagra import (  # noqa: E402
    DATASET,
    POLICY_FORCE_CPU,
    POLICY_HETERO,
    SCALES,
    bind_serialization,
    load_fixture,
    recall_at_10,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import ovvs  # noqa: E402

DEFAULT_MODULE = Path("out/comparators/hnswlib-v0.8.0-msvc-avx2-final-a/module")
EF_LADDER = [24, 32, 48, 64, 80, 96, 112, 128, 160, 200, 256, 320, 384, 512]
MUT_COUNTS = {"10k": 1_000, "100k": 5_000, "1m": 10_000}
SINGLE_OPS = 200
CRUD_ROUNDS = 3


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def rss() -> dict:
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    ok = psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    return {
        "working_set": int(pmc.WorkingSetSize),
        "peak_working_set": int(pmc.PeakWorkingSetSize),
        "private": int(pmc.PrivateUsage),
    }


def pct(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def lat_summary(ms: list[float]) -> dict:
    return {
        "n": len(ms),
        "p50_ms": round(pct(ms, 0.50), 4),
        "p95_ms": round(pct(ms, 0.95), 4),
        "p99_ms": round(pct(ms, 0.99), 4),
    }


def timed_round(search, chunks, nq):
    """One full pass; returns (qps, per-chunk ms, concatenated ids)."""
    per_ms = []
    rows = []
    started = time.perf_counter()
    for chunk in chunks:
        t0 = time.perf_counter()
        ids = search(chunk)
        per_ms.append((time.perf_counter() - t0) * 1000.0)
        rows.append(ids)
    elapsed = time.perf_counter() - started
    return nq / elapsed, per_ms, np.concatenate(rows, axis=0)


def load_queries_full(nq: int) -> np.ndarray:
    with h5py.File(DATASET, "r") as f:
        return np.ascontiguousarray(f["test"][:nq], dtype=np.float32)


def open_ovvs(graph: Path, dim: int, n: int):
    resource = ovvs.Resources()
    handle = ctypes.c_void_p()
    rc = ovvs._lib.ovvsCagraDeserialize(resource._h, os.fsencode(graph), ctypes.byref(handle))
    if rc != 0:
        raise RuntimeError(f"CAGRA deserialize failed with status {rc}")
    index = ovvs.CagraIndex(
        resource, handle, dim, own_res=False, destroy=ovvs._lib.ovvsCagraDestroy, n=n
    )
    return resource, index


def perf(args) -> int:
    os.environ["OVVS_CAGRA_PATIENCE"] = "0"
    sys.path.insert(0, str(args.module.resolve()))
    import hnswlib

    bind_serialization()
    base, queries1k, truth = load_fixture(args.scale, 1000, Path("out/quick"))
    n, dim = base.shape
    queries_bulk = load_queries_full(10_000)

    resource, index = open_ovvs(args.graph, dim, n)
    resource.set_policy(POLICY_HETERO)

    hnsw = hnswlib.Index(space="l2", dim=dim)
    hnsw.init_index(max_elements=n, ef_construction=200, M=16, random_seed=7)
    hnsw.set_num_threads(args.threads)
    t0 = time.perf_counter()
    hnsw.add_items(base, np.arange(n), num_threads=args.threads)
    hnsw_build_seconds = time.perf_counter() - t0
    hnsw_index_path = Path(f"out/matrix/hnsw-{args.scale}.bin")
    hnsw_index_path.parent.mkdir(parents=True, exist_ok=True)
    hnsw.save_index(str(hnsw_index_path))
    print(f"# hnswlib built in {hnsw_build_seconds:.1f}s, saved {hnsw_index_path}", flush=True)

    def ovvs_search(chunk):
        ids, _ = index.search(chunk, k=10, itopk_size=args.itopk, search_width=args.width)
        return np.asarray(ids, dtype=np.int64).reshape(chunk.shape[0], 10)

    def hnsw_search(chunk):
        labels, _ = hnsw.knn_query(chunk, k=10, num_threads=args.threads)
        return np.asarray(labels, dtype=np.int64).reshape(chunk.shape[0], 10)

    # --- recall + ef calibration (untimed) --------------------------------
    chunks1k = [queries1k]
    _, _, ids = timed_round(ovvs_search, chunks1k, 1000)  # also pays SYCL JIT
    ovvs_recall = recall_at_10(ids, truth)
    picked = None
    for ef in EF_LADDER:
        hnsw.set_ef(ef)
        _, _, ids = timed_round(hnsw_search, chunks1k, 1000)
        r = recall_at_10(ids, truth)
        if picked is None or abs(r - ovvs_recall) < abs(picked[1] - ovvs_recall):
            picked = (ef, r)
        print(f"# ef={ef} recall={r:.4f} (ovvs {ovvs_recall:.4f})", flush=True)
        if r >= ovvs_recall:
            break
    ef, hnsw_recall = picked
    hnsw.set_ef(ef)
    print(f"# matched: ovvs {args.itopk}/{args.width} p0 recall={ovvs_recall:.4f}  "
          f"hnsw ef={ef} recall={hnsw_recall:.4f}", flush=True)

    # --- query: three load points, alternating rounds ---------------------
    query_out = {}
    for batch, qset in ((1, queries1k), (32, queries1k), (1000, queries_bulk)):
        nq = qset.shape[0]
        chunks = [qset[i : i + batch] for i in range(0, nq, batch)]
        timed_round(ovvs_search, chunks, nq)  # warmup
        timed_round(hnsw_search, chunks, nq)
        lanes = {"ovvs": {"qps": [], "ms": []}, "hnswlib": {"qps": [], "ms": []}}
        # The gate metric is SUSTAINED throughput, so each engine gets an untimed
        # warm touch before its timed round. Without it the b1000 point is bistable:
        # the ~430ms of CPU-heavy hnsw between ovVS rounds sits right at the iGPU's
        # park threshold, and a 10000-query call that STARTS parked runs entirely at
        # low clocks (measured 17.2K vs 41.2K hot in the same window, ratio flipping
        # 0.75x vs 1.76x run to run).
        warm = chunks[: max(1, len(chunks) // 16)]
        for _ in range(args.rounds):
            timed_round(ovvs_search, warm, sum(c.shape[0] for c in warm))
            q, ms, _ = timed_round(ovvs_search, chunks, nq)
            lanes["ovvs"]["qps"].append(q)
            lanes["ovvs"]["ms"].extend(ms)
            timed_round(hnsw_search, warm, sum(c.shape[0] for c in warm))
            q, ms, _ = timed_round(hnsw_search, chunks, nq)
            lanes["hnswlib"]["qps"].append(q)
            lanes["hnswlib"]["ms"].extend(ms)
        point = {}
        for name, lane in lanes.items():
            point[name] = {
                "qps_median": round(statistics.median(lane["qps"]), 1),
                "qps_rounds": [round(v, 1) for v in lane["qps"]],
                **lat_summary(lane["ms"]),
            }
        ratio = point["ovvs"]["qps_median"] / max(1e-9, point["hnswlib"]["qps_median"])
        point["ovvs_over_hnsw"] = round(ratio, 3)
        query_out[f"b{batch}"] = point
        print(f"# b{batch}: ovvs {point['ovvs']['qps_median']:.0f} qps "
              f"(p50 {point['ovvs']['p50_ms']:.3f}ms) vs hnsw {point['hnswlib']['qps_median']:.0f} "
              f"(p50 {point['hnswlib']['p50_ms']:.3f}ms) -> x{ratio:.2f}", flush=True)

    # --- CRUD: identical mutation streams, alternating engines ------------
    rng = np.random.default_rng(7)
    mut = MUT_COUNTS[args.scale]
    pool = rng.permutation(n)
    span = CRUD_ROUNDS * mut + SINGLE_OPS
    upd_pool, del_pool = pool[:span], pool[span : 2 * span]

    def fresh(count):
        return np.ascontiguousarray(
            base[rng.integers(0, n, size=count)] + rng.normal(0, 4, (count, dim)),
            dtype=np.float32,
        )

    hnsw.resize_index(n + span)
    hnsw_next = n

    def ovvs_mutate(fn, *fn_args):
        resource.set_policy(POLICY_FORCE_CPU)
        t0 = time.perf_counter()
        fn(*fn_args)
        elapsed = time.perf_counter() - t0
        resource.set_policy(POLICY_HETERO)
        return elapsed

    def timed(fn, *fn_args):
        t0 = time.perf_counter()
        fn(*fn_args)
        return time.perf_counter() - t0

    crud = {}
    for op in ("insert", "update", "delete"):
        lanes = {
            "ovvs": {"batch_s": [], "single_ms": []},
            "hnswlib": {"batch_s": [], "single_ms": []},
        }
        for r in range(CRUD_ROUNDS):
            lo, hi = r * mut, (r + 1) * mut
            if op == "insert":
                vecs = fresh(mut)
                lanes["ovvs"]["batch_s"].append(ovvs_mutate(index.extend_ex, vecs))
                labels = np.arange(hnsw_next, hnsw_next + mut)
                hnsw_next += mut
                lanes["hnswlib"]["batch_s"].append(
                    timed(hnsw.add_items, vecs, labels, args.threads))
            elif op == "update":
                targets = upd_pool[lo:hi]
                vecs = fresh(mut)
                lanes["ovvs"]["batch_s"].append(
                    ovvs_mutate(index.update, targets.astype(np.int64), vecs))
                lanes["hnswlib"]["batch_s"].append(
                    timed(hnsw.add_items, vecs, targets, args.threads))
            else:
                victims = del_pool[lo:hi].astype(np.int64)
                lanes["ovvs"]["batch_s"].append(ovvs_mutate(index.delete, victims))

                def hnsw_del(ids):
                    for v in ids:
                        hnsw.mark_deleted(int(v))

                lanes["hnswlib"]["batch_s"].append(timed(hnsw_del, victims))
        # singles: alternate engines per op so both see the same thermal window
        tail = CRUD_ROUNDS * mut
        single_vecs = fresh(SINGLE_OPS)
        for i in range(SINGLE_OPS):
            v = single_vecs[i : i + 1]
            if op == "insert":
                lanes["ovvs"]["single_ms"].append(1000.0 * ovvs_mutate(index.extend_ex, v))
                lanes["hnswlib"]["single_ms"].append(
                    1000.0 * timed(hnsw.add_items, v, np.array([hnsw_next]), 1))
                hnsw_next += 1
            elif op == "update":
                t = upd_pool[tail + i : tail + i + 1]
                lanes["ovvs"]["single_ms"].append(
                    1000.0 * ovvs_mutate(index.update, t.astype(np.int64), v))
                lanes["hnswlib"]["single_ms"].append(1000.0 * timed(hnsw.add_items, v, t, 1))
            else:
                t = int(del_pool[tail + i])
                lanes["ovvs"]["single_ms"].append(
                    1000.0 * ovvs_mutate(index.delete, np.array([t], dtype=np.int64)))
                lanes["hnswlib"]["single_ms"].append(1000.0 * timed(hnsw.mark_deleted, t))
        entry = {}
        for name, lane in lanes.items():
            ops_s = [mut / s for s in lane["batch_s"]]
            entry[name] = {
                "batch_size": mut,
                "batch_ops_per_s": round(statistics.median(ops_s), 1),
                "batch_ops_per_s_rounds": [round(v, 1) for v in ops_s],
                "single": lat_summary(lane["single_ms"]),
            }
        entry["ovvs_over_hnsw"] = round(
            entry["ovvs"]["batch_ops_per_s"] / max(1e-9, entry["hnswlib"]["batch_ops_per_s"]), 3)
        crud[op] = entry
        print(f"# {op}: ovvs {entry['ovvs']['batch_ops_per_s']:.0f} ops/s vs "
              f"hnsw {entry['hnswlib']['batch_ops_per_s']:.0f} ops/s -> x{entry['ovvs_over_hnsw']}",
              flush=True)

    live, deleted = index.counts()
    report = {
        "phase": "perf",
        "scale": args.scale,
        "n": int(n),
        "dim": int(dim),
        "library": os.environ.get("OVVS_LIBRARY", "<unset>"),
        "graph": str(args.graph),
        "ovvs_config": {"policy": "hetero", "itopk": args.itopk, "width": args.width, "patience": 0},
        "hnsw_config": {"M": 16, "ef_construction": 200, "seed": 7, "threads": args.threads,
                        "ef": ef, "build_seconds": round(hnsw_build_seconds, 2)},
        "recall_at_10": {"ovvs": round(ovvs_recall, 4), "hnswlib": round(hnsw_recall, 4)},
        "query": query_out,
        "crud": crud,
        "ovvs_counts_after": {"live": live, "deleted": deleted},
    }
    out = args.out or Path(f"out/matrix/perf-{args.scale}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"# wrote {out}", flush=True)
    index.close()
    resource.close()
    return 0


def mem(args) -> int:
    os.environ["OVVS_CAGRA_PATIENCE"] = "0"
    sys.path.insert(0, str(args.module.resolve()))
    import hnswlib

    bind_serialization()
    n = SCALES[args.scale]
    queries = load_queries_full(1000)
    dim = queries.shape[1]
    gc.collect()
    base_rss = rss()

    if args.engine == "ovvs":
        resource, index = open_ovvs(args.graph, dim, n)
        resource.set_policy(POLICY_HETERO)
        file_bytes = args.graph.stat().st_size
        idle = rss()
        index.search(queries, k=10, itopk_size=args.itopk, search_width=args.width)
        index.search(queries, k=10, itopk_size=args.itopk, search_width=args.width)
    else:
        path = Path(f"out/matrix/hnsw-{args.scale}.bin")
        idx = hnswlib.Index(space="l2", dim=dim)
        idx.load_index(str(path), max_elements=n)
        idx.set_num_threads(args.threads)
        idx.set_ef(args.ef)
        file_bytes = path.stat().st_size
        idle = rss()
        idx.knn_query(queries, k=10, num_threads=args.threads)
        idx.knn_query(queries, k=10, num_threads=args.threads)
    active = rss()

    report = {
        "phase": "mem",
        "scale": args.scale,
        "engine": args.engine,
        "ef": args.ef if args.engine == "hnswlib" else None,
        "file_bytes": int(file_bytes),
        "baseline": base_rss,
        "idle_delta_ws": idle["working_set"] - base_rss["working_set"],
        "idle_delta_private": idle["private"] - base_rss["private"],
        "active_delta_ws": active["working_set"] - base_rss["working_set"],
        "active_delta_private": active["private"] - base_rss["private"],
        "peak_working_set": active["peak_working_set"],
    }
    print(json.dumps(report, indent=2))
    out = args.out or Path(f"out/matrix/mem-{args.scale}-{args.engine}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("perf", "mem"))
    ap.add_argument("--scale", choices=tuple(SCALES), required=True)
    ap.add_argument("--graph", type=Path, default=None)
    ap.add_argument("--module", type=Path, default=DEFAULT_MODULE)
    ap.add_argument("--engine", choices=("ovvs", "hnswlib"), default="ovvs")
    ap.add_argument("--itopk", type=int, default=32)
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--ef", type=int, default=96)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.graph is None:
        args.graph = Path(f"out/quick/sift{args.scale}-d32.ovvs")
    return perf(args) if args.phase == "perf" else mem(args)


if __name__ == "__main__":
    raise SystemExit(main())
