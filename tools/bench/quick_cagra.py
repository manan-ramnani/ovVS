"""Search-only CAGRA iteration harness.

Build the SIFT1M graph once, then re-measure search against any ovvs.dll without
paying the construction wall again.

    set OVVS_LIBRARY=C:\\Personal\\ioVS\\build-icpx\\bin\\ovvs.dll
    python tools/bench/quick_cagra.py build --graph out/quick/sift1m.ovvs
    python tools/bench/quick_cagra.py search --graph out/quick/sift1m.ovvs

hnswlib is not run here and its settings are never touched; compare against the
numbers already measured on this machine.
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

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import ovvs  # noqa: E402

DATASET = Path(__file__).resolve().parents[2] / "data" / "sift-128-euclidean.hdf5"
POLICY_AUTO = 0
POLICY_FORCE_GPU = 5
POLICY_FORCE_CPU = 6

# Rungs of the scale ladder. Win at each before climbing.
SCALES = {"100k": 100_000, "1m": 1_000_000}

# (itopk_size, search_width, query_batch_size)
# Batch 256/1024 matter: hnswlib's binding collapses to ONE thread when
# rows <= num_threads*4 (bindings.cpp:630), so only the large-batch points
# compare against a fully-threaded hnswlib.
DEFAULT_POINTS = [
    (32, 1, 32),
    (64, 2, 32),
    (128, 4, 32),
    (32, 1, 256),
    (64, 2, 256),
    (32, 1, 1024),
    (64, 2, 1024),
    (64, 2, 1),
]


def exact_truth(base: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    """Exact top-k by brute force. Required for any subset of the base, because the
    HDF5 `neighbors` field is only valid for the complete 1M base."""
    try:
        import faiss

        flat = faiss.IndexFlatL2(base.shape[1])
        flat.add(base)
        _, ids = flat.search(queries, k)
        return np.ascontiguousarray(ids, dtype=np.int64)
    except ImportError:
        out = np.empty((queries.shape[0], k), dtype=np.int64)
        base_sq = np.einsum("ij,ij->i", base, base)
        for start in range(0, queries.shape[0], 64):
            block = queries[start : start + 64]
            scores = base_sq - 2.0 * (block @ base.T)
            out[start : start + block.shape[0]] = np.argpartition(scores, k, axis=1)[:, :k]
        for i in range(queries.shape[0]):
            d = np.linalg.norm(base[out[i]] - queries[i], axis=1)
            out[i] = out[i][np.argsort(d)]
        return out


def load_fixture(scale: str, nq: int, cache_dir: Path):
    rows = SCALES[scale]
    with h5py.File(DATASET, "r") as f:
        base = np.ascontiguousarray(f["train"][:rows], dtype=np.float32)
        queries = np.ascontiguousarray(f["test"][:nq], dtype=np.float32)
        if rows >= f["train"].shape[0]:
            truth = np.ascontiguousarray(f["neighbors"][:nq, :10], dtype=np.int64)
            return base, queries, truth

    cache = cache_dir / f"truth-{scale}-nq{nq}.npy"
    if cache.exists():
        return base, queries, np.load(cache)
    truth = exact_truth(base, queries, 10)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, truth)
    return base, queries, truth


def bind_serialization():
    ovvs._lib.ovvsCagraSerialize.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    ovvs._lib.ovvsCagraSerialize.restype = ctypes.c_int32
    ovvs._lib.ovvsCagraDeserialize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ovvs._lib.ovvsCagraDeserialize.restype = ctypes.c_int32


def recall_at_10(ids: np.ndarray, truth: np.ndarray) -> float:
    hits = 0
    for row, want in zip(ids, truth):
        hits += len(set(int(v) for v in row) & set(int(v) for v in want))
    return hits / (truth.shape[0] * truth.shape[1])


def count_work(resource, index, queries, itopk, width, batch):
    """One extra UNTIMED pass with counters on, so timing is never perturbed."""
    if not resource.set_cagra_walk_counters(True):
        return None
    try:
        resource.reset_cagra_walk_counters()
        for i in range(0, queries.shape[0], batch):
            index.search(queries[i : i + batch], k=10, itopk_size=itopk, search_width=width)
        raw = resource.cagra_walk_counters()
    finally:
        resource.set_cagra_walk_counters(False)
    q = max(1, raw["queries"])
    return {
        "queries": raw["queries"],
        "evals_per_query": round(raw["evals"] / q, 1),
        "seed_evals_per_query": round(raw["seed_evals"] / q, 1),
        "seed_share": round(raw["seed_evals"] / max(1, raw["evals"]), 4),
        "iters_per_query": round(raw["iterations"] / q, 2),
        "max_iterations": raw["max_iterations"],
        "admits_per_query": round(raw["admits"] / q, 1),
        "survivors_per_query": round(raw["survivors"] / q, 1),
        "survivor_share": round(raw["survivors"] / max(1, raw["admits"]), 4),
        "table_full": raw["table_full"],
    }


def run_point(index, queries, truth, itopk, width, batch, warmup, passes):
    nq = queries.shape[0]
    chunks = [queries[i : i + batch] for i in range(0, nq, batch)]

    for _ in range(warmup):
        for chunk in chunks:
            index.search(chunk, k=10, itopk_size=itopk, search_width=width)

    pass_qps = []
    batch_ms = []
    ids_last = None
    for _ in range(passes):
        rows = []
        started = time.perf_counter()
        for chunk in chunks:
            t0 = time.perf_counter()
            ids, _ = index.search(chunk, k=10, itopk_size=itopk, search_width=width)
            batch_ms.append((time.perf_counter() - t0) * 1000.0)
            rows.append(np.asarray(ids, dtype=np.int64).reshape(chunk.shape[0], 10))
        elapsed = time.perf_counter() - started
        pass_qps.append(nq / elapsed)
        ids_last = np.concatenate(rows, axis=0)

    batch_ms.sort()

    def pct(p):
        if not batch_ms:
            return 0.0
        return batch_ms[min(len(batch_ms) - 1, int(round(p * (len(batch_ms) - 1))))]

    return {
        "itopk_size": itopk,
        "search_width": width,
        "query_batch_size": batch,
        "recall_at_10": round(recall_at_10(ids_last, truth), 4),
        "qps_median": round(statistics.median(pass_qps), 1),
        "qps_passes": [round(v, 1) for v in pass_qps],
        "batch_p50_ms": round(pct(0.50), 4),
        "batch_p99_ms": round(pct(0.99), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("build", "search"))
    ap.add_argument("--scale", choices=tuple(SCALES), default="100k")
    ap.add_argument("--graph", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--passes", type=int, default=5)
    ap.add_argument("--graph-degree", type=int, default=16)
    ap.add_argument("--intermediate-degree", type=int, default=32)
    ap.add_argument("--search-policy", choices=("gpu", "cpu", "auto"), default="gpu")
    ap.add_argument(
        "--counters",
        action="store_true",
        help="add one untimed counting pass per point (never perturbs the timed passes)",
    )
    ap.add_argument(
        "--points",
        default=None,
        help="semicolon list of itopk,width,batch (default: the standard five)",
    )
    args = ap.parse_args()

    library = os.environ.get("OVVS_LIBRARY", "<unset>")
    if args.graph is None:
        args.graph = Path(f"out/quick/sift{args.scale}-base.ovvs")
    bind_serialization()

    if args.points:
        points = []
        for chunk in args.points.split(";"):
            if not chunk.strip():
                continue
            itopk, width, batch = (int(v) for v in chunk.split(","))
            points.append((itopk, width, batch))
    else:
        points = DEFAULT_POINTS

    base, queries, truth = load_fixture(args.scale, args.queries, args.graph.parent)
    resource = ovvs.Resources()
    index = None
    try:
        if args.mode == "build":
            resource.set_policy(POLICY_AUTO)
            started = time.perf_counter()
            index = ovvs.neighbors.cagra.build(
                base,
                graph_degree=args.graph_degree,
                intermediate_degree=args.intermediate_degree,
                resources=resource,
            )
            build_seconds = time.perf_counter() - started
            args.graph.parent.mkdir(parents=True, exist_ok=True)
            rc = ovvs._lib.ovvsCagraSerialize(index._h, os.fsencode(args.graph))
            if rc != 0:
                raise RuntimeError(f"CAGRA serialize failed with status {rc}")
            print(json.dumps({
                "mode": "build",
                "scale": args.scale,
                "library": library,
                "build_seconds": round(build_seconds, 3),
                "graph": str(args.graph),
                "graph_bytes": args.graph.stat().st_size,
                "n": int(base.shape[0]),
                "dim": int(base.shape[1]),
                "graph_degree": args.graph_degree,
                "intermediate_degree": args.intermediate_degree,
            }, indent=2))
            return 0

        handle = ctypes.c_void_p()
        rc = ovvs._lib.ovvsCagraDeserialize(
            resource._h, os.fsencode(args.graph), ctypes.byref(handle)
        )
        if rc != 0:
            raise RuntimeError(f"CAGRA deserialize failed with status {rc}")
        index = ovvs.CagraIndex(
            resource,
            handle,
            int(base.shape[1]),
            own_res=False,
            destroy=ovvs._lib.ovvsCagraDestroy,
            n=int(base.shape[0]),
        )

        policy = {
            "gpu": POLICY_FORCE_GPU,
            "cpu": POLICY_FORCE_CPU,
            "auto": POLICY_AUTO,  # both engines permitted; OVVS_HYBRID_WALK splits them
        }[args.search_policy]
        resource.set_policy(policy)
        results = []
        for itopk, width, batch in points:
            row = run_point(index, queries, truth, itopk, width, batch, args.warmup, args.passes)
            row["last_device"] = resource.last_device()
            if args.counters:
                row["work"] = count_work(resource, index, queries, itopk, width, batch)
            results.append(row)
            print(
                f"itopk={itopk:<4} width={width:<2} batch={batch:<4} "
                f"recall={row['recall_at_10']:.4f}  qps={row['qps_median']:>9.1f}  "
                f"p50={row['batch_p50_ms']:>8.3f}ms  p99={row['batch_p99_ms']:>8.3f}ms",
                flush=True,
            )
            w = row.get("work")
            if w:
                print(
                    f"    work: evals/q={w['evals_per_query']:<8} "
                    f"seeds/q={w['seed_evals_per_query']:<7} (={w['seed_share']:.1%} of evals)  "
                    f"iters/q={w['iters_per_query']:<7} max_iters={w['max_iterations']:<5} "
                    f"table_full={w['table_full']}  "
                    f"survivors/q={w['survivors_per_query']:<7} (={w['survivor_share']:.1%})",
                    flush=True,
                )

        payload = {
            "mode": "search",
            "scale": args.scale,
            "library": library,
            "graph": str(args.graph),
            "policy": args.search_policy,
            "queries": int(queries.shape[0]),
            "warmup_passes": args.warmup,
            "measured_passes": args.passes,
            "points": results,
        }
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        if index is not None:
            index.close()
        resource.close()


if __name__ == "__main__":
    raise SystemExit(main())
