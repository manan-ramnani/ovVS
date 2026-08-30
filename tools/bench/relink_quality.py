"""Does low relink effort damage the graph that queries then walk?

Two indexes from the same base, identical mutation streams (updates + inserts),
env-selected relink effort per index -- then alternating query rounds in one window.
If the low-effort graph queries slower or recalls worse HERE, the cost is structural;
across separate processes it would be indistinguishable from thermal drift.

    python tools/bench/relink_quality.py
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from churn import live_truth, recall_at_k
from quick_cagra import POLICY_FORCE_CPU, bind_serialization, load_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import ovvs  # noqa: E402

UPD = 2_000
INS = 2_000
DEL = 2_000
CFG_A = (64, 2)
CFG_B = (32, 1)


def set_effort(cfg):
    os.environ["OVVS_CAGRA_RELINK_ITOPK"] = str(cfg[0])
    os.environ["OVVS_CAGRA_RELINK_WIDTH"] = str(cfg[1])


def open_index(res, path: Path, dim: int, n: int):
    handle = ctypes.c_void_p()
    rc = ovvs._lib.ovvsCagraDeserialize(res._h, os.fsencode(path), ctypes.byref(handle))
    if rc != 0:
        raise RuntimeError(f"deserialize failed: {rc}")
    return ovvs.CagraIndex(res, handle, dim, own_res=False,
                           destroy=ovvs._lib.ovvsCagraDestroy, n=n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=("100k", "1m"), default="100k")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    base, queries, _truth = load_fixture(args.scale, 500, Path("out/quick"))
    n, dim = base.shape
    ROUNDS = args.rounds

    res = ovvs.Resources()
    t0 = time.perf_counter()
    if args.scale == "1m":
        bind_serialization()
        fixture = Path("out/quick/sift1m-d32.ovvs")
        ix_a = open_index(res, fixture, dim, n)
        ix_b = open_index(res, fixture, dim, n)
    else:
        ix_a = ovvs.neighbors.cagra.build(base, graph_degree=32, intermediate_degree=64,
                                          resources=res)
        ix_b = ovvs.neighbors.cagra.build(base, graph_degree=32, intermediate_degree=64,
                                          resources=res)
    print(f"# indexes ready: {time.perf_counter() - t0:.1f}s", flush=True)
    res.set_policy(POLICY_FORCE_CPU)

    rng = np.random.default_rng(7)
    vectors = base.copy()
    total = n + ROUNDS * INS
    vectors = np.vstack([vectors, np.zeros((total - n, dim), dtype=np.float32)])
    rows = n
    alive = np.zeros(total, dtype=bool)
    alive[:n] = True
    # Logical row -> public id, per index. Deleted slots get recycled by extend_ex with
    # NEW public ids, so identity mapping breaks after round 1 (churn.py's pubid trick).
    pub_a = np.arange(total, dtype=np.int64)
    pub_b = np.arange(total, dtype=np.int64)

    for r in range(ROUNDS):
        live_ids = np.flatnonzero(alive[:rows])
        victims = rng.choice(live_ids, size=DEL, replace=False).astype(np.int64)
        alive[victims] = False
        live_ids = np.flatnonzero(alive[:rows])
        targets = rng.choice(live_ids, size=UPD, replace=False).astype(np.int64)
        upd_vecs = np.ascontiguousarray(
            base[rng.integers(0, n, size=UPD)] + rng.normal(0, 4, (UPD, dim)),
            dtype=np.float32)
        ins_vecs = np.ascontiguousarray(
            base[rng.integers(0, n, size=INS)] + rng.normal(0, 4, (INS, dim)),
            dtype=np.float32)
        vectors[targets] = upd_vecs
        vectors[rows : rows + INS] = ins_vecs
        alive[rows : rows + INS] = True
        set_effort(CFG_A)
        ix_a.delete(pub_a[victims])
        ix_a.update(pub_a[targets], upd_vecs)
        pub_a[rows : rows + INS] = ix_a.extend_ex(ins_vecs)
        set_effort(CFG_B)
        ix_b.delete(pub_b[victims])
        ix_b.update(pub_b[targets], upd_vecs)
        pub_b[rows : rows + INS] = ix_b.extend_ex(ins_vecs)
        rows += INS
    live_rows = np.flatnonzero(alive[:rows])
    print(f"# churn done: rows={rows} live={live_rows.size}", flush=True)

    if not np.array_equal(pub_a[live_rows], pub_b[live_rows]):
        raise RuntimeError("public id streams diverged between indexes")
    truth_pub = live_truth(vectors[:rows], live_rows, queries, 10)
    remap = np.full(total, -1, dtype=np.int64)
    remap[live_rows] = pub_a[live_rows]
    truth = remap[truth_pub]

    def qround(ix):
        t0 = time.perf_counter()
        ids, _ = ix.search(queries, k=10, itopk_size=64, search_width=4)
        el = time.perf_counter() - t0
        return queries.shape[0] / el, np.asarray(ids, dtype=np.int64).reshape(-1, 10)

    qa, qb = [], []
    ra = rb = None
    for _ in range(5):
        q, ids = qround(ix_a)
        qa.append(q)
        ra = recall_at_k(ids, truth)
        q, ids = qround(ix_b)
        qb.append(q)
        rb = recall_at_k(ids, truth)
    ma, mb = statistics.median(qa), statistics.median(qb)
    print(f"A {CFG_A}: {ma:.0f} qps recall {ra:.4f}  rounds={[round(v) for v in qa]}")
    print(f"B {CFG_B}: {mb:.0f} qps recall {rb:.4f}  rounds={[round(v) for v in qb]}")
    print(f"B/A qps: {mb / ma:.3f}  recall delta: {rb - ra:+.4f}")
    ix_a.close()
    ix_b.close()
    res.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
