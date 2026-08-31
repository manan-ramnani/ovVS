"""Package energy per query: ovVS vs the frozen hnswlib comparator.

ovvsResourcesEnergyUj reads package microjoules (EMI RAPL on this box). The iGPU is
on-package, so HETERO's GPU walk is charged fairly against hnswlib's CPU-only walk.
Interleaved timed windows of b32 queries; joules per 1000 queries, watts, and QPS per
engine; an idle window brackets the ambient draw (not subtracted -- reported).

    python tools/bench/energy.py
"""

from __future__ import annotations

import ctypes
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quick_cagra import POLICY_HETERO, bind_serialization, load_fixture
from matrix import DEFAULT_MODULE

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import ovvs  # noqa: E402

WINDOW_S = 5.0
ROUNDS = 3

_energy = ovvs._lib.ovvsResourcesEnergyUj
_energy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
_energy.restype = ctypes.c_int32


def read_uj():
    v = ctypes.c_int64()
    rc = _energy(None, ctypes.byref(v))
    if rc != 0:
        raise RuntimeError(f"energy counter unavailable: {rc}")
    return v.value


def window(fn):
    """Runs fn repeatedly for WINDOW_S; returns (ops, joules, seconds)."""
    e0, t0 = read_uj(), time.perf_counter()
    ops = 0
    while time.perf_counter() - t0 < WINDOW_S:
        ops += fn()
    t1, e1 = time.perf_counter(), read_uj()
    return ops, (e1 - e0) / 1e6, t1 - t0


def main() -> int:
    bind_serialization()
    sys.path.insert(0, str(DEFAULT_MODULE.resolve()))
    import hnswlib

    for scale, hnsw_bin, ef in (("100k", "out/matrix/hnsw-100k.bin", 96),
                                ("1m", "out/matrix/hnsw-1m.bin", 64)):
        base, queries, _ = load_fixture(scale, 1000, Path("out/quick"))
        n, dim = base.shape
        res = ovvs.Resources()
        handle = ctypes.c_void_p()
        fixture = f"out/quick/sift{scale}-d32-f16.ovvs"
        os.environ["OVVS_CAGRA_F16"] = "1"
        rc = ovvs._lib.ovvsCagraDeserialize(res._h, os.fsencode(fixture), ctypes.byref(handle))
        if rc != 0:
            raise RuntimeError(f"deserialize failed: {rc}")
        index = ovvs.CagraIndex(res, handle, dim, own_res=False,
                                destroy=ovvs._lib.ovvsCagraDestroy, n=n)
        res.set_policy(POLICY_HETERO)

        hnsw = hnswlib.Index(space="l2", dim=dim)
        hnsw.load_index(hnsw_bin, max_elements=n)
        hnsw.set_ef(ef)
        hnsw.set_num_threads(20)

        qn = queries.shape[0]
        state = {"off": 0}

        def ovvs_op():
            off = state["off"] = (state["off"] + 32) % (qn - 32)
            index.search(queries[off : off + 32], k=10, itopk_size=32, search_width=2)
            return 32

        def hnsw_op():
            off = state["off"] = (state["off"] + 32) % (qn - 32)
            hnsw.knn_query(queries[off : off + 32], k=10, num_threads=20)
            return 32

        # warm both
        for _ in range(20):
            ovvs_op()
            hnsw_op()
        _, idle_j, idle_s = window(lambda: (time.sleep(0.05), 0)[1] or 0)

        lanes = {"ovvs": [], "hnswlib": []}
        for _ in range(ROUNDS):
            lanes["ovvs"].append(window(ovvs_op))
            lanes["hnswlib"].append(window(hnsw_op))

        print(f"--- {scale} (idle window: {idle_j / idle_s:.1f} W)")
        stats = {}
        for name, rows in lanes.items():
            jper = statistics.median([j / ops * 1000 for ops, j, s in rows])
            watts = statistics.median([j / s for ops, j, s in rows])
            qps = statistics.median([ops / s for ops, j, s in rows])
            stats[name] = jper
            print(f"  {name:>8}: {qps:8.0f} qps  {watts:5.1f} W  {jper:7.2f} J/kq")
        print(f"  ovvs energy per query vs hnswlib: x{stats['ovvs'] / stats['hnswlib']:.2f}")
        index.close()
        res.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
