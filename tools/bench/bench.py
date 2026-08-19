#!/usr/bin/env python3
"""Compare ioVS brute-force against FAISS/hnswlib when those packages exist."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import ctypes
from ctypes import POINTER, c_float, c_int32, c_int64, c_void_p

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def load_iovs():
    os.environ.setdefault("IOVS_LIBRARY", str(ROOT / "build" / "bin" / "iovs.dll"))
    if not Path(os.environ["IOVS_LIBRARY"]).exists():
        for p in (ROOT / "build" / "bin").glob("*.dll"):
            if p.name.lower().startswith("iovs"):
                os.environ["IOVS_LIBRARY"] = str(p)
    import iovs as m

    return m


def main() -> int:
    try:
        import numpy as np
    except ImportError:
        print("numpy missing; bench skipped")
        return 0

    rng = np.random.default_rng(0)
    n, dim, nq, k = 200, 16, 8, 5
    data = rng.standard_normal((n, dim), dtype=np.float32)
    queries = rng.standard_normal((nq, dim), dtype=np.float32)

    lib = load_iovs()
    res = lib.Resources()
    idx = lib.neighbors.brute_force.build(data, dim=dim, resources=res)
    t0 = time.perf_counter()
    nb, ds = idx.search(queries, k=k)
    t1 = time.perf_counter()
    print(f"iovs brute ms={(t1 - t0) * 1000:.3f} first={nb[:k]}")

    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatL2(dim)
        index.add(data)
        t0 = time.perf_counter()
        _, I = index.search(queries, k)
        t1 = time.perf_counter()
        print(f"faiss brute ms={(t1 - t0) * 1000:.3f} first={list(I[0])}")
    except Exception as e:
        print(f"faiss unavailable: {e}")

    try:
        import hnswlib  # type: ignore

        p = hnswlib.Index(space="l2", dim=dim)
        p.init_index(max_elements=n, ef_construction=50, M=8)
        p.add_items(data)
        t0 = time.perf_counter()
        labels, _ = p.knn_query(queries, k=k)
        t1 = time.perf_counter()
        print(f"hnswlib ms={(t1 - t0) * 1000:.3f} first={list(labels[0])}")
    except Exception as e:
        print(f"hnswlib unavailable: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
