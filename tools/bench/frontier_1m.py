"""Interleaved QPS-vs-recall frontier at 1M: is the ovVS lead bigger at lower recall?

Times an effort ladder on both engines in one process, alternating point-by-point.
ovVS runs HETERO (which at 1M routes bulk to the GPU whole); hnswlib is the frozen
comparator with only ef moving. hnswlib loads from the index the matrix run saved.
"""

from __future__ import annotations

import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quick_cagra import POLICY_HETERO, bind_serialization, load_fixture, recall_at_10
from matrix import DEFAULT_MODULE, open_ovvs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import ovvs  # noqa: E402

OVVS_POINTS = [(16, 1), (16, 2), (24, 2), (32, 2), (48, 2), (64, 2)]
EF_POINTS = [24, 32, 48, 64, 80, 96, 128]
ROUNDS = 3
BATCH = 1000


def main() -> int:
    os.environ["OVVS_CAGRA_PATIENCE"] = "0"
    sys.path.insert(0, str(DEFAULT_MODULE.resolve()))
    import hnswlib

    bind_serialization()
    base, queries, truth = load_fixture("1m", 1000, Path("out/quick"))
    n, dim = base.shape
    resource, index = open_ovvs(Path("out/quick/sift1m-d32.ovvs"), dim, n)
    resource.set_policy(POLICY_HETERO)

    hnsw = hnswlib.Index(space="l2", dim=dim)
    hnsw.load_index("out/matrix/hnsw-1m.bin", max_elements=n)
    hnsw.set_num_threads(20)

    def ovvs_point(itopk, width):
        ids, _ = index.search(queries, k=10, itopk_size=itopk, search_width=width)
        qps, ids_last = [], None
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            ids_last, _ = index.search(queries, k=10, itopk_size=itopk, search_width=width)
            qps.append(1000.0 / (time.perf_counter() - t0))
        got = np.asarray(ids_last, dtype=np.int64).reshape(1000, 10)
        return recall_at_10(got, truth), statistics.median(qps)

    def hnsw_point(ef):
        hnsw.set_ef(ef)
        hnsw.knn_query(queries, k=10, num_threads=20)
        qps, ids_last = [], None
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            ids_last, _ = hnsw.knn_query(queries, k=10, num_threads=20)
            qps.append(1000.0 / (time.perf_counter() - t0))
        return recall_at_10(np.asarray(ids_last, dtype=np.int64), truth), statistics.median(qps)

    out = {"ovvs": [], "hnswlib": []}
    for i in range(max(len(OVVS_POINTS), len(EF_POINTS))):
        if i < len(OVVS_POINTS):
            itopk, width = OVVS_POINTS[i]
            r, q = ovvs_point(itopk, width)
            out["ovvs"].append({"itopk": itopk, "width": width, "recall": round(r, 4), "qps": round(q, 1)})
            print(f"ovvs {itopk}/{width}: recall={r:.4f} qps={q:.0f}", flush=True)
        if i < len(EF_POINTS):
            r, q = hnsw_point(EF_POINTS[i])
            out["hnswlib"].append({"ef": EF_POINTS[i], "recall": round(r, 4), "qps": round(q, 1)})
            print(f"hnsw ef={EF_POINTS[i]}: recall={r:.4f} qps={q:.0f}", flush=True)

    Path("out/matrix/frontier-1m.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    index.close()
    resource.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
