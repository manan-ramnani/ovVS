"""B20 at scale: fp16 ovVS vs frozen hnswlib on a non-integer 1M corpus.

SIFT1M scaled by 1/3: every value inexact in fp16, no int8 mirror possible, ranking
unchanged so the HDF5 truth stays valid. ovVS runs fp16 primary storage (the GPU walks
the DS16 path); hnswlib stores fp32 as always, frozen settings, ef matched to recall.
Alternating timed rounds in one process, ab_g1 discipline.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quick_cagra import POLICY_HETERO, load_fixture, recall_at_10
from matrix import DEFAULT_MODULE, rss

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import ovvs  # noqa: E402


def main() -> int:
    os.environ["OVVS_CAGRA_F16"] = "1"
    os.environ["OVVS_CAGRA_PATIENCE"] = "0"
    sys.path.insert(0, str(DEFAULT_MODULE.resolve()))
    import hnswlib

    base, queries, truth = load_fixture("1m", 1000, Path("out/quick"))
    base = np.ascontiguousarray(base / 3.0, dtype=np.float32)
    queries = np.ascontiguousarray(queries / 3.0, dtype=np.float32)
    n, dim = base.shape

    res = ovvs.Resources()
    t0 = time.perf_counter()
    index = ovvs.neighbors.cagra.build(base, graph_degree=32, intermediate_degree=64,
                                       resources=res)
    print(f"# ovvs f16 build: {time.perf_counter() - t0:.1f}s rss={rss()['working_set']/1e6:.0f}MB",
          flush=True)
    res.set_policy(POLICY_HETERO)

    hnsw = hnswlib.Index(space="l2", dim=dim)
    hnsw.init_index(max_elements=n, ef_construction=200, M=16, random_seed=7)
    hnsw.set_num_threads(20)
    t0 = time.perf_counter()
    hnsw.add_items(base, np.arange(n), num_threads=20)
    print(f"# hnsw build: {time.perf_counter() - t0:.1f}s", flush=True)

    def ovvs_round():
        t0 = time.perf_counter()
        ids, _ = index.search(queries, k=10, itopk_size=32, search_width=2)
        el = time.perf_counter() - t0
        return 1000.0 / el, np.asarray(ids, dtype=np.int64).reshape(1000, 10)

    def hnsw_round():
        t0 = time.perf_counter()
        labels, _ = hnsw.knn_query(queries, k=10, num_threads=20)
        el = time.perf_counter() - t0
        return 1000.0 / el, np.asarray(labels, dtype=np.int64)

    # recall probe + ef match
    _, ids = ovvs_round()
    r_ovvs = recall_at_10(ids, truth)
    picked = None
    for ef in (48, 64, 80, 96, 128):
        hnsw.set_ef(ef)
        _, got = hnsw_round()
        r = recall_at_10(got, truth)
        if picked is None or abs(r - r_ovvs) < abs(picked[1] - r_ovvs):
            picked = (ef, r)
        if r >= r_ovvs:
            break
    hnsw.set_ef(picked[0])
    print(f"# ovvs recall={r_ovvs:.4f}  hnsw ef={picked[0]} recall={picked[1]:.4f}", flush=True)

    o_qps, h_qps = [], []
    for _ in range(5):
        q, _ = ovvs_round()
        o_qps.append(q)
        q, _ = hnsw_round()
        h_qps.append(q)
    om, hm = statistics.median(o_qps), statistics.median(h_qps)
    print(f"ovvs f16 hetero: {om:.0f} qps  rounds={[round(v) for v in o_qps]}")
    print(f"hnswlib fp32:    {hm:.0f} qps  rounds={[round(v) for v in h_qps]}")
    print(f"ovvs ahead: x{om / hm:.2f} at recall {r_ovvs:.4f} vs {picked[1]:.4f}")
    index.close()
    res.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
