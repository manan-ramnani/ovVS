"""B20 preview: fp16 primary storage on a genuinely non-integer corpus.

Scales SIFT by 1/3 -- every value becomes inexact in fp16, but scaling preserves
neighbor order, so the cached exact truth still judges recall. Builds the same d32
index twice (fp32 mode, then OVVS_CAGRA_F16=1), same seed, and compares recall and
CPU walk throughput. The int8 mirror rejects a scaled corpus, so the f16 lane runs
the pure fp16 distance path -- exactly what a real embedding corpus would exercise.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quick_cagra import POLICY_FORCE_CPU, load_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import ovvs  # noqa: E402


def run_mode(base, queries, truth, label):
    from quick_cagra import recall_at_10

    res = ovvs.Resources()  # AUTO: GPU-assisted build
    t0 = time.perf_counter()
    index = ovvs.neighbors.cagra.build(base, graph_degree=32, intermediate_degree=64,
                                       resources=res)
    build_s = time.perf_counter() - t0
    res.set_policy(POLICY_FORCE_CPU)
    ids, _ = index.search(queries, k=10, itopk_size=32, search_width=2)  # warmup
    qps = []
    for _ in range(5):
        t0 = time.perf_counter()
        ids, _ = index.search(queries, k=10, itopk_size=32, search_width=2)
        qps.append(queries.shape[0] / (time.perf_counter() - t0))
    got = np.asarray(ids, dtype=np.int64).reshape(queries.shape[0], 10)
    r = recall_at_10(got, truth)
    print(f"{label}: recall={r:.4f} qps_median={statistics.median(qps):.0f} "
          f"build={build_s:.1f}s", flush=True)
    index.close()
    res.close()
    return r


def main() -> int:
    base, queries, truth = load_fixture("100k", 1000, Path("out/quick"))
    base = np.ascontiguousarray(base / 3.0, dtype=np.float32)
    queries = np.ascontiguousarray(queries / 3.0, dtype=np.float32)

    os.environ.pop("OVVS_CAGRA_F16", None)
    r32 = run_mode(base, queries, truth, "fp32 storage")
    os.environ["OVVS_CAGRA_F16"] = "1"
    r16 = run_mode(base, queries, truth, "fp16 storage")
    print(f"recall delta (fp16 - fp32): {r16 - r32:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
