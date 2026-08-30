"""Insert-effort sweep: how much search does mutation repair actually need?

The relink search behind insert/update runs at a fixed effort (itopk = 2*degree,
width = 2). This ladder measures update throughput at 100K across effort configs,
round-robin interleaved so thermal drift and graph churn hit every config equally.
Recall-under-churn for the winner is judged separately by churn.py (live truth).

    python tools/bench/relink_sweep.py
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

CONFIGS = [(64, 2), (48, 2), (32, 2), (64, 1), (48, 1), (32, 1)]
PASSES = 3
COUNT = 5_000


def main() -> int:
    base, _queries, _truth = load_fixture("100k", 1000, Path("out/quick"))
    n, dim = base.shape

    res = ovvs.Resources()
    t0 = time.perf_counter()
    index = ovvs.neighbors.cagra.build(base, graph_degree=32, intermediate_degree=64,
                                       resources=res)
    print(f"# build: {time.perf_counter() - t0:.1f}s", flush=True)
    res.set_policy(POLICY_FORCE_CPU)

    rng = np.random.default_rng(7)
    pool = rng.permutation(n)
    results = {c: [] for c in CONFIGS}
    visit = 0
    for p in range(PASSES):
        for cfg in CONFIGS:
            os.environ["OVVS_CAGRA_RELINK_ITOPK"] = str(cfg[0])
            os.environ["OVVS_CAGRA_RELINK_WIDTH"] = str(cfg[1])
            lo = (visit * COUNT) % (n - COUNT)
            targets = pool[lo : lo + COUNT].astype(np.int64)
            vecs = np.ascontiguousarray(
                base[rng.integers(0, n, size=COUNT)] + rng.normal(0, 4, (COUNT, dim)),
                dtype=np.float32)
            t0 = time.perf_counter()
            index.update(targets, vecs)
            el = time.perf_counter() - t0
            results[cfg].append(COUNT / el)
            visit += 1
        print(f"# pass {p + 1}/{PASSES} done", flush=True)

    base_med = statistics.median(results[CONFIGS[0]])
    print(f"\n{'itopk':>6} {'width':>6} {'ops/s':>9} {'vs (64,2)':>10}  rounds")
    for cfg in CONFIGS:
        med = statistics.median(results[cfg])
        print(f"{cfg[0]:>6} {cfg[1]:>6} {med:>9.0f} {med / base_med:>9.2f}x  "
              f"{[round(v) for v in results[cfg]]}")
    index.close()
    res.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
