"""Query tail latency under mixed read/write load, 1M.

Neither engine supports concurrent read-during-write, so "mixed" is an interleaved
single-threaded stream: rounds of a write burst (update 500 + insert 500) followed by
300 query ops (5/6 b32, 1/6 b1), identical schedules for both engines, alternating
per round in one window. A read-only phase first gives the reference tail. Queries
right after a write burst are tagged separately -- that is where a stall would show
(mirror rebuilds, cache effects, GPU state transitions).

ovVS runs fp16 primary storage under HETERO (b32 -> GPU at 1M, b1 -> CPU, mutation
GPU-assisted); hnswlib runs the frozen comparator at matched-recall ef.

    python tools/bench/mixed_load.py
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

ROUNDS = 12
UPD = 500
INS = 500
QUERY_OPS = 300
B1_EVERY = 6          # every 6th query op is a single
POST_WRITE_TAG = 3    # first N query ops after a write burst
EF = 64
THREADS = 20


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def summary(ms):
    return (f"p50 {pct(ms, 0.50):7.3f}  p95 {pct(ms, 0.95):7.3f}  "
            f"p99 {pct(ms, 0.99):7.3f}  max {max(ms):7.3f}")


def main() -> int:
    os.environ["OVVS_CAGRA_F16"] = "1"
    bind_serialization()
    sys.path.insert(0, str(DEFAULT_MODULE.resolve()))
    import hnswlib

    base, queries, _ = load_fixture("1m", 10_000, Path("out/quick"))
    n, dim = base.shape

    res = ovvs.Resources()
    handle = ctypes.c_void_p()
    rc = ovvs._lib.ovvsCagraDeserialize(
        res._h, os.fsencode("out/quick/sift1m-d32-f16.ovvs"), ctypes.byref(handle))
    if rc != 0:
        raise RuntimeError(f"deserialize failed: {rc}")
    index = ovvs.CagraIndex(res, handle, dim, own_res=False,
                            destroy=ovvs._lib.ovvsCagraDestroy, n=n)
    res.set_policy(POLICY_HETERO)

    hnsw = hnswlib.Index(space="l2", dim=dim)
    t0 = time.perf_counter()
    hnsw.load_index("out/matrix/hnsw-1m.bin", max_elements=n + ROUNDS * INS)
    hnsw.set_ef(EF)
    hnsw.set_num_threads(THREADS)
    print(f"# engines ready (hnsw load {time.perf_counter() - t0:.1f}s)", flush=True)

    rng = np.random.default_rng(7)
    # shared schedule: per round, query slices and mutation payloads
    sched = []
    for r in range(ROUNDS):
        targets = rng.integers(0, n, size=UPD).astype(np.int64)
        upd_vecs = np.ascontiguousarray(
            base[rng.integers(0, n, size=UPD)] + rng.normal(0, 4, (UPD, dim)),
            dtype=np.float32)
        ins_vecs = np.ascontiguousarray(
            base[rng.integers(0, n, size=INS)] + rng.normal(0, 4, (INS, dim)),
            dtype=np.float32)
        qops = []
        for i in range(QUERY_OPS):
            if i % B1_EVERY == 0:
                qops.append(rng.integers(0, queries.shape[0] - 1))          # b1 offset
            else:
                qops.append(-1 - rng.integers(0, queries.shape[0] - 32))    # b32 marker
        sched.append((targets, upd_vecs, ins_vecs, qops))
    warm_qops = sched[0][3]

    def make_lane(name, query_b1, query_b32, do_update, do_insert):
        lane = {"name": name, "ro_b1": [], "ro_b32": [], "mx_b1": [], "mx_b32": [],
                "post_b1": [], "post_b32": [], "upd_ms": [], "ins_ms": []}

        def qop(code, sink_b1, sink_b32, post=0, post_b1=None, post_b32=None):
            t0 = time.perf_counter()
            if code >= 0:
                query_b1(code)
                ms = 1000.0 * (time.perf_counter() - t0)
                (post_b1 if post else sink_b1).append(ms)
            else:
                query_b32(-1 - code)
                ms = 1000.0 * (time.perf_counter() - t0)
                (post_b32 if post else sink_b32).append(ms)

        lane["qop"] = qop
        lane["update"] = do_update
        lane["insert"] = do_insert
        return lane

    def ovvs_b1(off):
        index.search(queries[off : off + 1], k=10, itopk_size=32, search_width=2)

    def ovvs_b32(off):
        index.search(queries[off : off + 32], k=10, itopk_size=32, search_width=2)

    def hnsw_b1(off):
        hnsw.knn_query(queries[off : off + 1], k=10, num_threads=THREADS)

    def hnsw_b32(off):
        hnsw.knn_query(queries[off : off + 32], k=10, num_threads=THREADS)

    ov = make_lane("ovvs", ovvs_b1, ovvs_b32,
                   lambda t, v: index.update(t, v),
                   lambda v: index.extend_ex(v))
    hn = make_lane("hnswlib", hnsw_b1, hnsw_b32,
                   lambda t, v: hnsw.add_items(v, t, num_threads=THREADS),
                   None)
    hn_next = [n]

    def hn_insert(v):
        labels = np.arange(hn_next[0], hn_next[0] + v.shape[0])
        hn_next[0] += v.shape[0]
        hnsw.add_items(v, labels, num_threads=THREADS)

    hn["insert"] = hn_insert

    # Drop mirror8 up front with one noisy update: the mixed phase's first write would
    # drop it anyway, and the read-only reference must run on the same (DS16) path.
    index.update(np.array([0], dtype=np.int64),
                 np.ascontiguousarray(base[:1] + 0.5, dtype=np.float32))

    # warm-up + read-only reference phase (interleaved engines)
    for lane in (ov, hn):
        for code in warm_qops[:60]:
            lane["qop"](code, [], [])
    for lane in (ov, hn):
        for code in warm_qops:
            lane["qop"](code, lane["ro_b1"], lane["ro_b32"])
    print("# read-only phase done", flush=True)

    for r in range(ROUNDS):
        targets, upd_vecs, ins_vecs, qops = sched[r]
        order = [ov, hn] if r % 2 == 0 else [hn, ov]
        for lane in order:
            t0 = time.perf_counter()
            lane["update"](targets, upd_vecs)
            lane["upd_ms"].append(1000.0 * (time.perf_counter() - t0))
            t0 = time.perf_counter()
            lane["insert"](ins_vecs)
            lane["ins_ms"].append(1000.0 * (time.perf_counter() - t0))
            seen = 0
            for code in qops:
                post = 1 if seen < POST_WRITE_TAG else 0
                lane["qop"](code, lane["mx_b1"], lane["mx_b32"], post,
                            lane["post_b1"], lane["post_b32"])
                seen += 1
    print("# mixed phase done", flush=True)

    for lane in (ov, hn):
        nm = lane["name"]
        print(f"\n{nm}:")
        print(f"  b32 read-only : {summary(lane['ro_b32'])}")
        print(f"  b32 mixed     : {summary(lane['mx_b32'])}")
        print(f"  b32 post-write: {summary(lane['post_b32'])}")
        print(f"  b1  read-only : {summary(lane['ro_b1'])}")
        print(f"  b1  mixed     : {summary(lane['mx_b1'])}")
        print(f"  b1  post-write: {summary(lane['post_b1'])}")
        print(f"  update 500    : {summary(lane['upd_ms'])}")
        print(f"  insert 500    : {summary(lane['ins_ms'])}")
    index.close()
    res.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
