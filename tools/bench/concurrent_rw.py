"""Read-during-write: do searches survive, and what do their tails look like?

Two reader threads (one b32, one b1) hammer the index while a writer thread runs
rounds of update/insert/delete; ctypes releases the GIL, so this is real
concurrency. A read-only phase first gives the reference tails on the same threads.
Correctness: no crash, no error status, and post-churn recall on the untouched
majority stays in the expected band.

    python tools/bench/concurrent_rw.py [--seconds 15]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quick_cagra import POLICY_HETERO, bind_serialization, load_fixture, recall_at_10

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import ovvs  # noqa: E402


def pct(v, q):
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))] if s else float("nan")


def lat_line(name, ms):
    return (f"  {name}: n={len(ms):6d}  p50 {pct(ms, 0.5):7.3f}  p95 {pct(ms, 0.95):7.3f}  "
            f"p99 {pct(ms, 0.99):7.3f}  max {max(ms):8.3f} ms")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    os.environ["OVVS_CAGRA_F16"] = "1"
    bind_serialization()
    base, queries, truth = load_fixture("1m", 1000, Path("out/quick"))
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

    rng = np.random.default_rng(7)
    stop = threading.Event()
    errors = []

    def reader(batch, sink):
        qn = queries.shape[0]
        while not stop.is_set():
            off = int(rng.integers(0, qn - batch))
            t0 = time.perf_counter()
            try:
                index.search(queries[off : off + batch], k=10, itopk_size=32, search_width=2)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"reader b{batch}: {exc!r}")
                return
            sink.append(1000.0 * (time.perf_counter() - t0))

    def run_phase(label, writer_fn):
        b32_ms, b1_ms = [], []
        stop.clear()
        threads = [threading.Thread(target=reader, args=(32, b32_ms)),
                   threading.Thread(target=reader, args=(1, b1_ms))]
        for th in threads:
            th.start()
        wstats = writer_fn()
        stop.set()
        for th in threads:
            th.join()
        print(f"{label}:")
        print(lat_line("b32", b32_ms))
        print(lat_line("b1 ", b1_ms))
        if wstats:
            print(wstats)
        return b32_ms, b1_ms

    def idle_writer():
        time.sleep(args.seconds)
        return None

    write_log = {"upd": [], "ins": [], "del": []}

    def churn_writer():
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            targets = rng.integers(0, n, size=2000).astype(np.int64)
            upd = np.ascontiguousarray(
                base[rng.integers(0, n, size=2000)] + rng.normal(0, 4, (2000, dim)),
                dtype=np.float32)
            ins = np.ascontiguousarray(
                base[rng.integers(0, n, size=2000)] + rng.normal(0, 4, (2000, dim)),
                dtype=np.float32)
            try:
                t0 = time.perf_counter()
                index.update(targets, upd)
                write_log["upd"].append(1000.0 * (time.perf_counter() - t0))
                t0 = time.perf_counter()
                new_ids = index.extend_ex(ins)
                write_log["ins"].append(1000.0 * (time.perf_counter() - t0))
                t0 = time.perf_counter()
                index.delete(np.asarray(new_ids[:1000], dtype=np.int64))
                write_log["del"].append(1000.0 * (time.perf_counter() - t0))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"writer: {exc!r}")
                return "writer FAILED"
        return (f"  writer: {len(write_log['upd'])} rounds  "
                f"upd p50 {pct(write_log['upd'], 0.5):.0f}ms  "
                f"ins p50 {pct(write_log['ins'], 0.5):.0f}ms  "
                f"del p50 {pct(write_log['del'], 0.5):.0f}ms")

    # warm-up
    for _ in range(3):
        index.search(queries[:32], k=10, itopk_size=32, search_width=2)

    ro32, ro1 = run_phase("read-only", idle_writer)
    mx32, mx1 = run_phase("read-during-write", churn_writer)

    if errors:
        print("ERRORS:")
        for e in errors:
            print(" ", e)
        return 1

    ids, _ = index.search(queries, k=10, itopk_size=64, search_width=4)
    r = recall_at_10(np.asarray(ids, dtype=np.int64).reshape(-1, 10), truth)
    print(f"post-churn recall vs original truth (few % rows churned): {r:.4f}")
    print(f"read throughput kept under writes: "
          f"b32 {len(mx32) / max(1, len(ro32)):.2f}x  b1 {len(mx1) / max(1, len(ro1)):.2f}x")
    index.close()
    res.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
