"""G2/G4 harness: does ovVS behave like a living store?

G2 asks whether query / insert / update / delete are all practical -- no operation may be a
stall. G4 asks whether recall survives accumulated mutation.

The measurement that matters here is recall against the **live** set, recomputed after every
round. Comparing against the original ground truth would be meaningless once rows have been
deleted or overwritten: the right answer changes as the store changes, and an index that
correctly stops returning a deleted row would look like it had lost recall.

    python tools/bench/churn.py --scale 100k --rounds 5 --delete 2000 --insert 2000 --update 2000

Deleted rows are excluded from the live set; updated rows keep their id but get a new vector,
so truth has to be rebuilt from the live vectors each round either way.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quick_cagra import POLICY_FORCE_CPU, POLICY_FORCE_GPU, SCALES, bind_serialization, load_fixture

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import ovvs  # noqa: E402


def live_truth(vectors, live_rows, queries, k):
    """Exact top-k over just the live rows, returned as original row ids."""
    live = vectors[live_rows]
    truth = np.empty((queries.shape[0], k), dtype=np.int64)
    step = 256
    for start in range(0, queries.shape[0], step):
        chunk = queries[start : start + step]
        d = (
            np.einsum("ij,ij->i", chunk, chunk)[:, None]
            - 2.0 * chunk @ live.T
            + np.einsum("ij,ij->i", live, live)[None, :]
        )
        idx = np.argpartition(d, k, axis=1)[:, :k]
        rows = np.take_along_axis(d, idx, axis=1).argsort(axis=1)
        truth[start : start + chunk.shape[0]] = live_rows[np.take_along_axis(idx, rows, axis=1)]
    return truth


def recall_at_k(got, truth):
    hits = 0
    for row, want in zip(got, truth):
        hits += len(set(int(v) for v in row if v >= 0) & set(int(v) for v in want))
    return hits / (truth.shape[0] * truth.shape[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=tuple(SCALES), default="100k")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--delete", type=int, default=2000)
    ap.add_argument("--insert", type=int, default=2000)
    ap.add_argument("--update", type=int, default=2000)
    ap.add_argument("--itopk", type=int, default=64)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--search-policy", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--reuse", action="store_true",
                    help="insert via extend_ex, which recycles rows freed by delete")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    bind_serialization()
    base, queries, _ = load_fixture(args.scale, args.queries, Path("out/quick"))
    queries = np.ascontiguousarray(queries[: args.queries], dtype=np.float32)
    n0, dim = base.shape

    resource = ovvs.Resources()
    resource.set_policy(POLICY_FORCE_CPU)
    started = time.perf_counter()
    index = ovvs.neighbors.cagra.build(base, graph_degree=16, intermediate_degree=32, resources=resource)
    build_seconds = time.perf_counter() - started
    search_policy = POLICY_FORCE_GPU if args.search_policy == "gpu" else POLICY_FORCE_CPU

    # Mutation has no GPU path and is refused outright under FORCE_GPU, so the policy is flipped
    # around every mutating call. A caller that pins FORCE_GPU for search cannot mutate at all --
    # worth knowing, and a G2 wrinkle in its own right.
    def mutate(fn, *fn_args):
        resource.set_policy(POLICY_FORCE_CPU)
        t0 = time.perf_counter()
        fn(*fn_args)
        elapsed = time.perf_counter() - t0
        resource.set_policy(search_policy)
        return elapsed

    resource.set_policy(search_policy)

    # Mirror of what the store should contain, so truth can be recomputed independently of ovVS.
    vectors = base.copy()
    alive = np.ones(n0, dtype=bool)
    # Public id per slot. Only diverges from the slot number once a row has been reused, since a
    # reused row's generation is packed into the high half of the id.
    pubid = np.arange(n0, dtype=np.int64)

    def query_round():
        t0 = time.perf_counter()
        ids, _ = index.search(
            queries, k=args.k, itopk_size=args.itopk, search_width=args.width
        )
        elapsed = time.perf_counter() - t0
        raw = np.asarray(ids, dtype=np.int64).reshape(args.queries, args.k)
        # Search returns public ids; the mirror is indexed by slot, so strip the generation.
        got = np.where(raw >= 0, raw & 0xFFFFFFFF, -1)
        live_rows = np.flatnonzero(alive)
        truth = live_truth(vectors, live_rows, queries, args.k)
        returned = float((got >= 0).sum()) / (args.queries * args.k)
        return recall_at_k(got, truth), args.queries / elapsed, returned

    rows = []
    recall, qps, returned = query_round()
    live, deleted = index.counts()
    rows.append({"round": 0, "op": "baseline", "recall": round(recall, 4),
                 "query_qps": round(qps, 1), "returned_frac": round(returned, 4),
                 "live": live, "deleted": deleted})
    print(json.dumps(rows[-1]), flush=True)

    for r in range(1, args.rounds + 1):
        timings = {}

        if args.delete:
            victims = rng.choice(np.flatnonzero(alive), size=min(args.delete, int(alive.sum())),
                                 replace=False)
            elapsed = mutate(index.delete, pubid[victims])
            timings["delete_ms"] = round(elapsed * 1000.0, 3)
            timings["delete_per_op_us"] = round(elapsed * 1e6 / max(1, len(victims)), 2)
            alive[victims] = False

        if args.update:
            targets = rng.choice(np.flatnonzero(alive), size=min(args.update, int(alive.sum())),
                                 replace=False)
            fresh = np.ascontiguousarray(
                base[rng.integers(0, n0, size=len(targets))] + rng.normal(0, 4, (len(targets), dim)),
                dtype=np.float32)
            elapsed = mutate(index.update, pubid[targets], fresh)
            timings["update_ms"] = round(elapsed * 1000.0, 3)
            timings["update_per_op_us"] = round(elapsed * 1e6 / max(1, len(targets)), 2)
            vectors[targets] = fresh

        if args.insert:
            extra = np.ascontiguousarray(
                base[rng.integers(0, n0, size=args.insert)] + rng.normal(0, 4, (args.insert, dim)),
                dtype=np.float32)
            if args.reuse:
                assigned = []

                def do_insert():
                    assigned.extend(index.extend_ex(extra))

                elapsed = mutate(do_insert)
                for j, new_id in enumerate(assigned):
                    slot = int(new_id) & 0xFFFFFFFF
                    if slot < vectors.shape[0]:
                        vectors[slot] = extra[j]
                        alive[slot] = True
                        pubid[slot] = int(new_id)
                    else:
                        vectors = np.vstack([vectors, extra[j : j + 1]])
                        alive = np.concatenate([alive, [True]])
                        pubid = np.concatenate([pubid, [int(new_id)]])
            else:
                elapsed = mutate(index.extend, extra)
                vectors = np.vstack([vectors, extra])
                alive = np.concatenate([alive, np.ones(args.insert, dtype=bool)])
                pubid = np.concatenate(
                    [pubid, np.arange(vectors.shape[0] - args.insert, vectors.shape[0], dtype=np.int64)])
            timings["insert_ms"] = round(elapsed * 1000.0, 3)
            timings["insert_per_op_us"] = round(elapsed * 1e6 / max(1, args.insert), 2)

        recall, qps, returned = query_round()
        live, deleted = index.counts()
        row = {"round": r, "recall": round(recall, 4), "query_qps": round(qps, 1),
               "returned_frac": round(returned, 4), "live": live, "deleted": deleted,
               "rows_total": int(alive.size), "slots": live + deleted, **timings}
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {"scale": args.scale, "n0": int(n0), "dim": int(dim),
              "build_seconds": round(build_seconds, 3),
              "itopk": args.itopk, "width": args.width, "k": args.k, "rounds": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
