#!/usr/bin/env python3
"""Public Python/C ABI consumer: brute-force search vs an oracle computed here."""

from __future__ import annotations

import array
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
dll = ROOT / "build" / "bin" / "iovs.dll"
os.environ["IOVS_LIBRARY"] = str(dll)
sys.path.insert(0, str(ROOT / "python"))

import iovs  # noqa: E402


def l2sq(a, b, dim):
    s = 0.0
    for i in range(dim):
        t = a[i] - b[i]
        s += t * t
    return s


def main() -> int:
    n, dim, k = 12, 4, 3
    try:
        import numpy as np

        data = np.array([((i * 17) % 100) / 50.0 - 1.0 for i in range(n * dim)], dtype=np.float32).reshape(
            n, dim
        )
        q = np.array([[0.1, -0.2, 0.3, 0.0]], dtype=np.float32)
        use_np = True
    except ImportError:
        data = array.array("f", [((i * 17) % 100) / 50.0 - 1.0 for i in range(n * dim)])
        q = array.array("f", [0.1, -0.2, 0.3, 0.0])
        use_np = False
    res = iovs.Resources()
    idx = iovs.neighbors.brute_force.build(data, dim=dim, resources=res)
    nb, ds = idx.search(q, k=k)
    # oracle
    dists = []
    qrow = q[0] if use_np else q
    for i in range(n):
        row = data[i] if use_np else data[i * dim : (i + 1) * dim]
        dists.append((l2sq(qrow, row, dim), i))
    dists.sort()
    truth = [i for _, i in dists[:k]]
    got = list(nb.ravel()[:k]) if hasattr(nb, "ravel") else list(nb[:k])
    if got != truth:
        print("mismatch", got, truth, file=sys.stderr)
        return 1
    print("python consumer ok neighbors", got, "version", iovs.version(), "numpy", use_np)
    if use_np:
        # DLPack ingest: build+search from __dlpack__ capsules, same independent L2 oracle.
        d_dl = np.from_dlpack(data)
        q_dl = np.from_dlpack(q)
        idx2 = iovs.neighbors.brute_force.build(d_dl, dim=dim, resources=res)
        nb2, _ = idx2.search(q_dl, k=k)
        got2 = list(nb2.ravel()[:k])
        if got2 != truth:
            print("dlpack mismatch", got2, truth, file=sys.stderr)
            return 1
        print("python consumer dlpack ok neighbors", got2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
