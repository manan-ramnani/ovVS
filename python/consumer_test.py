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
    data = array.array("f", [((i * 17) % 100) / 50.0 - 1.0 for i in range(n * dim)])
    q = array.array("f", [0.1, -0.2, 0.3, 0.0])
    res = iovs.Resources()
    idx = iovs.neighbors.brute_force.build(data, dim=dim, resources=res)
    nb, ds = idx.search(q, k=k)
    # oracle
    dists = []
    for i in range(n):
        row = data[i * dim : (i + 1) * dim]
        dists.append((l2sq(q, row, dim), i))
    dists.sort()
    truth = [i for _, i in dists[:k]]
    if list(nb[:k]) != truth:
        print("mismatch", nb[:k], truth, file=sys.stderr)
        return 1
    print("python consumer ok neighbors", list(nb[:k]), "version", iovs.version())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
