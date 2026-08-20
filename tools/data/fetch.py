#!/usr/bin/env python3
"""Download optional ANN datasets into ./data (not committed)."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data"
ROOT.mkdir(exist_ok=True)

SIFT_URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"


def main() -> int:
    print("ioVS dataset kit")
    print(f"target: {ROOT}")
    marker = ROOT / "README.txt"
    marker.write_text(
        "Optional datasets. Unit tests use generated tensors.\n"
        "If sift-128-euclidean.hdf5 is present, tools/bench uses its base/query slices.\n",
        encoding="utf-8",
    )
    dest = ROOT / "sift-128-euclidean.hdf5"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"already have {dest} bytes={dest.stat().st_size}")
        return 0
    print(f"fetch {SIFT_URL}")
    try:
        req = urllib.request.Request(SIFT_URL, headers={"User-Agent": "ioVS-fetch/0.2"})
        with urllib.request.urlopen(req, timeout=60) as r, dest.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print(f"wrote {dest} bytes={dest.stat().st_size}")
        return 0
    except Exception as e:
        print(f"SIFT1M download failed: {type(e).__name__}: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        print("bench will use a documented generated stand-in.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
