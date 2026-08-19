#!/usr/bin/env python3
"""Download optional ANN datasets into ./data (not committed)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data"
ROOT.mkdir(exist_ok=True)


def main() -> None:
    print("ioVS dataset kit")
    print(f"target: {ROOT}")
    print("Tests use generated tensors and do not require this download.")
    print("Optional: place sift-128-euclidean.hdf5 here for bench.")
    marker = ROOT / "README.txt"
    marker.write_text("Optional datasets. Not required for unit tests.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
