#!/usr/bin/env python3
"""Crash-isolated stock-hnswlib smoke for an ovVS CAGRA export."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def _child(path: Path) -> int:
    import hnswlib
    import numpy as np
    import ovvs

    rng = np.random.default_rng(470)
    base = rng.standard_normal((256, 16), dtype=np.float32)
    queries = np.ascontiguousarray(base[:16])

    resources = ovvs.Resources()
    cagra = ovvs.neighbors.cagra.build(
        base,
        graph_degree=16,
        intermediate_degree=32,
        resources=resources,
    )
    exported = ovvs.neighbors.hnsw.from_cagra(cagra, resources=resources)
    try:
        exported.serialize(path)
    finally:
        exported.close()
        cagra.close()
        resources.close()

    index = hnswlib.Index(space="l2", dim=base.shape[1])
    index.load_index(str(path), max_elements=len(base))
    if index.M != 16:
        raise AssertionError(f"unexpected exported hnswlib M: {index.M}")
    index.set_ef(64)
    labels, distances = index.knn_query(queries, k=10, num_threads=1)
    if labels.shape != (len(queries), 10) or distances.shape != labels.shape:
        raise AssertionError("unexpected hnswlib result shape")
    if np.any(labels < 0) or np.any(labels >= len(base)) or not np.isfinite(distances).all():
        raise AssertionError("hnswlib returned invalid neighbors or distances")

    saved = path.with_suffix(".hnswlib.bin")
    index.save_index(str(saved))
    reloaded = hnswlib.Index(space="l2", dim=base.shape[1])
    reloaded.load_index(str(saved), max_elements=len(base))
    reloaded.set_ef(64)
    labels2, distances2 = reloaded.knn_query(queries, k=10, num_threads=1)
    if not np.array_equal(labels, labels2) or not np.array_equal(distances, distances2):
        raise AssertionError("hnswlib save/load changed deterministic search results")

    print(
        json.dumps(
            {
                "status": "success",
                "count": int(index.element_count),
                "M": int(index.M),
                "file_bytes": path.stat().st_size,
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        return _child(Path(sys.argv[2]))

    try:
        import hnswlib  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        print(f"SKIP: optional hnswlib interoperability dependencies unavailable: {exc}")
        return 77

    with tempfile.TemporaryDirectory(prefix="ovvs-hnswlib-") as directory:
        exported = Path(directory) / "ovvs-cagra.hnsw"
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(exported)],
            capture_output=True,
            text=True,
            timeout=120,
            env=os.environ.copy(),
            check=False,
        )
    if result.returncode != 0:
        print(f"stock hnswlib child failed with exit code {result.returncode}", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return 1
    print(result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
