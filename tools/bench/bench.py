#!/usr/bin/env python3
"""Recall-QPS: ioVS brute / IVF vs FAISS-CPU, CAGRA vs hnswlib when packages exist."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def load_iovs():
    os.environ.setdefault("IOVS_LIBRARY", str(ROOT / "build" / "bin" / "iovs.dll"))
    if not Path(os.environ["IOVS_LIBRARY"]).exists():
        for p in (ROOT / "build" / "bin").glob("*.dll"):
            if p.name.lower().startswith("iovs"):
                os.environ["IOVS_LIBRARY"] = str(p)
    import iovs as m

    return m


def recall_at_k(got, truth, nq, k):
    hit = 0
    for q in range(nq):
        tset = set(int(truth[q * k + j]) for j in range(k))
        for i in range(k):
            if int(got[q * k + i]) in tset:
                hit += 1
    return hit / float(nq * k)


def main() -> int:
    try:
        import numpy as np
    except ImportError:
        print("numpy missing; bench skipped")
        return 0

    rng = np.random.default_rng(0)
    n, dim, nq, k = 800, 32, 32, 10
    data = rng.standard_normal((n, dim), dtype=np.float32)
    queries = rng.standard_normal((nq, dim), dtype=np.float32)
    sift = ROOT / "data" / "sift-128-euclidean.hdf5"
    if sift.exists():
        try:
            import h5py  # type: ignore

            with h5py.File(sift, "r") as h:
                data = np.asarray(h["train"][:2000], dtype=np.float32)
                queries = np.asarray(h["test"][:32], dtype=np.float32)
            n, dim = int(data.shape[0]), int(data.shape[1])
            nq = int(queries.shape[0])
            print(f"using SIFT hdf5 n={n} dim={dim} nq={nq}")
        except Exception as e:
            print(f"SIFT hdf5 present but unused: {e}; generated 800x32 stand-in")
    else:
        print("SIFT hdf5 absent; generated stand-in n=800 dim=32 nq=32")

    lib = load_iovs()
    res = lib.Resources()
    res.set_policy(6)  # FORCE_CPU for comparable FAISS/hnswlib host bench
    idx = lib.neighbors.brute_force.build(data, dim=dim, resources=res)
    t0 = time.perf_counter()
    nb, _ = idx.search(queries, k=k)
    t1 = time.perf_counter()
    brute_ms = (t1 - t0) * 1000
    print(f"iovs brute n={n} dim={dim} nq={nq} ms={brute_ms:.3f} qps={nq / (t1 - t0 + 1e-9):.1f}")

    faiss_ok = False
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatL2(dim)
        index.add(data)
        t0 = time.perf_counter()
        _, I = index.search(queries, k)
        t1 = time.perf_counter()
        rec = recall_at_k(np.asarray(nb).reshape(-1), I.reshape(-1), nq, k)
        print(f"faiss brute ms={(t1 - t0) * 1000:.3f} iovs-vs-faiss-recall={rec:.3f}")
        nlist, nprobe = 32, 8
        quant = faiss.IndexFlatL2(dim)
        ivf = faiss.IndexIVFFlat(quant, dim, nlist)
        ivf.train(data)
        ivf.add(data)
        ivf.nprobe = nprobe
        t0 = time.perf_counter()
        _, Iivf = ivf.search(queries, k)
        t1 = time.perf_counter()
        print(f"faiss ivf-flat nlist={nlist} nprobe={nprobe} ms={(t1 - t0) * 1000:.3f}")
        ivf_ix = lib.neighbors.ivf_flat.build(data, dim=dim, nlist=nlist, resources=res)
        t0 = time.perf_counter()
        inb, _ = ivf_ix.search(queries, k=k, nprobe=nprobe)
        t1 = time.perf_counter()
        rec_i = recall_at_k(np.asarray(inb).reshape(-1), Iivf.reshape(-1), nq, k)
        print(f"iovs ivf-flat ms={(t1 - t0) * 1000:.3f} vs-faiss-recall={rec_i:.3f}")
        pq_m, nbits, krefine = 8, 8, 32
        quant_pq = faiss.IndexFlatL2(dim)
        faiss_pq = faiss.IndexIVFPQ(quant_pq, dim, nlist, pq_m, nbits)
        faiss_pq.train(data)
        faiss_pq.add(data)
        faiss_pq.nprobe = nprobe
        t0 = time.perf_counter()
        _, Ipq = faiss_pq.search(queries, k)
        t1 = time.perf_counter()
        print(f"faiss ivf-pq nlist={nlist} nprobe={nprobe} M={pq_m} nbits={nbits} ms={(t1 - t0) * 1000:.3f}")
        pq_ix = lib.neighbors.ivf_pq.build(
            data, dim=dim, nlist=nlist, pq_m=pq_m, pq_nbits=nbits, resources=res
        )
        t0 = time.perf_counter()
        pnb, _ = pq_ix.search(queries, k=k, nprobe=nprobe, krefine=krefine)
        t1 = time.perf_counter()
        rec_p = recall_at_k(np.asarray(pnb).reshape(-1), Ipq.reshape(-1), nq, k)
        print(f"iovs ivf-pq ms={(t1 - t0) * 1000:.3f} vs-faiss-recall={rec_p:.3f}")
        faiss_ok = True
    except Exception as e:
        print(f"faiss unavailable: {e}")

    try:
        import hnswlib  # type: ignore

        p = hnswlib.Index(space="l2", dim=dim)
        p.init_index(max_elements=n, ef_construction=50, M=16)
        p.add_items(data)
        p.set_ef(32)
        t0 = time.perf_counter()
        labels, _ = p.knn_query(queries, k=k)
        t1 = time.perf_counter()
        print(f"hnswlib M=16 ef=32 ms={(t1 - t0) * 1000:.3f}")
        cg = lib.neighbors.cagra.build(data, dim=dim, graph_degree=16, intermediate_degree=32, resources=res)
        t0 = time.perf_counter()
        cnb, _ = cg.search(queries, k=k, itopk_size=32, search_width=2)
        t1 = time.perf_counter()
        rec_c = recall_at_k(np.asarray(cnb).reshape(-1), np.asarray(labels).reshape(-1), nq, k)
        print(f"iovs cagra ms={(t1 - t0) * 1000:.3f} vs-hnswlib-recall={rec_c:.3f}")
    except Exception as e:
        print(f"hnswlib unavailable: {e}")

    if not faiss_ok:
        print("ioVS-only numbers emitted; FAISS comparator parked after install retry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
