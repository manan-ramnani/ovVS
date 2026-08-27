"""True recall vs brute for CAGRA / IVF-PQ / FAISS / hnswlib on the SIFT slice."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
os.environ["OVVS_LIBRARY"] = str(ROOT / "build-icpx" / "bin" / "ovvs.dll")
import ovvs as m  # noqa: E402


def rec(got, truth, nq, k):
    hit = 0
    for q in range(nq):
        tset = set(int(truth[q, j]) for j in range(k))
        for i in range(k):
            if int(got[q, i]) in tset:
                hit += 1
    return hit / float(nq * k)


def main():
    sift = ROOT / "data" / "sift-128-euclidean.hdf5"
    import h5py

    with h5py.File(sift, "r") as h:
        data = np.asarray(h["train"][:2000], dtype=np.float32)
        queries = np.asarray(h["test"][:32], dtype=np.float32)
    n, dim = data.shape
    nq, k = queries.shape[0], 10
    print(f"n={n} dim={dim} nq={nq} k={k}")

    res = m.Resources()
    res.set_policy(6)
    brute = m.neighbors.brute_force.build(data, dim=dim, resources=res)
    gt, _ = brute.search(queries, k=k)
    gt = np.asarray(gt)

    cg = m.neighbors.cagra.build(data, dim=dim, graph_degree=16, intermediate_degree=32, resources=res)
    for itopk, sw in [(32, 2), (64, 4), (128, 8), (64, 16)]:
        got, _ = cg.search(queries, k=k, itopk_size=itopk, search_width=sw)
        print(f"cagra itopk={itopk} sw={sw} vs-brute={rec(np.asarray(got), gt, nq, k):.3f}")

    pq = m.neighbors.ivf_pq.build(data, dim=dim, nlist=32, pq_m=8, pq_nbits=8, resources=res)
    for nprobe, kr in [(8, 32), (16, 64), (32, 128)]:
        got, _ = pq.search(queries, k=k, nprobe=nprobe, krefine=kr)
        print(f"ivf-pq nprobe={nprobe} krefine={kr} vs-brute={rec(np.asarray(got), gt, nq, k):.3f}")

    try:
        import faiss

        ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, 32)
        ivf.train(data)
        ivf.add(data)
        ivf.nprobe = 8
        _, I = ivf.search(queries, k)
        print(f"faiss ivf-flat nprobe=8 vs-brute={rec(I, gt, nq, k):.3f}")
        fpq = faiss.IndexIVFPQ(faiss.IndexFlatL2(dim), dim, 32, 8, 8)
        fpq.train(data)
        fpq.add(data)
        fpq.nprobe = 8
        _, I = fpq.search(queries, k)
        print(f"faiss ivf-pq nprobe=8 vs-brute={rec(I, gt, nq, k):.3f}")
    except Exception as e:
        print("faiss", e)

    try:
        import hnswlib

        p = hnswlib.Index(space="l2", dim=dim)
        p.init_index(max_elements=n, ef_construction=50, M=16)
        p.add_items(data)
        p.set_ef(32)
        labels, _ = p.knn_query(queries, k=k)
        print(f"hnswlib M=16 ef=32 vs-brute={rec(labels, gt, nq, k):.3f}")
    except Exception as e:
        print("hnswlib", e)


if __name__ == "__main__":
    main()
