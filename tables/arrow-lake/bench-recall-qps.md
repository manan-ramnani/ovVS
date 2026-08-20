# Arrow Lake 265K recall–QPS (icx SYCL + oneMKL)

Source: `tools/bench/bench.py` against `build-icpx` `iovs.dll` (`IOVS_WITH_SYCL=ON`, `IOVS_WITH_MKL=ON`).
Dataset: `data/sift-128-euclidean.hdf5` slice n=2000 dim=128 nq=32 k=10.
Comparators: faiss-cpu 1.15.0, hnswlib 0.8.0.

| Run | ioVS brute CPU ms / QPS | ioVS brute NPU | ioVS brute GPU | FAISS brute ms | ioVS vs FAISS recall | FAISS IVF-Flat ms | ioVS IVF-Flat ms / vs-faiss-recall | FAISS IVF-PQ ms | ioVS IVF-PQ ms / vs-faiss-recall | hnswlib ms | ioVS CAGRA ms / vs-hnswlib-recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.895 / 35750 | 607.301 / 52.7 (last=NPU) | 82.072 / 389.9 (last=GPU) | 3.845 | 1.000 | 0.306 | 1.414 / 0.981 | 0.339 | 2.069 / 0.637 | 0.400 | 1.956 / 0.984 |
| 2 | 0.871 / 36722 | 626.270 / 51.1 (last=NPU) | 74.270 / 430.9 (last=GPU) | 2.300 | 1.000 | 0.141 | 1.119 / 0.981 | 0.362 | 2.057 / 0.637 | 0.428 | 2.071 / 0.984 |

NPU last_device=2 and GPU last_device=3 match FORCE_* (cpu=1, npu=2, gpu=3).
FAISS IVF-PQ train warns that 2000 points is below 256-centroid advice; recall is still reported.
ioVS IVF-PQ vs-faiss 0.637 is competitor-overlap (FAISS IVF-PQ vs-brute on this slice is 0.637); ioVS IVF-PQ nprobe=8 krefine=32 vs-brute is 0.947.
CAGRA (graph_degree=16, itopk=32, search_width=2) vs-brute is 0.991 vs hnswlib M=16 ef=32 vs-brute 0.994. vs-hnswlib-recall is ID overlap with hnswlib, not vs-brute.
The bench does **not** emit `FAISS comparator parked`.
Python must load `build-icpx/bin/iovs.dll` (`IOVS_LIBRARY`); `python/iovs/__init__.py` honors that env var and prefers `build-icpx` over `build/bin`.
