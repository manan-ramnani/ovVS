# Arrow Lake 265K recall–QPS (icx SYCL + oneMKL)

Source: `tools/bench/bench.py` against `build-icpx` `ovvs.dll` (`OVVS_WITH_SYCL=ON`, `OVVS_WITH_MKL=ON`).
Dataset: `data/sift-128-euclidean.hdf5` slice n=2000 dim=128 nq=32 k=10.
Comparators: faiss-cpu 1.15.0, hnswlib 0.8.0.

| Run | ovVS brute CPU ms / QPS | ovVS brute NPU | ovVS brute GPU | FAISS brute ms | ovVS vs FAISS recall | FAISS IVF-Flat ms | ovVS IVF-Flat ms / vs-faiss-recall | FAISS IVF-PQ ms | ovVS IVF-PQ ms / vs-faiss-recall | hnswlib ms | ovVS CAGRA ms / vs-hnswlib-recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.895 / 35750 | 607.301 / 52.7 (last=NPU) | 82.072 / 389.9 (last=GPU) | 3.845 | 1.000 | 0.306 | 1.414 / 0.981 | 0.339 | 2.069 / 0.637 | 0.400 | 1.956 / 0.984 |
| 2 | 0.871 / 36722 | 626.270 / 51.1 (last=NPU) | 74.270 / 430.9 (last=GPU) | 2.300 | 1.000 | 0.141 | 1.119 / 0.981 | 0.362 | 2.057 / 0.637 | 0.428 | 2.071 / 0.984 |

NPU last_device=2 and GPU last_device=3 match FORCE_* (cpu=1, npu=2, gpu=3).
FAISS IVF-PQ train warns that 2000 points is below 256-centroid advice; recall is still reported.
ovVS IVF-PQ vs-faiss 0.637 is competitor-overlap (FAISS IVF-PQ vs-brute on this slice is 0.637); ovVS IVF-PQ nprobe=8 krefine=32 vs-brute is 0.947.
CAGRA (graph_degree=16, itopk=32, search_width=2) vs-brute is 0.991 vs hnswlib M=16 ef=32 vs-brute 0.994. vs-hnswlib-recall is ID overlap with hnswlib, not vs-brute.
The bench does **not** emit `FAISS comparator parked`.
Python must load `build-icpx/bin/ovvs.dll` (`OVVS_LIBRARY`); `python/ovvs/__init__.py` honors that env var and prefers `build-icpx` over `build/bin`.

Package RAPL µJ/query (`Resources.energy_uj`, EMI `RAPL_Package0_PKG`, ~150 ms repeat window) on the same slice:

| Path | last_device | uj/query |
|---|---|---|
| brute CPU | 1 | 1049 |
| brute NPU | 2 | 4638 |
| brute GPU | 3 | 17893 |
| IVF-Flat CPU | 1 | 1239 |
| IVF-PQ CPU | 1 | 2186 |
| CAGRA CPU | 1 | 2990 |

These are **package** joules (CPU+iGPU+uncore), not NPU-isolated. Short CPU searches are noisy; NPU/GPU walls are long enough that the delta is real. NPU brute is slower *and* more package µJ/query than CPU on this small slice.
