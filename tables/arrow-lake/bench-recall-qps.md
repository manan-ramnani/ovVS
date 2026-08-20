# Arrow Lake 265K recall–QPS (icx SYCL + oneMKL)

Source: `tools/bench/bench.py` against `build-icpx` `iovs.dll` (`IOVS_WITH_SYCL=ON`, `IOVS_WITH_MKL=ON`).
Dataset: `data/sift-128-euclidean.hdf5` slice n=2000 dim=128 nq=32 k=10.
Comparators: faiss-cpu 1.15.0, hnswlib 0.8.0.

| Run | ioVS brute CPU ms / QPS | ioVS brute NPU | ioVS brute GPU | FAISS brute ms | ioVS vs FAISS recall | FAISS IVF-Flat ms | ioVS IVF-Flat ms / vs-faiss-recall | FAISS IVF-PQ ms | ioVS IVF-PQ ms / vs-faiss-recall | hnswlib ms | ioVS CAGRA ms / vs-hnswlib-recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.519 / 12704 | 605.935 / 52.8 (last=NPU) | 1272.439 / 25.1 (last=GPU, includes JIT) | 2.383 | 1.000 | 0.746 | 2.197 / 0.981 | 0.473 | 2.867 / 0.637 | 0.483 | 2.042 / 0.713 |
| 2 | 2.402 / 13324 | 708.367 / 45.2 (last=NPU) | 65.409 / 489.2 (last=GPU) | 1.717 | 1.000 | 0.156 | 1.808 / 0.981 | 0.328 | 2.714 / 0.637 | 0.404 | 1.984 / 0.713 |

GPU brute run 1 includes SYCL/oneMKL first-touch JIT. Run 2 is the steady number.
NPU last_device=2 and GPU last_device=3 match FORCE_* (cpu=1, npu=2, gpu=3).
FAISS IVF-PQ train warns that 2000 points is below 256-centroid advice; recall is still reported.
The bench does **not** emit `FAISS comparator parked`.
