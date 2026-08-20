# Devices, mixer, and FORCE_* policy

ioVS picks a device per primitive (`gemm`, `topk`, `gather`, pairwise). Algorithms only call `iovs::prim`.

## Policy (`iovsResourcesSetPolicy`)

| Policy | Behavior |
|---|---|
| `IOVS_POLICY_AUTO` | NPU if the op is large enough and NPU is not marked busy; else GPU; else CPU |
| `IOVS_POLICY_FORCE_CPU` | CPU only |
| `IOVS_POLICY_FORCE_NPU` | NPU only. If the NPU does not run the op, the call returns `IOVS_STATUS_DEVICE_UNAVAILABLE`. CPU is **not** reported as an NPU win. |
| `IOVS_POLICY_FORCE_GPU` | GPU only (OpenVINO GPU plugin, or SYCL if `IOVS_WITH_SYCL=ON`). Same unavailable rule. |

`iovsResourcesLastDevice` is the device that **actually ran** the last primitive. Tests and bakeoff tables compare `last_device` to the forced policy.

`iovsResourcesSetNpuBusy(1)` is mixer v2: AUTO skips NPU (LLM or another graph occupying the NPU) and still completes on GPU/CPU.

Large GEMM (`flops >= 1e5×32×768` by default) uses the winner in `tables/<sku>/gemm_large.json` when that file is found (`IOVS_TABLES` or `../../tables` from the binary). On Arrow Lake 265K that is GPU, then NPU, then CPU.

`iovsResourcesEnergyUj` returns package microjoules from Linux RAPL sysfs or Intel Power Gadget on Windows. If neither exists, status is `unsupported` (no RAPL MSR/ETW energy counter on this host after probing Power Gadget DLLs and `GetSystemPowerStatus`).

## This SKU

Bakeoff JSON lives in `tables/<sku>/`. This lab machine writes `tables/arrow-lake/` (Core Ultra 7 265K). Lunar Lake files are only created if that CPU brand string is actually probed — do not copy Arrow Lake numbers.

Tiny GEMM (`B` small, `K` small) often wins on CPU because of NPU launch tax (`tables/arrow-lake/gemm.json`). Large GEMM `1e5 × 32 × 768` on this 265K: GPU 689 ms, NPU 775 ms, CPU 923 ms — both accelerators beat CPU. Large topk/gather still CPU-win (launch tax). See `tables/arrow-lake/gemm_large.json`.

## CAGRA walk

Search is `prim_graph_walk`: fused SYCL iGPU kernel when `IOVS_WITH_SYCL=ON` (SLM itopk, 256-slot hashmap, `search_width` seeds). Build that path with intel/llvm nightly `clang++ -fsycl` if oneAPI `icpx` is not installed. Otherwise a host walk whose gather/pairwise go through OpenVINO GPU (or NPU) under FORCE_*. `iovsSyclEnabled()` reports the compile-time path.

## FP16 / INT8

`iovsBruteForceBuildTyped` converts F16 (IEEE) or I8 (value as float) into the fp32 workspace used by prims. I8 search ids match an independent L2 oracle on the same integer-valued floats. F16 search ids match an independent L2 oracle on IEEE-decoded binary16 vectors (not the original fp32 dataset); L2 distances use atol **2e-2** (binary16 mantissa ~1e-3, accumulated over typical test `dim`).

Python `neighbors.*.build/search` accept NumPy arrays and DLPack exporters (`np.from_dlpack` / `__dlpack__`). Search results are NumPy when NumPy is importable.

ScaNN-like indexes train IVF-PQ in anisotropic (per-dim std) space and **re-rank candidates on the original unscaled vectors**.

`iovsBitsetFromAllowList(n, ids, nids, bitset)` fills a `(n+7)/8` bitset for filtered search. IVF-PQ and IVF-RaBitQ support serialize/deserialize/extend like IVF-Flat.

When `IOVS_WITH_SYCL=ON`, GEMM/TopK/Gather run on SYCL **USM shared** scratch first, then fall back to the OpenVINO GPU plugin. NPU Gather/TopK HostCompile tiles rows (32) and gather indices (128) when a full-shape compile fails.

C++ wrappers cover brute-force, IVF-Flat, IVF-PQ, IVF-RaBitQ, and CAGRA (`include/iovs/iovs.hpp`), including serialize/extend. Python `search(..., allow_list=ids)` builds the filter bitset via `iovsBitsetFromAllowList`.

## HNSW file

`iovsHnswSerialize` writes the [hnswlib](https://github.com/nmslib/hnswlib) `saveIndex` layout (size_t header, then per-element level-0 links + float data + label). Load with `hnswlib.Index(space='l2', dim=D).load_index(path)` when the package is installed, or `iovsHnswDeserialize`.
