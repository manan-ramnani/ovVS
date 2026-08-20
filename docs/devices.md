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

## This SKU

Bakeoff JSON lives in `tables/<sku>/`. This lab machine writes `tables/arrow-lake/` (Core Ultra 7 265K). Lunar Lake files are only created if that CPU brand string is actually probed — do not copy Arrow Lake numbers.

Tiny GEMM (`B` small, `K` small) often wins on CPU because of NPU launch tax (`tables/arrow-lake/gemm.json`). Large GEMM `1e5 × 32 × 768` on this 265K: GPU 689 ms, NPU 775 ms, CPU 923 ms — both accelerators beat CPU. Large topk/gather still CPU-win (launch tax). See `tables/arrow-lake/gemm_large.json`.

## CAGRA walk

Search is `prim_graph_walk`: fused SYCL iGPU kernel when `IOVS_WITH_SYCL=ON`, otherwise a host walk whose gather/pairwise go through OpenVINO GPU (or NPU) under FORCE_*. `iovsSyclEnabled()` reports the compile-time path.

## FP16 / INT8

`iovsBruteForceBuildTyped` converts F16 (IEEE) or I8 (value as float) into the fp32 workspace used by prims. Compare I8 search to an oracle on the same integer-valued floats. F16 neighbor ids match a dequantized-fp16 oracle; expect ~1e-3 distance atol vs true fp32.

## HNSW file

`iovsHnswSerialize` writes the [hnswlib](https://github.com/nmslib/hnswlib) `saveIndex` layout (size_t header, then per-element level-0 links + float data + label). Load with `hnswlib.Index(space='l2', dim=D).load_index(path)` when the package is installed, or `iovsHnswDeserialize`.
