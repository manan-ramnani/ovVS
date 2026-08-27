# Devices, mixer, and FORCE_* policy

ovVS picks a device per primitive (`gemm`, `topk`, `gather`, pairwise). Algorithms only call `ovvs::prim`.

## Policy (`ovvsResourcesSetPolicy`)

| Policy | Behavior |
|---|---|
| `OVVS_POLICY_AUTO` | NPU if the op is large enough and NPU is not marked busy; else GPU; else CPU |
| `OVVS_POLICY_FORCE_CPU` | CPU only |
| `OVVS_POLICY_FORCE_NPU` | NPU only. If the NPU does not run the op, the call returns `OVVS_STATUS_DEVICE_UNAVAILABLE`. CPU is **not** reported as an NPU win. |
| `OVVS_POLICY_FORCE_GPU` | GPU only (OpenVINO GPU plugin, or SYCL if `OVVS_WITH_SYCL=ON`). Same unavailable rule. |

`ovvsResourcesLastDevice` is the device that **actually ran** the last primitive. Tests and bakeoff tables compare `last_device` to the forced policy.

`ovvsResourcesSetNpuBusy(1)` is mixer v2: AUTO skips NPU (LLM or another graph occupying the NPU) and still completes on GPU/CPU.

Large GEMM (`flops >= 1e5×32×768` by default) uses the winner in `tables/<sku>/gemm_large.json` when that file is found (`OVVS_TABLES` or `../../tables` from the binary). On Arrow Lake 265K that is GPU, then NPU, then CPU.

`ovvsResourcesEnergyUj` returns package microjoules. On Windows the path is the inbox **Intel Processor** driver (`intelppm.sys` / `cpu.inf`) Energy Metering Interface (EMI v2, OEM `Microsoft` / model `PPM`) on the ACPI processor device, channel `RAPL_Package0_PKG`. Units are picowatt-hours → µJ via ×0.0036. Fallback is PDH `\Energy Meter(RAPL_Package0_PKG)\Energy` (same EMI channels), then Intel Power Gadget if installed. Linux uses RAPL sysfs. `GetSystemPowerStatus` is not a package counter. Intel IPF (`ipf_cpu`, PCI `8086:AD03`) and PMT (`IntcPMT`, PCI `8086:AD0D`) are present on this SKU but are DTT/telemetry, not the RAPL EMI publisher. Power Gadget is not required.

## This SKU

Bakeoff JSON lives in `tables/<sku>/`. This lab machine writes `tables/arrow-lake/` (Core Ultra 7 265K). Lunar Lake files are only created if that CPU brand string is actually probed — do not copy Arrow Lake numbers.

Tiny GEMM (`B` small, `K` small) often wins on CPU because of NPU launch tax (`tables/arrow-lake/gemm.json`). Large GEMM `1e5 × 32 × 768` on icx SYCL+oneMKL (warmup included): NPU 117 ms, GPU 150 ms, CPU 191 ms — **NPU is the AUTO large-GEMM winner**. Large topk/gather still CPU-win (launch tax). See `tables/arrow-lake/gemm_large.json`.

## CAGRA walk

Search is `prim_graph_walk`: fused SYCL iGPU kernel when `OVVS_WITH_SYCL=ON` (SLM itopk, 256-slot hashmap, `search_width` seeds). Build that path with intel/llvm nightly `clang++ -fsycl` if oneAPI `icpx` is not installed. Otherwise a host walk whose gather/pairwise go through OpenVINO GPU (or NPU) under FORCE_*. `ovvsSyclEnabled()` reports the compile-time path.

## FP16 / INT8 / FP8 / FP4

`ovvsBruteForceBuildTyped` still converts F16/I8 datasets into an fp32 workspace for search (index storage). **Compute** can be lower-bit: `ovvsGemmEx(..., compute)` keeps host buffers fp32 and runs the MatMul as that type on the device.

- **F16 (NPU):** OpenVINO `Convert→f16 MatMul→f32` with static shapes so the NPU MAC/SRAM path is FP16. AUTO large GEMM (`m*n*k ≥ 64*64*32`) tries this before fp32.
- **F16 (iGPU):** SYCL `sycl::half` kernel when `OVVS_WITH_SYCL=ON`; else the same Convert graph on OpenVINO GPU.
- **I8 (NPU):** host per-tensor absmax quantize to int8, `i8 MatMul`, dequant by `sa*sb`. Falls back to `DEVICE_UNAVAILABLE` if the compiler rejects i8 MatMul.
- **FP8 e4m3 / FP4 e2m1:** probed at `ovvsProbeJson` (`npu_matmul_f8e4m3`, `npu_matmul_f4e2m1`). Arrow Lake NPU may `compile_fail`; Lunar Lake / NPU4+ is the intended win.

`ovvsResourcesLastComputeDtype` reports the type that actually ran. I8 search ids on `BuildTyped` still match an independent L2 oracle on integer-valued floats. F16 `BuildTyped` ids match an L2 oracle on IEEE-decoded binary16 (atol **2e-2**).

Python `neighbors.*.build/search` accept NumPy arrays and DLPack exporters (`np.from_dlpack` / `__dlpack__`). Search results are NumPy when NumPy is importable.

ScaNN-like indexes train IVF-PQ in anisotropic (per-dim std) space and **re-rank candidates on the original unscaled vectors**.

`ovvsBitsetFromAllowList(n, ids, nids, bitset)` fills a `(n+7)/8` bitset for filtered search. IVF-PQ and IVF-RaBitQ support serialize/deserialize/extend like IVF-Flat.

When `OVVS_WITH_SYCL=ON`, GEMM tries oneMKL GPU then hand SYCL USM (winner cached at first call), then OpenVINO GPU. Dataset/graph allocations use shared USM so iGPU GEMM/TopK/Gather/CAGRA skip a host memcpy when the pointer is already shared. NPU still binds host-visible tiles. Hamming and Lp pairwise have a SYCL kernel (FORCE_GPU is honest) and an OpenVINO NPU graph (FORCE_NPU is honest): Hamming is GreaterEqual→LogicalXor→ReduceSum; Lp is Subtract→Abs→Power→ReduceSum→Power. Those lower to Intel ActShave (`eltwise_logical_xor`, `activation_abs`, …) plus DPU ReduceSum. PQ ADC on NPU is OpenVINO Gather+ReduceSum in tiles of 128 codes; `PERF_COUNT` shows those tiles run as ActShave + DPU (`shave_silicon_load: compiler_actshave`). A raw SHAVE ELF32 is not a firmware load object (`invalid_native_binary`); the compiler embeds kernel `.text` in a graph ELF64. See `compiler/shave/README.md`.

NPU Gather/TopK HostCompile tiles rows (32) and gather indices (128) when a full-shape compile fails.

C++ wrappers cover brute-force, IVF-Flat, IVF-PQ, IVF-RaBitQ, and CAGRA (`include/ovvs/ovvs.hpp`), including serialize/extend. Python `search(..., allow_list=ids)` builds the filter bitset via `ovvsBitsetFromAllowList`.

## HNSW file

`ovvsHnswSerialize` writes the [hnswlib](https://github.com/nmslib/hnswlib) `saveIndex` layout (size_t header, then per-element level-0 links + float data + label). Load with `hnswlib.Index(space='l2', dim=D).load_index(path)` when the package is installed, or `ovvsHnswDeserialize`.
