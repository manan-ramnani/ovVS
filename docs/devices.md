# Devices, mixer, and FORCE_* policy

ovVS picks a device per primitive (`gemm`, `topk`, `gather`, pairwise). Algorithms only call `ovvs::prim`.

## Policy (`ovvsResourcesSetPolicy`)

| Policy | Behavior |
|---|---|
| `OVVS_POLICY_AUTO` | Per `docs/hw-split.md`: GEMM/TopK/Gather → CPU oneMKL on this SKU; CAGRA walk → iGPU; ADC → NPU |
| `OVVS_POLICY_FORCE_CPU` | CPU only |
| `OVVS_POLICY_FORCE_NPU` | NPU only. If the NPU does not run the op, the call returns `OVVS_STATUS_DEVICE_UNAVAILABLE`. CPU is **not** reported as an NPU win. |
| `OVVS_POLICY_FORCE_GPU` | GPU only (OpenVINO GPU plugin, or SYCL if `OVVS_WITH_SYCL=ON`). Same unavailable rule. |

`ovvsResourcesLastDevice` is the device that **actually ran** the last primitive. Tests and bakeoff tables compare `last_device` to the forced policy.

`ovvsResourcesSetNpuBusy(1)` is mixer v2: AUTO skips NPU (LLM or another graph occupying the NPU) and still completes on GPU/CPU.

Large GEMM (`flops >= 1e5×32×768` by default) uses the winner in `tables/<sku>/gemm_large.json` when that file is found (`OVVS_TABLES` or `../../tables` from the binary). On Arrow Lake 265K the measured order is CPU oneMKL, then NPU, then iGPU; AUTO therefore selects CPU.

`ovvsResourcesEnergyUj` returns package microjoules. On Windows the path is the inbox **Intel Processor** driver (`intelppm.sys` / `cpu.inf`) Energy Metering Interface (EMI v2, OEM `Microsoft` / model `PPM`) on the ACPI processor device, channel `RAPL_Package0_PKG`. Units are picowatt-hours → µJ via ×0.0036. Fallback is PDH `\Energy Meter(RAPL_Package0_PKG)\Energy` (same EMI channels), then Intel Power Gadget if installed. Linux uses RAPL sysfs. `GetSystemPowerStatus` is not a package counter. Intel IPF (`ipf_cpu`, PCI `8086:AD03`) and PMT (`IntcPMT`, PCI `8086:AD0D`) are present on this SKU but are DTT/telemetry, not the RAPL EMI publisher. Power Gadget is not required.

## This SKU

Bakeoff JSON lives in `tables/<sku>/`. This lab machine writes `tables/arrow-lake/` (Core Ultra 7 265K). Lunar Lake files are only created if that CPU brand string is actually probed — do not copy Arrow Lake numbers.

Tiny and large GEMM on this SKU both win on **CPU oneMKL** (`cblas_sgemm`): 64×128×32 is 0.006 ms; `1e5×32×768` is **18 ms** vs NPU 45 ms vs GPU 221 ms (`gemm_large.json`). Search-shaped 32×1e5×768 is the same story (CPU 18 vs NPU 36, `gemm_search.json`). NPU DPU is 5.3 ms; the 45 ms wall is Parameter DMA. AUTO GEMM is CPU. Large TopK/Gather stay CPU. CAGRA walk is iGPU. ADC is NPU. See `docs/hw-split.md`.

## CAGRA walk

Search is `prim_graph_walk`. With `OVVS_WITH_SYCL=ON`, the current iGPU checkpoint assigns one work-group per query, cooperatively computes distance, keeps candidates/expansion state in SLM, and uses a traversal-budget-sized global visited hash. Query chunks bound hash allocation near 64 MiB. CAGRA dataset and graph vectors attempt shared-USM allocation so the normal path binds them directly; heap fallback causes explicit per-search device copies. `ovvsResourcesCagraTransferStats` exposes coherent resource-local counters for completed batched walk calls, direct index pointers, upload calls, and upload bytes. Bounded n=4,200/n=2,000 and the SIFT1M matched gate report zero explicit index uploads; SIFT1M produced `192/192/0/0`. These counters do not measure driver-managed shared-memory page migration. Query-content hashing gives CPU and GPU the same deterministic, batch/order-invariant seed stream; it is not yet a graph-aware entry scheme. The SIFT1M gate still fails quality at 0.4057 recall@10 versus hnswlib 0.8903, so builder quality, effort controls, graph-aware landmarks, and leader-serial selection/sort remain backlog B2 before performance tuning. Oversized shapes reject explicitly. FORCE_GPU returns `DEVICE_UNAVAILABLE` rather than crossing into the host walk; FORCE_NPU graph search is unavailable because the NPU has no graph-walk implementation. CAGRA-Q remains a host PQ-ADC walk (B10), so both forced accelerator policies reject it as unavailable. Adaptive policies may use host walks and report CPU. `ovvsSyclEnabled()` reports whether the compiled SYCL path has a usable GPU at runtime.

## NN-Descent build

Through 4,096 rows, NN-Descent uses the exact pairwise/top-k primitive path. Above that threshold, AUTO/GPU_IF_FASTER/HETERO and FORCE_GPU use the B5 sampled iGPU local join for L2, L2-sqrt, inner product, and cosine: one work-group per vertex, deterministic unique seeds, double-buffered graph rows, cooperative distance, and bounded SLM candidates with Bloom-assisted duplicate checks. The output remains a host graph; standalone NN-Descent does not retain a second dataset copy. FORCE_NPU is unavailable, and a present GPU kernel failure does not silently fall into the large host loop; OOM and runtime errors retain their statuses. Unsupported adaptive metrics retain the CPU reference path. AUTO CAGRA build can use this initializer but still ends in host `prune_graph` and reports CPU for that final stage. Full FORCE_GPU CAGRA build therefore returns `DEVICE_UNAVAILABLE` until T13.3 moves prune to GPU. The first SIFT1M build completed, but downstream CAGRA recall was only 0.4057 at the matched gate; convergence/new-edge/reverse-candidate quality and tuning therefore remain open.

## FP16 / INT8 / FP8 / FP4

`ovvsBruteForceBuildTyped` still converts F16/I8 datasets into an fp32 workspace for search (index storage). **Compute** can be lower-bit: `ovvsGemmEx(..., compute)` keeps host buffers fp32 and runs the MatMul as that type on the device.

- **INT8 (NPU):** NCE MACs are INT8-native (Arrow Lake 3720 TOPS are INT8; FP16 is half-rate). The OpenVINO graph contract is **NNCF Low Precision IR** (FakeQuantize on activations **and** Constant weights), not two runtime `f32` Parameters. Raw `si8` MatMul is rejected. `PERF_COUNT` on this SKU: DPU MatMul is 21 µs for 64³ and 5.3 ms for `1e5×32×768`; SHAVE DMA of the big Parameter is 35 ms **when the C++ path used `set_tensor` on host pointers**. The NPU plugin allocates Level Zero IO at `create_infer_request`; `get_input_tensor`/`get_output_tensor` + `memcpy` is the documented zero-extra-copy feed. `set_tensor(host*)` forces a plugin copy. Baking B as Constant makes DPU faster (2 µs) and **wall slower** (8–24 ms) — Level Zero/UMD, not the MAC array. Scratchpad is 4 MB, so large datasets never stay on-die. Compile hints: `NPU_TURBO`, `PERFORMANCE_HINT=LATENCY`, `NPU_COMPILATION_MODE_PARAMS=optimization-level=2 performance-hint-override=latency`. See `tables/arrow-lake/npu-gemm-dpu-vs-wall.md` and AGENTS.md “NPU contract”. FORCE_NPU still runs FakeQuantize INT8; AUTO low-bit prefers iGPU XMX because that path is a real GEMM.
- **FP16 (iGPU):** Xe Matrix Extensions (XMX/DPAS) do FP16 and INT8. `ovvsGemmEx(..., F16)` AUTO/FORCE_GPU uses oneMKL `sycl::half` GEMM (XMX), then a SYCL-half loop, then OpenVINO GPU.
- **INT8 (iGPU):** oneMKL `int8×int8→float` GEMM (XMX, 2× FP16 issue rate on Xe-core). AUTO I8 prefers this; FORCE_NPU still runs FakeQuantize.
- **FP16 (NPU):** still available via `ovvsGemmEx(..., F16)` FORCE_NPU (`Convert→f16 MatMul`). Not the AUTO large-GEMM default.
- **NPU range safety:** Arrow Lake NPU f32 graph IO can still exhibit FP16-range saturation. GEMM/TopK reject non-finite inputs, conservative bounds at or above 65,504, and invalid outputs. FORCE_NPU returns `DEVICE_UNAVAILABLE` for such shapes until range-scaled execution exists.
- **FP8 e4m3 / FP4 e2m1:** probed (`npu_matmul_f8e4m3`, `npu_matmul_f4e2m1`). Arrow Lake 3720 `compile_fail`.

`ovvsResourcesLastComputeDtype` reports the type that actually ran. I8 search ids on `BuildTyped` still match an independent L2 oracle on integer-valued floats. F16 `BuildTyped` ids match an L2 oracle on IEEE-decoded binary16 (atol **2e-2**).

Python `neighbors.*.build/search` accept NumPy arrays and DLPack exporters (`np.from_dlpack` / `__dlpack__`). Search results are NumPy when NumPy is importable.

ScaNN-like indexes train IVF-PQ in anisotropic (per-dim std) space and **re-rank candidates on the original unscaled vectors**.

`ovvsBitsetFromAllowList(n, ids, nids, bitset)` fills a `(n+7)/8` bitset for filtered search. IVF-PQ and IVF-RaBitQ support serialize/deserialize/extend like IVF-Flat.

When `OVVS_WITH_SYCL=ON`, GEMM tries oneMKL GPU then hand SYCL USM (winner cached at first call), then OpenVINO GPU. Device-accessible inputs avoid an extra copy. CAGRA's index-owned vectors use shared USM when allocation succeeds; the host-heap fallback remains correct but copies those buffers for each GPU search. NPU binds host-visible tiles. Hamming and Lp pairwise have a SYCL kernel (FORCE_GPU is honest) and an OpenVINO NPU graph (FORCE_NPU is honest): Hamming is GreaterEqual→LogicalXor→ReduceSum; Lp is Subtract→Abs→Power→ReduceSum→Power. Those lower to Intel ActShave (`eltwise_logical_xor`, `activation_abs`, …) plus DPU ReduceSum. PQ ADC on NPU is OpenVINO Gather+ReduceSum in tiles of 128 codes; `PERF_COUNT` shows those tiles run as ActShave + DPU (`shave_silicon_load: compiler_actshave`). A raw SHAVE ELF32 is not a firmware load object (`invalid_native_binary`); the compiler embeds kernel `.text` in a graph ELF64. See `compiler/shave/README.md`.

NPU Gather/TopK HostCompile tiles rows (32) and gather indices (128) when a full-shape compile fails.

C++ wrappers cover brute-force, IVF-Flat, IVF-PQ, IVF-RaBitQ, and CAGRA (`include/ovvs/ovvs.hpp`), including serialize/extend. Python `search(..., allow_list=ids)` builds the filter bitset via `ovvsBitsetFromAllowList`.

## HNSW file

`ovvsHnswSerialize` writes the [hnswlib](https://github.com/nmslib/hnswlib) `saveIndex` layout (size_t header, then per-element level-0 links + float data + label). Load with `hnswlib.Index(space='l2', dim=D).load_index(path)` when the package is installed, or `ovvsHnswDeserialize`.
