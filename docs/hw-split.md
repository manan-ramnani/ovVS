# Hardware split: CPU / iGPU / NPU

This is the product rule for ovVS on Intel client SoCs. Measured on Arrow Lake Core Ultra 7 265K (NPU 3720 ~13 TOPS INT8, Xe-LPG iGPU, 8P+12E). Other SKUs re-bake `tables/<sku>/` and keep the same roles unless a table disagrees.

Goal: **put each op on the device that is actually fastest for that op**, then overlap devices only when the pipeline has independent stages. Do not send work to the NPU or iGPU as a branding exercise.

## What each device is

| Device | Native strength | Native weakness |
|---|---|---|
| **NPU** | Compiled static MatMul / Conv on DPU. INT8 full-rate, FP16 half. Weight-stationary graphs. ~4 MB SRAM scratchpad. | Launch tax. No pointer-chasing. `set_tensor(host*)` SHAVE-copies. Two live fp32 Parameters do not light INT8 MACs. Dataset does not fit SRAM. |
| **iGPU** | XMX/DPAS GEMM (FP16 and INT8 via oneMKL). SYCL graph walk (CAGRA). USM shared with CPU. Parallel convert/pack. | TopK/Gather launch tax vs CPU. Serial host fp32→half is a self-inflicted stall. Shared DRAM, not HBM. |
| **CPU** | Tiny GEMM, TopK, Gather, control plane, heaps, HostCompile, k-means reduction, filters. AVX-512 / oneMKL `cblas_sgemm`. | Naive triple-loop GEMM is not a device (we do not ship that). Dense GEMM remains memory-bandwidth-bound and must be re-baked per SKU. |

## Primitive winners (Arrow Lake, icx + oneMKL + OpenVINO)

Sources: `tables/arrow-lake/gemm.json`, `gemm_large.json`, `gemm_f16.json`, `gemm_i8.json`, `topk.json`, `topk_large.json`, `gather.json`, `gather_large.json`, `npu-gemm-dpu-vs-wall.md`. Re-run `ovvs_bakeoff` after this split; `ms` is cold (includes L0 fill), `ms_hot` is repeated infer with fingerprint-sticky operands.

| Op | Shape | CPU | NPU | iGPU | AUTO |
|---|---|---|---|---|---|
| GEMM f32 | 64×128×32 | **0.006 ms** | 0.57 ms | 0.034 ms | CPU |
| GEMM f32 | 256×256×128 / 512³ / IVF 32×1024×768 | **wins** | ~0.8–1.1 ms launch | slower | CPU |
| GEMM f32 | 1e5×32×768 | **18 ms oneMKL** | 45 ms (DPU 5.3 + DMA) | 221 ms | CPU |
| GEMM f32 | search 32×1e5×768 | **18 ms** | 36 ms (sticky B, still DMA) | 226 ms | CPU |
| GEMM f16 | 1e5×32×768 | 19 ms (f32 cblas) | 39 ms Convert+f16 | 81 ms XMX | CPU |
| GEMM i8 | 1e5×32×768 | 17 ms (f32 cblas) | 223 ms FQ | 170 ms XMX | CPU |
| TopK | tiny and 32×1e5 k=10 | **1.9 ms** | 114 ms | 750 ms | CPU |
| Gather | 4096 rows from 1e5×768 | **4 ms** | 213 ms | 24 ms | CPU |
| Graph walk | CAGRA | host fallback | no | **SYCL fused** | GPU |
| PQ ADC | tiles of 128 | SHAVE C | **Gather+ReduceSum** | — | NPU |
| Hamming / Lp | pairwise graph | AVX | GreaterEqual/Xor or Abs/Power | SYCL | FORCE_* honest; AUTO follows size |

Search-shaped GEMM is `nq × n × dim` with **B = dataset** (large, immutable after build) and **A = queries** (small, changes). That is the NPU sticky-B win. The bakeoff `large` shape (`1e5 × 32 × 768`) is the inverse (huge A); `ovvs_bakeoff search` is `32 × 1e5 × 768`.

## Algorithm mapping

```
Brute-force search
  pairwise L2/IP/cos  → prim_gemm AUTO = CPU oneMKL on this SKU
  epilogue norms      → CPU
  topk                → CPU

IVF-Flat
  coarse assign       → CPU oneMKL (32×1024×768 is 0.11 ms)
  list gather         → CPU
  refine pairwise     → CPU
  topk                → CPU

IVF-PQ / RaBitQ
  coarse assign       → CPU GEMM
  ADC                 → NPU Gather+ReduceSum tiles (this is the NPU search win)
  refine              → CPU

CAGRA search
  graph walk          → iGPU SYCL (work-group/query checkpoint; B2 quality/perf gate open)
  candidate scoring inside the walk → fused in the SYCL kernel

CAGRA build
  NN-Descent init n>4096 → iGPU SYCL under AUTO
  prune               → CPU until T13.3 (full FORCE_GPU build is unavailable)

Vamana / NN-Descent
  NN-Descent n>4096  → iGPU SYCL sampled local join (B5 bounded checkpoint)
  Vamana prune/walk  → host control path (B21 remains open)

k-means
  assign              → CPU oneMKL GEMM
  reduce new centroids→ CPU

HNSW export / serialize / bitset / batcher
  CPU
```

Arrow Lake dense GEMM is an AVX-512 / AMX problem that oneMKL already owns. Shipping it to the NPU made sense when CPU GEMM was a triple loop (192 ms vs NPU 54 ms). After `cblas_sgemm`, NPU wall 36–45 ms is slower than CPU 18 ms. The DPU is still 5.3 ms; we cannot feed it 307 MB Parameters faster than MKL reads DRAM. Do not AUTO-route GEMM to NPU on this SKU. Lunar Lake (48 TOPS, 2× DRAM) must re-bake `gemm_large.json`.

## Feed path (NPU)

1. Compile once. One `InferRequest` per `(op, shape, dtype)`.
2. `memcpy` into `get_input_tensor` L0 buffers. Never `set_tensor` on malloc/USM host pointers.
3. Skip that memcpy when `nbytes ≥ 64 KiB` and a 32-sample fingerprint matches (search dataset, k-means X). Queries and centroids are small or change — they copy.
4. Do not bake B as an IR Constant. DPU got faster; wall got slower.
5. INT8 on NPU is NNCF FakeQuantize+Constant weights, not raw `si8`. Until that IR exists for a given op, AUTO I8 is iGPU XMX.
6. Compile: `NPU_TURBO`, `PERFORMANCE_HINT=LATENCY`, `optimization-level=2 performance-hint-override=latency`.
7. Fail closed when conservative GEMM/TopK bounds or outputs reach FP16 range (65,504); scaled execution is not implemented yet.

Remaining NPU gap: after fingerprint-sticky inputs skip the hot host memcpy, wall still remains about `45–54 ms` versus `5.3 ms` of profiled DPU work on 1e5×32×768. The remaining cost is principally device DMA/tile movement into a 4 MB scratchpad; a roughly 150 MB f16 operand cannot remain resident there. A canonical remote L0 tensor may reduce application copies, duplicate request buffers, or cold-path ownership cost, but it does not by itself prove lower steady-state DMA or wall time. Unsigned SHAVE ELF inject is still unsupported; ActShave already runs inside compiler graphs.

### Runtime opportunities and boundaries (not measured in ovVS)

The current measured Windows build links OpenVINO **2025.3.0** with Intel NPU driver **32.0.100.4841**. Its installed headers already expose NPU `ZeroContext`, `create_l0_host_tensor`, shared native-buffer wrapping, and mapped-file tensors. OpenVINO 2026 adds broader documented paths such as CPU virtual-address import and optional strided IO. All remain experiments until ovVS probes and measures them:

- Start B7 on the current stack: compile a minimal `create_l0_host_tensor`/shared-buffer probe, bind it to a compiled request, validate lifetime and correctness, then compare allocated bytes, application copies, cold wall, and hot wall against the current request-owned `get_input_tensor` path. Upgrade in an isolated build only for capabilities absent from 2025.3. A remote tensor is not evidence that the NPU scratchpad retains the full index between inferences.
- Optional strided NPU IO can avoid repacking non-contiguous buffers, but only when `ov::supported_properties` advertises it. Test packed IVF list/code layouts; do not enable all ports by default because the device docs warn of a performance cost.
- OpenVINO model caching is already enabled through `CACHE_DIR`. An explicit exported blob is useful only if measured startup/import improves and the cache key records OpenVINO, compiler, driver, device, shape, dtype, and compile properties. It does not improve steady-state inference by itself.
- The current NPU plugin reports one optimal request in latency mode and four in throughput mode. A small reusable request pool plus `start_async` is immediately testable for independent ADC tiles; the current single cached request and mutex deliberately serialize a shape. This is a primitive concurrency experiment, not a HETERO pipeline claim. Measure queueing, overlap, energy, and tail latency before retaining a pool.
- The in-process request map is currently unbounded and keyed by exact operation/shape/dtype strings, not every compile property. Before adding throughput variants, bound its residency and key by performance mode plus effective compile properties. `compile_on` also retries after removing optimization parameters and then turbo; expose the effective property set and fallback counter instead of silently benchmarking a weakened compile.
- An OpenVINO custom `ov::Op` supplies graph semantics, shape inference, serialization, and optional CPU evaluation. It does **not** provide a public NPU kernel implementation. GPU custom operations separately require OpenCL kernel code; ovVS keeps ANN hot loops in SYCL and the compiler/SHAVE track rather than treating `add_extension` as NPU support.

The archived Intel NPU Acceleration Library is reference material only; Intel directs users to OpenVINO/OpenVINO GenAI. Its useful confirmation is architectural: static/tiled graphs, managed DMA/cache, and quantized/mixed-precision paths. ovVS will not add it as a dependency. From OpenVINO GenAI, only the bounded scheduler, persistent compiled state, and explicit performance telemetry generalize. Token/KV/prefix caches, paged attention, and speculative decoding are LLM-specific and are not ANN index techniques.

Primary references: [Intel NPU Acceleration Library EOL](https://github.com/intel/intel-npu-acceleration-library), [OpenVINO NPU device and model caching](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html), [NPU Remote Tensor API](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device/remote-tensor-api-npu-plugin.html), [custom OpenVINO operations](https://docs.openvino.ai/2026/documentation/openvino-extensibility/custom-openvino-operations.html), and [OpenVINO GenAI continuous batching](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#continuous-batching-with-llmpipeline).

## Feed path (iGPU)

1. Dataset/graph in **USM shared**. Skip host→USM copies when the pointer is already shared.
2. GEMM is oneMKL SYCL (`float`, `sycl::half`, `int8→float`). Naive `parallel_for` is the fallback, not the product path.
3. fp32→half / fp32→int8 packing is a SYCL kernel, not a host `for`.
4. CAGRA walk is the fused SYCL kernel. Do not lower the walk to NPU.

## Mixer (`choose_device`)

AUTO:

- `topk` / `gather` → CPU
- GEMM / pairwise → CPU unless `flops ≥ large_gemm_flops` and `gemm_large.json` names NPU or GPU (Arrow Lake: CPU)
- CAGRA walk → GPU (`prim_graph_walk`)
- PQ ADC → NPU
- `FORCE_NPU` / `FORCE_GPU` still hit that device or return `DEVICE_UNAVAILABLE`

`FORCE_NPU` / `FORCE_GPU` still hit that device or return `DEVICE_UNAVAILABLE`. CPU is never reported as an NPU/GPU win.

## Hetero stage DAG (NPU ∥ iGPU)

Not a third GEMM backend. It is **stage overlap**:

- `OVVS_POLICY_HETERO` currently equals AUTO. `prim_pq_adc` is an IVF-PQ stage and `prim_graph_walk` is a CAGRA stage; merely running them concurrently would mix algorithms rather than form a same-query pipeline.
- Define the producer/consumer DAG, data ownership, and event dependencies before implementation. The first concrete candidate is NPU ADC tile *i+1* while iGPU scans/selects IVF-PQ tile *i* after B3. A CAGRA candidate-slab scoring pipeline is a later candidate after B9.
- Use bounded queues and reusable device requests/events. Report per-stage execution, queue wait, copied bytes, and overlap; a faster isolated NPU primitive is not a hetero win unless total query wall or energy improves.
- Do not ping-pong the same buffer NPU↔iGPU inside one stage. Scores stay host-visible USM; TopK is CPU.

## What not to do

- Do not send TopK or Gather to NPU/GPU on AUTO. The tables are not close.
- Do not treat NPU TOPS as a BLAS rating. TOPS are INT8 MACs behind a compiled graph.
- Do not skip memcpy by pointer identity alone (k-means/ScaNN).
- Do not put the CAGRA walk on NPU.
- Do not replace CPU GEMM with a triple loop when oneMKL is linked.
- Do not invent Lunar Lake numbers by copying this file.
