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
| PQ ADC | bounded fixed-bucket batch fallback | SHAVE C | **Batched Gather+ReduceSum when range-safe** | fused FORCE_GPU scan/select | Existing safe AUTO attempt / unsafe CPU fallback; no Arrow Lake NPU promotion |
| Hamming / Lp | pairwise graph | AVX | GreaterEqual/Xor or Abs/Power | SYCL | FORCE_* honest; AUTO follows size |

Search-shaped GEMM is `nq × n × dim` with **B = dataset** (large, immutable after build) and **A = queries** (small, changes). That geometry permits sticky-B reuse, but on Arrow Lake it is not a wall-time win: CPU remains faster after graph/DMA costs. The bakeoff `large` shape (`1e5 × 32 × 768`) is the inverse (huge A); `ovvs_bakeoff search` is `32 × 1e5 × 768`.

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
  ADC                 → NPU Gather+ReduceSum only when the LUT accumulation bound is safe; CPU fallback otherwise
  refine              → CPU

CAGRA search
  graph walk          → iGPU SYCL (work-group/query; SIFT1M recall-closeness gate passed, QPS open)
  candidate scoring inside the walk → fused in the SYCL kernel

CAGRA build
  NN-Descent init n>4096 → iGPU SYCL under AUTO (90.444 s / 90.36% of clean SIFT1M median build; synchronization/readback is first target)
  prune               → CPU until T13.3 (9.034 s clean SIFT1M median; full FORCE_GPU build is unavailable)
  attribution         → success-only V1 stages plus call-local GPU lifecycle counters; external API wall remains the end-to-end measure

Vamana / NN-Descent
  NN-Descent n>4096  → iGPU SYCL bounded NEW/OLD forward+reverse join (SIFT1M CAGRA-vs-hnswlib closeness passed; T12.5 IVF-PQ comparison/prune open)
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
7. Fail closed when conservative GEMM/TopK/PQ-ADC bounds or outputs reach FP16 range (65,504). Unsafe AUTO PQ ADC still uses CPU. FORCE_NPU may affine-center each LUT subspace and scale its total span to 60,000 only when the scale is at least 0.5; wider transforms remain unavailable. Every request-owned output is validated before bias restoration and whole-call publication.

Remaining NPU boundary: after fingerprint-sticky inputs skip the hot host memcpy, wall still remains about `45–54 ms` versus `5.3 ms` of profiled DPU work on 1e5×32×768. Corrected direct Level Zero did not remove graph/DMA, input expansion, full score production, or readback and lost reused OpenVINO at both sizes and depths 1/2/4 when refill and output consumption were included. Two 524K prefilled-only depth-2/4 cells slightly favored direct but are not end-to-end gains. A roughly 150 MB f16 operand cannot remain resident in the 4 MB scratchpad. Remote L0 tensors may reduce an application copy or duplicate request buffer on another runtime, but Arrow Lake index-home work is parked. Unsigned SHAVE ELF inject remains unsupported; ActShave already runs inside compiler graphs.

### Retained runtime opportunities and boundaries

The current measured Windows build links OpenVINO **2025.3.0** with Intel NPU driver **32.0.100.4841**. Its installed headers already expose NPU `ZeroContext`, `create_l0_host_tensor`, shared native-buffer wrapping, and mapped-file tensors. OpenVINO 2026 adds broader documented paths such as CPU virtual-address import and optional strided IO. All remain experiments until ovVS probes and measures them:

- B7 is parked on Arrow Lake after the corrected direct-Level-Zero refill+execute+consume loss. If a future SKU/runtime reopens it, compile a minimal `create_l0_host_tensor`/shared-buffer probe, validate lifetime and correctness, and compare application copies, allocated bytes, cold wall, and hot wall against request-owned tensors. A remote tensor is not evidence that the NPU scratchpad retains the full index between inferences.
- Optional strided NPU IO can avoid repacking non-contiguous buffers, but only when `ov::supported_properties` advertises it. Test packed IVF list/code layouts; do not enable all ports by default because the device docs warn of a performance cost.
- OpenVINO model caching is already enabled through `CACHE_DIR`. An explicit exported blob is useful only if measured startup/import improves and the cache key records OpenVINO, compiler, driver, device, shape, dtype, and compile properties. It does not improve steady-state inference by itself.
- A current-stack direction probe reported one optimal request in latency mode and four in throughput mode, but refill+execute+consume walls and joined composition measurements made the Arrow Lake directions non-promotable. The synchronous depth-one correctness path remains; depth 2/4 and request-cache expansion are parked.
- The in-process request map is currently unbounded and keyed by exact operation/shape/dtype strings, not every compile property. If a future SKU reopens throughput variants, first bound residency, include effective compile properties in the key, and expose property fallback counters; never benchmark a silently weakened compile.
- An OpenVINO custom `ov::Op` supplies graph semantics, shape inference, serialization, and optional CPU evaluation. It does **not** provide a public NPU kernel implementation. GPU custom operations separately require OpenCL kernel code; ovVS keeps ANN hot loops in SYCL and the compiler/SHAVE track rather than treating `add_extension` as NPU support.

The archived Intel NPU Acceleration Library is reference material only; Intel directs users to OpenVINO/OpenVINO GenAI. Its useful confirmation is architectural: static/tiled graphs, managed DMA/cache, and quantized/mixed-precision paths. ovVS will not add it as a dependency. From OpenVINO GenAI, only the bounded scheduler, persistent compiled state, and explicit performance telemetry generalize. Token/KV/prefix caches, paged attention, and speculative decoding are LLM-specific and are not ANN index techniques.

The B3/B6/B7 escape-route experiments were software-only and required no BIOS or firmware change. Resizable BAR and firmware power tuning have no measured ovVS benefit on this machine and remain excluded. The joined-stage and refill+execute+consume evidence parks further Arrow Lake Q/T, direct-Level-Zero, remote-index-home, and nonzero-NPU-partition work; future authoritative device capability plus a controlled end-to-end benchmark may reopen them.

Current B3 evidence is in `tables/arrow-lake/ivfpq-b3.md`. The fail-closed guard preserves bounded SIFT-prefix AUTO/FORCE_CPU recall parity. Persistent list-major CSR removes per-list code repacking, and bounded query/list descriptors feed synchronous fixed-bucket NPU graphs while preserving candidate order and whole-search publication. Unfiltered search avoids the dense 8-byte candidate-ID payload and resolves only `krefine` offsets. Lifecycle tests keep `IPQ1` v1 byte-compatible and extend transactional. The forced affine transform is correctness-only: the scaled NPU primitive remains 73–106× slower than CPU in current diagnostic shapes. A fused FORCE_GPU path scans persistent codes, filters, and selects without a candidate-sized score array; strict parity, tail, atomicity, lifecycle, and concurrent-resource tests pass.

Complete-call V1 telemetry attributes successful search wall and explicit GPU work. At bounded `nprobe=8`, FORCE_CPU reached 32,074 QPS and spent 73.6% of native wall constructing LUTs. FORCE_GPU reached 2,284 QPS and spent 45.5% in fused ADC plus 40.0% in final per-query TopK, while issuing 99 allocations, 120 logical kernel submissions, and 230 waits per call. AUTO remains on CPU. The subsequent certified residual-PQ Q/T implementation preserved bounded correctness but the best joined-stage path was 1.93×/1.79× slower than the direct scalar oracle at 131K/524K rows. That synthetic scope excludes coarse assignment and exact-vector refinement; Q/T is parked. Active B3 work is hierarchical iGPU top-k with one global merge/readback, packed AoSoA 4/6/8-bit codes, direct residual FP32 or validated-FP16 SLM LUTs, a workspace-byte planner, and batched/fused exact refinement with bounded persistent workspaces. Repeated matched SIFT1M wall, recall, tails, bytes, submissions, waits, peak memory, and energy remain the promotion gate.

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
- IVF-PQ ADC → the existing range-safe NPU attempt with unsafe CPU fallback on AUTO; fused iGPU scan/select is FORCE_GPU-only, and no Arrow Lake NPU expansion is active until a complete per-SKU table wins
- `FORCE_NPU` / `FORCE_GPU` still hit that device or return `DEVICE_UNAVAILABLE`

`FORCE_NPU` / `FORCE_GPU` still hit that device or return `DEVICE_UNAVAILABLE`. CPU is never reported as an NPU/GPU win.

## Hetero stage DAG (CPU + iGPU on Arrow Lake)

Not a third GEMM backend. It is **stage overlap**:

- `OVVS_POLICY_HETERO` currently equals AUTO. Arrow Lake NPU+iGPU composition is parked because every measured nonzero NPU partition lost the complete experimental composition scope; `prim_pq_adc` and `prim_graph_walk` also belong to different algorithms and cannot be called a pipeline.
- The active design is size-/latency-/deadline-gated CPU+iGPU work over resident shared-USM state: CPU handles control and measured small work, while bounded iGPU batches perform scan/walk/select and publish once.
- Define producer/consumer ownership and event dependencies before implementation. Use bounded queues and reusable workspaces/events; report execution, queue wait, copied bytes, submissions, peak memory, overlap, p50/p99, and energy.
- Future SKUs may add an NPU producer only after its isolated primitive and joined pipeline both win. Do not ping-pong the same buffer between devices inside one stage.

## What not to do

- Do not send TopK or Gather to NPU/GPU on AUTO. The tables are not close.
- Do not treat NPU TOPS as a BLAS rating. TOPS are INT8 MACs behind a compiled graph.
- Do not skip memcpy by pointer identity alone (k-means/ScaNN).
- Do not put the CAGRA walk on NPU.
- Do not replace CPU GEMM with a triple loop when oneMKL is linked.
- Do not invent Lunar Lake numbers by copying this file.
