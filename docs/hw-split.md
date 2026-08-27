# Hardware split: CPU / iGPU / NPU

This is the product rule for ovVS on Intel client SoCs. Measured on Arrow Lake Core Ultra 7 265K (NPU 3720 ~13 TOPS INT8, Xe-LPG iGPU, 8P+12E). Other SKUs re-bake `tables/<sku>/` and keep the same roles unless a table disagrees.

Goal: **put each op on the device that is actually fastest for that op**, then overlap devices only when the pipeline has independent stages. Do not send work to the NPU or iGPU as a branding exercise.

## What each device is

| Device | Native strength | Native weakness |
|---|---|---|
| **NPU** | Compiled static MatMul / Conv on DPU. INT8 full-rate, FP16 half. Weight-stationary graphs. ~4 MB SRAM scratchpad. | Launch tax. No pointer-chasing. `set_tensor(host*)` SHAVE-copies. Two live fp32 Parameters do not light INT8 MACs. Dataset does not fit SRAM. |
| **iGPU** | XMX/DPAS GEMM (FP16 and INT8 via oneMKL). SYCL graph walk (CAGRA). USM shared with CPU. Parallel convert/pack. | TopK/Gather launch tax vs CPU. Serial host fp32→half is a self-inflicted stall. Shared DRAM, not HBM. |
| **CPU** | Tiny GEMM, TopK, Gather, control plane, heaps, HostCompile, k-means reduction, filters. AVX-512 / oneMKL `cblas_sgemm`. | Naive triple-loop GEMM is not a device (we do not ship that). Large dense GEMM loses to NPU once the feed path is L0 `get_tensor`. |

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

CAGRA / Vamana / NN-Descent walk
  graph walk          → iGPU SYCL (this is the iGPU search win; NPU cannot pointer-chase)
  candidate scoring inside the walk → fused in the SYCL kernel

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

Remaining NPU gap: cold wall vs DPU (`54 ms` vs `5.3 ms` on 1e5×32×768) is the first memcpy of A into L0. Closing it means the **canonical dataset lives in an NPU L0 tensor** (remote `create_l0_host_tensor`), not a second copy. Unsigned SHAVE ELF inject is still unsupported; ActShave already runs inside compiler graphs.

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

## Hetero (NPU ∥ iGPU)

Not a third GEMM backend. It is **stage overlap**:

- Independent queries in a batch: NPU can score query *i+1* while iGPU walks query *i* (CAGRA). Not wired yet; `OVVS_POLICY_HETERO` currently equals AUTO.
- Do not ping-pong the same buffer NPU↔iGPU inside one stage. Scores stay host-visible USM; TopK is CPU.

## What not to do

- Do not send TopK or Gather to NPU/GPU on AUTO. The tables are not close.
- Do not treat NPU TOPS as a BLAS rating. TOPS are INT8 MACs behind a compiled graph.
- Do not skip memcpy by pointer identity alone (k-means/ScaNN).
- Do not put the CAGRA walk on NPU.
- Do not replace CPU GEMM with a triple loop when oneMKL is linked.
- Do not invent Lunar Lake numbers by copying this file.
