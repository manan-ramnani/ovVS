# AGENTS.md — ovVS

Intel NPU + iGPU vector search library. The goal is a **complete NVIDIA cuVS equivalent** on Intel client hardware (Core Ultra NPU + Arc iGPU), not a research toy and not a CPU FAISS wrapper.

Canonical plan (read before coding):

`.claude/plans/2026-08-20-ovvs-intel-cuvs-equivalent.md`

That document is the spec: architecture, device policy, feature matrix, and sequenced tasks T0.x–T23.x. If this file and the plan disagree on design, **the plan wins** until someone updates both.

Remaining work (what is still open vs v1.0 “Accelerated”): `.claude/backlog.md`. If this file or the plan disagrees with the backlog on *status*, the backlog wins until both are updated. Do not add more C ABI algorithm names — v0.2 already has them.

---

## Mission for agents

- Implement ovVS to cuVS feature parity (brute-force, IVF-Flat, IVF-PQ, IVF-RaBitQ, CAGRA, NN-Descent, Vamana, ScaNN, all-neighbors, HNSW export, filters, dynamic batching, k-means, SLINK, spectral, quantizers, PCA, pairwise, k-selection, C/C++/Python/Rust/Go/Java).
- **Do not drop a feature because the NPU cannot express it.** Use the punch-through ladder (OpenVINO graph → compiler rewrite → SHAVE kernel → HostCompile tiles → iGPU SYCL → CPU last). CPU-only for a hot loop is a defect unless the bakeoff proves it is fastest **and** a GPU/NPU path still exists for large batch.
- **NPU wherever it lifts**, else iGPU. Record that decision in `tables/<sku>/`. Never hard-code folklore without a bakeoff file.
- Custom kernels are in-scope: SYCL, oneMKL, Level Zero, OpenVINO graphs, `npu_compiler` SHAVE/DPU, HostCompile.

---

## Architecture (short)

```
bindings → C ABI (libovvs) → C++ algorithms → mixer/planner
  → ovvs::prim (gemm, dist, topk, gather, ...)
    → npu | gpu | cpu backends
  → ovvs::rt (Level Zero NPU + GPU, OpenVINO, USM, blob cache)
```

- Algorithms call `ovvs::prim`, never OpenVINO or SYCL directly.
- Default tensor home: USM shared (CPU + iGPU). NPU sees tiles/bound host buffers, not a second index copy as source of truth.
- Graph ANN (CAGRA/Vamana/NN-Descent walk) lives on **iGPU**. NPU may score a padded candidate slab if bakeoff says so.
- Dense GEMM / coarse IVF assign / k-means assign: **CPU oneMKL** on Arrow Lake (bakeoff). PQ ADC: range-safe LUTs may use NPU; FORCE_NPU can affine-center and scale a bounded unsafe span, while AUTO keeps the CPU fallback until the complete search wins. Graph walk: **iGPU**. NPU Parameter MatMul is kept for FORCE_NPU and for SKUs whose `gemm_large.json` names NPU.

**Device split (canonical):** `docs/hw-split.md`. AUTO GEMM / TopK / Gather → CPU (oneMKL `cblas_sgemm` beats NPU wall and iGPU XMX on Arrow Lake). CAGRA walk → iGPU. Range-safe PQ ADC may use NPU; unsafe AUTO tables use CPU, while bounded FORCE_NPU affine execution runs or fails unavailable. `FORCE_*` never silently falls back.

Details, knobs, and CAGRA mapping: the plan §4–§6.

---

## NPU contract (do not forget)

The DPU is **not** broken. Arrow Lake 265K `PERF_COUNT`: MatMul is **21 µs** on DPU for 64³ and **5.3 ms** for `1e5×32×768`. Wall times of 0.5–170 ms are SHAVE copies + Level Zero + **new InferRequest per call**. Table: `tables/arrow-lake/npu-gemm-dpu-vs-wall.md`.

Intel added an **inference engine** (compiled graph, weight-stationary, ~4 MB scratchpad), not cuBLAS.

1. **Compile once, reuse `InferRequest`.** Never `create_infer_request` on the hot path.
2. **Feed L0 tensors via `get_input_tensor` / `get_output_tensor`.** The plugin allocates Level Zero buffers at request creation. `memcpy` into those. `set_tensor` on a malloc/USM host pointer **forces a SHAVE copy** (intel_npu README). That copy is the 35 ms “SHAVE A” in `PERF_COUNT`, not a broken DPU.
3. **Reuse the request’s L0 buffers.** `memcpy` unless `nbytes ≥ 64 KiB` and a 32-sample fingerprint matches (search dataset). Pointer-identity alone is unsafe: k-means/ScaNN mutate centroids. Scratchpad is **4 MB** (`1e5×768` f16 ≈ 150 MB does not fit).
4. **INT8 is NNCF Low Precision IR** (FakeQuantize on activations **and Constant weights**). Raw `si8` MatMul is illegal. Two live fp32 Parameters will not light INT8 MACs. TOPS on this SKU are INT8; FP16 is half-rate. Compile with `NPU_TURBO` + `optimization-level=2 performance-hint-override=latency`.
5. **Device split:** `docs/hw-split.md`. AUTO dense GEMM / TopK / Gather → CPU oneMKL on Arrow Lake (18 ms vs NPU 45 ms vs GPU 221 ms at 1e5×32×768). Graph walk → iGPU. Range-safe ADC may use NPU; unsafe AUTO ADC falls back to CPU. The forced centered/scaled ADC lane is correctness evidence, not an AUTO win. NPU DPU is still 5.3 ms; Parameter DMA is why wall loses to MKL.
6. **Do not bake B as an IR Constant just to be clever.** That made DPU 10× faster and wall 10× *slower* (UMD/constant reload). Sticky Parameter + reused request + L0 `get_tensor` is the path.
7. **Fail closed on numeric range.** Arrow Lake f32 graph IO can exhibit FP16-range saturation. GEMM and TopK reject unsafe bounds. PQ ADC uses raw LUTs below `sum_m max_code(abs(lut[m, code])) < 65,504`; FORCE_NPU may subtract each subspace minimum, scale the remaining total span to 60,000, and restore the bias only when the scale is at least 0.5. Non-finite values, wider transforms, or invalid outputs return `DEVICE_UNAVAILABLE` without publication. AUTO deliberately retains the unsafe-range CPU fallback.

If a change “uses the NPU” but creates a request per call, `set_tensor`s host pointers, or DMA’s the dataset every query, it is a defect.

---

## Task order

Work the plan’s phases in sequence. **v0.2.0 already has C ABI symbols through the plan’s algorithm list** (brute, IVF family, CAGRA, NN-Descent, Vamana, ScaNN, HNSW export, cluster, quantizers, pairwise). Do not scaffold more names.

`prim::gemm`, `prim::topk`, and `prim::gather` exist. Remaining work is **accelerated quality and scale**, not first implementations. Current critical path (2026-08-28, `.claude/backlog.md`):

1. B1 harness core and the current-code matched SIFT1M CAGRA gate are complete; full curves/energy and the real 100K×768 corpus remain open.
2. B2/B5 pass the SIFT1M recall-closeness gate (0.9036 versus hnswlib 0.8915), but ≥0.95 recall, CAGRA QPS, NN-Descent convergence, and GPU prune remain open.
3. **Next implementation:** IVF-PQ persistent CSR, fixed ADC buckets, the bounded affine NPU lane, shortlist-only IDs, fused FORCE_GPU scan/select, and complete-call telemetry are complete. At bounded `nprobe=8`, LUT construction is 73.6% of CPU wall; fused ADC plus final per-query TopK is 85.5% of GPU wall. Continue B3 with certified exact Q/T factorization, a workspace-byte block budget, persistent GPU workspaces, and batched/fused refinement. Bound/version the NPU request cache before any depth 1/2/4 experiment, and require joined end-to-end wall before giving NPU Q work capacity. Then correct and accelerate RaBitQ (B4). Keep this software-only; do not require BIOS changes.
4. Continue CAGRA throughput work only from profiling evidence; never weaken hnswlib settings.
5. HETERO stage overlap (B6); NPU L0 dataset-home experiment (B7); Lunar Lake bakeoff when hardware exists (B8)
6. Then plan v1.0 polish (B9–B20). Vamana/ScaNN/SLINK/bindings are v1.1 (B21+) — do not start them ahead of B2

Phase 23 (npu_compiler / SHAVE) is a **parallel** track, not a reason to stall iGPU kernels. Unsigned SHAVE inject is parked after a real attempt; ActShave already runs inside compiler graphs.

---

## Engineering rules

- C++20. C ABI is the stability boundary (`include/ovvs/`). Bindings wrap C, not the reverse.
- SYCL code compiled with Intel DPC++ (`icpx`) Level Zero backend. Host may be MSVC or clang.
- OpenVINO NPU plugin for NPU graphs. Pin compiler/driver versions; cache keys must include them.
- Tests: golden vs CPU/FAISS; `FORCE_NPU` / `FORCE_GPU` / hetero; no silent compile-fail fallback without a counter and a log.
- Do not add proprietary Intel SVS blobs to the core. Reimplement ideas if needed.
- Match cuVS names for public APIs where they exist (`cagra.build`, `itopk_size`, `nprobe`, …) so ports are mechanical.
- Keep comments factual. No changelog comments. No unused feature flags as a way to skip work.
- Windows 11 and Ubuntu 24.04 are both first-class.

---

## Commands (once the tree exists)

```text
cmake -B build -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DOVVS_WITH_SYCL=ON
cmake --build build
ctest --test-dir build -L cpu
# Windows oneAPI: in one `cmd.exe` session, call `VsDevCmd.bat -arch=amd64`, then "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" and use icx (MSVC-like). icpx is GNU-like and breaks Ninja+MSVC flags.
# Fallback: intel/llvm nightly clang++ -fsycl at %USERPROFILE%\intel\sycl-nightly
tools/probe > probe.json
tools/bakeoff gemm --sku auto
```

Device tests: `-L npu`, `-L gpu`, `-L hetero`. Skip only if the device is absent; do not skip if compile fails.

---

## What not to do

- Do not replace the library with “call FAISS on CPU”.
- Do not implement only brute-force and call it done.
- Do not assume ONNX Runtime op tables mean the **NPU** compiler supports TopK/Gather — measure in bakeoff.
- Do not bounce every buffer NPU↔iGPU; the mixer must avoid ping-pong.
- Do not invent CUDA. This is SYCL + OpenVINO + SHAVE.

---

## Repo map (target)

See plan §4.8. Until those directories exist, create them as you implement the phase that needs them. Do not scaffold empty algorithm files ahead of their phase.

Update the plan’s §14 decision log if you change architecture. Do not silently fork the design.
