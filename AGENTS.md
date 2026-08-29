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
- Graph ANN (CAGRA/Vamana/NN-Descent walk) lives on **iGPU**. Arrow Lake NPU candidate-slab offload is parked under the zero-capacity decision; no retained artifact claims a CAGRA-specific candidate-slab benchmark. Another SKU may reopen it through a new bakeoff.
- Dense GEMM / coarse IVF assign / k-means assign: **CPU oneMKL** on Arrow Lake (bakeoff). PQ ADC retains its existing range-safe AUTO attempt, unsafe CPU fallback, and FORCE_NPU numeric-correctness coverage, but no Arrow Lake NPU expansion is active after direct-Level-Zero refill+execute+consume, NPU+iGPU composition, and Q/T joined-stage losses. Graph walk: **iGPU**. NPU Parameter MatMul remains for FORCE_NPU and SKUs whose `gemm_large.json` names NPU.

**Device split (canonical):** `docs/hw-split.md`. Arrow Lake AUTO GEMM / TopK / Gather → CPU; CAGRA walk → iGPU. Existing range-safe PQ ADC may still attempt NPU before CPU fallback, while the fused iGPU and bounded FORCE_NPU paths remain honest forced-device coverage; none is a measured AUTO winner. `FORCE_*` never silently falls back.

Details, knobs, and CAGRA mapping: the plan §4–§6.

---

## NPU contract (do not forget)

The DPU is **not** broken. Arrow Lake 265K `PERF_COUNT`: MatMul is **21 µs** on DPU for 64³ and **5.3 ms** for `1e5×32×768`. Historical 0.5–170 ms walls included SHAVE copies, Level Zero work, and new-request overhead. Reused requests and request-owned L0 buffers are mandatory but insufficient: corrected direct-Level-Zero refill+execute+consume still loses reused OpenVINO. Table: `tables/arrow-lake/npu-gemm-dpu-vs-wall.md`; disposition: `tables/arrow-lake/npu-igpu-escape-routes.md`.

Intel added an **inference engine** (compiled graph, weight-stationary, ~4 MB scratchpad), not cuBLAS.

1. **Compile once, reuse `InferRequest`.** Never `create_infer_request` on the hot path.
2. **Feed L0 tensors via `get_input_tensor` / `get_output_tensor`.** The plugin allocates Level Zero buffers at request creation. `memcpy` into those. `set_tensor` on a malloc/USM host pointer **forces a SHAVE copy** (intel_npu README). That copy is the 35 ms “SHAVE A” in `PERF_COUNT`, not a broken DPU.
3. **Reuse the request’s L0 buffers.** `memcpy` unless `nbytes ≥ 64 KiB` and a 32-sample fingerprint matches (search dataset). Pointer-identity alone is unsafe: k-means/ScaNN mutate centroids. Scratchpad is **4 MB** (`1e5×768` f16 ≈ 150 MB does not fit).
4. **INT8 is NNCF Low Precision IR** (FakeQuantize on activations **and Constant weights**). Raw `si8` MatMul is illegal. Two live fp32 Parameters will not light INT8 MACs. TOPS on this SKU are INT8; FP16 is half-rate. Compile with `NPU_TURBO` + `optimization-level=2 performance-hint-override=latency`.
5. **Device split:** `docs/hw-split.md`. AUTO dense GEMM / TopK / Gather → CPU oneMKL on Arrow Lake (GEMM: 18 ms CPU vs 45 ms NPU vs 221 ms GPU at 1e5×32×768). Graph walk → iGPU. Existing range-safe PQ ADC may still attempt NPU; unsafe AUTO uses CPU and forced routes stay honest. Corrected direct-Level-Zero lost every refill+execute+consume cell, every nonzero NPU+iGPU partition lost its joined experimental scope, and certified Q/T lost its joined stage, so Arrow Lake receives zero active NPU optimization capacity. Keep correctness paths and immutable evidence, but do not manufacture a nonzero share. NPU DPU is still 5.3 ms; graph/DMA/tile movement makes wall lose to MKL.
6. **Do not bake B as an IR Constant just to be clever.** That made DPU 10× faster and wall 10× *slower* (UMD/constant reload). Sticky Parameter + reused request + L0 `get_tensor` is the path.
7. **Fail closed on numeric range.** Arrow Lake f32 graph IO can exhibit FP16-range saturation. GEMM and TopK reject unsafe bounds. PQ ADC uses raw LUTs below `sum_m max_code(abs(lut[m, code])) < 65,504`; FORCE_NPU may subtract each subspace minimum, scale the remaining total span to 60,000, and restore the bias only when the scale is at least 0.5. Non-finite values, wider transforms, or invalid outputs return `DEVICE_UNAVAILABLE` without publication. AUTO deliberately retains the unsafe-range CPU fallback.

If a change “uses the NPU” but creates a request per call, `set_tensor`s host pointers, or DMA’s the dataset every query, it is a defect.

---

## Task order

Work the plan’s phases in sequence. **v0.2.0 already has C ABI symbols through the plan’s algorithm list** (brute, IVF family, CAGRA, NN-Descent, Vamana, ScaNN, HNSW export, cluster, quantizers, pairwise). Do not scaffold more names.

`prim::gemm`, `prim::topk`, and `prim::gather` exist. Remaining work is **accelerated quality and scale**, not first implementations. Current critical path (2026-08-29, `.claude/backlog.md`):

1. The clean same-index SIFT1M curve is complete. Cached-worst tracking, prefix-only publication sorting, canonical 64-ID frontier tiling, and exact subgroup min-location selection for multi-pick search are promoted with repeated complete-wall and bitwise evidence. The graph reaches 0.9609 recall at `itopk=64/search_width=2` and 3,243.5 QPS, versus hnswlib 0.9593 and 6,784.5 QPS; batch one is still much worse. The exact stable one-pass pick was measured and rejected. The proposed four-workgroup owner/helper route is also parked on Arrow Lake after its exact 128-thread/2,640-byte named-kernel gate reported one cooperative workgroup versus four required and launched nothing. Preserve the current one-workgroup path; do not add cross-workgroup synchronization without a materially changed capability result. Never weaken hnswlib settings. Evidence: `tables/arrow-lake/cagra-cooperative-pick-v1.md` and `tables/arrow-lake/gpu-root-group-canary-v1.md`.
2. The frozen CAGRA build telemetry ABI and six clean DLL-bound measurements are complete. At SIFT1M the median 100.09-second build spends 90.44 seconds (90.36%) in GPU NN-Descent and 9.03 seconds in host optimize/prune/merge; all runs hit the six-iteration cap. Three bounded synchronization designs are parked: fixed-`N` device-resident active counts regressed complete build 8.91%; a convergence-only single-work-group reduction regressed it 2.14% despite 27.27% fewer D2H bytes; and target-owned `heads` reset regressed it 5.55% despite 17.82% fewer submissions. Preserve dynamic active-count launches, bulk head fills, and the original convergence path. Next chain producer completion into the scalar active-count copy without changing consumer work, then run a fresh exact parent/candidate gate. Evidence: `tables/arrow-lake/nndescent-active-count-v1.md`, `tables/arrow-lake/nndescent-convergence-reduction-v1.md`, and `tables/arrow-lake/nndescent-heads-reset-v1.md`.
3. Move graph optimize/prune/merge to the iGPU only after the initializer experiment, or in a separately measured branch. The stock-hnswlib export is valid and fully reachable, but its measured SIFT100K build and nearest comparable-recall search both lost; do not tune it as the active acceleration route.
4. Continue IVF-PQ with hierarchical iGPU top-k plus one global merge/readback, packed AoSoA 4/6/8-bit codes, direct residual FP32 or validated-FP16 SLM LUTs, a workspace-byte planner, and batched/fused exact refinement with bounded persistent workspaces. Q/T and Arrow Lake NPU request-depth work are retained negative evidence, not active tasks.
5. Add latency-/throughput-/deadline-aware dynamic batching, size-gated CPU+iGPU composition, nonnegative squared-L2 early abandonment, and a held-out multi-objective per-SKU tuner. Correct and accelerate RaBitQ without assuming Arrow Lake NPU promotion.
6. In parallel, repeat/interleave the CAGRA curve for any promotion candidate, finish the other B1 algorithms/policies and the real 100K×768 corpus. Bake off Lunar Lake only when that hardware exists, then continue v1.0/v1.1 work in plan order.

Phase 23 (npu_compiler / SHAVE) is parked on Arrow Lake after the public gate found the real MoviTools/source/descriptor boundary. ActShave already runs inside compiler graphs, but that is not a public custom-kernel path. Reopen only with the missing public toolchain inputs or a new SKU/runtime capability; algorithms continue on iGPU/CPU.

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
