# AGENTS.md — ioVS

Intel NPU + iGPU vector search library. The goal is a **complete NVIDIA cuVS equivalent** on Intel client hardware (Core Ultra NPU + Arc iGPU), not a research toy and not a CPU FAISS wrapper.

Canonical plan (read before coding):

`.claude/plans/2026-08-20-iovs-intel-cuvs-equivalent.md`

That document is the spec: architecture, device policy, feature matrix, and sequenced tasks T0.x–T23.x. If this file and the plan disagree, **the plan wins** until someone updates both.

---

## Mission for agents

- Implement ioVS to cuVS feature parity (brute-force, IVF-Flat, IVF-PQ, IVF-RaBitQ, CAGRA, NN-Descent, Vamana, ScaNN, all-neighbors, HNSW export, filters, dynamic batching, k-means, SLINK, spectral, quantizers, PCA, pairwise, k-selection, C/C++/Python/Rust/Go/Java).
- **Do not drop a feature because the NPU cannot express it.** Use the punch-through ladder (OpenVINO graph → compiler rewrite → SHAVE kernel → HostCompile tiles → iGPU SYCL → CPU last). CPU-only for a hot loop is a defect unless the bakeoff proves it is fastest **and** a GPU/NPU path still exists for large batch.
- **NPU wherever it lifts**, else iGPU. Record that decision in `tables/<sku>/`. Never hard-code folklore without a bakeoff file.
- Custom kernels are in-scope: SYCL, oneMKL, Level Zero, OpenVINO graphs, `npu_compiler` SHAVE/DPU, HostCompile.

---

## Architecture (short)

```
bindings → C ABI (libiovs) → C++ algorithms → mixer/planner
  → iovs::prim (gemm, dist, topk, gather, ...)
    → npu | gpu | cpu backends
  → iovs::rt (Level Zero NPU + GPU, OpenVINO, USM, blob cache)
```

- Algorithms call `iovs::prim`, never OpenVINO or SYCL directly.
- Default tensor home: USM shared (CPU + iGPU). NPU sees tiles/bound host buffers, not a second index copy as source of truth.
- Graph ANN (CAGRA/Vamana/NN-Descent walk) lives on **iGPU**. NPU may score a padded candidate slab if bakeoff says so.
- Dense/static GEMM, coarse IVF, PQ ADC tables, k-means assignment, binary/RaBitQ GEMM: **NPU first**.

Details, knobs, and CAGRA mapping: the plan §4–§6.

---

## Task order

Work the plan’s phases in sequence. Critical path:

1. Phase 0–1: repo, probe, Resources, NPU MatMul, SYCL USM
2. Phase 2–5: bakeoff + primitives (gemm, topk, gather, …)
3. Phase 6: brute-force + Python (first user-visible)
4. Phase 7–11: quantizers, k-means, IVF family
5. Phase 12–14: NN-Descent, CAGRA, HNSW export
6. Rest as specified

Do not start CAGRA before `prim::gemm`, `prim::topk`, and `prim::gather` exist.

Phase 23 (npu_compiler / SHAVE) is a **parallel** track from Phase 4 onward, not a reason to stall iGPU kernels.

---

## Engineering rules

- C++20. C ABI is the stability boundary (`include/iovs/`). Bindings wrap C, not the reverse.
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
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
cmake --build build
ctest --test-dir build -L cpu
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
