# ovVS: Complete cuVS Equivalent on Intel NPU + iGPU

**Status:** implementing — **v0.2.0** C ABI and Arrow Lake bakeoff tables in-tree. Algorithm *names* exist; plan §5 “Accelerated” and §16 v1.0 are **not** met. Remaining work is kernel quality + scale (not more symbols): `.claude/backlog.md`.
**Date:** 2026-08-20 (snapshot 2026-08-29)
**Non-negotiable:** feature-complete vs NVIDIA cuVS. Hard problems are sequenced, not dropped. NPU is used wherever it lifts; otherwise iGPU; CPU only as control plane or when it actually wins latency. Custom kernels, compiler patches, and SHAVE/DPU work are in-scope.

This document is the implementation spec for agents and humans. Read it before writing code.

---

## 1. Mission

Build **ovVS** (this repo): a vector-search and clustering library with the same job as [NVIDIA cuVS](https://github.com/NVIDIA/cuvs), targeting Intel client SoCs that have both:

- an **NPU** (Meteor Lake NPU 3720, Lunar Lake NPU 4000, Arrow Lake, Panther Lake NPU5, Wildcat Lake)
- an **Arc iGPU** (Xe-LPG / Xe2) programmable via SYCL and Level Zero

The library must expose a cuVS-shaped API (resources, build/search/extend/serialize, DLPack tensors, C ABI + language bindings) and implement every public algorithm class cuVS ships, not a subset.

cuVS is GPU-native. ovVS is **heterogeneous-native**. Equivalence means:

| Dimension | cuVS | ovVS |
|---|---|---|
| Algorithms | brute-force, IVF-Flat, IVF-PQ, IVF-RaBitQ, CAGRA, CAGRA-Q, NN-Descent, Vamana, ScaNN, all-neighbors, HNSW export, filtered search, dynamic batching, k-means, SLINK, spectral clustering, PCA, PQ/SQ/binary quantizers, spectral embedding, pairwise distances, k-selection | same surface |
| Build + search | GPU | NPU and/or iGPU and/or CPU, chosen per op by a cost model |
| Programming | CUDA kernels | OpenVINO graphs + SHAVE/DPU kernels + SYCL kernels + AVX-512 |
| Memory | device VRAM | shared DRAM + NPU SRAM tiles + iGPU SLM/L1/L2 |
| Bindings | C, C++, Python, Java, Rust, Go | same, via a stable C ABI |
| Interop | DLPack, CAGRA→HNSW, FAISS | DLPack, CAGRA→HNSW, FAISS/SVS import-export |

"Equivalent" does **not** mean "same kernel". CAGRA on Xe2 will not be a CUDA warp port. It means the same user-visible algorithms, the same quality knobs (`itopk_size`, `nprobe`, `graph_degree`, …), and competitive or better **on this hardware** than FAISS-CPU / hnswlib / Intel SVS, with NPU energy wins called out as first-class metrics.

---

## 2. Non-negotiables

1. **Do not drop a cuVS feature because the NPU cannot express it.** Lower it to iGPU. If iGPU is awkward, write the SYCL kernel. If SYCL is not enough, write Level Zero SPIR-V or IGC custom sequences. CPU is a last-mile correctness and tiny-batch path, not an excuse to skip acceleration.
2. **NPU first when it lifts.** Every primitive and algorithm stage has a bakeoff: NPU vs iGPU vs CPU on the target SKU. The winner is wired into the mixer. Re-run bakeoffs when drivers/compilers move.
3. **Custom kernels are expected.** OpenVINO graph ops, `npu_compiler` SHAVE runtime kernels, HostCompile CPU+NPU split, SYCL subgroups, oneMKL GEMM, IGC intrinsics — all in scope.
4. **No silent fallback that hides a missing kernel.** If NPU compile fails, log, count, and take the next device in the ladder. Tests fail if a required stage has no accelerated path on a machine that has the device.
5. **Correctness before speed, then both.** Recall vs FAISS/brute-force ground truth is a gate. Throughput, latency, and joules/query are the optimization objective.
6. **One C ABI.** All languages wrap `libovvs`. Do not reimplement algorithms per language.

---

## 3. Hardware model (what we actually program)

### 3.1 NPU

Intel client NPU is a Movidius-descended tiled ML accelerator, not a GPGPU.

- Neural Compute Engines: MAC arrays (INT8 full rate, FP16 half). MatMul/Conv are the native ops.
- SHAVE DSPs: general vector work the MAC array cannot do (gather glue, transcendentals, some reductions).
- LEON SPARC cores: command processor / RTOS, not for user kernels.
- Software-managed SRAM per tile (Meteor Lake: 2 MB/NCE). Compiler tiles DRAM↔SRAM via DMA.
- Lunar Lake: 6 engines, 12 SHAVE, 48 TOPS, 2× NPU DRAM bandwidth vs Meteor Lake, 8 MB compute-tile side cache.
- Panther Lake: NPU5, ~50 TOPS.

**Programming today**

1. OpenVINO IR → `npu_compiler` (MLIR: IE → VPU → VPUIP → ELF) → Level Zero Graph execute.
2. Static shapes. Compile once, infer many. Mutable command lists swap arguments, not control flow.
3. There is no public CUDA-like kernel language. We add one by owning SHAVE kernels inside `npu_compiler` (`sw_runtime_kernels`) and/or HostCompile (CPU SCF orchestrates NPU ELF blobs).

**NPU physics we design around**

- Launch cost is large. Small GEMMs lose to CPU. Bakeoff must include `B=1`.
- Host copy measured ~10 GB/s vs iGPU ~19 GB/s on Meteor Lake. Keep data in place; do not bounce NPU↔iGPU every hop.
- DRAM latency through NPU ~1 µs. Pointer-chasing graphs on NPU are hostile unless the working set is in SRAM.
- INT8 is the throughput native type. FP16 is fine. FP32 is a conversion tax. FP64 does not exist.

### 3.2 iGPU (the CUDA analogue)

Arc iGPU is the device that can actually run CAGRA-class kernels.

- Xe / Xe2: SIMD8 (older) or SIMD16 (Xe2). Map CUDA warp-32 teams onto subgroup-16 (two subgroups ≈ one warp).
- Shared Local Memory (SLM): 192 KB L1/SLM per Xe-core class; Lunar Lake can run work with up to ~128 KB SLM per Xe-core concurrently.
- USM shared: CPU and iGPU see the same DRAM. Use this as the default tensor home.
- XMX: matrix engines. Prefer oneMKL GEMM over hand-rolled for large dense products.
- Atomics, subgroup shuffle, barriers: available in SYCL 2020 + Intel extensions.
- Bandwidth is **shared DRAM** (LPDDR5/5x, on the order of 60–150 GB/s depending on SKU), not HBM. Algorithms that assume TB/s will be rewritten to cache-blocked / quantized forms.

**Programming:** SYCL 2020 via Intel oneAPI DPC++, Level Zero backend. OpenCL only as a diagnostic path. No CUDA.

### 3.3 CPU

- AVX-512 / AMX on the same package.
- Owns: compiler invocation, graph metadata, heaps that are tiny, `B=1` latency if bakeoff says so, HostCompile loops, correctness oracles.
- Intel SVS is a competitor on this device, not a dependency we must link. We may steal ideas (LVQ, LeanVec) and reimplement under Apache-2.0-compatible terms. Do not link proprietary SVS blobs into the core.

### 3.4 Target SKUs (lab matrix)

Must run and bake off on at least:

| Codename | NPU | iGPU | Must-pass |
|---|---|---|---|
| Meteor Lake (Core Ultra 100H/U) | 3720, ~11 TOPS | Xe-LPG | yes |
| Lunar Lake (Core Ultra 200V) | 4000, 48 TOPS | Xe2 | yes (primary) |
| Arrow Lake | 3720-class | Xe-LPG / Xe | yes |
| Panther Lake (Core Ultra 300) | NPU5, ~50 TOPS | Xe3-class | yes when hardware exists |

Windows 11 and Ubuntu 24.04 are both first-class. Drivers: Intel NPU driver + compute-runtime (Level Zero GPU) + OpenVINO matching the compiler.

---

## 4. Tentative architecture

### 4.1 Layer cake

```
+------------------------------------------------------------------+
| Bindings: Python, C++, C, Rust, Go, Java                         |
|  (all call libovvs C ABI; tensors via DLPack)                    |
+------------------------------------------------------------------+
| Public C++ API  ovvs::neighbors / cluster / preprocess / distance|
+------------------------------------------------------------------+
| Planner / Device Mixer                                           |
|  stage graph -> cost model -> device assignment -> executable    |
+------------------------------------------------------------------+
| Algorithm IR  (stages, not devices)                              |
|  e.g. CAGRA = InitGraph | OptimizeGraph | SearchWalk | DistEval  |
+------------------------------------------------------------------+
| Primitive library  ovvs::prim  (RAFT analogue)                   |
|  gemm, gemv, dist, topk, sort, scan, hist, gather, scatter,      |
|  reduce, bitset, hashmap, subgroup ops                           |
+------------------------+------------------+----------------------+
| npu::                   | gpu::            | cpu::                |
| OpenVINO graphs         | SYCL kernels     | AVX-512 / AMX       |
| SHAVE/DPU kernels       | oneMKL / XMX     | HostCompile host    |
| HostCompile tiles       | Level Zero queues|                     |
+------------------------+------------------+----------------------+
| Runtime  ovvs::rt                                                |
|  Level Zero loader, NPU UMD, GPU UMD, events, command lists,     |
|  USM, NPU graph cache, compiled-blob cache, stream/event model   |
+------------------------------------------------------------------+
| Memory  ovvs::mem                                                |
|  host DRAM (source of truth), USM shared, USM device,            |
|  NPU graph weights, tile arenas, pinned staging                  |
+------------------------------------------------------------------+
```

### 4.2 Runtime (`ovvs::rt`) — analogue of `raft::device_resources`

One `ovvs::Resources` object per thread-pool / search worker:

- Level Zero driver handles for GPU and NPU (both are L0 devices on this stack).
- One GPU command queue (compute) + copy queue if the driver exposes it.
- NPU graph command queue (plugin currently uses one queue per compiled model; we pool as OpenVINO 2026 allows).
- Event timeline so NPU GEMM and iGPU walk can overlap.
- Compiled-graph cache keyed by `(op, shapes, dtypes, arch, compiler_version)`.
- Scratch arenas per device so we do not allocate per query.

NPU and iGPU **do not share a coherent device address space**. Crossing devices is a host-visible USM/DRAM buffer plus an explicit dependency. The mixer is responsible for not ping-ponging.

### 4.3 Memory (`ovvs::mem`)

Default layout:

- Dataset, graph, IVF lists, PQ codes live in **USM shared** (iGPU + CPU).
- NPU never owns the canonical index. NPU sees **tiles**: a compiled graph with inputs bound to host pointers / L0 NPU buffers for the duration of an inference.
- Working sets that fit NPU SRAM (centroids, PQ codebooks, a padded candidate slab) are promoted to NPU-resident weights with `defer_weights_load` respected.
- iGPU SLM is kernel-private (candidate heaps, hashmaps, shared Q tiles).

Alignment: 64-byte for AVX, 128-byte for iGPU block loads, NPU compiler’s preferred layout (typically NHWC-ish for activations; we will pack GEMM as `[M,K] x [K,N]` with K multiple of 32/64 after bakeoff).

### 4.4 Primitive library (`ovvs::prim`) — the RAFT we have to write

Every primitive has three backends and a mixer entry. No algorithm talks to OpenVINO or SYCL directly.

| Primitive | NPU path | iGPU path | CPU path | Notes |
|---|---|---|---|---|
| `gemm` | OpenVINO MatMul FP16/INT8 | oneMKL SYCL GEMM / XMX | oneMKL / AVX | NPU wins large static; iGPU wins medium; CPU wins tiny |
| `gemv` | often lose (launch) | SYCL | AVX | bakeoff |
| `l2_expand` / `ip` / `cosine` | fused into GEMM graph | fused SYCL | AVX | precompute norms on home device |
| `topk` / `k_select` | TopK op if compiler accepts; else output scores to iGPU | bitonic / radix / warp-select analogue (subgroup) | `pdqselect` | k-selection is a cuVS primitive; we must own it |
| `reduce` / `norm` | ReduceSum/Max graph | SYCL | AVX | |
| `gather` / `scatter` | Gather op, static index tensor | SYCL, the real one | AVX | IVF list gather lives or dies here |
| `sort` | not native; do not force it | SYCL radix/merge | IPS4o / AVX | |
| `scan` / `hist` | SHAVE kernel if needed | SYCL | CPU | k-means counts, IVF list offsets |
| `bitset` | no | SYCL ballot/atomic | AVX | filtered search |
| `hashmap` | no | SLM + global (CAGRA visit set) | CPU | graph search |
| `popcount` / `hamming` | INT8/binary GEMM or SHAVE | SYCL | AVX-512 VPOPCNT | RaBitQ / binary quant |

**NPU kernel punch-through ladder** (mandatory order when an op is “NPU-shaped” but does not compile):

1. Express as stock OpenVINO opset (MatMul, Gather, Reduce, TopK, Convert, Eltwise).
2. Rewrite the graph (pad, transpose, split, QDQ) until `npu_compiler` accepts it.
3. Add a custom OpenVINO op and a lowering in `npu_compiler` (fork or patch; track upstream).
4. Write a SHAVE C kernel under `sw_runtime_kernels` and wire it as `VPU.SW.Kernel`.
5. HostCompile: CPU `scf.for` tiles the dynamic dimension; each tile is a static NPU ELF blob.
6. Declare NPU incapable for this stage; mixer sends it to iGPU. **This is not a feature drop.**

**iGPU kernel punch-through ladder:**

1. oneMKL / oneDPL.
2. SYCL 2020 nd-range kernel, subgroup size 16 (Xe2) or 8 (Xe-LPG), SLM explicitly sized.
3. Intel SYCL extensions (named subgroup ops, joint_matrix / XMX).
4. Level Zero + SPIR-V / IGC-level if a shuffle/atomic is missing.
5. CPU. Logged as a defect until an iGPU kernel exists, unless bakeoff proves CPU is faster **and** we still have an iGPU kernel for the large-batch case.

### 4.5 Device mixer (`ovvs::plan`)

Algorithms are **stage graphs**, not monoliths.

Example: IVF-PQ search

```
Q ──► [coarse GEMM Q x centroids] ──► [select nprobe lists]
        NPU (static, small K=nlist)        CPU/iGPU (tiny)
   ──► [gather PQ codes for probed lists] ──► [ADC table lookup + accumulate]
        iGPU (variable list len)              NPU if padded static slab;
                                              else iGPU
   ──► [topk] ──► optional [refine on raw vectors]
        iGPU/CPU         NPU/iGPU GEMM on candidates
```

Example: CAGRA search

```
Q ──► [seed] ──► loop:
        iGPU
          [expand neighbors from graph]     iGPU (pointer chase)
          [gather vectors of candidates]    iGPU (USM)
          [distance tile]                   NPU if tile >= B_npu
                                            else iGPU fused in the walk kernel
          [itopk update + hashmap]          iGPU SLM
      ──► output top-k
```

The mixer has:

- **Offline tables** per SKU, produced by the bakeoff harness (`tools/bakeoff`). Checked into `tables/`.
- **Online guards:** NPU busy (Copilot/other graphs), thermal, `turbo` hint, query batch size, whether a compiled blob exists for this shape.
- **Policy:** `NPU_IF_FASTER | GPU_IF_FASTER | HETERO | FORCE_NPU | FORCE_GPU | FORCE_CPU` on `Resources`. Tests use FORCE_* .

Rule of thumb encoded in tables, not folklore:

- Static GEMM with `M*N*K` above SKU threshold and known shapes → NPU.
- Data-dependent gather, graph walk, variable-length IVF, atomics, hashmaps → iGPU.
- `B=1` and working set in L3 → CPU often wins; still keep GPU/NPU paths for `B>=16`.

### 4.6 Algorithm IR

Each index is a C++ class plus a serialized blob.

```
ovvs::neighbors::Index
  metadata: metric, dtype, n, dim, device_affinity
  storage:  ovvs::mem::Buffer views
  stages:   vector<Stage> for build and for search
```

Serialization format (`*.ovvs`): little-endian, versioned, mmap-friendly. Must round-trip CAGRA graphs, IVF lists, PQ codebooks. Must import/export:

- CAGRA graph → HNSW (hnswlib-compatible layout, matching cuVS)
- FAISS IVF-PQ / IVF-Flat where layouts are documented
- DLPack for tensors in/out

### 4.7 Public API (cuVS-shaped)

C ABI in `include/ovvs/` (stable):

```
ovvsResourcesCreate / Destroy
ovvsBruteForceIndex* / Search
ovvsIvfFlat*
ovvsIvfPq*
ovvsIvfRabitq*
ovvsCagra*  (Build, Search, Extend, Serialize, Deserialize)
ovvsNnDescent*
ovvsVamana*
ovvsScann*
ovvsAllNeighbors*
ovvsHnswFromCagra / ovvsHnswSearch
ovvsKMeans*
ovvsSlink*
ovvsSpectral*
ovvsPq / ovvsSq / ovvsBinaryQuantizer / ovvsPca
ovvsPairwiseDistance
ovvsKSelection
ovvsFilter  (bitset)
```

C++ in `include/ovvs/*.hpp` wrapping the C ABI or vice versa (C ABI is the compatibility boundary, like cuVS).

Python: `ovvs.neighbors.cagra.build(...)` mirroring cuVS names so ports are mechanical.

### 4.8 Directory layout (target)

```
ovVS/
  AGENTS.md
  CMakeLists.txt
  .claude/plans/
  include/ovvs/                 C + C++ headers
  src/
    rt/                         Level Zero, OpenVINO, queues, cache
    mem/                        USM, arenas
    prim/
      npu/                      OpenVINO graphs, SHAVE
      gpu/                      SYCL kernels
      cpu/                      AVX
      mixer.cpp
    neighbors/                  brute, ivf_*, cagra, nndescent, vamana, scann, all_neighbors, hnsw
    cluster/
    preprocess/
    distance/
    plan/                       stage graphs, cost model
  compiler/                     optional: submodule/fork of npu_compiler patches
  python/  rust/  go/  java/  c/
  tools/
    bakeoff/                    primitive + algorithm bakeoff
    bench/                      cuvs_bench-compatible runner
  tests/
    golden/                     vs FAISS / brute force
    device/                     NPU compile, SYCL, hetero
  tables/                       checked-in bakeoff results per SKU
  third_party/                  DLPack, json, ...
```

Build: CMake, C++20, Intel DPC++ (icpx) for GPU code, MSVC/clang for host if needed. OpenVINO as a package. oneAPI oneMKL. Catch2 or GoogleTest. Python via scikit-build or pybind11 around the C ABI.

---

## 5. Feature matrix (definition of done)

Every row must reach **Accelerated** on a Lunar Lake machine. **Accelerated** means the hot loop is NPU and/or iGPU, not “CPU works”. The device columns require those paths to exist; they do not override per-SKU AUTO bakeoffs. Arrow Lake AUTO uses CPU oneMKL for coarse/dense GEMM and NPU PQ ADC only when its numeric bound is safe.

### 5.1 Neighbors

| Algorithm | cuVS role | ovVS NPU | ovVS iGPU | ovVS CPU | Done when |
|---|---|---|---|---|---|
| Brute-force | exact kNN | GEMM+topk graph | GEMM+topk kernel | AVX GEMM | recall=1, beats naive NumPy, NPU used for large NxD |
| IVF-Flat | inverted lists, full vectors | coarse GEMM path | list scan + dist | metadata + SKU winner | recall vs FAISS IVF-Flat within 1% at same nprobe |
| IVF-PQ | IVF + PQ ADC | coarse path + range-safe padded ADC | ADC + refine | train + unsafe-range fallback | recall vs FAISS IVF-PQ within 1% at same params |
| IVF-RaBitQ | binary/quant IVF | native INT/binary GEMM | scan | train | recall-QPS curve vs paper/cuVS ballpark |
| CAGRA | GPU graph ANN | optional dist tiles | walk + build | export | recall vs CAGRA/HNSW; iGPU walk fused |
| CAGRA-Q | compressed attached dataset | ADC | walk on codes | — | same API as cuVS compression flag |
| CAGRA extend | incremental insert | — | rebuild local edges | — | extend API works, recall holds |
| NN-Descent | kNN graph construction | dist tiles | iteration kernels | — | graph used as CAGRA init |
| Vamana | disk/large graph | — | build + search | — | API + recall |
| ScaNN-like | partition + anisotropic quant + refine | GEMM/quant | prune + refine | — | API + recall |
| All-neighbors | full kNN graph | brute/IVF tiles | merge | — | API |
| HNSW from CAGRA | CPU search interop | — | — | search | hnswlib-compatible serialize |
| Filtered search | bitset prefilter | — | walk honors bits | — | bitset API on CAGRA + IVF |
| Dynamic batching | gather small queries | NPU likes bigger B | persistent-ish | queue | small-q QPS up vs eager |
| Serialize | blob | — | — | mmap | round trip |

### 5.2 Clustering and preprocess

| Algorithm | NPU | iGPU | Done when |
|---|---|---|---|
| K-means | assignment GEMM | update / hist | vs cuVS/FAISS kmeans inertia band |
| SLINK | — | MST/NN-graph path | vs scipy/cuSLINK labels on small sets |
| Spectral clustering | GEMM bits | eigensolver (oneMKL) | API + tests |
| PCA | SVD/GEMM | oneMKL | vs sklearn atol |
| Product quantizer | train assist | k-means on subspaces | used by IVF-PQ |
| Scalar quantizer | convert graph | kernel | roundtrip error bound |
| Binary quantizer | packed GEMM | hamming | API |
| Spectral embedding | — | eigensolver | API |
| Pairwise distances | GEMM | kernel | metric matrix matches CPU |
| K-selection | TopK op | kernel | matches `np.argpartition` |

### 5.3 Metrics

Implement at least: `L2Expanded`, `L2SqrtExpanded`, `InnerProduct`, `CosineExpanded`, `BitwiseHamming`, `LpUnexpanded` (Minkowski via `metric_arg`). Others from cuVS are queued in the same enum and implemented when a caller needs them — do not delete enum entries.

### 5.4 Bindings

C ABI first. Then C++ header-only/inline wrappers. Then Python. Then Rust, Go, Java. Feature flags in CI: `OVVS_BINDINGS=python` etc. A binding is not done until one algorithm (brute-force search) round-trips in that language.

---

## 6. CAGRA on Xe2 (hardest kernel — we still do it)

Porting CUDA CAGRA blindly will lose. Redesign for subgroup-16 and DRAM.

### 6.1 Build

1. **Initial graph**
   - IVF-PQ path: reuse ovVS IVF-PQ (SKU-selected coarse assignment + iGPU/NPU ADC), batched queries of the dataset against itself in chunks.
   - NN-Descent path: iGPU kernels (see §7 phase 12).
   - Iterative CAGRA-search path: after search kernel exists.
2. **Optimize/prune** on iGPU: each vertex’s `I` neighbors → reverse edges → prune to degree `G`. Sort and unique in SLM if `I` fits, else global.
3. Graph stored as `uint32[N, G]` in USM, optionally degree-padded.

NPU during build: retain large-GEMM paths for SKUs whose bakeoff selects them. Arrow Lake AUTO uses CPU oneMKL for IVF-PQ assignment and batched dense scores.

### 6.2 Search kernel (iGPU)

Map cuVS knobs:

| cuVS | ovVS iGPU |
|---|---|
| `team_size` 8/16/32 | subgroup 8 or 16; 32 = two subgroups cooperating |
| `itopk_size` | SLM array per query |
| `search_width` | number of vertices expanded per iteration |
| `SINGLE_CTA` | one work-group per query (latency) |
| `MULTI_CTA` | many work-groups per query (large itopk) |
| hashmap | SLM bloom + global fallback |
| persistent kernel | L0 queue + submitted batches; true persistent is a later stretch |

**Fused distance in the walk** is the default (one kernel, less DRAM). **NPU distance offload** is a second implementation: iGPU produces a padded `[Q_tile, Cands, D]` slab, NPU MatMul, scores return. Enable only when bakeoff says the slab is large enough to beat fused iGPU. Keep both.

### 6.3 Filtered search

Bitset in USM. On expand, skip edges whose bit is 0. Increase `itopk_size` from `filtering_rate` the same way cuVS documents.

### 6.4 Extend

Local NN for new points (brute or IVF against dataset), insert vertices, repair edges. iGPU. Correctness tests: search recall after many extends vs rebuild.

---

## 7. Exhaustive task sequence

Do these in order. A later phase may start on a second machine only when its prerequisites say so. Each task has **exit criteria**. If a task is blocked on hardware, keep a CPU oracle so software above can proceed, but the task stays **open** until the accelerated path exists.

Legend: `[P]` prerequisite, `[N]` NPU, `[G]` iGPU, `[C]` CPU, `[H]` hetero, `[K]` custom kernel work.

---

### Phase 0 — Repo, toolchain, lab

**Goal:** any engineer or agent can build, detect devices, and run a no-op test.

- **T0.1** CMake skeleton, C++20, `icpx` + host compiler, sanitizers (host), clang-format, pre-commit. Exit: `cmake --build` hello world.
- **T0.2** Dependency pin: OpenVINO (NPU plugin), oneAPI DPC++/oneMKL, Level Zero loader, Intel NPU driver, compute-runtime. Document exact versions in `docs/toolchain.md` (create when pinning). Exit: version table for Win/Linux.
- **T0.3** Device probe tool `tools/probe`: list L0 GPU, L0 NPU, OpenVINO devices, SRAM/GOPS properties, driver versions. Exit: JSON dump on MTL and LNL.
- **T0.4** CI: CPU-only unit tests on any runner; device tests tagged `npu` / `gpu` / `hetero` and skipped if missing. Exit: GitHub Actions or local script.
- **T0.5** Dataset kit: SIFT1M, gist, a 768-d embedding set (e.g. dbpedia-openai subset), random fp16/int8. Download script. Exit: `python tools/data/fetch.py` fills `data/`.
- **T0.6** License Apache-2.0, SECURITY, CONTRIBUTING pointers (keep short).

**Do not start algorithms before T0.3 works on at least one NPU machine.**

---

### Phase 1 — Runtime and memory

**Goal:** `ovvs::Resources` can allocate USM, submit a GPU kernel, and run an OpenVINO NPU MatMul.

- **T1.1** `[G]` Level Zero GPU context, USM shared malloc, event. Exit: SYCL vector add correctness.
- **T1.2** `[N]` OpenVINO `compile_model(..., "NPU")` for a static MatMul. Cache blob to disk. Exit: numerical vs NumPy fp16.
- **T1.3** `[N]` Bind NPU infer request to external host pointers (zero extra copy). Exit: probe bandwidth; record GB/s.
- **T1.4** `[H]` Two-device timeline: GPU kernel → host visible → NPU infer → host. Events, no deadlock. Exit: unit test.
- **T1.5** Compiled-graph cache (`~/.cache/ovvs/` + in-process). Key includes shapes + driver/compiler versions. Exit: second run skips compile.
- **T1.6** Scratch arenas, `Resources` thread contract documented. Exit: TSAN-clean host tests (no device TSAN).
- **T1.7** Error model: `ovvsStatus` codes for compile fail, unsupported op, OOM, shape mismatch. No exceptions across C ABI.

---

### Phase 2 — Primitive bakeoff harness

**Goal:** we never guess which device wins.

- **T2.1** `tools/bakeoff` CLI: op name, shapes, dtypes, devices, repeats, warmup, power if ETW/RAPL available.
- **T2.2** Metrics: latency p50/p99, GB/s, TOPS achieved, compile time, energy if available.
- **T2.3** Emitter for `tables/<sku>/<op>.json`.
- **T2.4** Mixer loads tables at `Resources` init; missing table → conservative heuristic + warning.

Exit: one checked-in table for `gemm` on the primary laptop.

---

### Phase 3 — GEMM and pairwise distance  `[P]` for almost everything

- **T3.1** `[N]` Static GEMM graphs: fp16, int8×int8→fp16, batched. Shape set: powers of two and “real” (768, 1024, 1536) × N in {1e3,1e4,1e5,1e6 tiles}.
- **T3.2** `[G]` oneMKL SYCL GEMM same shapes.
- **T3.3** `[C]` oneMKL/AVX GEMM.
- **T3.4** Mixer + bakeoff. Exit: documented crossover curves.
- **T3.5** Pairwise L2 expanded = norms + GEMM + fuse. IP, cosine. All three devices.
- **T3.6** `[K][N]` If OpenVINO MatMul misses a layout, add transpose/reorder graph or SHAVE pack kernel. Do not accept 2× slowdown vs theoretical without a punch-through attempt logged in `tables/notes`.

---

### Phase 4 — Top-k / k-selection

- **T4.1** `[C]` reference `topk` (correct).
- **T4.2** `[G]` SYCL: small-k bitonic in SLM; large-k block radix select. Subgroup shuffles.
- **T4.3** `[N]` Try OpenVINO TopK on score matrix. If compile fails: punch-through ladder. If still fails: NPU emits full scores, iGPU selects — **hetero is success**, not a skip.
- **T4.4** Bakeoff vs `k`, `n`, `B`.

Exit: primitive API `ovvs::prim::topk` used by brute-force.

---

### Phase 5 — Gather / scatter / sort / scan / hist / bitset

- **T5.1** `[G]` gather (variable indices), scatter-add, inclusive scan, histogram, bitset test/set, popcount.
- **T5.2** `[N]` Gather with **static** index tensor (compiled). HostCompile loop for dynamic indices (CPU feeds tiles).
- **T5.3** `[G]` radix sort for IVF list build and SLINK support.
- **T5.4** Unit tests vs CPU for random and adversarial (all indices 0, out-of-range rejected).

---

### Phase 6 — Exact kNN (brute-force index)

First user-visible algorithm.

- **T6.1** API: build (store dataset USM), search `(queries, k)`.
- **T6.2** Pipeline: tile dataset along N so GEMM tiles fit NPU SRAM / iGPU cache.
- **T6.3** Mixer: large tiles NPU, remainder iGPU, merge top-k on CPU/iGPU.
- **T6.4** Metrics + dtypes fp32 in, compute fp16 with error bound; int8 dataset path.
- **T6.5** Golden: recall@k = 1 vs fp32 CPU for fp32 path; documented error for fp16/int8.
- **T6.6** Python binding for this one algorithm (forces C ABI to exist).

Exit: `pytest` search SIFT1M subset, k=10, recall 1.0 fp32.

---

### Phase 7 — Quantizers (needed by IVF-PQ, CAGRA-Q, binary)

- **T7.1** Scalar quantizer (per-tensor, per-channel). NPU Convert/QDQ.
- **T7.2** Binary quantizer (sign bits), Hamming via popcount GEMM.
- **T7.3** Product quantizer train: subspace k-means (Phase 8 k-means may be stubbed with CPU for train only if Phase 8 is parallelized after 7.3’s CPU oracle — but assignment must move to NPU/iGPU in Phase 8).
- **T7.4** Encode/decode APIs. Error histograms in tests.
- **T7.5** `[K]` LVQ-like / LeanVec-inspired residual quant **as our own code** (not SVS blobs). Optional but scheduled after PQ; needed to beat SVS on CPU-side refine.

---

### Phase 8 — K-means

- **T8.1** Assignment = brute-force 1-NN to centroids → **reuse Phase 6** (NPU GEMM).
- **T8.2** `[G]` Update: scatter-add / hist of members, divide. Atomic add in SYCL.
- **T8.3** Empty cluster repair, k-means++, mini-batch option.
- **T8.4** Golden vs FAISS kmeans inertia band on SIFT.

Exit: IVF training can call this.

---

### Phase 9 — IVF-Flat

- **T9.1** Train: k-means coarse, assign lists, prefix-sum offsets, pack lists (CSR-like).
- **T9.2** Search: coarse GEMM through the SKU-selected primitive (**CPU oneMKL on Arrow Lake; NPU path retained**), top `nprobe` **CPU/iGPU**, scan lists **iGPU** fused dist.
- **T9.3** `[H][N]` Padded-list NPU scan: pad each probed list to `max_list_size`, pack `[B*nprobe, max_list, D]`, NPU GEMM. Bakeoff vs fused iGPU. Keep winner per SKU.
- **T9.4** Add/remove optional (lists rebuild or extra list). If cuVS IVF-Flat has extend, match it.
- **T9.5** Serialize. Filtered search bitset while scanning.
- **T9.6** Recall vs FAISS IVF-Flat same `nlist, nprobe`.

---

### Phase 10 — IVF-PQ

- **T10.1** Train: coarse k-means + per-subspace PQ (Phase 7+8).
- **T10.2** ADC tables: for each query, `nprobe` lists, compute distance tables (`dsub x 256`) — GEMM-shaped; **NPU candidate**.
- **T10.3** Scan codes: iGPU (variable length). NPU padded scan if T9.3 pattern wins.
- **T10.4** Refine: re-rank top `krefine` on raw vectors (NPU/iGPU GEMM).
- **T10.5** Polar-quant / by-residual layouts as FAISS compatibility flags.
- **T10.6** Recall vs FAISS IVF-PQ M=8/16, nbits=8.

---

### Phase 11 — IVF-RaBitQ / binary IVF

This is the NPU-native ANN, not a consolation prize.

- **T11.1** Implement RaBitQ encode (follow the paper cuVS integrated).
- **T11.2** `[N]` Packed binary/INT GEMM or popcount kernel (SHAVE if MatMul cannot express it).
- **T11.3** IVF lists of codes + optional add-on factors.
- **T11.4** Bakeoff vs IVF-PQ and vs CAGRA on the same recall target.

Exit: at least one SKU where this is the recommended index for on-device RAG (documented in mixer heuristics).

---

### Phase 12 — NN-Descent  `[G]` primarily

- **T12.1** NN-Descent local join in SYCL: bounded sampled forward/reverse NEW and OLD lists, NEW-NEW + NEW-OLD comparisons, bidirectional proposals, cached distances, and a replacement-count convergence threshold. Compute distance on iGPU; bake off **NPU tiles** only when a join slab is large enough to amortize transfer.
- **T12.2** Bloom / seen-edge filter on iGPU.
- **T12.3** Host graph buffers as in cuVS NN-Descent (degree I).
- **T12.4** No public query API (match cuVS); only graph output for CAGRA init and all-neighbors.
- **T12.5** Quality: overlap with exact kNN graph on small N; on SIFT1M, downstream CAGRA recall not worse than IVF-PQ-init by more than a documented delta.

---

### Phase 13 — CAGRA

Depends on 6, 10, 12, 4, 5.

- **T13.1** Index params matching cuVS names (`graph_degree`, `intermediate_graph_degree`, `graph_build_params`).
- **T13.2** Init graph: IVF-PQ (default heuristic), NN-Descent, iterative (after T13.5).
- **T13.3** Optimize/prune SYCL: detour-count rank reorder, fixed-degree prune, capped reverse graph, and deterministic forward/reverse interleave. Nearest-only truncation is not CAGRA graph optimization.
- **T13.4** Search kernel v1: one work-group per query, fused L2, SLM itopk, global hashmap. Correctness over speed.
- **T13.5** Search kernel v2: multi-work-group, search_width, team_size mapping, hashmap modes.
- **T13.6** `[H]` NPU dist-offload variant. Bakeoff.
- **T13.7** Search params: `itopk_size`, `min/max_iterations`, `filtering_rate`.
- **T13.8** Filtered bitset.
- **T13.9** Extend.
- **T13.10** Serialize/deserialize + attach/detach dataset.
- **T13.11** CAGRA-Q: PQ/binary codes attached; search in code space + optional refine.
- **T13.12** Persistent/batched submission (dynamic batching Phase 17 can feed this).
- **T13.13** Recall-QPS vs hnswlib and vs FAISS CAGRA if a GPU is available for oracle; if not, vs hnswlib at matched build cost.

**Do not ship CAGRA until T13.4 recall is within 2% of hnswlib at similar M/ef on SIFT1M.** Speed comes after.

---

### Phase 14 — HNSW interop

- **T14.1** `from_cagra` hierarchy GPU/CPU analogue: we built on iGPU, search may run CPU.
- **T14.2** Serialize to hnswlib-compatible file (cuVS rust `serialize_to_hnswlib` behavior).
- **T14.3** CPU HNSW search (own or vendored hnswlib) for the export path.
- **T14.4** Tests: CAGRA search vs HNSW-from-CAGRA search recall close, as cuVS claims.

---

### Phase 15 — Vamana

- **T15.1** Graph build: robust prune (α), long-range edges. iGPU with optional NPU distances.
- **T15.2** Search: greedy beam, iGPU.
- **T15.3** Optional SSD/mmap path (DiskANN-like): I/O on CPU, distance on iGPU/NPU for fetched vectors. This is the “large-scale / disk-backed” cuVS Vamana role.
- **T15.4** API + serialize + filters.

---

### Phase 16 — ScaNN-style index

- **T16.1** Partition (k-means / tree) — reuse IVF coarse.
- **T16.2** Anisotropic vector quantization (AVQ) train on CPU/iGPU.
- **T16.3** Score-aware in-register scan on iGPU; NPU if it reduces to GEMM on quantized codes.
- **T16.4** Re-rank. API names aligned with cuVS ScaNN guide.

---

### Phase 17 — All-neighbors, dynamic batching, filters as a library

- **T17.1** All-neighbors: brute or IVF or NN-Descent switch, output kNN graph CSR.
- **T17.2** Dynamic batcher: thread-safe queue, max wait, max B; prefers NPU-friendly B.
- **T17.3** Common `ovvsFilter` bitset used by IVF + CAGRA + Vamana.
- **T17.4** Allow-list API (Go cuVS style) converting to bitset.

---

### Phase 18 — Clustering beyond k-means

- **T18.1** SLINK / single-linkage: kNN graph (Phase 17) + MST on iGPU (Boruvka SYCL) + dendrogram. Golden vs scipy on N≤10k.
- **T18.2** Spectral clustering: kNN graph → Laplacian → oneMKL eigensolver (GPU if available) → k-means on embeddings.
- **T18.3** Spectral embedding preprocess API.

---

### Phase 19 — Preprocess finish + pairwise + k-selection public APIs

- **T19.1** PCA via oneMKL SVD/eig; project on NPU MatMul.
- **T19.2** Public pairwise distance API (all metrics).
- **T19.3** Public k-selection API (not just internal).
- **T19.4** FP16/INT8 dataset attach helpers.

---

### Phase 20 — Mixer v2 and contention

- **T20.1** Read NPU/GPU occupancy (Level Zero info, Windows Task Manager counters if needed).
- **T20.2** If NPU is busy with an LLM, search must still hit SLA on iGPU. Tests: synthetic NPU load.
- **T20.3** `turbo` hint passthrough for NPU.
- **T20.4** Energy metric in bench: RAPL package + NPU/GPU if exposed.

---

### Phase 21 — Bindings

- **T21.1** Freeze C ABI versioning (`OVVS_VERSION`, `ovvsGetVersion`).
- **T21.2** C++ headers.
- **T21.3** Python (pybind11/nanobind) + DLPack + numpy.
- **T21.4** Rust crate.
- **T21.5** Go cgo.
- **T21.6** Java Panama or JNI.
- **T21.7** Examples per language mirroring cuVS README snippets.

---

### Phase 22 — Benchmarks, packaging, docs

- **T22.1** `ovvs_bench` compatible with cuVS bench datasets and recall-QPS plots.
- **T22.2** Comparison scripts: FAISS CPU, hnswlib, Intel SVS if present, ovVS NPU, ovVS GPU, ovVS hetero.
- **T22.3** Packaging: pip wheel, CMake install, vcpkg/conan later.
- **T22.4** User guide: which index on which SKU, how mixer decides, how to FORCE device.
- **T22.5** Paper-ready: IVF-RaBitQ-on-NPU + CAGRA-on-Xe2 + hetero energy.

---

### Phase 23 — Compiler fork (parallel track, starts after first NPU GEMM pain)

Not optional if stock OpenVINO cannot TopK/Gather/ADC.

- **T23.1** Submodule `compiler/npu_compiler` at a tagged commit.
- **T23.2** Inventory `sw_runtime_kernels`; hello-world SHAVE kernel running on device.
- **T23.3** Custom TopK SHAVE/DPU.
- **T23.4** Custom packed-list ADC kernel.
- **T23.5** HostCompile_Interpreter path for dynamic N (list length, candidate count) with bounded tiles.
- **T23.6** Upstream patches where possible; keep a `patches/` dir with rebase notes.

This phase runs **in parallel from Phase 4 onward**. Assign it to a compiler-shaped workstream. Algorithms must not wait forever: they use iGPU until the SHAVE kernel lands, then mixer flips.

---

## 8. Testing gates (every algorithm)

1. **Oracle:** CPU fp32 brute-force on a slice.
2. **Recall@k** at documented params vs FAISS or hnswlib.
3. **Bitwise or atol** for primitives.
4. **Device:** FORCE_NPU and FORCE_GPU both run if the stage claims that device; hetero path too.
5. **Shape cache:** second search does not recompile NPU graphs.
6. **Determinism:** same seeds → same graph build for tests that seed RNG.
7. **Leak:** NPU/GPU allocations returned (L0 / OpenVINO infer request destroy).

CI labels: `cpu`, `npu`, `gpu`, `hetero`, `slow`.

---

## 9. Performance targets (Lunar Lake class, revise after bakeoff)

These are **targets**, not excuses to cut scope. If missed, we kernel-tweak, we do not remove CAGRA.

| Workload | Target |
|---|---|
| Brute-force 1e5 x 768, B=32, k=10 | faster than OpenVINO-CPU and faster than naive iGPU OpenCL; NPU used for GEMM |
| IVF-PQ SIFT1M, nprobe typical, recall@10 ≥ 0.95 | beat FAISS CPU QPS at same recall; energy < iGPU-only |
| CAGRA SIFT1M, recall@10 ≥ 0.95 | beat hnswlib QPS on the same laptop; build faster than hnswlib `efC=200` |
| IVF-RaBitQ on-device RAG 1e6 x 768 int | lowest joules/query among ovVS indexes at recall ≥ 0.9 |
| B=1 CAGRA | p99 latency competitive with hnswlib (iGPU fused walk, not NPU) |

---

## 10. Risks and punch-through (not stop signs)

| Risk | What we do |
|---|---|
| NPU static shapes | pad; HostCompile tiles; compile a shape family `{1,8,32,128}` |
| NPU no TopK | SHAVE kernel or iGPU select |
| NPU Gather poor | iGPU gather; NPU only for dense GEMM |
| NPU launch tax | batch; dynamic batcher; never NPU for B=1 dist of 32-d |
| iGPU DRAM bandwidth | quantize; keep codes; SLM blocking; fuse walk+dist |
| Subgroup 16 vs warp 32 | two-subgroup teams; re-tune team_size |
| OpenVINO/NPU driver blobs incompatible | pin versions; cache keyed by compiler; CI matrix |
| SHAVE SDK not public enough | work inside open `npu_compiler`; if embargoed headers block us, implement the stage on iGPU **and keep a ticket** until SHAVE is usable |
| Shared DRAM contention with LLM | mixer occupancy; prefer NPU for embed + iGPU for search or vice versa |
| Windows vs Linux driver gaps | both first-class; feature-detect |
| Numerical fp16 | document; optional fp32 iGPU refine |

---

## 11. Tentative internal APIs (so work can start in parallel)

```cpp
namespace ovvs {
enum class Device { Cpu, Npu, Gpu, Auto };

struct Resources { /* L0, OV, arenas, tables */ };

namespace prim {
  void gemm(Resources&, Tensor A, Tensor B, Tensor C, GemmEpiloque);
  void pairwise(Resources&, Metric, Tensor X, Tensor Y, Tensor out);
  void topk(Resources&, Tensor scores, int k, Tensor idx, Tensor val);
  void gather(Resources&, Tensor src, Tensor idx, Tensor out);
  // ...
}

namespace neighbors {
  struct BruteForceIndex { /* ... */ };
  BruteForceIndex build(Resources&, IndexParams, Tensor dataset);
  void search(Resources&, SearchParams, const BruteForceIndex&,
              Tensor queries, Tensor neighbors, Tensor distances,
              Filter = {});
}
}
```

Mixer lives behind `prim::*`. Algorithms call `prim`, not OpenVINO.

---

## 12. Workstream split (agents)

When implementing with multiple agents, split along **phases**, not files at random:

| Stream | Phases | Needs hardware |
|---|---|---|
| A Runtime | 0–2 | yes |
| B Primitives | 3–5, 23 | yes |
| C IVF family | 6–11 | yes |
| D Graphs | 12–17 | yes |
| E Cluster/preprocess | 8, 18–19 | yes |
| F Bindings/bench | 21–22 | can mock then device |

Stream B is on the critical path of everything. Do not start D before `topk` + `gather` + `gemm` exist.

---

## 13. First 30 days (concrete)

Week 1: Phase 0 + T1.1–T1.2 (GPU add, NPU MatMul).  
Week 2: T1.3–T1.7, Phase 2 harness, T3.1–T3.4 GEMM bakeoff.  
Week 3: Phase 4 topk, Phase 6 brute-force + Python.  
Week 4: Phase 8 k-means assignment on NPU, start IVF-Flat coarse path.

If week 2 bakeoff shows NPU GEMM never beating iGPU on that SKU, **keep the NPU path** and still complete IVF-RaBitQ — Lunar Lake / Panther Lake may flip the table. Mixer tables are per SKU.

---

## 14. Decision log

| Decision | Choice | Why |
|---|---|---|
| Language core | C++20 + C ABI | matches cuVS; SYCL and OpenVINO are C++ |
| GPU API | SYCL/DPC++ on Level Zero | supported on Arc iGPU; oneMKL |
| NPU API | OpenVINO + npu_compiler + L0 Graph | only production path |
| Tensor interchange | DLPack | cuVS does this; Python/JAX/torch |
| Default memory | USM shared | CPU+iGPU zero copy; NPU tiles from host |
| Graph ANN home | iGPU | NPU cannot pointer-chase. Candidate-distance offload remains a per-SKU option, but Arrow Lake NPU offload is parked under the zero-capacity decision; no retained artifact claims a CAGRA-specific candidate-slab benchmark. |
| Quant IVF home | NPU + iGPU | NPU-shaped; must exist |
| CPU role | control, tiny-B, HostCompile, HNSW export | not the library’s identity |
| SVS | competitor / idea source | no proprietary blob in core |
| Equivalence | API + algorithms + quality knobs | not CUDA kernel isomorphism |
| iGPU GEMM without DPC++ | OpenVINO GPU plugin | historical; official oneAPI `icx` 2025.1.1 is the Windows Ninja compiler. SYCL-first USM is the default iGPU path when `OVVS_WITH_SYCL=ON`; OpenVINO GPU remains fallback |
| First measured SKU | Arrow Lake 265K | lab machine; Lunar Lake tables still TBD |
| FORCE_* honesty | DEVICE_UNAVAILABLE if the requested device did not run | bakeoff last_device must match requested device on success |
| iGPU topk/gather without DPC++ | OpenVINO TopK/Gather on GPU plugin | SYCL kernels remain for icpx |
| CAGRA walk | `prim_graph_walk`: fused SYCL when built; adaptive host fallback otherwise | **B2 ≥0.95 recall point observed; throughput open:** one work-group/query, cooperative distance, SLM candidate/expanded state, traversal-budget global visited hash, chunked hash allocation, and strict device USM are implemented. Cached-worst tracking, prefix-only publication sorting, and canonical 64-ID frontier tiling are promoted. Exact subgroup min-location selection for `search_width > 1` passed a separate six-process gate: multi-pick QPS improved 17.8–29.9%, p50 fell 15.2–22.9%, p99 fell 15.4–20.8%, recall remained exact, and old/new DLL output bytes matched on a retained serialized graph. Current `64/2` reaches 0.9609 recall and 3,243.5 QPS versus hnswlib 0.9593 and 6,784.5 QPS, so the product gap remains 2.09× there and 3.13× near 0.989 recall. The exact stable one-pass pick was rejected after regressing three effort points. The proposed four-workgroup owner/helper route is also parked on the measured Arrow Lake stack: its exact 128-thread/2,640-byte named-kernel gate reported `max_num_work_groups=1`, exited 77, and launched nothing. Preserve the promoted one-workgroup path and continue only with software designs that avoid unsupported cross-workgroup synchronization. Forced GPU rejection never host-fallbacks; FORCE_NPU graph walk is unavailable. Evidence: `tables/arrow-lake/cagra-cooperative-pick-v1.md` and `tables/arrow-lake/gpu-root-group-canary-v1.md`. |
| NN-Descent build | exact primitive path through n=4096; bounded sampled iGPU NEW/OLD join above it | **B5 partial; CAGRA-vs-hnswlib gate passed, T12.5 open:** deterministic cached-distance NEW/OLD tagging, bounded forward/reverse samples, NEW–NEW and NEW–OLD reciprocal proposals, target-owned merges, and exact changed/pending-NEW accounting execute on SYCL. Three clean SIFT1M AUTO builds completed in 99.874–100.525 s (100.093 s median) within 2,343.2 MiB median isolated peak process RSS and supported 0.9036 downstream CAGRA recall. The GPU initializer owns 90.444 s (90.36%) of median complete wall; host optimize/prune/merge owns 9.034 s. All runs hit the six-iteration cap with about 18.5% changed/pending edges, so standalone convergence and the T12.5 comparison against IVF-PQ initialization remain open. Two bounded synchronization changes are retained negative evidence: device-resident active-count/fixed-`N` consumers regressed initializer wall 9.85% and complete build wall 8.91%; a convergence-only single-work-group reduction cut D2H bytes 27.27% but regressed initializer wall 1.91% and complete build wall 2.14%. Production retains dynamic consumer ranges and the original convergence path. Next initialize `heads[N]` once per iteration and have each unique target-owned consumer reset only touched rows. Host CAGRA optimization is still CPU, so full FORCE_GPU build rejects until the T13.3 SYCL port. K64 scale remains unproven. Evidence: `tables/arrow-lake/cagra-build-v1.md`, `tables/arrow-lake/nndescent-b5.md`, `tables/arrow-lake/nndescent-active-count-v1.md`, `tables/arrow-lake/nndescent-convergence-reduction-v1.md`. |
| CAGRA build telemetry | frozen additive V1 snapshot; external API wall remains canonical | `ovvsResourcesCagraBuildStatsV1` publishes 31 resource-local cumulative counters only after a successful build is ready to publish. Four named stages, exact logical bytes, initializer-final device, and call-local GPU NN-Descent lifecycle counters are separated from the final CPU attribution. `total_wall_ns` is the inner wall through persistent graph materialization and excludes telemetry merge, caller-handle publication, and serialization. Fixed benchmark modes require a fresh, fully instrumented AUTO GPU NN-Descent initializer and a fingerprinted loaded library matching explicit `--library`. This enables attribution; it is not an acceleration result. |
| CAGRA graph optimization | detour-rank prune plus capped reverse-edge interleave | The current M=16/intermediate-32 graph reaches 0.9609 on the same-index `64/2` search point, so changing graph geometry is not the first response to the ≥0.95 target. Record initializer and post-prune recall/connectivity before any future construction-quality change; retain 16/32 as the low-memory baseline and treat 32/64 as a later controlled quality/scale experiment. |
| HostCompile | NPU GEMM tiles of M=256, TopK rows of 32, Gather nidx of 128 when full-shape compile fails | in `npu_gemm` / `npu_topk` / `npu_gather_rows`; SHAVE ADC/TopK C still host-linked until unsigned ELF is loadable |
| HNSW serialize | honest single-layer hnswlib `saveIndex` layout | Stock hnswlib load/query/save/reload and strict native parsing are covered. Three clean SIFT100K runs produced a deterministic 65,600,096-byte graph with 100% entry reachability, but hybrid construction was 5.779× slower and nearest comparable-recall search delivered only 0.582×/0.602× native throughput. B24 remains partial for full SIFT1M recall closeness; no acceleration claim. Evidence: `tables/arrow-lake/hnsw-export.md`. |
| Mixer v2 | `ovvsResourcesSetNpuBusy` skips NPU on AUTO/HETERO, including range-safe PQ ADC | competing occupancy APIs are not exposed; explicit FORCE_NPU overrides the advisory flag. A live regression proves busy AUTO IVF-PQ submits zero NPU requests and completes on CPU. |
| Dynamic batcher | thread-safe waiter queue; flush at max_B or max_wait; `ovvsBatcherLastBatchSize` | concurrent nq=1 submits coalesce; results match eager brute |
| ScaNN | anisotropic IVF-PQ + original-space refine | nprobe=all + krefine=n matches brute L2 |
| Python tensors | NumPy + DLPack (`np.from_dlpack` / `__dlpack__`) | consumer searches from DLPack views; `allow_list=` / `bitset_from_allow_list` |
| C++ API | header wrappers beyond brute-force | IVF-Flat / IVF-PQ / IVF-RaBitQ / CAGRA + serialize/extend in `ovvs.hpp` |
| Allow-list filter | `ovvsBitsetFromAllowList` | fills `(n+7)/8` bitset; missing ids stay 0 |
| IVF-PQ / RaBitQ persist | magic `IPQ1` / `RQB1`; IVF-PQ derives a list-major CSR at lifecycle boundaries | `IPQ1` v1 bytes remain canonical and unchanged. IVF-PQ build/load/extend validate unique ID/list/code alignment before publishing derived offsets, IDs, and contiguous codes; unfiltered search uses those spans directly and resolves only shortlisted offsets to persistent IDs. Extend is transactional. Concurrent mutation/serialization/destruction requires exclusive index access, and search workers use distinct `Resources`. The fused iGPU scan/select is implemented for FORCE_GPU but loses the bounded CPU route. Transformed-NPU, direct-Level-Zero, and NPU+iGPU measurements are retained negative Arrow Lake evidence; matched GPU evidence remains open. |
| Large-GEMM AUTO | load `tables/<sku>/gemm_large.json` | Arrow Lake after oneMKL CPU: **CPU 18 ms** vs NPU 45 vs GPU 221. NPU DPU 5.3 ms; Parameter DMA is the tax. Canonical split: `docs/hw-split.md`. |
| TopK/Gather AUTO | CPU | `topk_large.json` CPU 3.4 ms vs NPU 99 vs GPU 758. |
| CPU GEMM | oneMKL `cblas_sgemm` | Naive triple loop was 192 ms; MKL is 18 ms. That flip made CPU the dense GEMM device on this SKU. |
| NPU programming model | Inference engine, not BLAS | DPU MatMul is µs; request-owned L0 buffers and reused InferRequests avoid known SHAVE-copy/request-creation defects and remain mandatory. They are not sufficient for an Arrow Lake win: corrected direct Level Zero still paid graph/DMA, i32 expansion, full score production, and readback, and lost reused OpenVINO at both sizes/depths 1/2/4 in refill+execute+consume scope. Two 524K prefilled-only cells slightly favored direct but are not end-to-end gains. Copy inputs unless a payload is at least 64 KiB and its 32-sample fingerprint still matches. INT8 = NNCF FQ+Constant weights. Scratchpad 4 MB. Compile: TURBO + `optimization-level=2`. Evidence: `tables/arrow-lake/npu-gemm-dpu-vs-wall.md` and `tables/arrow-lake/npu-igpu-escape-routes.md`. |
| NPU PQ ADC numeric contract | Raw execution requires `sum_m max_code(abs(lut[m, code])) < 65,504`; FORCE_NPU may affine-center and scale a remaining span to 60,000 only at scale ≥0.5; AUTO keeps the unsafe CPU fallback | Real SIFT residual LUT sums reached roughly 85k–100k and full 128-row tiles saturated at 65,504, collapsing bounded AUTO recall to 0.3781 versus CPU 0.9469. The raw guard restored parity. The bounded transform rejects wider spans atomically and retained 32/32 candidates on a scale-0.588 fixture, but ran 73–106× slower than CPU. It remains FORCE_NPU correctness coverage with zero active Arrow Lake optimization capacity. Evidence: `tables/arrow-lake/ivfpq-b3.md`. |
| IVF-PQ ADC batching | bounded ordered query/list descriptors; fixed NPU buckets `{128,256,512,1024,2048}`; fused iGPU scan/select | Equal NPU buckets pack into capacity-bounded request-owned tensors; only real rows are published. FORCE_GPU scans persistent codes, filters, and emits an exact shortlist without a dense score buffer. At bounded `nprobe=8`, complete-call telemetry attributes 73.6% of CPU wall to LUT construction; GPU fused ADC plus final TopK consumes 85.5% and incurs 99 allocations/120 kernels/230 waits. Certified Q/T then lost its joined-stage direct oracle by 1.93×/1.79× at 131K/524K; that synthetic scope excludes coarse assignment and exact-vector refinement. Direct Level Zero lost refill+execute+consume at depths 1/2/4, and all nonzero NPU compositions lost their joined experimental scope. Active work is a workspace-byte planner, hierarchical iGPU top-k/global merge, packed AoSoA codes, direct residual LUT placement, and batched/fused refinement with bounded workspaces. Evidence: `tables/arrow-lake/ivfpq-b3.md`. |
| Arrow Lake NPU escape-route disposition | retain forced correctness/evidence; allocate zero active optimization capacity | Corrected direct Level Zero loses reused OpenVINO at both sizes/depths 1/2/4 in refill+execute+consume scope; two 524K prefilled-only cells slightly favor direct but are not end-to-end gains. Every nonzero NPU+iGPU partition loses its complete experimental scope; certified Q/T loses its synthetic joined stage, and NPU-Q is slower than CPU-Q/iGPU-Q. No integration or acceleration claim. Reopen only for a materially new runtime/SKU capability. Evidence: `tables/arrow-lake/npu-igpu-escape-routes.md`. |
| Public custom ActShave boundary | fail closed until sources, descriptors, MoviTools, and a matching isolated build are public | The pinned public compiler clone has 770 prebuilt ELFs but zero tracked kernel sources or descriptors; `moviCompile`, `moviLLD`, runtime archives, and the matching isolated OpenVINO/CiP build are unavailable. `add_extension`, host C, renamed ELFs, and caller markers are not NPU kernels. The retained gate exits `BASELINE_BLOCKED` with explicit blockers. |
| Remaining work after v0.2 ABI | Kernel quality + scale, not more algorithm C ABI names | 2026-08-29 status: symbols cover the plan’s algorithm list; the additive IVF-PQ and CAGRA telemetry ABIs bring the export count to 112. Three clean SIFT1M low-effort CAGRA gates pass recall closeness. Cached-worst tracking, prefix-only publication sorting, canonical frontier tiling, and cooperative multi-pick selection are promoted; the latest gate retains 0.9609 recall while raising `64/2` median QPS to 3,243.5. Full-strength hnswlib still reaches 6,784.5 QPS at comparable recall. The four-workgroup owner/helper small-batch route is parked after a fail-closed Arrow Lake capacity result of one group versus four required; the broader one-workgroup/SIMD16 portfolio remains open. NN-Descent fixed-`N` active counts and convergence-only single-work-group reduction are both rejected; sparse target-owned scratch reset is the next construction target. HNSW export interoperability is valid but loses build and nearest comparable-recall search at SIFT100K. Persistent IVF-PQ storage, buckets, numeric guards, shortlist-only IDs, fused FORCE_GPU scan/select, and both telemetry contracts are complete; Q/T and Arrow Lake NPU escape routes are negative within their documented scopes. Hierarchical/packed iGPU PQ, size-gated CPU+iGPU batching, other B1 algorithms/policies, and the real 100K×768 corpus remain open. Journal: `.claude/backlog.md`. |
| HETERO policy | equals AUTO until an explicit stage DAG lands | Not a third GEMM backend. Arrow Lake NPU+iGPU composition is parked after joined experimental-scope losses. Active work is size-/deadline-aware CPU+iGPU scheduling over resident shared-USM state with bounded queues/events and one publication. Future SKUs may add NPU only after isolated and joined wins. Documented in `docs/hw-split.md`; backlog B6. |
| v0.2 vs v1.0 | 0.2.0 is not §16 equivalent | v1.0 still requires Lunar Lake accelerated paths, published FAISS/hnswlib benches, mixer tables for that SKU. Toy-test recall floors (CAGRA-Q 0.35 on n=32, IVF-PQ 0.5 on n=64) are not those gates. |

---

## 17. Deviations / parked (only after a real attempt)

Filled in as installs finish. Placeholder until toolchain logs land:

- **SYCL/oneAPI toolkit:** elevated offline installer `intel-oneapi-base-toolkit-2025.1.3.8_offline.exe` exit 0. `icpx`/`icx` 2025.1.1 at `C:\Program Files (x86)\Intel\oneAPI\compiler\2025.1\bin\`. On Windows Ninja, **`icx` for both C and CXX** (`icpx` is GNU-like; CMake then passes `/nologo /EHsc` and the compiler test fails). `build-icpx` with `-DOVVS_WITH_SYCL=ON -DOVVS_WITH_OPENVINO=ON` links `sycl8.dll` and passes the configured native suite plus C++/C consumers, including `cagra_sycl_walk_n_over_4096`. `intel/llvm` nightly `clang++ -fsycl` remains a fallback (`build-sycl`).
- **SYCL fused CAGRA walk (B2 checkpoint):** one work-group/query, cooperative distance, SLM candidates, and bounded visited hash compile with oneAPI `icx`; `cagra_force_gpu_last_device`, forced-policy tests, and the n=4200 regression pass. Forced GPU failure returns unavailable; it never reports a host fallback as GPU. Nightly `clang++ -fsycl` remains an unverified fallback for this checkpoint.
- **SHAVE on NPU silicon:** ActShave **does** run. OpenVINO NPU ADC profiling shows `lut`/`Gather`/`Result` on Shave and `ReduceSum` on DPU. The compiler embeds Intel prebuilt ELF32 `.text` into a graph ELF64; a raw SHAVE ELF32 is `invalid_native_binary`, VCL rejects ELF-as-IR, and `ze_activation_kernel_desc_t` is compile-time data rather than an unsigned loader. The fail-closed public gate additionally found 770 prebuilt ELFs but no tracked kernel sources/descriptors, MoviTools compiler/linker, runtime archives, or matching isolated build. Park custom public ActShave until those inputs exist; do not accept `add_extension`, host C, renamed ELFs, or caller markers as a kernel. Evidence: `tables/arrow-lake/npu-igpu-escape-routes.md`.
- **Persistent CAGRA grid:** batched `prim_graph_walk` only; Level Zero resident kernel not required.
- **Energy/RAPL:** Windows package joules come from **intelppm EMI**, not Power Gadget. `GUID_DEVICE_ENERGY_METER` `{45BD8344-7ED6-49CF-A440-C276C933B053}` is registered on ACPI processor instance 0 (`cpu.inf` / `intelppm.sys`, Microsoft PPM v2). Channels: `RAPL_Package0_PKG`, `_PP0`, `_PP1`, `_DRAM` (DRAM is 0 on this desktop SKU). `ovvsResourcesEnergyUj` reads `IOCTL_EMI_GET_MEASUREMENT` (pWh × 0.0036 = µJ), then PDH Energy Meter, then Power Gadget, then Linux sysfs. Arrow Lake 265K: SUCCESS, increasing µJ. IPF/PMT are not this counter. Bakeoff `energy_probe` is `success` when EMI/PDH is present.
- **git push:** Repo renamed to [`manan-ramnani/ovVS`](https://github.com/manan-ramnani/ovVS). `origin` is `https://github.com/manan-ramnani/ovVS.git`. Full local `main` (including `.github/workflows/ci.yml`) replaced the old workflow-free snapshot `769ff5d` at `950c625`. C ABI remains `ovvs`.
- **FAISS/hnswlib pip:** `faiss-cpu` 1.15.0 and `hnswlib` 0.8.0 import. SIFT **n=2000** artifacts remain smoke only. Three clean strict SIFT1M low-effort pairs pass recall closeness: CAGRA 0.9036 versus hnswlib median 0.8911. The promoted cooperative-pick checkpoint raises current CAGRA `64/2` to 3,243.5 QPS at 0.9609 recall versus hnswlib 6,784.5 QPS at 0.9593; the 2.09× deficit remains negative product evidence. AUTO/CPU brute beats FAISS only on the bounded slice; the legacy NPU brute result is invalid due FP16-range saturation and now fails closed; IVF paths remain behind FAISS. Other policies/algorithms, matched FAISS curves, and the real 100K×768 corpus keep B1 open.
- **HNSW export B24 checkpoint:** the honest single-layer file loads, searches, saves, and reloads in stock hnswlib. Three clean SIFT100K runs produced a deterministic 65,600,096-byte file with 100% entry reachability, but time-to-searchable was 5.779× worse than native hnswlib and nearest comparable-recall throughput was 0.582×/0.602×. Interoperability is complete; full SIFT1M recall closeness and acceleration are not. Evidence: `tables/arrow-lake/hnsw-export.md`.
- **CAGRA SYCL B2 checkpoint:** work-group/query, SLM candidates, bounded hash, cooperative distance, and forced-policy honesty are implemented. Three clean full SIFT1M low-effort gates pass recall closeness at 0.9036 versus hnswlib median 0.8911. Resource-local counters prove direct GPU-accessible index pointers and zero explicit dataset/graph uploads; they do not measure driver-managed page migration. Query-content hashing fixes batch/order dependence with exact CPU/GPU parity regressions. Cached-worst tracking, prefix-only publication sorting, and canonical 64-ID frontier tiling are promoted. Exact subgroup min-location selection adds a separate 17.8–29.9% multi-pick QPS gain with exact old/new output fingerprints, but hnswlib remains 2.09×/3.13× faster at the closest batch-32 recall points. The exact stable one-pass pick was rejected. The proposed four-workgroup small-batch owner/helper route is parked after its Arrow Lake named-kernel gate reported one cooperative group versus four required and launched nothing. Graph-aware entry points and within-one-workgroup SIMD16 work remain separate gates. Evidence: `tables/arrow-lake/bench-recall-qps.md`, `tables/arrow-lake/cagra-cooperative-pick-v1.md`, `tables/arrow-lake/gpu-root-group-canary-v1.md`, `tables/arrow-lake/cagra-transfer-b2.md`, `tables/arrow-lake/cagra-build-v1.md`.
- **NN-Descent SYCL B5 checkpoint:** n>4096 L2-family/IP/cosine construction uses cached-distance NEW/OLD tags, bounded sampled forward/reverse lists, NEW–NEW and NEW–OLD reciprocal proposals, deterministic target-owned merges, and exact changed/pending-NEW accounting. Three clean SIFT1M AUTO builds have a 100.093 s median and 2,343.2 MiB median isolated peak RSS; the GPU initializer contributes 90.444 s. All runs exhaust six iterations, while downstream CAGRA recall 0.9036 passes matched hnswlib closeness. Fixed-`N` active-count consumers (+8.91% complete wall) and a convergence-only single-work-group reduction (+2.14%) are both rejected and restored. Production retains dynamic consumer ranges and the original convergence path; sparse target-owned `heads` reset is next. This does not close T12.5: standalone convergence, the IVF-PQ-init comparison, K64 scale, and the T13.3 GPU prune remain open. Evidence: `tables/arrow-lake/cagra-build-v1.md`, `tables/arrow-lake/nndescent-b5.md`, `tables/arrow-lake/nndescent-active-count-v1.md`, `tables/arrow-lake/nndescent-convergence-reduction-v1.md`.
- **IVF-PQ accelerated paths:** persistent CSR, synchronous NPU batching, bounded affine execution, shortlist-only IDs, fused iGPU scan/select, and complete-call telemetry are correctness guarded. At bounded `nprobe=8`, telemetry attributes 73.6% of CPU wall to LUT construction and 85.5% of GPU wall to fused ADC plus final TopK; the scaled NPU primitive is 73–106× slower than CPU. Certified Q/T then lost its synthetic joined stage by 1.93×/1.79× at 131K/524K; that stage excludes coarse assignment and exact-vector refinement. Direct Level Zero lost refill+execute+consume, and every nonzero NPU composition lost its complete experimental scope, so those Arrow Lake branches are parked. Active work is hierarchical/packed iGPU selection, a workspace-byte planner, and batched/fused refinement. No AUTO, competitor, or energy win is claimed. Evidence: `tables/arrow-lake/ivfpq-b3.md` and `tables/arrow-lake/npu-igpu-escape-routes.md`.
- **HETERO:** `OVVS_POLICY_HETERO` currently equals AUTO. Arrow Lake NPU+iGPU composition is measured negative and parked; CPU+iGPU size-/deadline-aware scheduling is unwired (backlog B6).
- **Lunar Lake tables:** still TBD. Do not copy Arrow Lake. v1.0 SKU (backlog B8).

---

## 15. Open implementation notes (not blockers)

- Exact OpenVINO opset support for TopK/Gather on NPU 4000 must be measured in T4.3 / T5.2, not assumed from ONNX EP tables (those mix CPU/GPU).
- HostCompile_Interpreter is the preferred dynamic-shape story on NPU 4000+ (`npu_compiler` docs). Use it for IVF list length.
- oneMKL GPU GEMM vs hand SYCL joint_matrix: start oneMKL, replace if bakeoff demands.
- Power: Windows EMI/PDH RAPL vs Linux RAPL; both implemented in `ovvsResourcesEnergyUj`. Allow missing only when neither EMI, PDH Energy Meter, Power Gadget, nor sysfs RAPL exists.

---

## 16. Definition of “v1.0 equivalent”

ovVS 1.0 ships when on Lunar Lake (Windows or Linux):

1. Brute-force, IVF-Flat, IVF-PQ, IVF-RaBitQ, CAGRA (build+search+filter+serialize), NN-Descent, k-means, PQ/SQ/binary, pairwise, topk all have accelerated paths.
2. Python + C++ + C work for those.
3. Bench vs FAISS/hnswlib published in-repo.
4. Mixer tables for that SKU checked in.
5. Vamana, ScaNN, SLINK, spectral, extra bindings may be 1.1 — **they stay on the roadmap and are not cancelled**. Start them as soon as CAGRA search v1 is correct.

v1.1: Vamana, ScaNN, SLINK, spectral, HNSW export polish, Java/Go/Rust, DiskANN-style mmap.

v2: SHAVE ADC/TopK in-tree, persistent iGPU search, multi-NPU if Intel ever enumerates two, discrete Arc dGPU as a bigger CAGRA machine.

---

*End of plan. Implement in task order. Update this file only when a decision in §14 changes.*
