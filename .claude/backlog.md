# ovVS backlog journal

Living remaining-work log. **Does not replace the spec.**

- Spec (architecture, feature matrix, T0–T23): `.claude/plans/2026-08-20-ovvs-intel-cuvs-equivalent.md`
- Agent rules: `AGENTS.md`
- Device split: `docs/hw-split.md`
- If this file and the plan disagree on *design*, the plan wins. If they disagree on *what is still open*, this file wins until someone updates both.

How to use: append dated entries; check items off in **Open items** (do not delete IDs); move finished work to **Done**. Do not add new algorithm names to the C ABI because a row in the plan exists — v0.2 already has those symbols.

---

## Snapshot (2026-08-28)

ovVS **0.2.0** is a cuVS-shaped library with real NPU (OpenVINO) and iGPU (SYCL) primitive backends and honest Arrow Lake bakeoffs. It is **not** a hardware-accelerated cuVS equivalent.

The C ABI covers brute-force, IVF-Flat/PQ/RaBitQ, CAGRA, NN-Descent, Vamana, ScaNN, HNSW-from-CAGRA, all-neighbors, filters, dynamic batcher, k-means, SLINK, spectral, PQ/SQ/binary, PCA, pairwise, k-selection. Python ctypes + NumPy/DLPack ingest exist. Toy tests pass (typically **n=24–80**). The published competitor bench is SIFT **n=2000**, not SIFT1M.

On Arrow Lake 265K AUTO search is still a **CPU oneMKL** library. The two remaining hardware bets (iGPU CAGRA walk, NPU PQ ADC) are wired but do not beat FAISS-CPU / hnswlib. Lunar Lake (plan v1.0 SKU) has no tables.

Evidence: `tables/arrow-lake/bench-recall-qps.md`, `gemm_large.json`, `docs/hw-split.md`.

| Path (SIFT n=2000 dim=128 nq=32 k=10) | Result |
|---|---|
| ovVS brute CPU | 0.89 ms / 36k QPS (beats FAISS brute) |
| ovVS brute NPU / GPU | 607 ms / 82 ms |
| ovVS IVF-Flat / IVF-PQ | slower than FAISS at same nprobe |
| ovVS CAGRA | ~2.0 ms vs hnswlib 0.40 ms (recall vs brute 0.99) |
| Package µJ/query brute | CPU 1049; NPU 4638; GPU 17893 |

v1.0 (plan §16) still requires Lunar Lake accelerated paths, published FAISS/hnswlib benches, and mixer tables for that SKU. v0.2 API completeness is not that gate.

---

## Open items

Priority: P0 = objective is blocked without it; P1 = v1.0 quality; P2 = v1.1 / productization. IDs are stable.

### P0 — hardware benefits (the original mission)

| ID | Item | Plan | Notes |
|---|---|---|---|
| B1 | SIFT1M / 1e5×768 recall–QPS harness vs FAISS-CPU and hnswlib; AUTO + FORCE_*; µJ/query | T22.1–T22.2, §8–§9 | `tools/bench/bench.py` is a 2k slice. `data/sift-128-euclidean.hdf5` is in-tree. Every later kernel change needs this number. |
| B2 | CAGRA search kernel T13.4: one **work-group** per query, SLM itopk, bounded hashmap (not `Seen[nq×n]`), graph-aware seeds, subgroup distance | T13.4, then T13.5–T13.6 | Current `gpu_cagra_walk` is `parallel_for(nq)` one WI/query, hash seeds `(s*9973+qi*13)%n`, scalar L2, linear heap. FORCE_GPU can be honest and still lose to hnswlib. Gate: SIFT1M recall within 2% of hnswlib at similar M/ef **before** chasing QPS. |
| B3 | IVF-PQ search rewrite: persistent list codes, ADC tables batched over `nq×nprobe`, iGPU variable-length scan; padded-list NPU bakeoff | T10.2–T10.4, T9.3 | `prim_pq_adc` on NPU is real. `ovvsIvfPqSearch` rebuilds residual + tables + packed codes **on the host per query per list**. That is why FAISS is ~6× faster on 2k points. |
| B4 | IVF-RaBitQ packed binary/INT GEMM (or SHAVE popcount), not scalar `rabitq_ip` | T11 | Plan: this is the NPU-native ANN, not a consolation prize. |
| B5 | NN-Descent iGPU local-join + bloom for n≫4096 | T12 | n≤4096 is exact kNN via `prim_pairwise`. Else a host nested loop. CAGRA IVF-PQ init uses `nlist = min(8, n/4)` — will not build 1e6 graphs. |
| B6 | Wire `OVVS_POLICY_HETERO` as stage overlap (NPU ADC ∥ iGPU walk), not a third GEMM backend | T20, `docs/hw-split.md` | Currently equals AUTO. |

### P1 — v1.0 quality and SKU coverage

| ID | Item | Plan | Notes |
|---|---|---|---|
| B7 | NPU index home: dataset / codebook as L0 tensor (`create_l0_host_tensor` or non-regressing Constant). Close wall vs DPU (5.3 vs ~45 ms on 1e5×32×768) | NPU contract, `npu-gemm-dpu-vs-wall.md` | Sticky Parameter + memcpy is current. Baking B as Constant made DPU faster and wall slower. Arrow Lake AUTO GEMM stays CPU until a SKU table flips. |
| B8 | Lunar Lake mixer tables (`tables/<sku>/`). Do not copy Arrow Lake numbers | T2, §3.4, §16 | v1.0 is defined on Lunar Lake. Meteor Lake / Panther Lake still required when hardware exists. |
| B9 | CAGRA knobs: `min/max_iterations`, `team_size`, hashmap mode, `filtering_rate` → itopk; NPU candidate-slab dist offload bakeoff | T13.5–T13.8 | Search ABI today: `itopk_size`, `search_width` only. |
| B10 | CAGRA-Q: codes in the iGPU walk + optional refine | T13.11 | API + host PQ-ADC walk; recall floor in tests is 0.35 on n=32. |
| B11 | CAGRA extend: iGPU local repair; recall vs rebuild at n≥1e5 | T13.9 | Host insert + `robust_prune`. Test is n=40. |
| B12 | IVF-Flat iGPU fused list scan; FAISS same `nlist/nprobe` recall within 1% on SIFT1M | T9.2–T9.6 | Search is host gather + `prim_pairwise` per query. |
| B13 | IVF-PQ polar-quant / by-residual FAISS flags; FAISS blob import/export | T10.5 | Custom `IPQ1` / `RQB1` / `IVF1` only. |
| B14 | k-means++ / empty-cluster repair / SYCL scatter-add; FAISS inertia band | T8.2–T8.4 | Random sample init; reduce is a host loop. |
| B15 | Brute-force tile along N so GEMM tiles fit; do not materialize full `nq×n` host scores as the only path | T6.2–T6.3 | Keep NPU/GPU paths for large-B even though AUTO GEMM is CPU on Arrow Lake. |
| B16 | Device-fused pairwise epilogue (norms still host loops after GEMM) | T3.5 | |
| B17 | `prim` sort / scan / hist / scatter-add / bitset / hashmap | T5 | Algorithms use `std::vector` / `std::partial_sort`. |
| B18 | Linux device CI (`npu` / `gpu` / `hetero` labels). Windows Actions today is CPU-only, no OpenVINO/SYCL/MKL | T0.4 | Skip only if the device is absent, not if compile fails. |
| B19 | DLPack on the **C ABI** (`DLManagedTensor*`), not only Python `np.from_dlpack` | §4.7, §14 | |
| B20 | 768-d embedding set in `data/` (plan T0.5) | T0.5 | |

### P2 — v1.1 / bindings / packaging (not cancelled)

| ID | Item | Plan | Notes |
|---|---|---|---|
| B21 | Vamana iGPU build/search; DiskANN-style mmap (I/O CPU, dist iGPU/NPU for fetched vectors) | T15 | Whole-file mmap exists; search is host. |
| B22 | ScaNN real AVQ / in-register scan (not per-dim std + IVF-PQ rename) | T16 | |
| B23 | SLINK Boruvka SYCL; spectral sparse Laplacian; scipy/sklearn gates | T18–T19 | Dense n×n Laplacian today. |
| B24 | HNSW-from-CAGRA recall closeness to CAGRA (cuVS claim), not only hnswlib `saveIndex` bytes | T14.4 | |
| B25 | Dynamic batcher over CAGRA/IVF, not brute only | T13.12, T17.2 | |
| B26 | Python pip wheel; CMake install package; user guide “which index on which SKU” | T21.3, T22.3–T22.4 | ctypes in-tree. |
| B27 | C++ wrappers: NN-Descent, all-neighbors, SLINK, spectral, PCA, PQ/SQ/binary | T21.2 | Indexes + k-means + batcher exist. |
| B28 | Rust beyond gemm/brute/k-means; non-Windows load | T21.4 | Windows `LoadLibrary` only. |
| B29 | Go Linux cgo (currently `errUnimplemented`); Java beyond Panama brute demo | T21.5–T21.6 | |
| B30 | Architecture leftovers: `src/mem/`, `src/plan/` stage IR (mixer is still `choose_device(op, flops)`) | §4.2–§4.5 | Not a reason to stall B2–B5. |
| B31 | Custom unsigned SHAVE (`moviCompile` + `VPU.SW.Kernel`) when MoviTools exists | T23 | Parked after a real attempt. ActShave **does** run inside compiler graphs. Algorithms must not wait. |
| B32 | INT8 NPU NNCF graphs that beat MKL on some SKU; FP8/FP4 on 3720 compile_fail | NPU contract | AUTO I8 is iGPU XMX and still loses to f32 `cblas_sgemm` on Arrow Lake. |

### Explicitly not backlog

- Replacing Arrow Lake AUTO GEMM with NPU (bakeoff: CPU 18 ms vs NPU 45 vs GPU 221). Keep FORCE_NPU.
- Copying Arrow Lake tables onto Lunar Lake.
- Linking Intel SVS blobs.
- Adding more C ABI algorithm names.
- CUDA.
- Multi-GPU (cuVS MG) — out of scope for a client SoC.
- Treating unsigned SHAVE as a blocker for CAGRA/IVF.

---

## Critical path (do in this order)

1. **B1** harness (so B2–B5 have a number).
2. **B2** CAGRA T13.4 kernel. Gate: SIFT1M recall vs hnswlib within 2% at similar M/ef.
3. **B5** NN-Descent iGPU (CAGRA init at scale).
4. **B3** then **B4** (IVF-PQ rewrite, then RaBitQ binary GEMM).
5. **B6** HETERO overlap; **B7** NPU L0 home experiment (does not flip Arrow Lake AUTO GEMM unless a table says so).
6. **B8** Lunar Lake bakeoff when that SKU exists.
7. Then B9–B20 (v1.0 polish), then B21–B32 (v1.1).

---

## Done (journal)

Nothing from the 2026-08-28 assessment is closed as v1.0 “Accelerated.” Already in-tree and **not** re-opened as new work:

- Phase 0–2 runtime, probe, bakeoff harness, Arrow Lake `tables/`.
- `prim` GEMM/TopK/Gather/pairwise/ADC with NPU + iGPU + CPU backends; FORCE_* honesty.
- NPU contract: InferRequest reuse, L0 `get_tensor`, fingerprint-sticky copies, no Constant-B.
- C ABI + toy tests for the plan’s algorithm list (quality/scale open — see B1–B5).
- Python neighbors.* + k-means + batcher (ctypes).
- HNSW `saveIndex` layout; Vamana file mmap; energy via Windows EMI / Linux RAPL.
- SHAVE unsigned inject attempted and parked (`compiler/shave/README.md`).

---

## Entries

### 2026-08-28 — Deep assessment vs cuVS-equivalent objective

Question: given the mission (port/adapt cuVS to Intel NPU+iGPU for hardware-accelerated benefits), what is pending?

Finding: **surface area is largely done; the reason to exist is not.** Remaining work is kernel quality and scale gates, not more symbols.

Shipped: C ABI 0.2.0, prim mixer, NPU OpenVINO path that obeys the NPU contract, SYCL iGPU path including a fused CAGRA walk entry, CPU oneMKL, Arrow Lake bakeoff that correctly flipped AUTO dense GEMM to CPU.

Not shipped relative to plan §5 “Accelerated” and §16 v1.0:

- CAGRA walk is not T13.4 (see B2). `docs/devices.md` previously claimed SLM itopk + 256-slot hashmap; the kernel uses USM heaps and `Seen[nq×n]`.
- IVF-PQ/RaBitQ/IVF-Flat search control planes are host loops (B3, B4, B12).
- NN-Descent / CAGRA init do not scale (B5).
- HETERO not wired (B6).
- Competitor benches are n=2000 (B1). Unit tests are n≤80 (CAGRA-Q n=32, recall floor 0.35).
- Lunar Lake unmeasured (B8).
- No FAISS import; DLPack is Python-only; Linux CI is CPU-only.
- Bindings other than C/C++/Python neighbors are demos.

Architecture unchanged. Decision logged in the plan §14: remaining critical path is B1→B2→B5→B3→B4, not new ABI. `OVVS_POLICY_HETERO` stays documented as unwired.

Canonical doc fixes made in the same change: plan status 0.2.0; AGENTS.md task order; `docs/devices.md` CAGRA walk description; README/CONTRIBUTING pointers here.
