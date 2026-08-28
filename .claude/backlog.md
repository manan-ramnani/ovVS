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

The C ABI covers brute-force, IVF-Flat/PQ/RaBitQ, CAGRA, NN-Descent, Vamana, ScaNN, HNSW-from-CAGRA, all-neighbors, filters, dynamic batcher, k-means, SLINK, spectral, PQ/SQ/binary, PCA, pairwise, k-selection. Python ctypes + NumPy/DLPack ingest exist. Most algorithm tests remain toy-sized (**n=24–80**); bounded NN-Descent GPU regressions now cover 4,097 and 16,384 rows. The published competitor bench is SIFT **n=2000**, not SIFT1M.

On Arrow Lake 265K AUTO search is still a **CPU oneMKL** library. The two remaining hardware bets (iGPU CAGRA walk, NPU PQ ADC) are wired but do not beat FAISS-CPU / hnswlib. Lunar Lake (plan v1.0 SKU) has no tables.

Evidence: `tables/arrow-lake/bench-recall-qps.md`, `gemm_large.json`, `docs/hw-split.md`.

| Path (SIFT n=2000 dim=128 nq=32 k=10) | Result |
|---|---|
| ovVS brute CPU | 0.89 ms / 36k QPS (beats FAISS brute) |
| ovVS brute NPU / GPU | NPU **invalid** (non-finite distances) / GPU 82 ms |
| ovVS IVF-Flat / IVF-PQ | slower than FAISS at same nprobe |
| ovVS CAGRA | legacy CPU ~2.0 ms; B2 GPU 4.3 ms vs hnswlib 0.44 ms at the 32-query smoke point (recall vs brute about 0.99) |
| Package µJ/query brute | CPU 1049; NPU invalid output; GPU 17893 |

v1.0 (plan §16) still requires Lunar Lake accelerated paths, published FAISS/hnswlib benches, and mixer tables for that SKU. v0.2 API completeness is not that gate.

---

## Open items

Priority: P0 = objective is blocked without it; P1 = v1.0 quality; P2 = v1.1 / productization. IDs are stable.

### P0 — hardware benefits (the original mission)

| ID | Item | Plan | Notes |
|---|---|---|---|
| B1 | SIFT1M / 1e5×768 recall–QPS harness vs FAISS-CPU and hnswlib; AUTO + FORCE_*; µJ/query | T22.1–T22.2, §8–§9 | **PARTIAL:** strict CLI/JSON harness and bounded smoke validation are complete. Construction now has one independent policy (AUTO by default) with status, timing, final-primitive, and fallback telemetry; search lanes apply their own policy afterward. SIFT1M is locally fetchable with a pinned hash, not committed. Full SIFT1M and a real 100K×768 corpus report remain open. |
| B2 | CAGRA search kernel T13.4: one **work-group** per query, SLM itopk, bounded hashmap (not `Seen[nq×n]`), graph-aware seeds, subgroup distance | T13.4, then T13.5–T13.6 | **PARTIAL:** work-group/query, cooperative distance, SLM candidates, and a bounded visited hash are implemented. Index dataset/graph storage attempts shared USM, but strict no-fallback/transfer-byte evidence at SIFT1M is still missing. Hash seeds and leader-serial selection/sort remain. The 2K smoke keeps recall but is roughly 10× slower than hnswlib at `itopk=32/search_width=1` and farther behind at `64/2`; the SIFT1M recall gate is still unrun. |
| B3 | IVF-PQ search rewrite: persistent list codes, ADC tables batched over `nq×nprobe`, iGPU variable-length scan; padded-list NPU bakeoff | T10.2–T10.4, T9.3 | `prim_pq_adc` on NPU is real. `ovvsIvfPqSearch` rebuilds residual + tables + packed codes **on the host per query per list**. That is why FAISS is ~6× faster on 2k points. |
| B4 | IVF-RaBitQ packed binary/INT GEMM (or SHAVE popcount), not scalar `rabitq_ip` | T11 | Plan: this is the NPU-native ANN, not a consolation prize. |
| B5 | NN-Descent iGPU local-join + bloom for n≫4096 | T12 | **PARTIAL:** n≤4096 remains exact kNN; larger L2/L2-sqrt/IP/cosine builds now use a deterministic, double-buffered SYCL sampled local join with SLM candidates/Bloom and honest forced-policy failures. AUTO C/Python CAGRA uses this initializer, then finishes with host prune; full FORCE_GPU CAGRA build therefore rejects until T13.3. SIFT1M quality/build evidence and convergence/new-edge scheduling remain open. The explicit IVF-PQ initializer independently retains its `nlist≤8` per-row-search scale defect. |
| B6 | Wire `OVVS_POLICY_HETERO` as stage overlap (NPU ADC ∥ iGPU walk), not a third GEMM backend | T20, `docs/hw-split.md` | Currently equals AUTO. |

### P1 — v1.0 quality and SKU coverage

| ID | Item | Plan | Notes |
|---|---|---|---|
| B7 | NPU index home: dataset / codebook as L0 tensor (`create_l0_host_tensor` or non-regressing Constant). Close wall vs DPU (5.3 vs ~45 ms on 1e5×32×768) | NPU contract, `npu-gemm-dpu-vs-wall.md` | Sticky Parameter + memcpy is current. Baking B as Constant made DPU faster and wall slower. Arrow Lake AUTO GEMM stays CPU until a SKU table flips. |
| B8 | Lunar Lake mixer tables (`tables/<sku>/`). Do not copy Arrow Lake numbers | T2, §3.4, §16 | v1.0 is defined on Lunar Lake. Meteor Lake / Panther Lake still required when hardware exists. |
| B9 | CAGRA knobs: `min/max_iterations`, `team_size`, hashmap mode, `filtering_rate` → itopk; NPU candidate-slab dist offload bakeoff | T13.5–T13.8 | Search ABI today: `itopk_size`, `search_width` only. |
| B10 | CAGRA-Q: codes in the iGPU walk + optional refine | T13.11 | API + host PQ-ADC walk; recall floor in tests is 0.35 on n=32. FORCE_GPU/FORCE_NPU reject this host-only walk instead of claiming accelerated success. |
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

1. **B1** harness core, then retain the full workload reports as an open gate.
2. **B2** CAGRA T13.4 kernel at bounded scale.
3. **B5** NN-Descent iGPU (CAGRA init at scale), then close B2's SIFT1M recall gate vs hnswlib within 2% at similar M/ef.
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

### 2026-08-28 — B1 benchmark harness checkpoint

`tools/bench/bench.py` now runs isolated, timeout-bounded ovVS, FAISS-CPU, and hnswlib lanes; uses one exact oracle; validates IDs, duplicates, and finite distances; records warm/repeated latency, QPS, build time, final primitive attribution, and package energy; and writes versioned JSON plus generated Markdown. Required full profiles fail closed on incomplete recall, timing, policy, or energy evidence unless `--allow-partial` is explicit. Benchmark dependencies and an HTTPS SIFT fetcher with pinned size/hash (plus schema validation when h5py is installed) are included.

Bounded Arrow Lake smoke validation completed for every algorithm on FORCE_CPU and for brute-force across AUTO, FORCE_CPU, FORCE_NPU, FORCE_GPU, and HETERO. The harness exposed a real negative result: FORCE_NPU brute returned non-finite distances. A range guard now fails that lane closed as `DEVICE_UNAVAILABLE`; the legacy timing is invalid evidence. These runs validate the harness, not SIFT1M performance.

B1 remains partial. No full SIFT1M or real 100K×768 report is published, the synthetic embedding profile is explicitly provisional, and CAGRA construction above 4,096 rows remains resource-gated by B5.

### 2026-08-28 — B2 CAGRA work-group checkpoint and NPU range guard

The iGPU CAGRA walk now assigns one SYCL work-group per query, cooperatively reduces dimensions, stores candidates/expansion state in SLM, and uses a traversal-budget-sized global visited hash instead of `Seen[nq×n]`. Device allocations are strict RAII USM, oversized local/hash shapes reject explicitly, and asynchronous failures propagate. FORCE_GPU never crosses into the host walk; FORCE_NPU graph search returns `DEVICE_UNAVAILABLE`; adaptive host fallback records CPU.

Bounded SIFT-prefix evidence (`n=2000`, `nq=32`, `k=10`) is a correctness check only: FORCE_GPU recall@10 was 0.990625 at `itopk=32/search_width=1` and 1.0 at `64/2`, versus hnswlib 0.99375/1.0. Median QPS was about 7.4K/2.9K versus 71.7K/49.2K, and the batch-size-one lane exposed the cost of copying dataset/graph buffers per search. B2 remains partial; graph-aware seeds, parallel candidate selection, persistent index buffers, B5 scale, and the SIFT1M gate are open. Reproducible details are in `tables/arrow-lake/bench-recall-qps.md`.

The B1 failure was traced separately to Arrow Lake NPU FP16-range behavior behind f32 tensors: large SIFT dot products clipped near 65,504 and TopK returned `+inf`. NPU GEMM/TopK now reject unsafe/non-finite ranges and validate outputs, so forced execution fails closed instead of returning corrupt success. Range-scaled execution remains future work.

### 2026-08-28 — B5 bounded NN-Descent iGPU checkpoint

For `n>4096`, NN-Descent now dispatches through the mixer to a synchronous iGPU kernel: one work-group owns each vertex, initialization is deterministic and unique, refinement reads/writes double-buffered graphs, dimensions reduce cooperatively, and bounded current/two-hop/exploration candidates use SLM plus a per-iteration Bloom-assisted duplicate filter. Transient graph storage is `O(n×degree)`; the kernel never allocates an `n²` score matrix. L2, L2-sqrt, inner product, and cosine are supported. FORCE_GPU either completes the kernel or returns unavailable, FORCE_NPU is unavailable and counted once, and adaptive execution does not silently enter the large host loop after an advertised GPU path fails.

The builder lifecycle now propagates primitive/device failures through standalone NN-Descent, default/explicit CAGRA initialization, IVF-PQ training/search, and Vamana; failed C ABI builds do not publish partial handles. GPU OOM and runtime failures remain distinct statuses, queued kernels drain before device storage is freed, and invalid metric/build enums reject. The standalone NN-Descent result retains only host graph IDs, not a duplicate dataset. The requested iteration count is honored instead of being silently raised to ten. The n≤4096 exact path now has an independent 1.0-overlap regression.

Bounded Arrow Lake evidence is deliberately below the product gate. Isolated-process medians were 107.1 ms for `n=4097, dim=8, degree=8, iterations=6` (5 runs; 106.8–1079.5 ms including first-use SYCL initialization) and 832.6 ms for `n=16384, dim=16, degree=16, iterations=4` (21 runs; 826.7–1333.2 ms). Exact-neighbor overlap remained 0.7266 on 64 sampled rows and 0.2832 on 32 rows. Both had valid unique non-self rows and GPU attribution; the 4,097-row graph was deterministic across repeats. AUTO CAGRA build plus FORCE_GPU search also passes at `n=4200`; the full build reports its final CPU prune and FORCE_GPU build rejects until T13.3. Details: `tables/arrow-lake/nndescent-b5.md`.

B5 remains partial. The SIFT1M downstream-CAGRA comparison, sample/iteration tuning, new-edge/convergence scheduling, peak-RSS evidence, and T13.3 GPU prune are open. GPU scale regressions are isolated CTest processes with 45/120-second timeouts and an Intel-GPU resource lock. The B1 harness now defaults construction to AUTO independently of search policy, but retains the explicit `--allow-unscalable-cagra` gate until a full measured build succeeds.

Checkpoint verification: 63/63 native tests, 6/6 CTest lanes, 19/19 benchmark-harness tests, and 6/6 SIFT-fetcher tests. Device-absent scale lanes report CTest SKIP rather than PASS.

### 2026-08-28 — Deep assessment vs cuVS-equivalent objective

Question: given the mission (port/adapt cuVS to Intel NPU+iGPU for hardware-accelerated benefits), what is pending?

Finding: **surface area is largely done; the reason to exist is not.** Remaining work is kernel quality and scale gates, not more symbols.

Shipped: C ABI 0.2.0, prim mixer, NPU OpenVINO path that obeys the NPU contract, SYCL iGPU path including a fused CAGRA walk entry, CPU oneMKL, Arrow Lake bakeoff that correctly flipped AUTO dense GEMM to CPU.

Not shipped relative to plan §5 “Accelerated” and §16 v1.0:

- At assessment time, CAGRA walk was not T13.4: it used USM heaps and `Seen[nq×n]`. The bounded work-group/SLM/hash checkpoint above supersedes that implementation; B2 remains open for quality and performance.
- IVF-PQ/RaBitQ/IVF-Flat search control planes are host loops (B3, B4, B12).
- NN-Descent / CAGRA init do not scale (B5).
- HETERO not wired (B6).
- Competitor benches are n=2000 (B1). Unit tests are n≤80 (CAGRA-Q n=32, recall floor 0.35).
- Lunar Lake unmeasured (B8).
- No FAISS import; DLPack is Python-only; Linux CI is CPU-only.
- Bindings other than C/C++/Python neighbors are demos.

Architecture unchanged. Decision logged in the plan §14: remaining critical path is B1→B2→B5→B3→B4, not new ABI. `OVVS_POLICY_HETERO` stays documented as unwired.

Canonical doc fixes made in the same change: plan status 0.2.0; AGENTS.md task order; `docs/devices.md` CAGRA walk description; README/CONTRIBUTING pointers here.
