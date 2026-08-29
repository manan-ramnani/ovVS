# ovVS backlog journal

Living remaining-work log. **Does not replace the spec.**

- Spec (architecture, feature matrix, T0–T23): `.claude/plans/2026-08-20-ovvs-intel-cuvs-equivalent.md`
- Agent rules: `AGENTS.md`
- Device split: `docs/hw-split.md`
- If this file and the plan disagree on *design*, the plan wins. If they disagree on *what is still open*, this file wins until someone updates both.

How to use: append dated entries; check items off in **Open items** (do not delete IDs); move finished work to **Done**. Do not add new algorithm names to the C ABI because a row in the plan exists — v0.2 already has those symbols.

---

## Snapshot (2026-08-29)

ovVS **0.2.0** is a cuVS-shaped library with real NPU (OpenVINO) and iGPU (SYCL) primitive backends and honest Arrow Lake bakeoffs. It is **not** a hardware-accelerated cuVS equivalent.

The C ABI covers brute-force, IVF-Flat/PQ/RaBitQ, CAGRA, NN-Descent, Vamana, ScaNN, HNSW-from-CAGRA, all-neighbors, filters, dynamic batcher, k-means, SLINK, spectral, PQ/SQ/binary, PCA, pairwise, k-selection. Python ctypes + NumPy/DLPack ingest exist. Most algorithm tests remain toy-sized (**n=24–80**); bounded NN-Descent GPU regressions cover 4,097 and 16,384 rows. Three clean current-code SIFT1M CAGRA gates pass the plan's recall-closeness condition at the low-effort point: 0.9036 recall@10 versus hnswlib median 0.8911. The same graph reaches the separate 0.95 target at CAGRA `itopk=64/search_width=2` (0.9609). Exact cached-worst tracking, prefix-only publication sorting, canonical 64-ID frontier tiling, and subgroup min-location selection for multi-pick search are promoted. The cooperative-pick checkpoint preserved bitwise IDs/distances and improved median QPS by 17.8–29.9% across the three multi-pick rows in three serial-baseline and three candidate SIFT1M processes; the width-one serial-selection semantics showed non-attributable +2.0% variation, treated as neutral. Product performance is still negative: at the nearest ≥0.95 point current CAGRA reaches 3,243.5 QPS versus hnswlib 6,784.5, a 2.09× deficit; near 0.989 recall the deficit is 3.13×, and the higher-recall hnswlib batch-one point remains 42.55× faster. The exported single-layer graph loads, searches, saves, and reloads through stock hnswlib, but three clean SIFT100K diagnostics found it slower to build and search than native hnswlib. The frozen 256-byte CAGRA build telemetry ABI attributes 90.36% of the clean 100.09-second SIFT1M median build to GPU NN-Descent and 9.03 seconds to host optimize/prune/merge. All three builds hit the six-iteration cap rather than convergence.

On Arrow Lake 265K, AUTO dense primitive routing remains **CPU oneMKL-dominated**. The current search hot-loop bets are real device paths but do not beat matched FAISS-CPU / hnswlib gates. Corrected escape-route experiments found direct Level Zero slower than reused OpenVINO requests in every refill+execute+consume cell, every measured nonzero NPU+iGPU partition slower in its joined experimental scope, and certified residual-PQ Q/T factorization 1.93×/1.79× slower than the direct scalar oracle in its joined stage at 131K/524K rows. The Q/T stage excludes coarse assignment and exact-vector refinement. Arrow Lake's active experimental NPU capacity is therefore **zero**; FORCE_NPU correctness paths and immutable evidence remain, but further Arrow Lake tuning for a nonzero NPU share is parked. Unfiltered IVF-PQ avoids the dense 8-byte candidate-ID copy and resolves only the exact-refinement shortlist. A fused FORCE_GPU variable-list scan/select removes the dense ADC score array, but a clean bounded run remains 13.65× slower than FORCE_CPU at `nprobe=8`; AUTO therefore remains on the prior measured route. Lunar Lake (plan v1.0 SKU) has no tables.

Evidence: `tables/arrow-lake/cagra-build-v1.md`, `tables/arrow-lake/cagra-search-v1.md`, `tables/arrow-lake/cagra-frontier-v1.md`, `tables/arrow-lake/cagra-cooperative-pick-v1.md`, `tables/arrow-lake/bench-recall-qps.md`, `tables/arrow-lake/hnsw-export.md`, `tables/arrow-lake/npu-igpu-escape-routes.md`, `tables/arrow-lake/gemm_large.json`, and `docs/hw-split.md`.

| Path (SIFT n=2000 dim=128 nq=32 k=10) | Result |
|---|---|
| ovVS brute CPU | 0.89 ms / 36k QPS (beats FAISS brute) |
| ovVS brute NPU / GPU | NPU **invalid** (non-finite distances) / GPU 82 ms |
| ovVS IVF-Flat / IVF-PQ | IVF-PQ bounded AUTO recall matches FORCE_CPU after the ADC range guard. The fused FORCE_GPU scan/select retains recall but reaches 2,336 QPS versus FORCE_CPU 31,885 QPS at bounded `nprobe=8`; it is a correctness path, not an AUTO winner. The refined ovVS/raw FAISS smoke is not a matched competitor gate. |
| ovVS CAGRA | legacy CPU ~2.0 ms; B2 GPU 4.3 ms vs hnswlib 0.44 ms at the 32-query smoke point (recall vs brute about 0.99) |
| Package µJ/query brute | CPU 1049; NPU invalid output; GPU 17893 |

v1.0 (plan §16) still requires Lunar Lake accelerated paths, published FAISS/hnswlib benches, and mixer tables for that SKU. v0.2 API completeness is not that gate.

---

## Open items

Priority: P0 = objective is blocked without it; P1 = v1.0 quality; P2 = v1.1 / productization. IDs are stable.

### P0 — hardware benefits (the original mission)

| ID | Item | Plan | Notes |
|---|---|---|---|
| B1 | SIFT1M / 1e5×768 recall–QPS harness vs FAISS-CPU and hnswlib; AUTO + FORCE_*; µJ/query | T22.1–T22.2, §8–§9 | **PARTIAL:** strict CLI/JSON harness, bounded smoke validation, independent construction/search policy, fail-closed matched SIFT1M and prefix gates, loaded-DLL fingerprints, and an opt-in CAGRA→stock-hnswlib lane are complete. The current cooperative-pick checkpoint adds a clean repeated/interleaved CAGRA A/B curve: three serial-baseline and three candidate processes, complete tails and package energy, exact truth, and unchanged hnswlib settings. Current CAGRA reaches 0.9609 recall and 3,243.5 QPS versus hnswlib 0.9593 and 6,784.5 QPS. Other algorithms/policies, matched FAISS comparisons, a native-versus-WSL control before any Google ScaNN comparator, and a real 100K×768 corpus report remain open. Evidence: `tables/arrow-lake/cagra-build-v1.md`, `tables/arrow-lake/cagra-search-v1.md`, `tables/arrow-lake/cagra-cooperative-pick-v1.md`, `tables/arrow-lake/hnsw-export.md`. |
| B2 | CAGRA search kernel T13.4: one **work-group** per query, SLM itopk, bounded hashmap (not `Seen[nq×n]`), graph-aware seeds, subgroup distance | T13.4, then T13.5–T13.6 | **PARTIAL / COOPERATIVE PICK PROMOTED; PERFORMANCE OPEN:** work-group/query, cooperative distance, SLM candidates, and a bounded visited hash are implemented. Exact cached-worst tracking first delivered 81.9–156.5% QPS gains; prefix-only sorting and canonical frontier tiling added 8.0–10.9%. Exact subgroup min-location selection for `search_width > 1` then passed a separate six-process gate: multi-pick QPS improved 17.8%, 24.6%, and 29.9%, p50 fell 15.2–22.9%, and p99 fell 15.4–20.8%; the width-one serial-selection semantics showed non-attributable +2.0% variation, treated as neutral. Recall stayed unchanged, the cross-DLL fixture matched ID and distance bytes, build wall changed -0.33%, and peak RSS was unchanged. Product performance still loses: hnswlib is 2.09× faster at the nearest ≥0.95 point, 3.13× near 0.989 recall, and 42.55× at the non-recall-matched batch-one point. A stable one-pass pick was exact but rejected after regressing three of four effort points. Next gate a bounded owner/helper multi-work-group route only for batch one or very small batches; preserve the current one-work-group path for larger batches and fail closed when root-group capability is absent. Evidence: `tables/arrow-lake/cagra-cooperative-pick-v1.md`. |
| B3 | IVF-PQ search rewrite: persistent list codes, ADC tables batched over `nq×nprobe`, iGPU variable-length scan; padded-list NPU bakeoff | T10.2–T10.4, T9.3 | **PARTIAL / COMPLETE-CALL TELEMETRY LANDED:** persistent CSR, synchronous fixed buckets, bounded affine FORCE_NPU, shortlist-only ID resolution, and fused FORCE_GPU scan/select are complete. The frozen additive 200-byte telemetry ABI publishes success-only resource-local stage/counter snapshots. At bounded `nprobe=8`, FORCE_CPU reached 32,074 QPS and spent 73.6% of native wall constructing LUTs; FORCE_GPU reached 2,284 QPS and spent 85.5% in fused ADC plus final TopK, with 99 allocations, 120 kernel submissions, and 230 waits per call. AUTO is not promoted. Certified Q/T factorization is retained negative evidence: the best joined-stage path was 1.93×/1.79× slower than the direct scalar oracle at 131K/524K rows; that synthetic scope excludes coarse assignment and exact-vector refinement. Arrow Lake NPU request-depth/cache promotion is parked after refill+execute+consume losses. Active work is a workspace-byte planner, hierarchical iGPU top-k with one global merge/readback, packed AoSoA 4/6/8-bit codes, direct residual FP32 or validated-FP16 SLM LUTs, and batched/fused exact refinement with bounded persistent workspaces. `IPQ1` v1 remains unchanged. Evidence: `tables/arrow-lake/ivfpq-b3.md` and `tables/arrow-lake/npu-igpu-escape-routes.md`. |
| B4 | IVF-RaBitQ packed binary/INT GEMM (or SHAVE popcount), not scalar `rabitq_ip` | T11 | Retain the device path and correct packed binary math. On Arrow Lake this follows the measured iGPU work; it is not an active attempt to manufacture a nonzero NPU share. Reopen NPU promotion only on a SKU or software stack whose complete wall wins. |
| B5 | NN-Descent iGPU local-join + bloom for n≫4096 | T12 | **PARTIAL / BUILD TELEMETRY COMPLETE; T12.5 OPEN:** n≤4096 remains exact kNN. Larger L2/L2-sqrt/IP/cosine builds use deterministic cached-distance NEW/OLD tagging, bounded forward/reverse samples, reciprocal proposals, target-owned merges, and exact changed/pending-NEW accounting on SYCL. Three clean SIFT1M AUTO constructions completed in 99.874–100.525 s (100.093 s median) within 2,343.2 MiB median isolated peak RSS and supported 0.9036 downstream CAGRA recall. The GPU initializer contributes 90.444 s (90.36%) of median complete wall; host optimize/prune/merge contributes 9.034 s. Each build reports 1,098 kernels, 2,458 submissions, 1,820 waits, 459 D2H calls, zero H2D, and six iterations without convergence. The next bounded change must reduce initializer synchronization/readback without changing graph IDs/quality; T12.5 IVF-PQ-initializer comparison, K64 scale, and T13.3 remain open. Evidence: `tables/arrow-lake/cagra-build-v1.md`. |
| B6 | Define and wire an explicit heterogeneous producer/consumer DAG, not a third GEMM backend | T20, `docs/hw-split.md` | Currently equals AUTO. Arrow Lake NPU+iGPU composition is parked because every measured nonzero NPU partition lost the complete experimental composition scope. The active composition work is size-/deadline-gated CPU+iGPU scheduling: dynamic batches, resident iGPU state, bounded queues/events, and one final publication/readback. Only repeated complete wall/energy wins count. Future SKUs may reopen an NPU stage after an isolated primitive and its joined pipeline both win. OpenVINO GenAI continuous batching is a scheduler reference only. |

### P1 — v1.0 quality and SKU coverage

| ID | Item | Plan | Notes |
|---|---|---|---|
| B7 | NPU index home: probe dataset/codebook in an L0 tensor (`create_l0_host_tensor` or non-regressing Constant); quantify copies, memory, and wall versus request-owned tensors | NPU contract, `npu-gemm-dpu-vs-wall.md` | **PARKED ON ARROW LAKE:** public remote tensors may reduce an application copy but do not remove graph/DMA or retain a ~150 MB operand in a 4 MB scratchpad. Corrected direct Level Zero lost reused OpenVINO requests at both sizes and depths 1/2/4 when refill and output consumption were included. Two 524K prefilled-only cells slightly favored direct and are retained as incomplete-scope evidence, not an end-to-end gain. Keep the probe design for a future SKU/runtime capability change. |
| B8 | Lunar Lake mixer tables (`tables/<sku>/`). Do not copy Arrow Lake numbers | T2, §3.4, §16 | v1.0 is defined on Lunar Lake. Meteor Lake / Panther Lake still required when hardware exists. |
| B9 | CAGRA knobs: `min/max_iterations`, `team_size`, hashmap mode, `filtering_rate` → itopk; kernel portfolio and dynamic batching | T13.5–T13.8 | Search ABI today exposes `itopk_size` and `search_width` only. Build a measured SIMD16 portfolio: multi-work-group/query for small batches, one-work-group/query for large batches, and a bounded fallback. Arrow Lake NPU candidate-slab offload is parked. |
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
| B24 | HNSW-from-CAGRA recall closeness to CAGRA (cuVS claim), not only hnswlib `saveIndex` bytes | T14.4 | **PARTIAL / INTEROP COMPLETE, PROMOTION FAILED:** the exporter writes an honest single-layer graph that stock hnswlib loads, queries, saves, and reloads. Three clean SIFT100K runs produced the same 65,600,096-byte SHA-256-stable graph with 100% entry reachability. Hybrid recall reached 0.9657 at ef=64, but native hnswlib reached 0.9665 at ef=40 with 19.54k QPS versus 11.77k, and hybrid time-to-searchable was 5.779× worse. The export path is parked as an acceleration candidate. Full SIFT1M CAGRA-vs-export recall closeness remains open. Evidence: `tables/arrow-lake/hnsw-export.md`. |
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
- Further Arrow Lake Q/T, direct-Level-Zero, remote-index-home, or nonzero-NPU-partition tuning without a new runtime/SKU capability. The retained joined-stage and refill+execute+consume evidence is negative.

---

## Critical path (do in this order)

1. Continue promoted **B2** search work one variable at a time from `c9bb312`: gate a bounded owner/helper multi-work-group route only for batch one or very small batches. Preserve the one-work-group route for larger batches, query and canary the required SYCL root-group capability, and fail closed or retain the current route when it is absent. Compare against the frozen cooperative-pick DLL with exact old/new outputs and complete p50/p99/QPS/recall evidence; never weaken hnswlib settings. The exact stable one-pass pick is rejected evidence and must not be revived without a materially different design.
2. Reduce the measured **GPU NN-Descent initializer** wall without changing graph quality: the clean SIFT1M median is 90.444 s of a 100.093 s build, with 2,458 submissions, 1,820 waits, and 459 D2H calls. Start with synchronization/readback removal; compare interleaved complete walls and preserve exact structural counters or document intentional changes.
3. Move the 9.034-second host graph optimize/prune/merge stage to the iGPU only as a separately attributable change. The measured export-only route is parked.
4. Continue **B3** with hierarchical iGPU PQ top-k and one global merge/readback, then packed AoSoA 4/6/8-bit codes, direct residual LUT placement, a workspace-byte planner, and batched/fused refinement with bounded persistent workspaces. Q/T and Arrow Lake NPU request-depth work are parked.
5. Implement latency-/throughput-/deadline-aware dynamic batching and size-gated CPU+iGPU composition, then test nonnegative squared-L2 early abandonment.
6. Build a held-out, multi-objective per-SKU tuner across CPU/iGPU/CPU+iGPU and algorithm knobs; retain all invalid/negative lanes.
7. In parallel, repeat/interleave the selected CAGRA curve for promotion candidates, finish the remaining **B1** algorithms/policies and the real 100K×768 corpus. Bake off **B8** only when Lunar Lake hardware exists, then continue B9–B20 and B21–B32 in plan order.

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

### 2026-08-29 — CAGRA build-stage telemetry checkpoint

The C ABI now exports 112 functions. `ovvsResourcesCagraBuildStatsV1` adds a frozen 256-byte, version-1 snapshot without changing an algorithm signature or serialized index format. The resource-local counters are coherent, cumulative, saturating, and merged only after a complete CAGRA build is ready to publish. Failed validation, device, numeric, allocation, or runtime paths leave the snapshot unchanged. All field offsets are compile-time frozen in C and Python.

The inner timer covers private dataset copy plus finite validation, the selected NN-Descent/IVF-PQ/iterative initializer, CPU optimizer/prune/merge, and persistent graph materialization. It explicitly excludes telemetry merge, caller-handle publication, and serialization; the harness separately retains complete Python/API build wall. Call-local NN-Descent telemetry counts algorithm-owned USM allocations/bytes, optional H2D, every D2H, library kernels, all queue submissions, literal waits including RAII drains, iterations/convergence, and peak owned bytes. It does not infer these from the older resource-global last-call snapshot.

Python exposes `build_algo={nndescent,ivf_pq,iterative}` with a backward-compatible NN-Descent default. Ordinary profiles report a missing old-library symbol explicitly; the fixed SIFT100K/SIFT1M modes fail closed unless one fresh AUTO build proves an instrumented GPU NN-Descent initializer with exact geometry and valid structural counters. Every lane that successfully loads ovVS fingerprints that actual library by resolved path, SHA-256, size, and mtime, including post-load failures; fixed modes require it to match an explicit `--library` path. Verification is 94/94 native tests with zero skips, 8/8 configured CTest lanes, and 87/87 harness tests. This is instrumentation and correctness evidence, not a construction speedup.

### 2026-08-29 — Clean CAGRA build-stage evidence

Six sequential isolated processes at clean commit `5dc0ec7` used the explicitly matched DLL SHA-256 `c4ee36a1f7540726e7686c138190d201bfb825112ede9fdd6249048fab319f8f`. Three SIFT100K admission diagnostics and three SIFT1M matched gates all passed their validation contracts, exact byte geometry, GPU initializer attribution, structural counters, and output checks. The raw JSON is retained under `tables/arrow-lake/evidence/cagra-build-v1/`.

At SIFT1M, low-effort CAGRA recall was 0.9036 in every run versus hnswlib 0.8899–0.8915, but product performance remains negative. Median build wall was 100.093 versus 76.869 s, median search throughput 3,465 versus 11,417 QPS, amortized p99 1.090 versus 0.101 ms/query, and peak RSS 2,343.2 versus 1,387.0 MiB. This fixed low-effort point remains below the separate 0.95 recall target; the later same-index curve below reaches it with more traversal work.

The initializer is now the measured first construction target: median GPU NN-Descent wall was 90.444 s (90.36% of complete build), followed by 9.034 s in host optimize/prune/merge. All three runs reported 1,098 kernels, 2,458 submissions, 1,820 waits, 459 D2H calls, zero H2D, and six iterations without convergence. The historical 185.3-second contaminated wall remains historical evidence but is superseded for current performance status. Full data and interpretation: `tables/arrow-lake/cagra-build-v1.md`.

### 2026-08-29 — Same-index SIFT1M CAGRA search-effort curve

One clean complete process at `acd67e2` reused a single degree-16/intermediate-32 CAGRA graph across `32/1`, `64/2`, and `128/4` batch-32 search plus `128/4` batch one. The full-strength hnswlib lane retained M=16, `ef_construction=200`, 20 threads, and `ef=32/64/128/256`. All nine points passed exact-output validation, five timing passes, and whole-package energy sampling; all 7,704 CAGRA calls were direct GPU walks with zero explicit index uploads. The immutable JSON is under `tables/arrow-lake/evidence/cagra-search-v1/`.

CAGRA recall rose from 0.9036 to 0.9609 and 0.9889 on that same graph, so 0.9036 is not a hard topology ceiling below the product target. This does not prove topology parity or isolate `itopk_size` from `search_width`, which increased together. Near 0.96 recall, hnswlib was 5.71× faster and CAGRA used 3.74× more package energy/query. Near 0.99 recall, the deficits were 10.82× QPS and 13.43× energy. The non-recall-matched batch-one rows exposed a 156.37× raw comparator QPS gap even though hnswlib had higher recall.

This is one benchmark invocation with two sequential isolated lane processes and no in-run clock, thermal, utilization, or background-load trace, not a promotion claim. It moves the primary CAGRA search target to exact cached-worst candidate tracking, then deterministic SIMD16 cooperative selection and a multi-work-group/query small-batch route. Construction synchronization/readback remains a separate measured build target. Details: `tables/arrow-lake/cagra-search-v1.md`.

### 2026-08-29 — CAGRA cached-worst promotion

Three old-source and three candidate SIFT1M processes completed strict CAGRA and unchanged hnswlib lanes with exact truth, five timing passes, tails, package energy, and zero failed/skipped/timed-out/unavailable cells. After the original published baseline, the five new processes alternated candidate/old/candidate/old/candidate. Across the four CAGRA points, candidate QPS improved 81.9–156.5%, p50 fell 44.8–61.8%, and p99 fell 43.7–61.8%. Recall was identical in all candidate processes. A retained serialized-graph fixture also produced identical ID and distance-byte hashes under the original, independently rebuilt old-source, and final candidate DLLs.

The change therefore passes its internal promotion gate, with only +0.49% median build-wall and +0.64% peak-RSS variation on paths it does not modify. It is not a product acceleration claim: at the nearest ≥0.95 comparison, CAGRA reaches 2,549.9 QPS versus hnswlib 6,746.8, and the gaps remain 4.19× near 0.989 recall and 60.8× at the non-recall-matched batch-one point. Raw artifacts, hashes, process ranges, and next exact checkpoints: `tables/arrow-lake/cagra-cached-worst-v1.md`.

### 2026-08-29 — CAGRA canonical-frontier promotion

Prefix-only publication sorting and a canonical 64-ID frontier reduce leader work and explicit barriers without changing candidate order. The frontier compacts accepted seed and expansion IDs in the original order, scores them with the unchanged work-group reductions, and inserts them in that same order. Explicit barriers fall from 7,361→593, 27,009→1,185, and 103,169→2,369 at `32/1`, `64/2`, and `128/4`; these counts exclude the still-required distance collectives. The final DLL matched frozen baseline ID and float-distance bytes at all three efforts, including a native regression whose 640-ID raw frontier spans ten tiles. Verification was 94/94 native tests, 8/8 configured CTest lanes, and 87/87 harness tests.

Three frozen-baseline and three candidate SIFT1M processes completed all 54 requested points with exact truth, five measured passes, tails, package energy, and no failed/skipped/timed-out/unavailable cell. Candidate QPS improved 8.0–10.9%, p50 fell 7.3–9.7%, and p99 fell 7.2–11.0%, with unchanged recall. At the nearest ≥0.95 point CAGRA now reaches 2,762.9 QPS versus hnswlib 6,678.4; the remaining gaps are 2.42× there, 3.84× near 0.989 recall, and 54.8× at the non-recall-matched batch-one point. Package energy improved in three heavier rows but regressed with overlapping ranges at `32/1`, so no blanket energy claim is made. A separate exact stable one-pass pick regressed three of four effort points and was reverted. Evidence and immutable hashes: `tables/arrow-lake/cagra-frontier-v1.md`.

### 2026-08-29 — CAGRA cooperative multi-pick promotion

For `search_width > 1`, the first subgroup now performs an exact two-stage
minimum reduction over `(distance, candidate-slot)` while width one retains the
prior serial path. The change adds no SLM, work-group barrier, allocation,
submission, wait, or transfer. The final DLL matched the frozen frontier DLL's
ID and float-distance bytes at `32/1`, `64/2`, and `128/4`; verification was
94/94 native tests with zero skips, 8/8 configured CTest lanes, and 87/87
harness tests.

Three serial-baseline and three candidate SIFT1M processes alternated under a
clean `c9bb312` runner. All 54 points completed with exact truth, five measured
passes, tails, package energy, and no failed/skipped/timed-out/unavailable
cell. Width-one retained serial-selection semantics and showed
non-attributable +2.0% variation, treated as neutral;
median-of-run-median QPS improved +17.8%, +24.6%, and +29.9% on the three
multi-pick rows, while multi-pick p50 fell 15.2–22.9% and
p99 fell 15.4–20.8%, with unchanged recall. The remaining hnswlib gaps are
2.09× at the nearest ≥0.95 point, 3.13× near 0.989 recall, and 42.55× at the
non-recall-matched batch-one point. Whole-package energy was mixed, so no
blanket energy claim is made. Evidence and immutable hashes:
`tables/arrow-lake/cagra-cooperative-pick-v1.md`.

---

## Entries

### 2026-08-28 — B1 benchmark harness checkpoint

`tools/bench/bench.py` now runs isolated, timeout-bounded ovVS, FAISS-CPU, and hnswlib lanes; uses one exact oracle; validates IDs, duplicates, and finite distances; records warm/repeated latency, QPS, build time, final primitive attribution, and package energy; and writes versioned JSON plus generated Markdown. Required full profiles fail closed on incomplete recall, timing, policy, or energy evidence unless `--allow-partial` is explicit. Benchmark dependencies and an HTTPS SIFT fetcher with pinned size/hash (plus schema validation when h5py is installed) are included.

Bounded Arrow Lake smoke validation completed for every algorithm on FORCE_CPU and for brute-force across AUTO, FORCE_CPU, FORCE_NPU, FORCE_GPU, and HETERO. The harness exposed a real negative result: FORCE_NPU brute returned non-finite distances. A range guard now fails that lane closed as `DEVICE_UNAVAILABLE`; the legacy timing is invalid evidence. These runs validate the harness, not SIFT1M performance.

B1 remains partial. A narrow matched SIFT1M CAGRA quality result is now published below, but the full SIFT1M curve/energy matrix and real 100K×768 report remain open. The synthetic embedding profile is explicitly provisional, and CAGRA above 4,096 rows remains an explicit opt-in while B5 quality is unresolved.

### 2026-08-28 — B2 CAGRA work-group checkpoint and NPU range guard

The iGPU CAGRA walk now assigns one SYCL work-group per query, cooperatively reduces dimensions, stores candidates/expansion state in SLM, and uses a traversal-budget-sized global visited hash instead of `Seen[nq×n]`. Device allocations are strict RAII USM, oversized local/hash shapes reject explicitly, and asynchronous failures propagate. FORCE_GPU never crosses into the host walk; FORCE_NPU graph search returns `DEVICE_UNAVAILABLE`; adaptive host fallback records CPU.

Bounded SIFT-prefix evidence (`n=2000`, `nq=32`, `k=10`) is a correctness check only: FORCE_GPU recall@10 was 0.990625 at `itopk=32/search_width=1` and 1.0 at `64/2`, versus hnswlib 0.99375/1.0. Median QPS was about 7.4K/2.9K versus 71.7K/49.2K. Resource-local counters report direct shared-USM index access and zero explicit index uploads, including `128/128/0/0` walk calls/direct calls/upload calls/upload bytes for the batch-size-one point; its QPS deficit therefore lies elsewhere in the current walk.

The strict SIFT1M gate (`n=1,000,000`, `nq=1,000`, `k=10`, M=16, AUTO build, FORCE_GPU search) then failed on quality: CAGRA `itopk=32/search_width=1` recalled 0.4057 versus hnswlib `ef=32` at 0.8903. The 0.4846 absolute gap exceeds the 0.0200 limit. Both lanes and exact truth validated; CAGRA reported GPU on every observed search, zero NPU fallbacks, and `192/192/0/0` transfer deltas. The observed 1,549.7 versus 10,568.7 median QPS and 830.0 versus 61.7 s build times were collected under about 62.7% unrelated iGPU 3D load and are diagnostic, not publishable performance baselines. B2/B5 quality diagnosis now precedes QPS work. Reproducible details are in `tables/arrow-lake/bench-recall-qps.md` and `tables/arrow-lake/cagra-transfer-b2.md`.

The follow-up walk audit found exact CPU/GPU ID parity on the same bounded graphs but a shared correctness defect: seed selection used the query ordinal inside each C ABI call, so partitioning or reordering a batch could change results. CPU and SYCL now hash the query float bits to derive the same deterministic seed stream. The native non-exhaustive regression now exercises `itopk=32/search_width=1` over 2,048 rows, verifies exact batched/single-query and reorder IDs, requires bitwise-equal GPU distances, and retains tolerant CPU/GPU distance parity. This fixes reproducibility only; it does not make the seeds graph-aware or close the SIFT1M recall gate.

The B1 failure was traced separately to Arrow Lake NPU FP16-range behavior behind f32 tensors: large SIFT dot products clipped near 65,504 and TopK returned `+inf`. NPU GEMM/TopK now reject unsafe/non-finite ranges and validate outputs, so forced execution fails closed instead of returning corrupt success. Range-scaled execution remains future work.

### 2026-08-28 — B5 bounded NN-Descent iGPU checkpoint

For `n>4096`, NN-Descent now dispatches through the mixer to a synchronous iGPU kernel: one work-group owns each vertex, initialization is deterministic and unique, refinement reads/writes double-buffered graphs, dimensions reduce cooperatively, and bounded current/two-hop/exploration candidates use SLM plus a per-iteration Bloom-assisted duplicate filter. Transient graph storage is `O(n×degree)`; the kernel never allocates an `n²` score matrix. L2, L2-sqrt, inner product, and cosine are supported. FORCE_GPU either completes the kernel or returns unavailable, FORCE_NPU is unavailable and counted once, and adaptive execution does not silently enter the large host loop after an advertised GPU path fails.

The builder lifecycle now propagates primitive/device failures through standalone NN-Descent, default/explicit CAGRA initialization, IVF-PQ training/search, and Vamana; failed C ABI builds do not publish partial handles. GPU OOM and runtime failures remain distinct statuses, queued kernels drain before device storage is freed, and invalid metric/build enums reject. The standalone NN-Descent result retains only host graph IDs, not a duplicate dataset. The requested iteration count is honored instead of being silently raised to ten. The n≤4096 exact path now has an independent 1.0-overlap regression.

Bounded Arrow Lake evidence is deliberately below the product gate. Isolated-process medians were 107.1 ms for `n=4097, dim=8, degree=8, iterations=6` (5 runs; 106.8–1079.5 ms including first-use SYCL initialization) and 832.6 ms for `n=16384, dim=16, degree=16, iterations=4` (21 runs; 826.7–1333.2 ms). Exact-neighbor overlap remained 0.7266 on 64 sampled rows and 0.2832 on 32 rows. Both had valid unique non-self rows and GPU attribution; the 4,097-row graph was deterministic across repeats. AUTO CAGRA build plus FORCE_GPU search also passes at `n=4200`; the full build reports its final CPU prune and FORCE_GPU build rejects until T13.3. Details: `tables/arrow-lake/nndescent-b5.md`.

B5 remains partial. The SIFT1M build now completes, but its downstream CAGRA quality gate fails; sample/iteration tuning, reverse/new-edge/convergence scheduling, peak-RSS evidence, and T13.3 GPU prune remain open. GPU scale regressions are isolated CTest processes with 45/120-second timeouts and an Intel-GPU resource lock. The B1 harness defaults construction to AUTO independently of search policy and retains the explicit `--allow-unscalable-cagra` gate while quality is unresolved.

Checkpoint verification: 64/64 native tests, 6/6 CTest lanes, 45/45 benchmark-harness tests, and 6/6 SIFT-fetcher tests. Device-absent scale lanes report CTest SKIP rather than PASS.

### 2026-08-28 — CAGRA host rank-optimizer checkpoint

Commit `48a4171` replaces nearest-only host truncation with deterministic detour-count rank ordering, a flat capped reverse graph, mutual-edge deduplication, and forward-first reverse interleave while preserving exact fixed degree. Optimizer failures propagate before an index is published, and the iterative initializer now requests one extra result before removing self. Public C headers and serialization layout are unchanged.

This is a CPU algorithm-semantics checkpoint, not T13.3 SYCL completion. AUTO build still reports its final CPU stage and FORCE_GPU build still rejects. No SIFT1M recall or performance rerun has been made, so the earlier 0.4057 gate remains the latest measured failure rather than evidence that the optimizer helped. Verification at the checkpoint was 67/67 native tests and 6/6 CTest lanes, including all three GPU/scale entries.

### 2026-08-28 — B5 NEW/OLD and reverse-join quality checkpoint

The n>4096 SYCL initializer now retains cached distances and explicit NEW/OLD state, constructs bounded deterministic reverse samples, evaluates NEW–NEW and NEW–OLD pairs, emits reciprocal proposals, and merges each target row under a single deterministic owner. Exact changed-edge and pending-NEW counts gate convergence; non-finite input or computed overflow fails before a graph is published. Public ABI and mixer policy are unchanged.

Five isolated-process runs measured 0.9727 sampled exact overlap at n=4,097/K=8 and 0.8750 at n=16,384/K=16, up from 0.7266 and 0.2832. Median build walls were 1,022.4 ms and 1,931.4 ms. Both fixtures exhausted their 6/4-iteration budgets with pending NEW edges, so this is a bounded quality repair rather than a convergence or throughput win. Owned device allocations were 5,670,256 and 45,219,848 bytes. Verification is 69/69 native tests and 6/6 CTest lanes; independent race review also reproduced deterministic high-in-degree and concurrent builds.

B5 remains partial. At SIFT1M/K64 the direct-dataset allocation estimate is 1.116 GiB and the conservative per-iteration work bound remains large. The full CAGRA gate has not been rerun, and the latest 0.4057 recall is from the old initializer/prune path. Run an intermediate-scale memory/runtime preflight before spending another full SIFT1M build; do not infer a product gain from the bounded overlap alone.

### 2026-08-28 — SIFT100K CAGRA scale preflight

`tools/bench/bench.py --preflight-only cagra-sift100k` now fixes a noncanonical intermediate checkpoint: the first 100,000 vectors and 1,000 queries from the checksum-pinned SIFT source, exact FAISS `IndexFlatL2` truth recomputed for that prefix, AUTO construction, FORCE_GPU CAGRA search, M=16/effort=32, one warmup, five measured passes, seed 7, and 20 explicit hnswlib threads. Completion requires both lanes, repeated timing, isolated-worker peak RSS, exact policy attribution for every search, and the derived `192/192/0/0` direct-walk/zero-upload delta. Oracle failures and timeouts propagate; exit zero means this contract completed, not that recall passed. The artifact remains partial and cannot become canonical B1 evidence.

Three independent current-code Arrow Lake runs completed. CAGRA recall@10 was exactly 0.9718 each time versus hnswlib 0.9504–0.9525 (0.9510 median). Median QPS was 3,923.6 versus 22,142.6, with ranges 3,874.5–3,941.5 and 22,104.8–22,978.9. Median build wall was 8.686 versus 1.682 s; isolated peak RSS was 398.4 versus 201.2 MiB. CAGRA reported GPU for all 192 search calls in every run, zero NPU fallbacks, and `192/192/0/0` transfer deltas; AUTO construction reported CPU as its final primitive because host graph optimization remains. This proves current-code 100K practicality and good intermediate recall, while also retaining a 5.64× median search-throughput and 5.16× median build-wall deficit. Energy was intentionally disabled, and no in-run device-occupancy trace was captured, so timing is diagnostic rather than a publishable isolated baseline.

B1, B2, and B5 remain partial. The following current-code SIFT1M gate supersedes the older 0.4057 result. Harness verification is 57/57 tests plus Python compilation; details are in `tables/arrow-lake/bench-recall-qps.md`, `tables/arrow-lake/cagra-transfer-b2.md`, and `tables/arrow-lake/nndescent-b5.md`.

### 2026-08-28 — Current-code SIFT1M CAGRA quality-gate pass

The strict full-base gate completed on the checksum-pinned one-million-vector SIFT index with 1,000 queries, `k=10`, M=16, AUTO construction, FORCE_GPU search at `itopk=32/search_width=1`, one warmup, five measured passes, seed 7, and hnswlib `ef_construction=200/ef=32` with 20 explicit threads. Exact HDF5 truth and both lanes validated. CAGRA recall@10 was 0.9036 versus hnswlib 0.8915, so `hnswlib-CAGRA=-0.0121` passes the maximum 0.0200 gap. This closes the plan's T13.4 recall-closeness gate; it does not meet the separate ≥0.95 product target.

CAGRA reported GPU on all 192 search calls, zero attribution failures, zero NPU fallbacks, and `192/192/0/0` transfer deltas. AUTO build ended with the CPU host optimizer. CAGRA/hnswlib median QPS was 1,839.1/11,069.0, build wall 185.3/96.7 s, and isolated peak RSS 2,342.5/1,383.0 MiB. In this run hnswlib was 6.02× faster in search, 1.92× faster to build, and 1.69× lighter by peak RSS. Energy was disabled. During CAGRA construction the Intel compute engine was saturated while unrelated 3D processes concurrently consumed roughly 50–60%, so timing is diagnostic; recall and completion remain valid.

The 0.4979 absolute CAGRA recall gain over the earlier 0.4057 result is an end-to-end current-code result across the seed, graph-optimizer, and NN-Descent changes; it is not attributed to any one change. At this checkpoint B1 remained partial for full curves/energy and the 100K×768 corpus, while B2 remained partial for ≥0.95 recall and throughput. The later 2026-08-29 same-index entry supersedes that quality status: one curve reaches ≥0.95, while throughput and broader/repeated B1 evidence remain open. B5 remains partial for standalone convergence/IVF-PQ-init comparison and T13.3 GPU prune. No BIOS change is required or planned.

### 2026-08-28 — B3 NPU ADC numeric-correctness checkpoint

The initial bounded SIFT-prefix smoke exposed a routing-dependent IVF-PQ defect on the same AUTO-built geometry: at `nprobe=8`, AUTO recall@10 was 0.378125 while FORCE_CPU reached 0.946875. Direct inspection of real `M=8, Ks=256` LUTs showed CPU candidate scores around 85,273.5–99,740.4 while full 128-row NPU tiles saturated at 65,504. Safe tile-boundary shapes at 127/128/129/255/256/257 codes retained a maximum observed error below 0.004, isolating numeric range rather than tile indexing.

At this checkpoint, NPU ADC rejected non-finite LUTs or a conservative accumulation bound `sum_m max_code(abs(lut[m, code])) >= 65,504`, validated every request-owned tile, and published the full result atomically only after all tiles succeeded. Unsafe AUTO calls executed the CPU oracle; FORCE_NPU returned `DEVICE_UNAVAILABLE` without publishing output. The later bounded forced transform supersedes only that FORCE_NPU behavior. The final exact smoke rerun restored AUTO/FORCE_CPU recall parity: 0.6375 at `nprobe=2` and 0.946875 at `nprobe=8`. This is a correctness checkpoint, not an acceleration win. At `nprobe=8`, AUTO median QPS was 3,441.7 versus CPU 23,463.9, a 6.82× deficit. The smoke also refines ovVS candidates while the FAISS lane is raw IVF-PQ, so it is not a matched competitor comparison.

Software-only probes identify the next experiments without changing BIOS or firmware. With request-owned tensors already filled, a four-request pool completed 512 bucket-128 tasks in 43.228 ms versus 99.219 ms at depth one (2.30× infer-only throughput). At depth four and a fixed 524,288-code workload, measured throughput rose from 1.539M codes/s with bucket 128 to 18.596M codes/s with bucket 2,048. Per-LUT scaling to 60,000 headroom produced 0.9990 mean candidate top-32 overlap on 32 queries, but scaling, packing, output consumption, selection, energy, and end-to-end IVF-PQ wall were excluded. These results justify persistent codes, fixed buckets, scaling validation, and bounded async request pools; they do not close B3.

Current benchmark code now emits `float32_cast_on_load`; raw SIFT values were never normalized. The baseline artifact predates that label correction and retains the stale normalized label; the final guarded artifact carries the corrected label. Verification: 73/73 native tests, 6/6 CTest lanes, 57/57 benchmark-harness tests, and 6/6 SIFT-fetcher tests. CTest re-runs native subsets plus the two consumers, so its count is not additive. Detailed evidence and caveats are in `tables/arrow-lake/ivfpq-b3.md`.

### 2026-08-28 — B3 persistent IVF-PQ CSR checkpoint

IVF-PQ now keeps a derived list-major CSR view (`offsets`, original IDs, contiguous PQ codes) alongside the canonical row-major `IPQ1` state. Build, deserialize, and extend validate that every row occurs exactly once in its assigned list and construct the derived view before publication. Unfiltered search passes persistent code spans directly to ADC; filtered search compacts only allowed IDs and their aligned codes. Internal resource telemetry, updated once per successful search, measured 384 direct rows and zero host list-code scratch bytes across two `n=64`, `nq=3`, full-probe searches. At this checkpoint final candidate-ID aggregation and backend/request copies were not measured; the later shortlist-only ID checkpoint below removes the unfiltered dense ID copy. An all-allowed filter copied the expected 768 bytes and returned exact ID/distance parity; a three-ID filter copied 12 bytes and preserved empty-result fill semantics.

Serialization remains byte-for-byte `IPQ1` v1. A hand-authored 268-byte legacy file deserializes, searches, and reserializes identically; unknown versions return `UNSUPPORTED`, while truncation and duplicate/out-of-range list IDs fail as I/O without publishing a handle or rebuild counter. Extend now propagates primitive failures before mutation, stages assignments/lists/codes/CSR, reserves the final dataset size only after those fallible steps, and commits with no-throw swaps. A forced NPU primitive failure leaves serialized bytes, search results, and rebuild telemetry unchanged; successful extend and reload preserve self-search parity. Index mutation/serialization/destruction require exclusive access, and concurrent search workers require distinct `Resources` objects.

Three repeated bounded SIFT-prefix smokes retained recall 0.6375/0.946875 at `nprobe=2/8`. Median-of-run-median QPS was 12,910.5/3,479.1 for AUTO and 74,888.8/31,815.5 for FORCE_CPU. FORCE_CPU ranges were 66,445.2–78,817.7 and 25,153.3–32,245.1 QPS, so the result is diagnostic rather than a controlled performance claim. AUTO remained 9.14× slower than CPU at `nprobe=8`; forced NPU remained explicitly unavailable, and the raw FAISS lane lacks equivalent refinement. B3 remains partial for batched work descriptors, fixed buckets, bounded request pools, scaled ADC, iGPU scan/select, and repeated matched SIFT1M recall/latency/QPS/RSS/energy evidence. No BIOS or firmware change was used or added to the plan.

Checkpoint verification: 76/76 accelerator-enabled native tests, 6/6 CTest lanes, 57/57 benchmark-harness tests, and 6/6 SIFT-fetcher tests. CTest re-runs native subsets plus the two consumers, so its count is not additive.

### 2026-08-28 — B3 synchronous fixed-bucket batching checkpoint

IVF-PQ search now converts nonempty query/list work into dense ordered descriptors, processes bounded complete-query blocks, and calls ADC once per block instead of once per list. One LUT is retained per logical list, filtered code storage is pre-sized before task pointers are published, and candidate order remains query/probe/list-row so TopK tie behavior does not change. Refinement workspaces are reused, and final neighbors/distances are copied to the caller only after every block succeeds. The public C ABI and `IPQ1` layout are unchanged.

The NPU backend splits lists into `{128,256,512,1024,2048}` buckets and packs equal buckets into adaptive power-of-two-capacity requests using `Gather(axis=1,batch_dims=1)` followed by `ReduceSum(axis=2)`. Inputs and outputs remain request-owned; padded indices select code zero and padded rows have no candidate IDs. The depth-one request is cached per fixed graph shape. A live six-chunk fixture submitted 1,792 capacity rows across three requests for 1,152 valid rows, including 384 intra-bucket and 256 inactive-slot padded rows; it mutated LUTs between passes and matched an independent oracle within 0.02 without CPU fallback. A second live fixture split a 2,049-row task, regrouped seven alternating chunks into four requests, and executed bucket 2,048 with exact output order. Invalid codes, unsafe forced execution, and FORCE_GPU fail before publishing output. The prior `[32,128]` latency probe reported one optimal request; the adaptive production shapes still need an explicit depth 1/2/4 bakeoff.

Three sequential clean-code SIFT-prefix runs at `aa716eb` retained recall 0.6375/0.946875 at `nprobe=2/8`. AUTO median-of-run-median QPS was 58,801.9/23,071.4 versus FORCE_CPU 83,095.3/32,228.8. CPU remains 1.41×/1.40× faster; AUTO's 4.55×/6.63× median ratios against the pre-batching checkpoint are not isolated batching or hardware wins because control-path/allocation changes also landed. The wide-range fixture still makes FORCE_NPU unavailable, AUTO's final primitive is CPU, raw FAISS is unrefined and recalls only 0.553125/0.6375, package energy is disabled, and the comparison with the preceding checkpoint is not simultaneous. The ovVS peak process RSS range was 141.3–141.8 MiB.

B3 remains partial for safe per-LUT scaling, a bounded cache keyed by effective compile/runtime/device identity, repeated request-depth 1/2/4 end-to-end evidence, iGPU variable-length scan/select, and matched SIFT1M recall/QPS/tail/RSS/energy. No BIOS or firmware change was used or added to the plan. Verification: 81/81 accelerator-enabled native tests with zero skips, 10/10 focused IVF-PQ/NPU tests, 6/6 CTest lanes, 57/57 benchmark-harness tests, and 6/6 SIFT-fetcher tests.

### 2026-08-28 — B3 forced affine-range experiment

FORCE_NPU PQ ADC now removes candidate-invariant magnitude before inference: each subspace minimum is subtracted, the sum of remaining subspace spans is scaled to 60,000 when necessary, and the sum of minima is restored after validating the request-owned output. The complete result remains staged. This first lane accepts only scales at or above 0.5; non-finite inputs, wider transforms, invalid codes, or invalid restored outputs fail without publication. AUTO deliberately retains the prior unsafe-range CPU fallback until a complete-search promotion gate passes. Resource-local counters distinguish transformed chunks and rows from ordinary range-safe NPU work.

Live NPU tests cover a 65,536 constant-offset result, mixed safe/transformed tasks, a material scale of 0.588, and a rejected scale of about 0.029. The scale-0.588 fixture measured 27.25 maximum absolute ADC error and 32/32 CPU shortlist overlap; the wider case returned `DEVICE_UNAVAILABLE`, preserved canaries, incremented one forced fallback, and did not count a compile failure. This is bounded synthetic correctness evidence, not production recall.

The new `ovvs_bakeoff pq-adc-scale` lane includes validation, affine transform, u8-to-i32 Gather-index expansion, request-owned tensor fills, synchronous inference, output restoration, and publication; it excludes IVF coarse assignment, selection, and exact refinement. Across three fresh processes, median hot wall was 0.4363 ms CPU versus 46.2097 ms NPU for 131,072 candidates, and 2.5255 ms versus 185.1460 ms for 524,288. NPU was about 106× and 73.3× slower. GPU is explicitly unavailable for this primitive. Energy spans the whole multi-policy process and is not lane-attributable. The result fails the current AUTO promotion gate and makes the iGPU fused scan/select plus removal of expanded indices/full score materialization the primary speed experiment. Request depth 2/4 remains a secondary experiment after cache hardening and stage telemetry.

Verification at this checkpoint is 82/82 accelerator-enabled native tests with zero skips and 6/6 CTest lanes. The benchmark harness and SIFT fetcher were unchanged and were not rerun. No BIOS, firmware, driver, or power setting changed.

### 2026-08-28 — B3 shortlist-only ID resolution checkpoint

Unfiltered IVF-PQ search no longer materializes or copies an `int64` candidate ID beside every ADC score. TopK continues to operate on the same dense score order; only its `krefine` offsets are mapped through the ordered block descriptors to persistent list-major IDs. The filtered compact-copy path is unchanged. Resource counters report avoided unfiltered ID bytes and selected-ID resolutions only after successful publication. A 33-query direct-versus-all-allowed-filter regression crosses the 32-query block boundary with exact ID/distance parity, preserving order and tie behavior.

Three complete clean baseline runs at `f4336a7` and three complete clean post-change runs at `4e0bf87` used the same bounded SIFT prefix, runner, build settings, one warmup, and five measured passes. Recall remained 0.6375/0.946875 at `nprobe=2/8`. Median-of-run-median QPS changed as follows:

| Policy | nprobe 2 baseline → post | Ratio | nprobe 8 baseline → post | Ratio |
|---|---:|---:|---:|---:|
| AUTO | 57,224.6 → 58,856.0 | 1.03× | 19,573.1 → 21,627.5 | 1.10× |
| FORCE_CPU | 80,120.2 → 80,685.8 | 1.01× | 29,157.2 → 32,679.7 | 1.12× |

Median ovVS process RSS changed from about 141.1/141.0 MiB to 140.8/140.8 MiB for AUTO/CPU. The result is a bounded diagnostic A/B, not SIFT1M or a matched FAISS win: the FAISS lane is unrefined, quality is unmatched, and energy was disabled. Current FORCE_CPU is still 1.37×/1.51× faster than AUTO at `nprobe=2/8`. Verification: 82/82 accelerator-enabled native tests with zero skips and 6/6 CTest lanes. No BIOS, firmware, driver, or power setting changed.

### 2026-08-29 — B3 fused iGPU scan/select correctness checkpoint

FORCE_GPU IVF-PQ now scans persistent list-major uint8 codes directly, stages each query/list LUT in local memory, applies the allow bitset in-kernel, keeps exact `(score, query/probe/list-row ordinal)` TopK order, merges list shortlists per query, and publishes only validated packed positions and counts. It no longer materializes the candidate-sized ADC score array or a filtered code buffer. The final original-space refinement remains unchanged. AUTO/HETERO routing and `IPQ1` v1 are unchanged.

The strict test target covers CPU/GPU parity, null/all/selective/empty filters including `n=65` tail bits, a telemetry-proven 32+1 query split, `krefine=1/128`, atomic rejection at 129, deterministic ties, caller canaries, serialize/deserialize/extend, and concurrent searches through two resources sharing one index. That concurrency test found a real process-global GPU scratch race; typed scratch is now thread-local with RAII cleanup. Shared-USM Gather also fails closed after a direct SYCL failure instead of re-uploading the full dataset through OpenVINO.

Three complete clean runs at `a23de6a` used the bounded SIFT prefix (`n=2,000`, `nq=32`, `dim=128`, `k=10`, `nlist=32`, `pq_m=8`, `krefine=32`), one warmup, five measured passes, and no energy pass:

| Policy | nprobe 2 QPS, median (range) | nprobe 8 QPS, median (range) | Recall@10 nprobe 2/8 |
|---|---:|---:|---:|
| FORCE_CPU | 80,665.5 (80,543.7–80,951.2) | 31,885.2 (31,834.5–31,894.8) | 0.6375 / 0.946875 |
| FORCE_GPU | 3,555.4 (3,481.8–3,566.4) | 2,336.0 (1,623.7–2,507.6) | 0.6375 / 0.946875 |

CPU is 22.69×/13.65× faster at `nprobe=2/8`, so the GPU path is not promoted. The bounded raw-candidate cap admits only about four smoke queries per fused call at `nprobe=8`; every block still performs 9–10 device allocations and queue drains, and exact refinement issues roughly three synchronized GPU primitives per query. Post-kernel validation is also more expensive than necessary. These are structural leads, not stage-attributed measurements. Next is complete-call stage telemetry, then a byte-budgeted block planner, exact query/list LUT factorization, persistent workspaces, and batched refinement. The raw FAISS lane has substantially lower recall and no equivalent refinement, so it is not a matched competitor result. No BIOS, firmware, driver, or power setting changed. Verification: 88/88 native tests with zero skips, 7/7 CTest lanes, and 57/57 benchmark-harness tests.

### 2026-08-29 — B3 complete-call telemetry checkpoint

Commit `4fee489` adds a frozen additive `ovvsIvfPqSearchStatsV1` C ABI and optional Python binding. The 200-byte snapshot is cumulative per resource, saturating, mutex-coherent, and merged only after a successful IVF-PQ search publishes its complete output. Nine non-overlapping stages sum exactly to native call wall; GPU counters record explicit library-visible allocations, queued transfers, logical kernel submissions, and literal waits. `candidate_rows` counts real index rows submitted or inspected, including raw GPU rows before in-kernel filtering but excluding artificial NPU bucket padding. Shared-USM migration and opaque oneMKL/OpenVINO work are outside the structural counters. The harness measures warmup, timed, energy, and whole-point scopes separately; the primary record is timed-only and fails the point when present telemetry violates repeat/query counts, row bounds, ABI, monotonicity, or the stage sum. Older libraries without the symbol remain explicit but nonblocking.

Three sequential clean runs at `6a0f4c5` used the same bounded SIFT prefix, one warmup, five measured passes, and no energy pass. All nine lanes completed. Median-of-run-median QPS at `nprobe=2/8` was 79,820/32,074 for FORCE_CPU and 2,807/2,284 for FORCE_GPU, with unchanged recall 0.6375/0.946875. At `nprobe=8`, the median native CPU stage shares were LUT 73.6%, ADC 8.7%, shortlist 10.5%, refine 3.9%, coarse 1.4%, and planning 1.8%. The median GPU shares were fused ADC 45.5%, final TopK 40.0%, gather 6.6%, distance 4.4%, LUT 4.3%, coarse 1.2%, shortlist 0.3%, and planning 0.3%. One GPU `nprobe=8` run fell to 1,064 QPS, so neither a telemetry regression nor a GPU speed claim is made. The CPU QPS change versus the earlier clean checkpoint was -1.05%/+0.59% at `nprobe=2/8`, within the observed run variance.

The current raw-row cap produces 3/11 blocks per 32-query call at `nprobe=2/8`. FORCE_GPU performs 27/99 allocations, 104/120 logical kernel submissions, and 134/230 waits per call. This evidence promotes exact Q/T LUT factorization on CPU and batched/fused GPU refinement ahead of workspace reuse alone. The Q/T algebra is exact over reals but changes f32 association, so production promotion requires a conservative error band versus the shortlist cutoff gap and a fail-closed direct-LUT fallback; fixed oversampling alone is insufficient. No BIOS, firmware, driver, or power setting changed. Verification: 92/92 accelerator-enabled native tests with zero skips, 7/7 CTest lanes, 66/66 benchmark-harness tests, Python compilation, a real Windows Python/FAISS/ovVS import, and clean diff checks.

### 2026-08-29 — Arrow Lake escape-route and HNSW-export disposition

Two fail-closed experimental checkpoints close the active Arrow Lake NPU branch without deleting its device paths or evidence. The public `npu_compiler` checkout contains prebuilt kernels but no public kernel sources, descriptors, MoviTools compiler/linker, or matching isolated OpenVINO build; `add_extension`, renamed ELFs, host C, and caller markers are not treated as an NPU kernel. Corrected direct Level Zero graph submission lost reused OpenVINO requests at 131,072 and 524,288 rows for request depths 1/2/4 when refill and output consumption were included; two 524K prefilled-only cells slightly favored direct but are not an end-to-end gain. Exact NPU+iGPU local-top32 union plus 64-ID rerank preserved correctness but every nonzero NPU partition lost in the complete experimental composition scope. Certified residual-PQ Q/T factorization also lost its joined stage: the best CPU-Q+iGPU path was 0.8631/2.7617 ms versus the direct scalar oracle at 0.4479/1.5405 ms, or 1.93×/1.79× slower. That synthetic scope includes Q production, handoff, iGPU scan/select, readback, rescore, and synchronization but excludes coarse assignment and exact-vector refinement. NPU-Q was slower again. These are retained negative results; Arrow Lake active NPU capacity is zero and no integration or acceleration claim is made. Evidence: `tables/arrow-lake/npu-igpu-escape-routes.md` plus eight immutable JSON artifacts.

The HNSW exporter now writes an honest single-layer hnswlib layout and has native plus stock-hnswlib load/query/save/reload coverage. Three clean SIFT100K runs completed 9/9 lanes; the exported graph was 100% reachable from its entry, 65,600,096 bytes, and SHA-256 identical across runs. Its median time-to-searchable was 9.449 s versus 1.635 s for native hnswlib (5.779× slower). At the nearest measured comparable-recall points, native ef=32 reached 0.9512 recall and 22.13k QPS while hybrid ef=56 reached 0.9583 and 12.88k QPS (0.582×); native ef=40 reached 0.9665 and 19.54k QPS while hybrid ef=64 reached 0.9657 and 11.77k QPS (0.602×). Energy was disabled and SIFT100K is prefix evidence, so B24 remains partial for the full SIFT1M recall-closeness contract. The export branch is valid interop but is parked as an acceleration route. Evidence: `tables/arrow-lake/hnsw-export.md`.

The clean build-stage evidence attributes the first construction target to GPU NN-Descent synchronization/readback, followed by host optimize/prune/merge. The later same-index curve moved the primary search target to cached-worst tracking, which subsequently passed its repeated complete-wall gate. Prefix-only publication sorting, canonical frontier tiling, and cooperative multi-pick selection then passed separate repeated gates; the exact stable one-pass pick was rejected after a negative admission screen. Search work now proceeds to a separately gated owner/helper small-batch experiment. Hierarchical/packed iGPU PQ and size-gated CPU+iGPU batching remain on the critical path. No BIOS, firmware, driver, or power setting changed.

The same checkpoint closes a mixer-policy hole: AUTO/HETERO IVF-PQ had bypassed `ovvsResourcesSetNpuBusy(1)` and attempted range-safe NPU ADC directly. The mixer now skips that attempt when busy, while explicit FORCE_NPU retains its override. A live device regression observed zero additional NPU requests, CPU ADC rows, CPU attribution, and valid results. Verification is 93/93 accelerator-enabled native tests, 8/8 configured CTest lanes, and 71/71 benchmark-harness tests; the four optional escape-route targets also compile against current main.

### 2026-08-29 — CAGRA cached-worst exactness checkpoint

The SYCL walk now caches the first strict maximum candidate slot in the existing four-int local state. Appends maintain that invariant across the `itopk`-to-beam capacity growth; full-beam rejections become one comparison, while accepted replacements rebuild the same first-maximum slot with the prior strict scan. Distance production, candidate acceptance, traversal order, final sorting, work-group geometry, local-memory size, barriers, allocations, submissions, waits, and transfers are unchanged.

The 2,048-row `32/1` non-exhaustive regression now requires bitwise GPU batch/single/reorder distances and exact IDs. The 4,200-row scale test also compares repeated output bytes. At this checkpoint, a local serialized-graph diagnostic produced by the pre-change DLL matched candidate IDs and distance bits at `32/1`, `64/2`, and `128/4` under the rebuilt DLL. Verification was 94/94 native tests, all eight configured CTest lanes after rerunning the sandbox-denied interop subprocess outside that boundary, and 87/87 benchmark-harness tests. The later promotion checkpoint supersedes this section's performance status and now retains the graph plus original/rebuilt-old/final-candidate fingerprints under `tables/arrow-lake/evidence/cagra-cached-worst-v1/`.

### 2026-08-28 — Deep assessment vs cuVS-equivalent objective

Question: given the mission (port/adapt cuVS to Intel NPU+iGPU for hardware-accelerated benefits), what is pending?

Finding: **surface area is largely done; the reason to exist is not.** Remaining work is kernel quality and scale gates, not more symbols.

Shipped: C ABI 0.2.0, prim mixer, NPU OpenVINO path that obeys the NPU contract, SYCL iGPU path including a fused CAGRA walk entry, CPU oneMKL, Arrow Lake bakeoff that correctly flipped AUTO dense GEMM to CPU.

Not shipped relative to plan §5 “Accelerated” and §16 v1.0:

- At assessment time, CAGRA walk was not T13.4: it used USM heaps and `Seen[nq×n]`. The bounded work-group/SLM/hash checkpoint above supersedes that implementation; B2 remains open for quality and performance.
- IVF-PQ/RaBitQ/IVF-Flat search control planes are host loops (B3, B4, B12).
- At assessment time NN-Descent / CAGRA init did not scale. The current bounded SYCL join now completes the full SIFT1M AUTO build and passes the CAGRA-vs-hnswlib recall-closeness gate; B5 remains partial because T12.5 convergence/IVF-PQ-init comparison and GPU prune are open.
- HETERO not wired (B6).
- Competitor benches are n=2000 (B1). Unit tests are n≤80 (CAGRA-Q n=32, recall floor 0.35).
- Lunar Lake unmeasured (B8).
- No FAISS import; DLPack is Python-only; Linux CI is CPU-only.
- Bindings other than C/C++/Python neighbors are demos.

The 2026-08-29 disposition above supersedes this entry's old B3→B4 ordering. `OVVS_POLICY_HETERO` remains unwired, but active Arrow Lake composition work is CPU+iGPU; NPU stages reopen only after a new SKU/runtime produces a complete-wall win.

Canonical doc fixes made in the same change: plan status 0.2.0; AGENTS.md task order; `docs/devices.md` CAGRA walk description; README/CONTRIBUTING pointers here.
