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

The C ABI covers brute-force, IVF-Flat/PQ/RaBitQ, CAGRA, NN-Descent, Vamana, ScaNN, HNSW-from-CAGRA, all-neighbors, filters, dynamic batcher, k-means, SLINK, spectral, PQ/SQ/binary, PCA, pairwise, k-selection. Python ctypes + NumPy/DLPack ingest exist. Most algorithm tests remain toy-sized (**n=24–80**); bounded NN-Descent GPU regressions cover 4,097 and 16,384 rows. The current-code matched SIFT1M CAGRA gate now passes the plan's recall-closeness condition: 0.9036 recall@10 versus hnswlib 0.8915, a 0.0121 advantage. This is a 0.4979 absolute recovery from the pre-change CAGRA result, but both lanes remain below the 0.95 product target and hnswlib search was 6.02× faster in the load-contaminated diagnostic run.

On Arrow Lake 265K, AUTO dense primitive routing remains **CPU oneMKL-dominated**. The two current search hot-loop bets (iGPU CAGRA walk, NPU PQ ADC) are wired but do not beat matched FAISS-CPU / hnswlib gates. NPU ADC is fail-closed for FP16-range saturation; persistent storage plus fixed-bucket batching preserve CPU-equivalent bounded recall, but clean-code AUTO remains 1.40× slower than FORCE_CPU at `nprobe=8` and the wide-range forced-NPU lane is unavailable. Lunar Lake (plan v1.0 SKU) has no tables.

Evidence: `tables/arrow-lake/bench-recall-qps.md`, `gemm_large.json`, `docs/hw-split.md`.

| Path (SIFT n=2000 dim=128 nq=32 k=10) | Result |
|---|---|
| ovVS brute CPU | 0.89 ms / 36k QPS (beats FAISS brute) |
| ovVS brute NPU / GPU | NPU **invalid** (non-finite distances) / GPU 82 ms |
| ovVS IVF-Flat / IVF-PQ | IVF-PQ bounded AUTO recall matches FORCE_CPU after the ADC range guard; after persistent storage and batching, AUTO is still 1.40× slower at nprobe=8. The refined ovVS/raw FAISS smoke is not a matched competitor gate. |
| ovVS CAGRA | legacy CPU ~2.0 ms; B2 GPU 4.3 ms vs hnswlib 0.44 ms at the 32-query smoke point (recall vs brute about 0.99) |
| Package µJ/query brute | CPU 1049; NPU invalid output; GPU 17893 |

v1.0 (plan §16) still requires Lunar Lake accelerated paths, published FAISS/hnswlib benches, and mixer tables for that SKU. v0.2 API completeness is not that gate.

---

## Open items

Priority: P0 = objective is blocked without it; P1 = v1.0 quality; P2 = v1.1 / productization. IDs are stable.

### P0 — hardware benefits (the original mission)

| ID | Item | Plan | Notes |
|---|---|---|---|
| B1 | SIFT1M / 1e5×768 recall–QPS harness vs FAISS-CPU and hnswlib; AUTO + FORCE_*; µJ/query | T22.1–T22.2, §8–§9 | **PARTIAL:** strict CLI/JSON harness, bounded smoke validation, independent construction/search policy, a fail-closed matched SIFT1M CAGRA recall gate, and a contract-checked 100K SIFT-prefix preflight are complete. The current full gate passes: CAGRA 0.9036 recall@10 versus hnswlib 0.8915 at M=16/effort=32, with exact full-base truth, both lanes valid, and zero explicit CAGRA index uploads. Median QPS was 1,839.1/11,069.0, build wall 185.3/96.7 s, and isolated peak RSS 2,342.5/1,383.0 MiB. Energy was disabled and concurrent unrelated iGPU 3D load makes timing diagnostic. Full SIFT1M curves/energy, the other algorithms/policies, the ≥0.95 target point, and a real 100K×768 corpus report remain open. The pinned SIFT file is local and not committed. |
| B2 | CAGRA search kernel T13.4: one **work-group** per query, SLM itopk, bounded hashmap (not `Seen[nq×n]`), graph-aware seeds, subgroup distance | T13.4, then T13.5–T13.6 | **PARTIAL / T13.4 RECALL GATE PASSED; PERFORMANCE OPEN:** work-group/query, cooperative distance, SLM candidates, and a bounded visited hash are implemented. On current code the full SIFT1M matched point reached CAGRA 0.9036 versus hnswlib 0.8915; `hnswlib-CAGRA=-0.0121` passes the maximum 0.0200 gap. All 192 searches reported GPU and `192/192/0/0` walk/direct/upload-call/upload-byte deltas; driver-managed shared-memory migration was not measured. Query-content seeds are batch/order invariant, and the host build uses deterministic detour-rank/reverse-interleave optimization. The result remains below the 0.95 product target; hnswlib was 6.02× faster in the load-contaminated diagnostic run. The seeds are not graph-aware, and effort controls plus leader-serial selection/sort remain. |
| B3 | IVF-PQ search rewrite: persistent list codes, ADC tables batched over `nq×nprobe`, iGPU variable-length scan; padded-list NPU bakeoff | T10.2–T10.4, T9.3 | **PARTIAL / PERSISTENT CSR + SYNCHRONOUS FIXED-BUCKET BATCHING COMPLETE:** range-safe ADC, validated list-major storage, transactional lifecycle, bounded query/list descriptors, reused refinement workspaces, and whole-search publication are implemented without changing the C ABI or `IPQ1` v1. Equal fixed buckets `{128,256,512,1024,2048}` are packed into adaptive capacity-bounded NPU Gather requests; a six-chunk fixture executes as three requests with exact real-row mapping, and a split/regroup fixture executes bucket 2,048 live. Three clean-code SIFT-prefix runs at `aa716eb` retained recall 0.6375/0.946875 and measured median-of-run-median QPS 58,801.9/23,071.4 for AUTO versus 83,095.3/32,228.8 for FORCE_CPU at `nprobe=2/8`. CPU remains 1.41×/1.40× faster, and this remains diagnostic: wide-range FORCE_NPU is unavailable, raw FAISS has lower recall and no equivalent refinement, energy is disabled, and SIFT1M is unrun. Per-LUT scaling, bounded/versioned request caches, depth 2/4 bakeoffs, iGPU scan/select, and matched SIFT1M evidence remain open. Evidence: `tables/arrow-lake/ivfpq-b3.md`. |
| B4 | IVF-RaBitQ packed binary/INT GEMM (or SHAVE popcount), not scalar `rabitq_ip` | T11 | Plan: this is the NPU-native ANN, not a consolation prize. |
| B5 | NN-Descent iGPU local-join + bloom for n≫4096 | T12 | **PARTIAL / CAGRA-VS-HNSWLIB GATE PASSED; T12.5 OPEN:** n≤4096 remains exact kNN. Larger L2/L2-sqrt/IP/cosine builds use deterministic cached-distance NEW/OLD tagging, bounded forward/reverse samples, NEW–NEW and NEW–OLD reciprocal proposals, target-owned merges, and exact changed/pending-NEW convergence accounting on SYCL. Bounded sampled exact overlap improved from 0.7266 to 0.9727 at n=4,097 and from 0.2832 to 0.8750 at n=16,384; both cases exhausted their 6/4-iteration budgets with pending NEW edges. Current-code SIFT1M AUTO construction completed in 185.3 s within 2,342.5 MiB isolated peak process RSS and supported 0.9036 downstream CAGRA recall, closing the matched hnswlib-closeness gate. This is not the T12.5 IVF-PQ-initialization comparison. The final optimizer primitive remains CPU, so full FORCE_GPU build still rejects until T13.3. Standalone convergence evidence and the IVF-PQ comparison remain open; the explicit IVF-PQ initializer independently retains its `nlist≤8` per-row-search scale defect. |
| B6 | Define and wire an explicit heterogeneous producer/consumer DAG, not a third GEMM backend | T20, `docs/hw-split.md` | Currently equals AUTO. `prim_pq_adc` and `prim_graph_walk` belong to different algorithms, so “NPU ADC ∥ iGPU walk” is not yet a same-query pipeline. First candidates are IVF-PQ NPU ADC tile i+1 overlapped with iGPU scan/select tile i after B3, or a CAGRA candidate-slab pipeline after B9. Specify buffer ownership and dependencies, then use bounded queues, reusable requests/events, and per-stage wall/queue/byte telemetry. A current-stack 1-vs-4 request-pool microbenchmark is useful but is not an end-to-end HETERO win. Bound the current exact-shape request cache, include performance/effective compile properties in its key, and expose property fallback counters before adding throughput variants. OpenVINO GenAI continuous batching is a scheduler reference only. |

### P1 — v1.0 quality and SKU coverage

| ID | Item | Plan | Notes |
|---|---|---|---|
| B7 | NPU index home: probe dataset/codebook in an L0 tensor (`create_l0_host_tensor` or non-regressing Constant); quantify copies, memory, and wall versus request-owned tensors | NPU contract, `npu-gemm-dpu-vs-wall.md` | Sticky Parameter + fingerprint reuse is current. The installed OpenVINO 2025.3 headers already expose `create_l0_host_tensor`, shared-buffer wrapping, and mapped-file tensors, so start with a current-stack compile/allocate/bind probe; use an isolated 2026 upgrade for CPU-address import, strided IO, or missing support. Remote host memory cannot make a ~150 MB operand resident in 4 MB scratchpad and is not presumed to close the 5.3 ms DPU versus ~45 ms wall gap. Baking B as Constant regressed wall. Arrow Lake AUTO GEMM stays CPU until an end-to-end SKU table flips. |
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

1. **B1** harness core is complete; full curves/energy and the real 100K×768 corpus remain open.
2. **B2/B5** now pass the current-code matched SIFT1M recall-closeness gate; ≥0.95 recall, CAGRA QPS, GPU prune, and standalone NN-Descent convergence remain open.
3. **B3** remains the next implementation step for scaled ADC, bounded/versioned request caching, request-depth bakeoffs, and iGPU scan/select; then **B4** RaBitQ binary GEMM.
4. Continue B2 throughput work from profiling evidence; do not weaken hnswlib settings.
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

B1 remains partial. A narrow matched SIFT1M CAGRA quality result is now published below, but the full SIFT1M curve/energy matrix and real 100K×768 report remain open. The synthetic embedding profile is explicitly provisional, and CAGRA above 4,096 rows remains an explicit opt-in while B5 quality is unresolved.

### 2026-08-28 — B2 CAGRA work-group checkpoint and NPU range guard

The iGPU CAGRA walk now assigns one SYCL work-group per query, cooperatively reduces dimensions, stores candidates/expansion state in SLM, and uses a traversal-budget-sized global visited hash instead of `Seen[nq×n]`. Device allocations are strict RAII USM, oversized local/hash shapes reject explicitly, and asynchronous failures propagate. FORCE_GPU never crosses into the host walk; FORCE_NPU graph search returns `DEVICE_UNAVAILABLE`; adaptive host fallback records CPU.

Bounded SIFT-prefix evidence (`n=2000`, `nq=32`, `k=10`) is a correctness check only: FORCE_GPU recall@10 was 0.990625 at `itopk=32/search_width=1` and 1.0 at `64/2`, versus hnswlib 0.99375/1.0. Median QPS was about 7.4K/2.9K versus 71.7K/49.2K. Resource-local counters report direct shared-USM index access and zero explicit index uploads, including `128/128/0/0` walk calls/direct calls/upload calls/upload bytes for the batch-size-one point; its QPS deficit therefore lies elsewhere in the current walk.

The strict SIFT1M gate (`n=1,000,000`, `nq=1,000`, `k=10`, M=16, AUTO build, FORCE_GPU search) then failed on quality: CAGRA `itopk=32/search_width=1` recalled 0.4057 versus hnswlib `ef=32` at 0.8903. The 0.4846 absolute gap exceeds the 0.0200 limit. Both lanes and exact truth validated; CAGRA reported GPU on every observed search, zero NPU fallbacks, and `192/192/0/0` transfer deltas. The observed 1,549.7 versus 10,568.7 median QPS and 830.0 versus 61.7 s build times were collected under about 62.7% unrelated iGPU 3D load and are diagnostic, not publishable performance baselines. B2/B5 quality diagnosis now precedes QPS work. Reproducible details are in `tables/arrow-lake/bench-recall-qps.md` and `tables/arrow-lake/cagra-transfer-b2.md`.

The follow-up walk audit found exact CPU/GPU ID parity on the same bounded graphs but a shared correctness defect: seed selection used the query ordinal inside each C ABI call, so partitioning or reordering a batch could change results. CPU and SYCL now hash the query float bits to derive the same deterministic seed stream. A native regression verifies exact batched/single-query and reorder ID invariance, tight distance parity, and exact CPU/GPU IDs at `itopk=32/search_width=1`. This fixes reproducibility only; it does not make the seeds graph-aware or close the SIFT1M recall gate.

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

The 0.4979 absolute CAGRA recall gain over the earlier 0.4057 result is an end-to-end current-code result across the seed, graph-optimizer, and NN-Descent changes; it is not attributed to any one change. B1 remains partial for full curves/energy and the 100K×768 corpus. B2 remains partial for ≥0.95 recall and throughput. B5 remains partial for standalone convergence/IVF-PQ-init comparison and T13.3 GPU prune. The next implementation checkpoint is B3 persistent/batched IVF-PQ ADC, using reusable NPU requests and bounded software-only overlap; no BIOS change is required or planned.

### 2026-08-28 — B3 NPU ADC numeric-correctness checkpoint

The initial bounded SIFT-prefix smoke exposed a routing-dependent IVF-PQ defect on the same AUTO-built geometry: at `nprobe=8`, AUTO recall@10 was 0.378125 while FORCE_CPU reached 0.946875. Direct inspection of real `M=8, Ks=256` LUTs showed CPU candidate scores around 85,273.5–99,740.4 while full 128-row NPU tiles saturated at 65,504. Safe tile-boundary shapes at 127/128/129/255/256/257 codes retained a maximum observed error below 0.004, isolating numeric range rather than tile indexing.

NPU ADC now rejects non-finite LUTs or a conservative accumulation bound `sum_m max_code(abs(lut[m, code])) >= 65,504`, validates every request-owned tile, and publishes the full result atomically only after all tiles succeed. Unsafe AUTO calls execute the CPU oracle; FORCE_NPU returns `DEVICE_UNAVAILABLE` without publishing output. The final exact smoke rerun restored AUTO/FORCE_CPU recall parity: 0.6375 at `nprobe=2` and 0.946875 at `nprobe=8`. This is a correctness checkpoint, not an acceleration win. At `nprobe=8`, AUTO median QPS was 3,441.7 versus CPU 23,463.9, a 6.82× deficit. The smoke also refines ovVS candidates while the FAISS lane is raw IVF-PQ, so it is not a matched competitor comparison.

Software-only probes identify the next experiments without changing BIOS or firmware. With request-owned tensors already filled, a four-request pool completed 512 bucket-128 tasks in 43.228 ms versus 99.219 ms at depth one (2.30× infer-only throughput). At depth four and a fixed 524,288-code workload, measured throughput rose from 1.539M codes/s with bucket 128 to 18.596M codes/s with bucket 2,048. Per-LUT scaling to 60,000 headroom produced 0.9990 mean candidate top-32 overlap on 32 queries, but scaling, packing, output consumption, selection, energy, and end-to-end IVF-PQ wall were excluded. These results justify persistent codes, fixed buckets, scaling validation, and bounded async request pools; they do not close B3.

Current benchmark code now emits `float32_cast_on_load`; raw SIFT values were never normalized. The baseline artifact predates that label correction and retains the stale normalized label; the final guarded artifact carries the corrected label. Verification: 73/73 native tests, 6/6 CTest lanes, 57/57 benchmark-harness tests, and 6/6 SIFT-fetcher tests. CTest re-runs native subsets plus the two consumers, so its count is not additive. Detailed evidence and caveats are in `tables/arrow-lake/ivfpq-b3.md`.

### 2026-08-28 — B3 persistent IVF-PQ CSR checkpoint

IVF-PQ now keeps a derived list-major CSR view (`offsets`, original IDs, contiguous PQ codes) alongside the canonical row-major `IPQ1` state. Build, deserialize, and extend validate that every row occurs exactly once in its assigned list and construct the derived view before publication. Unfiltered search passes persistent code spans directly to ADC; filtered search compacts only allowed IDs and their aligned codes. Internal resource telemetry, updated once per successful search, measured 384 direct rows and zero host list-code scratch bytes across two `n=64`, `nq=3`, full-probe searches. Final candidate-ID aggregation and backend/request copies are not measured. An all-allowed filter copied the expected 768 bytes and returned exact ID/distance parity; a three-ID filter copied 12 bytes and preserved empty-result fill semantics.

Serialization remains byte-for-byte `IPQ1` v1. A hand-authored 268-byte legacy file deserializes, searches, and reserializes identically; unknown versions return `UNSUPPORTED`, while truncation and duplicate/out-of-range list IDs fail as I/O without publishing a handle or rebuild counter. Extend now propagates primitive failures before mutation, stages assignments/lists/codes/CSR, reserves the final dataset size only after those fallible steps, and commits with no-throw swaps. A forced NPU primitive failure leaves serialized bytes, search results, and rebuild telemetry unchanged; successful extend and reload preserve self-search parity. Index mutation/serialization/destruction require exclusive access, and concurrent search workers require distinct `Resources` objects.

Three repeated bounded SIFT-prefix smokes retained recall 0.6375/0.946875 at `nprobe=2/8`. Median-of-run-median QPS was 12,910.5/3,479.1 for AUTO and 74,888.8/31,815.5 for FORCE_CPU. FORCE_CPU ranges were 66,445.2–78,817.7 and 25,153.3–32,245.1 QPS, so the result is diagnostic rather than a controlled performance claim. AUTO remained 9.14× slower than CPU at `nprobe=8`; forced NPU remained explicitly unavailable, and the raw FAISS lane lacks equivalent refinement. B3 remains partial for batched work descriptors, fixed buckets, bounded request pools, scaled ADC, iGPU scan/select, and repeated matched SIFT1M recall/latency/QPS/RSS/energy evidence. No BIOS or firmware change was used or added to the plan.

Checkpoint verification: 76/76 accelerator-enabled native tests, 6/6 CTest lanes, 57/57 benchmark-harness tests, and 6/6 SIFT-fetcher tests. CTest re-runs native subsets plus the two consumers, so its count is not additive.

### 2026-08-28 — B3 synchronous fixed-bucket batching checkpoint

IVF-PQ search now converts nonempty query/list work into dense ordered descriptors, processes bounded complete-query blocks, and calls ADC once per block instead of once per list. One LUT is retained per logical list, filtered code storage is pre-sized before task pointers are published, and candidate order remains query/probe/list-row so TopK tie behavior does not change. Refinement workspaces are reused, and final neighbors/distances are copied to the caller only after every block succeeds. The public C ABI and `IPQ1` layout are unchanged.

The NPU backend splits lists into `{128,256,512,1024,2048}` buckets and packs equal buckets into adaptive power-of-two-capacity requests using `Gather(axis=1,batch_dims=1)` followed by `ReduceSum(axis=2)`. Inputs and outputs remain request-owned; padded indices select code zero and padded rows have no candidate IDs. The depth-one request is cached per fixed graph shape. A live six-chunk fixture submitted 1,792 capacity rows across three requests for 1,152 valid rows, including 384 intra-bucket and 256 inactive-slot padded rows; it mutated LUTs between passes and matched an independent oracle within 0.02 without CPU fallback. A second live fixture split a 2,049-row task, regrouped seven alternating chunks into four requests, and executed bucket 2,048 with exact output order. Invalid codes, unsafe forced execution, and FORCE_GPU fail before publishing output. The prior `[32,128]` latency probe reported one optimal request; the adaptive production shapes still need an explicit depth 1/2/4 bakeoff.

Three sequential clean-code SIFT-prefix runs at `aa716eb` retained recall 0.6375/0.946875 at `nprobe=2/8`. AUTO median-of-run-median QPS was 58,801.9/23,071.4 versus FORCE_CPU 83,095.3/32,228.8. CPU remains 1.41×/1.40× faster; AUTO's 4.55×/6.63× median ratios against the pre-batching checkpoint are not isolated batching or hardware wins because control-path/allocation changes also landed. The wide-range fixture still makes FORCE_NPU unavailable, AUTO's final primitive is CPU, raw FAISS is unrefined and recalls only 0.553125/0.6375, package energy is disabled, and the comparison with the preceding checkpoint is not simultaneous. The ovVS peak process RSS range was 141.3–141.8 MiB.

B3 remains partial for safe per-LUT scaling, a bounded cache keyed by effective compile/runtime/device identity, repeated request-depth 1/2/4 end-to-end evidence, iGPU variable-length scan/select, and matched SIFT1M recall/QPS/tail/RSS/energy. No BIOS or firmware change was used or added to the plan. Verification: 81/81 accelerator-enabled native tests with zero skips, 10/10 focused IVF-PQ/NPU tests, 6/6 CTest lanes, 57/57 benchmark-harness tests, and 6/6 SIFT-fetcher tests.

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

Architecture unchanged. Decision logged in the plan §14: after the current-code B2/B5 SIFT1M quality-gate pass, the next implementation step is B3→B4 while B1 curves/energy and B2 throughput remain open. `OVVS_POLICY_HETERO` stays documented as unwired.

Canonical doc fixes made in the same change: plan status 0.2.0; AGENTS.md task order; `docs/devices.md` CAGRA walk description; README/CONTRIBUTING pointers here.
