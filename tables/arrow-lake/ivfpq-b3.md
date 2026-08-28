# Arrow Lake IVF-PQ B3 evidence

Status: **correctness guarded; persistent storage, synchronous fixed buckets, forced affine-range execution, shortlist-only ID resolution, and fused FORCE_GPU scan/select complete; throughput and end-to-end competitor gates open**. Hardware is the repository's Core Ultra 7 265K with OpenVINO 2025.3.0 and NPU driver 32.0.100.4841. No BIOS or firmware setting was changed.

The latest clean-code artifacts are `out/bench/ivfpq-fused-clean-r1.json` through `r3.json` at `a23de6a`. The matched shortlist-only A/B artifacts remain `out/bench/ivfpq-idmap-baseline-r1.json` through `r3.json` at `f4336a7` and `out/bench/ivfpq-idmap-clean-r1.json` through `r3.json` at `4e0bf87`. Earlier guarded, packed, and fixed-bucket artifacts remain local history. The latest runs are complete for FORCE_CPU, FORCE_GPU, and FAISS; energy was disabled. The ignored earliest baseline artifact predates the fixture-label correction and says `float32_normalized_on_load`; the loader only cast raw SIFT values. Later artifacts emit the corrected `float32_cast_on_load` label.

## Numeric defect and correction

The bounded checksum-pinned SIFT-prefix smoke uses `n=2,000`, `dim=128`, `nq=32`, `k=10`, `nlist=32`, `pq_m=8`, 8-bit codes, `krefine=32`, one warmup, and five measured passes. The comparison below is ovVS AUTO versus FORCE_CPU at identical search settings; it is not a publishable FAISS gate.

| Search | nprobe | Recall@10 | Median QPS |
|---|---:|---:|---:|
| AUTO before guard | 2 | 0.306250 | 782.1 |
| FORCE_CPU before guard | 2 | 0.637500 | 65,721.9 |
| AUTO before guard | 8 | 0.378125 | 197.7 |
| FORCE_CPU before guard | 8 | 0.946875 | 23,271.0 |
| AUTO after guard | 2 | 0.637500 | 12,589.5 |
| FORCE_CPU after guard | 2 | 0.637500 | 63,103.9 |
| AUTO after guard | 8 | 0.946875 | 3,441.7 |
| FORCE_CPU after guard | 8 | 0.946875 | 23,463.9 |

A one-off local per-list ADC diagnostic found CPU top candidate scores around 85,273.5–99,740.4 while full 128-row NPU tiles returned 65,504. Safe `M=8, Ks=256` tile-boundary probes at 127/128/129/255/256/257 codes had maximum absolute error below 0.004, so the primary defect was FP16-range saturation rather than code indexing or tail handling. No raw diagnostic artifact was preserved; these measurements are supporting diagnosis, not a benchmark result.

The original fail-closed guard checkpoint required:

- finite LUT entries;
- `sum_m max_code(abs(lut[m, code])) < 65,504`;
- finite request-owned output strictly below 65,504 before it is copied to caller memory.

At that checkpoint, unsafe AUTO execution fell back to CPU and unsafe FORCE_NPU returned `DEVICE_UNAVAILABLE`. The later bounded FORCE_NPU transform is documented below; AUTO deliberately retains the guard behavior. Multi-tile NPU results are staged and published atomically only after every tile validates. A same-index wide-range native regression verifies exact AUTO/CPU neighbor parity and distance parity within `1e-5 * max(1, abs(cpu_distance))`.

The guarded AUTO path is correct but not fast: at `nprobe=8`, FORCE_CPU was 6.82× faster. The smoke's ovVS lane uses exact refinement while its FAISS lane uses raw `IndexIVFPQ`, so no ovVS-versus-FAISS conclusion is drawn from this artifact.

## Persistent list-major storage

IVF-PQ now derives list offsets, original IDs, and contiguous PQ codes at build, deserialize, and extend. The row-major codes and list vectors remain the canonical `IPQ1` v1 state; the derived CSR is not serialized. Its additional storage is `n*(8+pq_m) + 8*(nlist+1)` bytes, about 15.27 MiB at SIFT1M with `pq_m=8` and `nlist=1,024`.

Focused native telemetry validates the intended runtime behavior:

| Operation | Direct ADC rows | Filter code-copy bytes | Result |
|---|---:|---:|---|
| Two unfiltered full-probe searches, `n=64`, `nq=3` | 384 | 0 | exact repeat ID/distance parity |
| All-allowed bitset, same three queries, `pq_m=4` | 0 | 768 | exact parity with unfiltered search |
| Three-ID allow list, one query | 0 | 12 | allowed-only results plus `-1/+inf` fill |

A hand-authored 268-byte legacy `IPQ1` file round-trips byte-for-byte. Truncation and invalid/duplicate list IDs fail without publishing state; version 2 reports `UNSUPPORTED`. Extend now stages assignments, lists, row-major codes, and CSR before mutation, propagates primitive failures, then appends the preserved input rows and commits via no-throw swaps. A forced-NPU failure leaves serialized bytes, search results, and rebuild telemetry unchanged.

Three repeated pre-batching smokes retained recall 0.6375 and 0.946875 at `nprobe=2/8`:

| Search policy | nprobe 2 QPS, three-run median (range) | nprobe 8 QPS, three-run median (range) |
|---|---:|---:|
| AUTO | 12,910.5 (12,381.5–13,015.5) | 3,479.1 (3,389.5–3,513.9) |
| FORCE_CPU | 74,888.8 (66,445.2–78,817.7) | 31,815.5 (25,153.3–32,245.1) |

One earlier guarded run measured FORCE_CPU at 63,103.9/23,463.9 QPS, but that single run plus the observed variance is not a controlled A/B. Direct persistent code spans and zero unfiltered host list-code scratch are proven; at this checkpoint final candidate-ID aggregation and backend/request copies were not measured. The shortlist-only ID checkpoint below later removes the unfiltered dense ID copy. At this checkpoint AUTO was 9.14× slower than FORCE_CPU at `nprobe=8`, consistent with repeated per-list AUTO attempts plus allocation and dispatch overhead. The fixed-bucket checkpoint below supersedes it while retaining the result as negative history; no retained stage timing isolates one cause.

## Batched descriptors and fixed buckets

IVF-PQ search now plans ordered query/list work descriptors in bounded complete-query blocks, stores one LUT per nonempty query/list, writes ADC scores into dense candidate slices, and invokes one internal batch primitive per block. The block admits at most 32 queries and caps raw candidate rows at index `n`, so a full-probe query remains a one-query block while narrower probes combine multiple queries. Candidate IDs retain the prior query/probe/list-row order, refinement workspaces are reused, and the complete `nq*k` result is staged before caller publication. Filters compact only allowed IDs/codes into pre-sized storage.

The NPU backend splits lists into fixed buckets `{128,256,512,1024,2048}` and packs equal buckets into capacity-bounded graph requests. A graph receives LUTs `[batch,M*Ks]` and flattened Gather indices `[batch,bucket,M]`, uses `Gather(axis=1,batch_dims=1)` and `ReduceSum(axis=2)`, fills padding with valid code-zero indices, and copies only real rows. The maximum capacity is the largest power of two not exceeding `min(32, 65536/(bucket*M))`; each request uses the smallest power-of-two capacity that fits its active group. Compilation and one request remain cached per `(M,Ks,bucket,capacity,latency)` shape. This checkpoint deliberately retains request depth one. An earlier latency-mode `[32,128]` direction probe reported one optimal request, but the adaptive production shapes have not been re-probed for their optimal count. Depth 2/4 remains a separate throughput bakeoff.

Native telemetry proves the grouping rather than inferring it from `last_device`: a six-task fixture covering 1,152 real rows and 384 intra-bucket padded rows executed as exactly three NPU requests using capacities 2/4/1 for buckets 128/256/512. It submitted 1,792 capacity rows: 1,152 valid, 384 intra-bucket padding, and 256 inactive-slot padding, with zero CPU fallback. The test mutates LUTs between passes, checks an independent scalar oracle within 0.02, validates output canaries, and confirms invalid codes and forced-policy failures publish nothing. A second live fixture alternates bucket classes and splits a 2,049-row task; seven chunks and 2,823 valid rows execute in four requests covering buckets 128/256/512/2,048, with 3,840 capacity rows and exact dense output order. Bucket 1,024 remains planner-covered but lacks equivalent retained live execution evidence. These are primitive correctness and routing facts, not an end-to-end speed claim.

Three sequential clean-code runs at commit `aa716eb` (`git.dirty=false`) on a SIFT prefix (`n=2,000`, `dim=128`, `nq=32`, `k=10`, `nlist=32`, `pq_m=8`, `krefine=32`, one warmup, five measured passes) produced:

| Search policy | nprobe 2 QPS, three-run median (range) | nprobe 8 QPS, three-run median (range) | Recall@10 at nprobe 2/8 | Peak process RSS range |
|---|---:|---:|---:|---:|
| AUTO | 58,801.9 (58,575.9–62,293.2) | 23,071.4 (22,709.5–23,338.9) | 0.6375 / 0.946875 | 141.3–141.6 MiB |
| FORCE_CPU | 83,095.3 (75,757.6–85,015.9) | 32,228.8 (32,154.3–32,817.1) | 0.6375 / 0.946875 | 141.4–141.8 MiB |

AUTO was 1.41×/1.40× slower than FORCE_CPU at `nprobe=2/8`; the `nprobe=8` ratio was 9.14× at the pre-batching checkpoint. AUTO's median ratio between checkpoints is 4.55×/6.63× while recall stayed exact, but this is not a simultaneous isolated A/B and includes control-path/allocation changes as well as batching. At that checkpoint, wide-range SIFT LUTs still failed the 65,504 bound, FORCE_NPU was explicitly unavailable in all three runs, and per-LUT scaling was unimplemented. AUTO attributed the final primitive to CPU.

Raw FAISS in the same clean runs had median-of-run-median QPS 143,626.6/149,045.2 at `nprobe=2/8`, with ranges 111,265.6–407,124.7 and 128,876.4–299,345.2. Recall was only 0.553125/0.6375; ovVS used `krefine=32` while FAISS was unrefined. This is retained as a high-variance unmatched competitor observation, not evidence that ovVS wins or loses at equal quality. All three artifacts remain partial because forced NPU is unavailable and package energy was disabled.

## Software-only direction probes

These one-off local probes prefilled request-owned tensors and exclude input packing, output consumption, candidate selection, memory growth, package energy, and full IVF-PQ wall. No raw artifact was preserved. They justify implementation experiments only.

| Reusable request depth | 512 bucket-128 tasks | Tasks/s |
|---:|---:|---:|
| 1 | 99.219 ms | 5,160 |
| 2 | 57.231 ms | 8,946 |
| 4 | 43.228 ms | 11,844 |

At depth four with a constant 524,288 scored codes, fixed buckets measured 1.539M, 2.869M, 5.918M, 10.927M, and 18.596M codes/s for bucket sizes 128, 256, 512, 1,024, and 2,048 respectively. Launch overhead dominates small tiles; SIFT1M with `nlist=1,024` averages about 977 codes per list, making bucket 1,024 the first end-to-end candidate rather than a selected winner.

An isolated per-LUT scaling probe used a conservative 60,000 headroom, then de-scaled each list before cross-list selection. Candidate top-32 overlap versus CPU averaged 0.9990 over 32 queries, with minimum 0.96875 and 31/32 queries exact. That probe motivated the bounded implementation below; it was not itself an end-to-end recall result.

## Forced affine-range experiment

The production backend now supports a deliberately bounded FORCE_NPU transform for LUTs whose raw absolute-sum guard fails. For subspace `m`, it subtracts `min_code(lut[m,code])`; it sums those minima as a task bias, and scales the sum of remaining subspace spans to at most 60,000. After synchronous inference, it requires each centered score to remain non-negative and within its planned span plus a conservative fp16 reduction envelope, divides by the scale, restores the bias, validates finite f32, and publishes only after the complete logical call succeeds. Exact arithmetic preserves every candidate score. NPU precision does not, so this first implementation rejects scales below 0.5. AUTO still falls back to CPU for raw-unsafe LUTs.

Live deterministic cases establish the bounded contract:

| Case | Result |
|---|---|
| Eight constant LUT offsets, exact score 65,536 | FORCE_NPU success; one transformed chunk and 129 transformed rows recorded |
| Mixed safe and constant-offset tasks | FORCE_NPU success; oracle match within `2e-4 * max(1,abs(score))` |
| Span 102,000, scale 0.588 | 27.25 maximum absolute error; 32/32 CPU top-32 overlap |
| Span 2,040,000, scale about 0.029 | `DEVICE_UNAVAILABLE`; output canaries preserved; one fallback; no compile failure |
| Zero span, finite LUT entries whose restored sum exceeds f32 | post-inference `DEVICE_UNAVAILABLE`; output canaries and prior attribution preserved; executed-request and runtime-failure telemetry retained |

`ovvs_bakeoff pq-adc-scale` uses the scale-0.588 table and includes validation, transformation, u8-to-i32 Gather-index construction, request tensor fills, inference, restoration, and publication. It excludes IVF list selection, final candidate selection, and refinement. Three fresh-process samples each ran one warmup, one timed call, and one hot call:

| Candidates, `M=8`, `Ks=256` | CPU hot median (range) | NPU hot median (range) | NPU/CPU wall |
|---:|---:|---:|---:|
| 131,072 | 0.4363 ms (0.4258–0.6352) | 46.2097 ms (46.0634–46.5929) | 105.9× |
| 524,288 | 2.5255 ms (2.5000–2.5357) | 185.1460 ms (184.8050–185.3009) | 73.3× |

GPU is unavailable for PQ ADC. Package-energy readings bracketed the complete multi-policy tool process and cannot be attributed to a lane. This is a strong negative result for the present Gather+ReduceSum dataflow, not for the NPU generally. The path still pays for expanded i32 indices, synchronous request traffic, and full score readback, but stage telemetry has not isolated their individual shares. The current diagnostic fails the AUTO promotion gate. The next speed path is iGPU variable-length fused scan/select, while the request cache is bounded/versioned and stage telemetry is added. Depth 2/4 remains a secondary bakeoff rather than a production default.

## Shortlist-only ID resolution

Unfiltered search no longer allocates and fills an `int64` ID array parallel to every ADC score. TopK still consumes the same dense score order. Only the selected `krefine` offsets are resolved through the ordered block descriptors to persistent list-major IDs; filtered search retains its compact allowed-ID/code storage. Resource counters record avoided candidate-ID bytes and selected-ID resolutions only after successful search publication.

Native regressions cover exact unfiltered parity, filtered-path counter isolation, and 33 queries crossing the 32-query block boundary. A matched three-run clean A/B used the bounded SIFT prefix (`n=2,000`, `nq=32`, `dim=128`, `k=10`, `nlist=32`, `pq_m=8`, `krefine=32`), one warmup, five measured passes, and identical runner settings:

| Policy | nprobe 2 baseline → post QPS | Ratio | nprobe 8 baseline → post QPS | Ratio | Recall@10 nprobe 2/8 |
|---|---:|---:|---:|---:|---:|
| AUTO | 57,224.6 → 58,856.0 | 1.03× | 19,573.1 → 21,627.5 | 1.10× | 0.6375 / 0.946875 |
| FORCE_CPU | 80,120.2 → 80,685.8 | 1.01× | 29,157.2 → 32,679.7 | 1.12× | 0.6375 / 0.946875 |

The values are medians of the three per-run medians. Post-change run ranges were 52,709.6–61,574.0 / 21,244.1–22,945.6 QPS for AUTO and 79,188.3–82,516.8 / 32,064.1–33,239.8 for FORCE_CPU at `nprobe=2/8`. Median process RSS was about 140.8 MiB for both policies versus 141.1/141.0 MiB before. Current CPU remains 1.37×/1.51× faster than AUTO. The result is a bounded diagnostic, not a SIFT1M or matched FAISS win: FAISS is unrefined, recall is unmatched, and energy was disabled.

## Fused iGPU scan/select

The FORCE_GPU path now scans the persistent list-major uint8 codes without publishing a candidate-sized ADC score array. One work-group handles each query/list task, stages its LUT in local memory, applies the optional allow bitset through persistent packed IDs, and maintains an exact `(score,dense ordinal)` list shortlist. A second kernel merges list shortlists per query. Only validated int32 packed positions and counts return to the host; original IDs are resolved for the exact-refinement shortlist. `krefine<=128` and `Ks<=256` are explicit current limits. AUTO, HETERO, the C ABI, and `IPQ1` v1 are unchanged.

The focused target covers CPU/GPU result parity, null/all/selective/empty filters, `n=65` tail bits, a telemetry-proven 32+1 query split, `krefine=1/128`, atomic rejection at 129, equal-score ordering, output canaries, serialize/deserialize/extend, and two concurrent resources searching one shared index. That last case exposed process-global typed GPU scratch; scratch is now thread-local and freed by a queue/context-owning RAII holder when the worker exits. A direct shared-USM Gather failure also returns unavailable instead of falling through to an OpenVINO full-dataset upload.

Three sequential clean runs at commit `a23de6a` used the same bounded SIFT prefix, one warmup, five measured passes, and no energy pass:

| Policy | nprobe 2 QPS, three-run median (range) | nprobe 8 QPS, three-run median (range) | Recall@10 nprobe 2/8 |
|---|---:|---:|---:|
| FORCE_CPU | 80,665.5 (80,543.7–80,951.2) | 31,885.2 (31,834.5–31,894.8) | 0.6375 / 0.946875 |
| FORCE_GPU | 3,555.4 (3,481.8–3,566.4) | 2,336.0 (1,623.7–2,507.6) | 0.6375 / 0.946875 |

CPU is 22.69×/13.65× faster at `nprobe=2/8`; FORCE_GPU reaches only 4.41%/7.33% of CPU throughput. The result is negative performance evidence, so no AUTO promotion is made. It is also bounded and energy-free. Raw FAISS reached lower recall because the harness comparator does not perform equivalent refinement; no competitor conclusion is drawn.

## Complete-call stage telemetry

Commit `4fee489` adds the frozen additive `ovvsIvfPqSearchStatsV1` ABI. Its 200-byte resource-local snapshot is cumulative, saturating, mutex-coherent, and merged only after the complete caller output is published successfully. Nine non-overlapping stage counters sum to native function wall. GPU structural counters cover explicit library-visible allocations, queued copies, logical kernel submissions, and literal waits; they intentionally exclude shared-USM page migration and opaque oneMKL/OpenVINO internals. `candidate_rows` counts real index rows submitted or inspected: the fused GPU path includes raw rows before its in-kernel filter, while synthetic NPU bucket padding is excluded.

The harness records warmup, timed, energy, and complete-point deltas separately. The primary record covers only the five timed passes and fails closed when a present record has the wrong ABI, moves backwards, is incomplete, disagrees with repeat/query counts, selects more rows than it scanned, or has a stage sum different from complete-call wall. Missing telemetry from an older library remains explicit and nonblocking.

Three fresh sequential processes at clean commit `6a0f4c5` used the bounded SIFT prefix (`n=2,000`, `nq=32`, `dim=128`, `k=10`, `nlist=32`, `pq_m=8`, `krefine=32`), one warmup, five measured passes, and no energy pass. Artifacts are `out/bench/ivfpq-stage-clean-r1.json` through `r3.json`; every ovVS and FAISS lane completed.

| Policy | nprobe | QPS, median of run medians | Recall@10 | Native wall/call, median run | Dominant median stage shares |
|---|---:|---:|---:|---:|---|
| FORCE_CPU | 2 | 79,820.4 | 0.6375 | 329.3 us | LUT 52.3%; shortlist 22.5%; ADC 8.2%; refine 10.2% |
| FORCE_CPU | 8 | 32,073.8 | 0.946875 | 927.1 us | LUT 73.6%; shortlist 10.5%; ADC 8.7%; refine 3.9% |
| FORCE_GPU | 2 | 2,807.2 | 0.6375 | 13,277.9 us | final TopK 62.6%; fused ADC 10.3%; gather 11.4%; distance 6.3% |
| FORCE_GPU | 8 | 2,283.8 | 0.946875 | 17,908.0 us | fused ADC 45.5%; final TopK 40.0%; gather 6.6%; distance 4.4% |

The native wall is the median of each run's cumulative timed delta divided by five calls; stage percentages are medians of per-run shares. Harness QPS includes the surrounding Python call boundary. One GPU `nprobe=8` run fell to 1,064.1 QPS and 30.06 ms batch p50, so the result supports bottleneck selection but not a speedup or regression claim. Against the earlier clean checkpoint, CPU median QPS changed by -1.05%/+0.59% at `nprobe=2/8`, within run variance. Raw FAISS remains lower recall because it lacks equivalent refinement.

The raw-row cap produces 3/11 blocks per 32-query call at `nprobe=2/8`, inspecting 143.2/536.2 real candidates per query. FORCE_GPU issues 27/99 allocations, 104/120 logical kernel submissions, and 134/230 waits per call; explicit queued traffic is 514.6/2,058.2 KiB H2D and 4.1/4.2 KiB D2H. These counts make the next order evidence-based: factor CPU LUT work first; batch/fuse GPU exact refinement before treating allocation reuse as sufficient.

Residual-PQ admits `||q-c_l-w||^2 = ||q-c_l||^2 - 2q·w + (||w||^2 + 2c_l·w)`, allowing one query term plus a derived persistent list term. The identity is exact over reals, but f32 reassociation and cancellation can change near ties. Promotion therefore requires a conservative score-error band smaller than the shortlist cutoff gap, followed by exact validation; otherwise the call must use the current direct LUT. Fixed oversampling is not a proof. Derived terms remain rebuildable from `IPQ1` v1 data, including after extend.

## Remaining B3 gate

1. Factor residual-PQ tables into persistent list terms plus one query term with a cutoff-gap certificate and direct-LUT fallback.
2. Replace the raw-candidate block cap with an explicit descriptor/LUT/workspace byte budget so batch size is not coupled to list fanout.
3. Batch or fuse exact GPU refinement, then reuse the stabilized layout through a bounded resource-owned GPU workspace.
4. Validate the bounded affine NPU transform on production LUTs through shortlist survival and final recall. Bound/version the request cache before measuring depths 1/2/4; depth one stays default.
5. Compare raw ovVS and raw FAISS at identical `nlist`, `nprobe`, `pq_m`, `nbits`, and `krefine=k`; refined comparisons need an equivalent FAISS refinement lane.
6. Require repeated end-to-end SIFT1M recall, QPS, tail latency, peak RSS, and package-energy evidence before claiming a win.

Latest checkpoint verification: 92/92 accelerator-enabled native tests with zero skips, 7/7 CTest lanes, and 66/66 benchmark-harness tests. Python compilation, a real Windows Python/FAISS/ovVS import, and diff checks also pass. Three clean benchmark artifacts completed with FORCE_CPU, FORCE_GPU, and FAISS all successful; energy was disabled. CTest re-runs native subsets plus the two consumers, so its count is not additive.
