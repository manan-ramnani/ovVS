# Arrow Lake IVF-PQ B3 evidence

Status: **correctness guarded; persistent storage and synchronous fixed-bucket batching complete; throughput and end-to-end competitor gates open**. Hardware is the repository's Core Ultra 7 265K with OpenVINO 2025.3.0 and NPU driver 32.0.100.4841. No BIOS or firmware setting was changed.

The current clean-code artifacts are `out/bench/ivfpq-b3-batched-sift-postcommit-r1.json` through `r3.json`, with sibling Markdown files. Earlier baseline, guarded, and packed artifacts remain local negative history. Every run is overall **partial** because the forced-NPU end-to-end lane is explicitly unavailable at `nprobe=2/8`; that lane is retained rather than filtered. The ignored baseline artifact predates the fixture-label correction and says `float32_normalized_on_load`; the loader only cast raw SIFT values. Later artifacts emit the corrected `float32_cast_on_load` label.

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

`npu_pq_adc` now requires:

- finite LUT entries;
- `sum_m max_code(abs(lut[m, code])) < 65,504`;
- finite request-owned output strictly below 65,504 before it is copied to caller memory.

Unsafe AUTO execution falls back to CPU. Unsafe FORCE_NPU returns `DEVICE_UNAVAILABLE`, increments the fallback counter, does not count a compile failure, and does not publish output. Multi-tile NPU results are staged and published atomically only after every tile validates. A same-index wide-range native regression verifies exact AUTO/CPU neighbor parity and distance parity within `1e-5 * max(1, abs(cpu_distance))`.

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

One earlier guarded run measured FORCE_CPU at 63,103.9/23,463.9 QPS, but that single run plus the observed variance is not a controlled A/B. Direct persistent code spans and zero unfiltered host list-code scratch are proven; final candidate-ID aggregation and backend/request copies are not measured. At this checkpoint AUTO was 9.14× slower than FORCE_CPU at `nprobe=8`, consistent with repeated per-list AUTO attempts plus allocation and dispatch overhead. The fixed-bucket checkpoint below supersedes it while retaining the result as negative history; no retained stage timing isolates one cause.

## Batched descriptors and fixed buckets

IVF-PQ search now plans ordered query/list work descriptors in bounded complete-query blocks, stores one LUT per nonempty query/list, writes ADC scores into dense candidate slices, and invokes one internal batch primitive per block. The block admits at most 32 queries and caps raw candidate rows at index `n`, so a full-probe query remains a one-query block while narrower probes combine multiple queries. Candidate IDs retain the prior query/probe/list-row order, refinement workspaces are reused, and the complete `nq*k` result is staged before caller publication. Filters compact only allowed IDs/codes into pre-sized storage.

The NPU backend splits lists into fixed buckets `{128,256,512,1024,2048}` and packs equal buckets into capacity-bounded graph requests. A graph receives LUTs `[batch,M*Ks]` and flattened Gather indices `[batch,bucket,M]`, uses `Gather(axis=1,batch_dims=1)` and `ReduceSum(axis=2)`, fills padding with valid code-zero indices, and copies only real rows. The maximum capacity is the largest power of two not exceeding `min(32, 65536/(bucket*M))`; each request uses the smallest power-of-two capacity that fits its active group. Compilation and one request remain cached per `(M,Ks,bucket,capacity,latency)` shape. This checkpoint deliberately retains request depth one. An earlier latency-mode `[32,128]` direction probe reported one optimal request, but the adaptive production shapes have not been re-probed for their optimal count. Depth 2/4 remains a separate throughput bakeoff.

Native telemetry proves the grouping rather than inferring it from `last_device`: a six-task fixture covering 1,152 real rows and 384 intra-bucket padded rows executed as exactly three NPU requests using capacities 2/4/1 for buckets 128/256/512. It submitted 1,792 capacity rows: 1,152 valid, 384 intra-bucket padding, and 256 inactive-slot padding, with zero CPU fallback. The test mutates LUTs between passes, checks an independent scalar oracle within 0.02, validates output canaries, and confirms invalid codes and forced-policy failures publish nothing. A second live fixture alternates bucket classes and splits a 2,049-row task; seven chunks and 2,823 valid rows execute in four requests covering buckets 128/256/512/2,048, with 3,840 capacity rows and exact dense output order. Bucket 1,024 remains planner-covered but lacks equivalent retained live execution evidence. These are primitive correctness and routing facts, not an end-to-end speed claim.

Three sequential clean-code runs at commit `aa716eb` (`git.dirty=false`) on a SIFT prefix (`n=2,000`, `dim=128`, `nq=32`, `k=10`, `nlist=32`, `pq_m=8`, `krefine=32`, one warmup, five measured passes) produced:

| Search policy | nprobe 2 QPS, three-run median (range) | nprobe 8 QPS, three-run median (range) | Recall@10 at nprobe 2/8 | Peak process RSS range |
|---|---:|---:|---:|---:|
| AUTO | 58,801.9 (58,575.9–62,293.2) | 23,071.4 (22,709.5–23,338.9) | 0.6375 / 0.946875 | 141.3–141.6 MiB |
| FORCE_CPU | 83,095.3 (75,757.6–85,015.9) | 32,228.8 (32,154.3–32,817.1) | 0.6375 / 0.946875 | 141.4–141.8 MiB |

AUTO was 1.41×/1.40× slower than FORCE_CPU at `nprobe=2/8`; the `nprobe=8` ratio was 9.14× at the pre-batching checkpoint. AUTO's median ratio between checkpoints is 4.55×/6.63× while recall stayed exact, but this is not a simultaneous isolated A/B and includes control-path/allocation changes as well as batching. Wide-range SIFT LUTs still fail the 65,504 bound, so FORCE_NPU remained explicitly unavailable in all three runs and AUTO attributed the final primitive to CPU. Per-LUT scaling remains unimplemented.

Raw FAISS in the same clean runs had median-of-run-median QPS 143,626.6/149,045.2 at `nprobe=2/8`, with ranges 111,265.6–407,124.7 and 128,876.4–299,345.2. Recall was only 0.553125/0.6375; ovVS used `krefine=32` while FAISS was unrefined. This is retained as a high-variance unmatched competitor observation, not evidence that ovVS wins or loses at equal quality. All three artifacts remain partial because forced NPU is unavailable and package energy was disabled.

## Software-only direction probes

These one-off local probes prefilled request-owned tensors and exclude input packing, output consumption, candidate selection, memory growth, package energy, and full IVF-PQ wall. No raw artifact was preserved. They justify implementation experiments only.

| Reusable request depth | 512 bucket-128 tasks | Tasks/s |
|---:|---:|---:|
| 1 | 99.219 ms | 5,160 |
| 2 | 57.231 ms | 8,946 |
| 4 | 43.228 ms | 11,844 |

At depth four with a constant 524,288 scored codes, fixed buckets measured 1.539M, 2.869M, 5.918M, 10.927M, and 18.596M codes/s for bucket sizes 128, 256, 512, 1,024, and 2,048 respectively. Launch overhead dominates small tiles; SIFT1M with `nlist=1,024` averages about 977 codes per list, making bucket 1,024 the first end-to-end candidate rather than a selected winner.

An isolated per-LUT scaling probe used a conservative 60,000 headroom, then de-scaled each list before cross-list selection. Candidate top-32 overlap versus CPU averaged 0.9990 over 32 queries, with minimum 0.96875 and 31/32 queries exact. This is not yet implemented and is not an end-to-end recall result.

## Remaining B3 gate

1. Validate per-LUT scaling at production geometry before re-enabling unsafe-range NPU work.
2. Bake off reusable request depths 1/2/4 with queue, cache, tail, copy, memory, and stage-device telemetry; depth one remains the production default until end-to-end evidence says otherwise.
3. Bound the compiled-request cache and include effective compile properties, runtime/compiler/driver identity, device ID, and performance mode in its key.
4. Add the iGPU variable-length scan/select path and compare it with padded NPU execution per SKU.
5. Compare raw ovVS and raw FAISS at identical `nlist`, `nprobe`, `pq_m`, `nbits`, and `krefine=k`; refined comparisons need an equivalent FAISS refinement lane.
6. Require repeated end-to-end SIFT1M recall, QPS, tail latency, peak RSS, and package-energy evidence before claiming a win.

Latest checkpoint verification: 81/81 accelerator-enabled native tests with zero skips, 10/10 focused IVF-PQ/NPU tests, 6/6 CTest lanes, 57/57 benchmark-harness tests, and 6/6 SIFT-fetcher tests. CTest re-runs native subsets plus the two consumers, so its count is not additive.
