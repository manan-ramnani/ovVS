# Arrow Lake IVF-PQ B3 evidence

Status: **correctness guarded; throughput and end-to-end competitor gates open**. Hardware is the repository's Core Ultra 7 265K with OpenVINO 2025.3.0 and NPU driver 32.0.100.4841. No BIOS or firmware setting was changed.

Local development artifacts are `out/bench/ivfpq-b3-baseline-smoke.json` and `out/bench/ivfpq-b3-guarded-smoke.json` with their sibling Markdown files. Both artifacts are overall **partial** because the forced-NPU end-to-end lane is explicitly unavailable at `nprobe=2/8`; that lane is retained rather than filtered. The ignored baseline artifact predates the fixture-label correction and says `float32_normalized_on_load`; the loader only cast raw SIFT values. The final guarded artifact emits the corrected `float32_cast_on_load` label.

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

1. Persist list offsets, IDs, and list-contiguous codes; repeated unfiltered searches must report zero repacks.
2. Validate per-LUT scaling at production geometry before re-enabling unsafe-range NPU work.
3. Batch query/list work descriptors and use fixed buckets 128/256/512/1,024/2,048 without allowing padded entries into selection.
4. Bake off reusable request depths 1/2/4 with queue, cache, tail, copy, memory, and stage-device telemetry.
5. Compare raw ovVS and raw FAISS at identical `nlist`, `nprobe`, `pq_m`, `nbits`, and `krefine=k`; refined comparisons need an equivalent FAISS refinement lane.
6. Require repeated end-to-end SIFT1M recall, QPS, tail latency, peak RSS, and package-energy evidence before claiming a win.

Latest checkpoint verification: 73/73 native tests, 6/6 CTest lanes, 57/57 benchmark-harness tests, and 6/6 SIFT-fetcher tests. CTest re-runs native subsets plus the two consumers, so its count is not additive.
