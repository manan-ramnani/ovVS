# Arrow Lake 265K CAGRA cooperative-pick V1 promotion evidence

Status: **promoted for multi-pick CAGRA search; full-strength hnswlib remains
faster**.

## Change and proof boundary

For `search_width > 1`, the first subgroup now performs the exact beam-pick
operation cooperatively. Each subgroup lane scans its strided candidate slots,
then two subgroup reductions select the minimum distance and lowest candidate
slot. The slot owner marks it expanded before the next pick. This preserves the
serial `(distance, slot)` order, including ties, while removing the repeated
single-lane scan of the full beam. `search_width == 1` retains the prior serial
selection semantics.

The change adds no local memory, work-group barrier, allocation, submission,
wait, or transfer. It adds two subgroup reductions per multi-pick attempt,
including a terminal attempt that finds no eligible slot.
The final candidate and the frozen frontier DLL emitted identical ID and
float-distance-byte hashes on the retained graph at `32/1`, `64/2`, and
`128/4`. Verification was 94/94 native tests with zero skips, 8/8 configured
CTest lanes, and 87/87 benchmark-harness tests.

## Repeated complete-wall result

Three serial-baseline and three candidate SIFT1M processes alternated
candidate/baseline three times. Every process used a clean `c9bb312` runner,
checksum-pinned exact HDF5 truth, a fresh degree-16/intermediate-32 graph, one
warmup, five measured passes, package energy, isolated peak RSS, and unchanged
hnswlib. All 12 lanes and 54 requested points succeeded; no cell failed,
skipped, timed out, or was unavailable.

Values are medians of the three process medians; parentheses are process
ranges.

| CAGRA point | Recall@10 | Serial QPS | Candidate QPS | QPS change | Serial / candidate p50 | Serial / candidate p99 |
|---|---:|---:|---:|---:|---:|---:|
| `32/1`, batch 32 | 0.9036 in all runs | 6,826.2 (6,784.0–6,848.3) | 6,961.3 (6,892.0–6,982.5) | +2.0% (neutral) | 4.544 / 4.444 ms (-2.2%) | 5.395 / 5.177 ms (-4.0%) |
| `64/2`, batch 32 | 0.9609 in all runs | 2,754.4 (2,731.4–2,763.5) | 3,243.5 (3,216.3–3,280.4) | **+17.8%** | 11.278 / 9.559 ms (**-15.2%**) | 12.318 / 10.420 ms (**-15.4%**) |
| `128/4`, batch 32 | 0.9889 in all runs | 972.0 (962.9–973.6) | 1,211.6 (1,201.6–1,218.3) | **+24.6%** | 32.062 / 25.744 ms (**-19.7%**) | 33.594 / 26.937 ms (**-19.8%**) |
| `128/4`, batch 1 | 0.9889 in all runs | 37.082 (36.946–37.093) | 48.167 (48.118–48.172) | **+29.9%** | 27.060 / 20.853 ms (**-22.9%**) | 31.316 / 24.813 ms (**-20.8%**) |

Width one retains serial-selection semantics and matched the retained fixture's
output bytes. Its +2.0% is non-attributable variation and is treated as
neutral, not as a cooperative-pick gain.

The ratio of cohort-median package energy/query changed by -8.5%, +4.5%,
-4.8%, and -17.6% in the same row order. Across the 12 paired rows, seven
improved and five regressed; even the sign differs by run for some points.
Package energy includes CPU, iGPU, and uncore, and the process ranges overlap
except at batch one, so this checkpoint makes no blanket or isolated-device
energy claim. Median CAGRA build wall changed from 100.268 to 99.937 seconds
(-0.33%), and peak RSS remained 2,343.1 MiB; the construction path and index
geometry are unchanged.

Across the six processes, all 46,608 CAGRA walks were direct GPU walks with
zero explicit index-upload calls or bytes. These counters do not measure
driver-managed shared-memory page migration.

## Comparator status

This promotion narrows but does not close the product deficit:

- At the nearest measured ≥0.95 comparison, CAGRA reaches 0.9609 recall and
  3,243.5 QPS versus hnswlib median 0.9593 recall and 6,784.5 QPS. hnswlib is
  still **2.09× faster**; CAGRA p99 is 1.96× hnswlib's.
- Near 0.989 recall, CAGRA reaches 1,211.6 QPS versus hnswlib 3,788.1 QPS at
  0.9882 recall. hnswlib is still **3.13× faster**; CAGRA p99 is 2.85×
  hnswlib's.
- The batch-one points are not recall matched: hnswlib has 0.9972 recall and
  remains **42.55× faster** by median QPS.
- Candidate construction remains slower: 99.937 seconds versus hnswlib 77.396
  seconds (1.29×), with 2,343.1 versus 1,458.4 MiB peak RSS.

hnswlib remained at `M=16`, `ef_construction=200`, 20 threads, and
`ef={32,64,128,256}`. The comparator figures above are medians of the three
candidate processes. Their median QPS changes versus the three baseline
processes were +1.2%, +1.3%, effectively 0%, effectively 0%, and +1.0% for the
five points. Candidate process 3 nevertheless contains a valid late CPU-tail
outlier at `ef=64`: 6,036.4 QPS and 14.509 ms p99 versus 6,784–6,799 QPS and
5.21–5.32 ms in candidate processes 1–2. The median remains robust, and every
candidate CAGRA multi-pick QPS range is disjoint from its serial baseline
range, so the promotion survives this disclosed comparator noise.

The next B2 experiment is a bounded owner/helper multi-work-group route for
batch-one or very small batches. It must fail closed when the required SYCL
root-group capability is absent and must keep the current one-work-group path
for larger batches. It is not part of this promotion.

Raw artifacts and producer hashes:
[`evidence/cagra-cooperative-pick-v1/`](evidence/cagra-cooperative-pick-v1/README.md).
