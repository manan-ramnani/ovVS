# Arrow Lake 265K CAGRA frontier V1 promotion evidence

Status: **promoted as an exact CAGRA search improvement; full-strength
hnswlib remains faster**.

Performance status is historical. The later cooperative-pick promotion
supersedes the current-product figures below; see
[`cagra-cooperative-pick-v1.md`](cagra-cooperative-pick-v1.md).

## Change and proof boundary

The candidate combines two adjacent exact cleanups after the promoted
cached-worst checkpoint:

1. final selection sorts only the published `min(candidate_count, k)` prefix;
2. seed and expansion IDs are compacted in canonical order into a 64-entry
   local frontier before scoring.

The frontier adds 256 bytes of local memory per work-group. Lane 0 still calls
the filter and visited-table insert in seed order and pick-major/edge-minor
order. The unchanged work-group distance reduction scores accepted IDs in that
same order, and lane 0 calls `consider` immediately in that order. Scoring and
beam insertion do not read or mutate the visited table, so moving later visits
ahead within one tile does not alter acceptance or traversal state.

At degree 16 and the full iteration budgets, explicit `item.barrier` calls fall
from 7,361 to 593 at `32/1`, 27,009 to 1,185 at `64/2`, and 103,169 to 2,369
at `128/4`. These are **explicit-barrier counts only**. Every accepted L2/IP
candidate still executes its original group reduction; cosine still executes
three. The structural counts are not a speed claim.

The final candidate and frozen baseline DLL emitted identical ID and
float-distance-byte hashes on the same retained graph at `32/1`, `64/2`, and
`128/4`. The native scale regression also exercises a 640-ID raw expansion
frontier across ten tiles. Verification at the checkpoint was 94/94 native
tests, 8/8 configured CTest lanes, and 87/87 benchmark-harness tests.

## Repeated complete-wall result

Three baseline and three candidate SIFT1M processes were compared. After the
retained clean baseline process, five new processes alternated
candidate/baseline/candidate/baseline/candidate. Each process built a fresh
degree-16/intermediate-32 graph, ran CAGRA `32/1`, `64/2`, and `128/4` at
batch 32 plus `128/4` at batch one, and ran unchanged hnswlib. Every point used
checksum-pinned exact HDF5 truth, one warmup, five measured passes, package
energy, and isolated peak RSS.

Values below are medians of the three process medians; parentheses are the
process ranges.

| CAGRA point | Recall@10 | Baseline QPS | Candidate QPS | QPS change | Baseline / candidate p50 | Baseline / candidate p99 |
|---|---:|---:|---:|---:|---:|---:|
| `32/1`, batch 32 | 0.9036 in all runs | 6,343.7 (6,332.2–6,348.1) | 6,848.2 (6,842.5–6,857.1) | **+8.0%** | 4.885 / 4.529 ms (**-7.3%**) | 5.693 / 5.071 ms (**-10.9%**) |
| `64/2`, batch 32 | 0.9609 in all runs | 2,557.6 (2,556.8–2,558.0) | 2,762.9 (2,757.8–2,769.4) | **+8.0%** | 12.144 / 11.214 ms (**-7.7%**) | 13.134 / 12.183 ms (**-7.2%**) |
| `128/4`, batch 32 | 0.9889 in all runs | 894.2 (892.2–895.0) | 975.1 (972.3–975.1) | **+9.0%** | 34.925 / 32.024 ms (**-8.3%**) | 36.278 / 33.407 ms (**-7.9%**) |
| `128/4`, batch 1 | 0.9889 in all runs | 33.435 (33.414–33.442) | 37.081 (37.077–37.108) | **+10.9%** | 29.972 / 27.061 ms (**-9.7%**) | 35.015 / 31.149 ms (**-11.0%**) |

Median package energy/query changed by +7.6%, -9.1%, -13.8%, and -4.8% in
the same row order. The lightest-point ranges overlap, and package energy
includes CPU, iGPU, and uncore, so this checkpoint makes no blanket energy
claim. Median CAGRA build wall changed from 100.272 to 99.637 seconds (-0.63%)
and peak RSS by less than 0.01%; construction and index geometry are unchanged.

## Comparator status

This promotion does not meet the product objective:

- At the nearest measured ≥0.95 comparison, CAGRA reaches 0.9609 recall and
  2,762.9 QPS versus hnswlib median 0.9602 recall and 6,678.4 QPS. hnswlib is
  still **2.42× faster**; CAGRA p99 remains 2.29× hnswlib's.
- Near 0.989 recall, hnswlib is still **3.84× faster**; CAGRA p99 remains
  3.37× hnswlib's.
- The batch-one points are not recall matched: hnswlib has higher recall and is
  still **54.8× faster** by median QPS.
- Candidate CAGRA construction remains slower: 99.637 seconds versus hnswlib
  76.972 seconds (1.29×).

An exact stable one-pass pick experiment was rejected after its complete
dirty-source admission screen regressed three of four effort points; its code
was not committed. At this checkpoint, the next B2 experiment was cooperative
integer/min-location selection; it later passed its separate gate. A later
bounded owner/helper capacity gate reported one cooperative workgroup versus
four required and parked that route on the measured Arrow Lake stack. See
[`gpu-root-group-canary-v1.md`](gpu-root-group-canary-v1.md).

Raw artifacts and producer hashes:
[`evidence/cagra-frontier-v1/`](evidence/cagra-frontier-v1/README.md).
