# Arrow Lake 265K CAGRA cached-worst promotion evidence

Status: **promoted as an exact CAGRA walk improvement; product performance is
still behind full-strength hnswlib**.

## Scope and method

Commit `c7e9c04` caches the current strict worst candidate slot in the existing
work-group state. A candidate that cannot replace that slot now rejects in
constant time; after an accepted replacement, the leader rescans the beam to
restore the exact old strict-`>` worst-slot choice. The change adds no SLM,
allocation, submission, wait, transfer, or floating-point operation and does
not change traversal, final ordering, or construction.

Three old-source and three candidate SIFT1M processes were compared. After the
published original old-source process, the five new processes alternated
candidate/old/candidate/old/candidate. Each process built a fresh
degree-16/intermediate-32 CAGRA graph, reused it for `32/1`, `64/2`, and `128/4`
batch-32 search plus `128/4` batch one, and ran an unchanged hnswlib 0.8.0
control at `M=16`, `ef_construction=200`, 20 threads, and
`ef={32,64,128,256}`. Every point used exact validated HDF5 truth, one warmup,
five measured passes, and whole-package energy sampling. Values below are the
median of the three process medians; parentheses give the process range.

## Old-source versus candidate

| CAGRA point | Recall@10 | Old QPS | Candidate QPS | QPS change | Old / candidate p50 | Old / candidate p99 |
|---|---:|---:|---:|---:|---:|---:|
| `32/1`, batch 32 | 0.9036 in all runs | 3,479.3 (3,470.4–3,480.2) | 6,330.2 (6,319.5–6,343.7) | **+81.9%** | 8.883 / 4.901 ms (**-44.8%**) | 10.213 / 5.754 ms (**-43.7%**) |
| `64/2`, batch 32 | 0.9609 in all runs | 1,166.3 (1,156.0–1,169.1) | 2,549.9 (2,548.7–2,558.0) | **+118.6%** | 26.623 / 12.176 ms (**-54.3%**) | 29.273 / 13.233 ms (**-54.8%**) |
| `128/4`, batch 32 | 0.9889 in all runs | 347.8 (347.7–347.9) | 892.2 (891.6–894.2) | **+156.5%** | 89.896 / 34.980 ms (**-61.1%**) | 94.845 / 36.270 ms (**-61.8%**) |
| `128/4`, batch 1 | 0.9889 in all runs | 13.069 (13.047–13.070) | 33.414 (33.349–33.448) | **+155.7%** | 78.500 / 29.977 ms (**-61.8%**) | 89.459 / 35.013 ms (**-60.9%**) |

Median whole-package energy/query fell 37.0%, 40.5%, 75.7%, and 61.2% in the
same row order. The raw process ranges are retained because package energy
includes CPU, iGPU, and uncore and varied more than QPS or latency. CAGRA build
wall changed from 99.821 to 100.314 seconds (+0.49%) and isolated peak RSS from
2,441,465,856 to 2,457,112,576 bytes (+0.64%); neither path is changed by the
search patch.

The bounded cross-DLL exactness seam uses one serialized graph produced by the original DLL.
The original DLL, an independently rebuilt old-source DLL, and the final
candidate DLL produced identical ID bytes and float-distance bits at all three
batch-32 effort points. Native verification remains 94/94 tests, configured
CTest coverage is 8/8 across the documented out-of-sandbox interop rerun, and
the benchmark harness is 87/87.

## Comparator status

This promotion closes only the internal cached-worst A/B gate. It does not meet
the product objective:

- At the nearest ≥0.95 comparison, CAGRA `64/2` reaches 0.9609 recall and
  2,549.9 QPS versus hnswlib `ef=64` median recall 0.9589 and 6,746.8 QPS;
  hnswlib remains **2.65× faster** and its p99 is 2.45× lower.
- Near 0.989 recall, hnswlib remains **4.19× faster** and its p99 is 3.70×
  lower.
- The batch-one points are not recall matched: hnswlib has higher recall and is
  still **60.8× faster** by median QPS.
- Candidate CAGRA construction remains slower: 100.314 seconds versus 74.334
  seconds for hnswlib in the three candidate processes.

This frozen cohort remains the promotion evidence for cached-worst tracking.
The later combined prefix-sort/frontier checkpoint supersedes its current
performance status; the stable one-pass pick was exact but rejected.
Cooperative integer/min-location selection and a bounded owner/helper route are
the next separate gates. No NPU share is implied on Arrow Lake. Current result:
[`cagra-frontier-v1.md`](cagra-frontier-v1.md).

Raw artifacts and producer hashes:
[`evidence/cagra-cached-worst-v1/`](evidence/cagra-cached-worst-v1/README.md).
