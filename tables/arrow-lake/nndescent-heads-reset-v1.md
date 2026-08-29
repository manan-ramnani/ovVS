# Arrow Lake NN-Descent target-owned head-reset experiment

## Decision

**Rejected; no acceleration claim.** Initializing `heads[N]` once per
iteration and clearing each touched entry from its unique target-owned consumer
removed 438 full-array fill submissions at SIFT1M, but made the complete CAGRA
build 5.55% slower and the GPU NN-Descent initializer 6.17% slower. Production
source and tests were restored. Raw artifacts and producer hashes are under
[`evidence/nndescent-heads-reset-v1/`](evidence/nndescent-heads-reset-v1/README.md).

## Change tested

The candidate retained every scalar active-count readback, dynamic consumer
range, convergence readback, and host termination decision. It filled the
shared `heads` array once at the start of each iteration instead of before each
reverse and proposal chunk. Atomic exchange already records each nonempty
target once, so its sole consumer reset that target's head after traversing the
linked inbox. Existing consumer waits remained the scratch-reuse boundary on
the out-of-order queue.

For `I` iterations, `R=ceil(N/196608)` reverse chunks, and
`P=ceil(N/16384)` proposal chunks, the expected submission reduction is
`I*(2R+P-1)`. Kernels, D2H traffic, waits, allocations, graph geometry, and
public ABIs are unchanged. Independent reviews found the ownership and wait
ordering sound. The 16,384-row bounded test was temporarily raised to 16,385
to exercise reuse across the proposal-chunk seam.

## Measurements

One fresh native Windows candidate/parent pair ran at each size with exact SIFT
truth, AUTO NN-Descent construction, FORCE_GPU search, one warmup, five measured
search passes, and unchanged hnswlib 0.8.0 settings (`M=16`,
`ef_construction=200`, 20 threads, seed 7, `ef=32`). The candidate ran first.
Package energy was disabled.

| SIFT1M metric | Parent | Candidate | Change |
|---|---:|---:|---:|
| Complete Python/API CAGRA build | 99,906.380 ms | 105,449.002 ms | **+5.55%** |
| Inner CAGRA build | 99,896.678 ms | 105,439.862 ms | **+5.55%** |
| GPU NN-Descent initializer | 90,502.095 ms | 96,081.871 ms | **+6.17%** |
| Host optimize/prune/merge | 9,229.313 ms | 9,201.898 ms | -0.30% |
| Isolated peak RSS | 2,456,121,344 B | 2,456,399,872 B | +0.01% |
| GPU submissions | 2,458 | 2,020 | **-17.82%** |
| Literal waits | 1,820 | 1,820 | unchanged |
| Kernel launches | 1,098 | 1,098 | unchanged |
| D2H calls / bytes | 459 / 176,003,272 | 459 / 176,003,272 | unchanged |
| Allocations / owned bytes | 12 / 685,748,744 | 12 / 685,748,744 | unchanged |
| Iterations | 6 | 6 | unchanged |
| Final changed / pending-NEW edges | 5,911,309 / 5,949,466 | 5,911,309 / 5,949,466 | unchanged |
| CAGRA recall@10 | 0.9036 | 0.9036 | unchanged |

The structural signature is exact: SIFT1M removed 438 submissions and changed
no other tracked GPU-work counter. hnswlib build wall was 78,063.239 ms in the
parent process and 77,433.518 ms in the earlier candidate process (-0.81%), so
the ovVS loss is not explained by a generally slower candidate run. Search
timings moved in opposite directions between ovVS and hnswlib and are not
attributed to this construction-only patch.

At SIFT100K, submissions fell 326→278 while all other tracked GPU-work counters
and recall 0.9718 remained exact. Initializer wall was neutral
(7,846.639→7,850.403 ms, +0.05%), but the complete build moved
8,619.488→8,923.144 ms (+3.52%) because unmodified post-initializer stages were
slower in that process. The prefix therefore served only as admission; it did
not predict the full-scale initializer regression.

## Correctness and scope

The candidate built with oneAPI 2025.1.1 and passed eight focused GPU/graph
tests with zero skips. Coverage included deterministic overlap, 4,097-row and
16,385-row scale, non-finite fail-closed behavior, structured iteration
progress, forced-policy failure, atomic telemetry publication, and downstream
CAGRA search. Every valid ovVS lane loaded the explicitly requested DLL and
recorded its matching hash.

The fixed harness has no raw NN-Descent or native CAGRA graph digest. Matching
recall, iterations, final convergence counters, and successful validation are
claimed; bitwise cross-DLL graph identity is not. One pair per size, no energy,
and no full search curve are sufficient for rejection but not promotion.

The result shows that fewer submissions alone is not a performance proxy. On
this Arrow Lake stack, the scattered-reset candidate lost to the baseline
bulk-fill design; these runs do not isolate whether stores, cache/coherency, or
submission behavior caused the loss. The next bounded synchronization
experiment should keep the bulk fills and dynamic ranges, then replace only
queue-wide producer barriers with explicit producer→active-count-copy
dependencies. T13.3 GPU optimize/prune/merge remains the next independent
construction stage after that bounded screen.
