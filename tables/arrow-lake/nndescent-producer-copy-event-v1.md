# Arrow Lake NN-Descent producer-to-copy event experiment

## Decision

**Rejected; no acceleration claim.** Replacing queue-wide waits after reverse
and proposal producers with explicit producer→active-count-copy dependencies
removed 444 waits at SIFT1M without changing any other tracked GPU-work or
transfer counter,
but made the complete CAGRA build 8.96% slower and the GPU NN-Descent
initializer 9.68% slower. Production source and tests were restored. Raw
artifacts and producer hashes are under
[`evidence/nndescent-producer-copy-event-v1/`](evidence/nndescent-producer-copy-event-v1/README.md).

## Change tested

The candidate retained every per-chunk bulk `heads`/scalar fill and its
pre-producer wait, every scalar D2H copy, each dynamic consumer range, each
post-consumer scratch-reuse wait, convergence, and host termination. The
reverse `parallel_for` and proposal `submit` returned producer events. The
existing scalar copies used the SYCL `queue::memcpy(..., depEvent)` overload,
and the host waited on those copy events instead of first draining the queue
after each producer.

For `I` iterations, `R=ceil(N/196608)` reverse chunks, and
`P=ceil(N/16384)` proposal chunks, waits fall by `I*(2R+P)`. Submissions,
kernels, D2H/H2D, allocations, launch ranges, graph geometry, and public ABIs
are unchanged. The 16,384-row bounded test was temporarily raised to 16,385
to exercise the proposal-chunk seam.

An initial implementation submitted each dependent copy through a handler with
`h.depends_on`. Its SIFT100K initializer was 6.42% slower, so it was not run at
SIFT1M. The final candidate used the direct standard queue overload; that made
the SIFT100K initializer neutral (+0.04%) before the full-scale screen.

## Measurements

One fresh native Windows direct-copy candidate/parent pair ran at each size
with exact SIFT truth, AUTO NN-Descent construction, FORCE_GPU search, one
warmup, five measured search passes, and unchanged hnswlib 0.8.0 settings
(`M=16`, `ef_construction=200`, 20 threads, seed 7, `ef=32`). The candidate ran
first. Package energy was disabled.

| SIFT1M metric | Parent | Candidate | Change |
|---|---:|---:|---:|
| Complete Python/API CAGRA build | 100,098.072 ms | 109,062.125 ms | **+8.96%** |
| Inner CAGRA build | 100,089.479 ms | 109,056.321 ms | **+8.96%** |
| GPU NN-Descent initializer | 90,601.501 ms | 99,367.258 ms | **+9.68%** |
| Dataset copy/validation | 123.595 ms | 460.617 ms | +337.022 ms |
| Host optimize/prune/merge | 9,354.462 ms | 9,219.326 ms | -1.44% |
| Isolated peak RSS | 2,456,256,512 B | 2,457,341,952 B | +0.04% |
| Literal waits | 1,820 | 1,376 | **-24.40%** |
| GPU submissions | 2,458 | 2,458 | unchanged |
| Kernel launches | 1,098 | 1,098 | unchanged |
| D2H calls / bytes | 459 / 176,003,272 | 459 / 176,003,272 | unchanged |
| Allocations / owned bytes | 12 / 685,748,744 | 12 / 685,748,744 | unchanged |
| Iterations | 6 | 6 | unchanged |
| Final changed / pending-NEW edges | 5,911,309 / 5,949,466 | 5,911,309 / 5,949,466 | unchanged |
| CAGRA recall@10 | 0.9036 | 0.9036 | unchanged |

The initializer loss remains 8,765.757 ms after excluding the slower dataset
stage. hnswlib built in 79,438.210 ms in the later parent process and
76,802.297 ms in the earlier candidate process (-3.32%). That opposite drift
argues against a global process slowdown, but one non-interleaved pair cannot
exclude phase-specific machine noise. Search throughput fell for both ovVS and
hnswlib in the candidate process; no search regression is attributed to this
construction-only patch.

At SIFT100K the direct candidate produced the exact 260→206 wait change with
all other tracked work counters and recall 0.9718 unchanged. Initializer wall
was 7,844.997→7,847.878 ms (+0.04%). Complete wall moved -1.79%, but large
variation in unmodified dataset/materialization/optimizer stages makes that
non-attributive. The handler-copy diagnostic had the same exact counters and
recall but regressed initializer wall 6.42% and complete wall 9.51%.

## Correctness and scope

Both variants built with oneAPI 2025.1.1. The final direct-copy candidate passed
eight focused GPU/graph tests with zero skips; the handler variant passed the
same set before its admission run. Coverage included deterministic overlap,
4,097-row and 16,385-row scale, non-finite fail-closed behavior, structured
iteration progress, forced-policy failure, atomic telemetry publication, and
downstream CAGRA search. Every valid ovVS lane loaded its explicit DLL and
recorded the matching hash.

The fixed harness has no raw NN-Descent or native CAGRA graph digest. Matching
recall, iterations, final convergence counters, and successful validation are
claimed; bitwise cross-DLL graph identity is not. One pair per size, no energy,
and no full search curve are sufficient for rejection but not promotion.

Four different synchronization/counter-reduction designs have now lost despite
meeting their structural targets. Further initializer work requires device-side
phase profiling and a reduction in actual algorithmic work, not another API
counter proxy. The next independent implementation target is the measured
9-second T13.3 host optimize/prune/merge stage on the iGPU; the current CPU path
remains production until a complete-wall winner passes exact graph validation.
