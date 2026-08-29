# Arrow Lake NN-Descent convergence-reduction experiment

## Decision

**Rejected; no acceleration claim.** Replacing each iteration's two full
convergence-array readbacks with a device reduction and one 24-byte summary cut
SIFT1M D2H volume 27.27%, but made the complete build 2.14% slower and the GPU
initializer 1.91% slower. Production source was restored. Raw artifacts and
producer hashes are under
[`evidence/nndescent-convergence-reduction-v1/`](evidence/nndescent-convergence-reduction-v1/README.md).

## Change tested

The candidate preserved every reverse/proposal active-count readback and
dynamic consumer range. The existing convergence shards still produced exact
per-row changed and pending-NEW counts. One dependent 128-work-item work-group
then validated and summed all rows into an aligned 24-byte typed summary; one
dependent D2H copy and event wait returned that summary to the unchanged host
termination decision. The out-of-order queue dependency covered every shard.

The summary included 64-bit totals, processed-row count, and range-error flags.
Host publication rejected flags, missing rows, graph-count overflow, or int64
overflow. The device control allocation grew by 24 bytes; allocation count and
serialized/public ABIs were unchanged.

## Measurements

One fresh native Windows candidate/parent pair ran at each size with exact SIFT
truth, AUTO NN-Descent construction, FORCE_GPU search, one warmup, five measured
search passes, and unchanged hnswlib 0.8.0 settings (`M=16`,
`ef_construction=200`, 20 threads, seed 7, `ef=32`). Package energy was disabled.

| SIFT1M metric | Parent | Candidate | Change |
|---|---:|---:|---:|
| Complete CAGRA build | 99,863.161 ms | 102,004.333 ms | **+2.14%** |
| GPU NN-Descent initializer | 90,515.058 ms | 92,240.123 ms | **+1.91%** |
| Host optimize/prune/merge | 9,214.876 ms | 9,231.340 ms | +0.18% |
| Isolated peak RSS | 2,342.508 MiB | 2,343.453 MiB | +0.04% |
| D2H calls | 459 | 453 | -1.31% |
| D2H bytes | 176,003,272 | 128,003,416 | **-27.27%** |
| GPU submissions | 2,458 | 2,458 | unchanged |
| Literal waits | 1,820 | 1,814 | -0.33% |
| Kernel launches | 1,098 | 1,104 | +0.55% |
| Algorithm-owned bytes | 685,748,744 | 685,748,768 | +24 bytes |
| Iterations | 6 | 6 | unchanged |
| Final changed / pending-NEW edges | 5,911,309 / 5,949,466 | 5,911,309 / 5,949,466 | unchanged |
| CAGRA recall@10 | 0.9036 | 0.9036 | unchanged |

The SIFT100K pair reached the same conclusion at smaller scale: initializer
wall changed only +0.04%, while complete build changed +1.42% because the
unmodified host optimizer was slower in that process. Recall remained 0.9718.
D2H bytes fell 27.27%, calls 69→63, waits 260→254, and kernels 136→142.

The candidate process ran first. Its dataset-copy stage was unusually slower at
SIFT1M, but that stage is outside the initializer comparison; the initializer
itself still lost 1.91%. hnswlib build changed only +0.72%. Search throughput
and tails moved in the same adverse direction for both ovVS and hnswlib, so no
search regression is attributed to this construction-only patch.

## Correctness and scope

Local verification outside the five immutable benchmark artifacts built the
candidate with oneAPI 2025.1.1 and passed eight focused GPU/graph tests with
zero skips. Those cover deterministic overlap, 4,097/16,384-row scale,
non-finite fail-closed behavior, structured iteration progress, forced-policy
failure, atomic telemetry publication, and downstream CAGRA search. Independent
audits found the event DAG, typed layout, validation, integer bounds, graph
publication, and exact counter formulas sound.

Only one valid build pair per size was run and no graph digest was captured.
That is sufficient for rejection: the candidate did not approach the required
complete-wall improvement and added a single-work-group strided scan.
The subsequent bounded B5 change attacked repeated work rather than readback
volume: it initialized the `heads` scratch once per iteration and had each
unique target-owned consumer reset the rows it touched. That candidate retained
dynamic active-count launches and the original convergence path, but was also
rejected after regressing the SIFT1M initializer 6.17%. Evidence:
[`nndescent-heads-reset-v1.md`](nndescent-heads-reset-v1.md).
