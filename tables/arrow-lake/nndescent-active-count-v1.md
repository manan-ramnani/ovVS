# Arrow Lake NN-Descent active-count experiment

## Decision

**Rejected; no acceleration claim.** Keeping reverse/proposal active counts on
the iGPU removed most scalar readbacks and queue waits, but the fixed-`N`
consumer launches made the complete SIFT1M CAGRA build 8.91% slower and the GPU
NN-Descent initializer 9.85% slower. Production source was restored to the
`e2f8a04` behavior. Raw inputs and producer hashes are under
[`evidence/nndescent-active-count-v1/`](evidence/nndescent-active-count-v1/README.md).

## Change tested

The candidate replaced each fill → producer → host scalar readback → dynamic
consumer sequence with explicit SYCL event dependencies and a consumer launched
over all `N` rows. Work-items at or above the device-resident active count
returned immediately. Sticky status bits and producer bounds guards kept the
path fail-closed. Convergence and final graph readback were unchanged.

This was a synchronization experiment, not a transfer-volume optimization.
At SIFT1M, 444 small active-count readbacks disappeared, but the two full
convergence arrays per iteration and the final 128 MB graph copy remained.

## Measurements

The current-parent and candidate DLLs were frozen separately. One valid
SIFT100K admission pair and one valid SIFT1M gate pair used fresh native Windows
processes, exact SIFT truth, AUTO NN-Descent construction, FORCE_GPU search, one
warmup, five measured search passes, and unchanged hnswlib 0.8.0 settings
(`M=16`, `ef_construction=200`, 20 threads, `ef=32`). Package energy was disabled.

| SIFT1M metric | Parent | Candidate | Change |
|---|---:|---:|---:|
| Complete CAGRA build | 100,335.287 ms | 109,275.077 ms | **+8.91%** |
| GPU NN-Descent initializer | 90,906.382 ms | 99,863.842 ms | **+9.85%** |
| Host optimize/prune/merge | 9,276.468 ms | 9,234.669 ms | -0.45% |
| Isolated peak RSS | 2,343.043 MiB | 2,342.980 MiB | -0.003% |
| D2H calls | 459 | 21 | **-95.42%** |
| D2H bytes | 176,003,272 | 176,000,032 | -0.00184% |
| GPU submissions | 2,458 | 2,020 | **-17.82%** |
| Literal waits | 1,820 | 488 | **-73.19%** |
| Kernel launches | 1,098 | 1,098 | unchanged |
| Iterations | 6 | 6 | unchanged |
| Final changed / pending-NEW edges | 5,911,309 / 5,949,466 | 5,911,309 / 5,949,466 | unchanged |
| CAGRA recall@10 | 0.9036 | 0.9036 | unchanged |

The SIFT100K pair was neutral: complete build changed from 8,904.682 to
8,887.655 ms (-0.19%) and initializer wall from 7,846.538 to 7,825.278 ms
(-0.27%), with recall 0.9718 and final counts unchanged. It did not predict the
full-scale loss.

hnswlib SIFT1M build wall changed only -0.68% between the two processes, while
the candidate regression was isolated in NN-Descent rather than host graph
optimization. Search QPS and tails moved with large same-process hnswlib noise,
so they are not attributed to this construction-only patch. The likely cost is
the 444 fixed-`N` consumer grids scheduling up to 444 million work-items; that is
an inference from the implementation and measurements, not a kernel-profiled
claim.

## Correctness and scope

Eight focused GPU/graph tests passed with zero skips after the final bounds
guard, covering deterministic overlap, 4,097/16,384-row scale, non-finite
fail-closed behavior, structured iteration progress, forced-policy failure,
telemetry publication, and downstream CAGRA search. The full gate preserved
recall, iterations, final changed/pending counts, allocation accounting, and
zero H2D. No cross-DLL raw graph hash was captured, so byte-identical graph IDs
are not claimed.

Only one valid SIFT1M pair was run. The admission result was already decisive
against the required improvement, so additional promotion repetitions would
have spent time refining a losing design. The subsequent convergence-only
experiment kept the original dynamic active-count consumer ranges and reduced
the large per-iteration readbacks to a validated summary, but also lost; see
[`nndescent-convergence-reduction-v1.md`](nndescent-convergence-reduction-v1.md).
