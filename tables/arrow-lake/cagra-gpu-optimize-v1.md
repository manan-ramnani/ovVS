# Arrow Lake CAGRA GPU optimizer V1

## Status and scope

T13.3 is promoted for CAGRA intermediate degree at most 64. The iGPU now runs
the existing exact detour-rank reorder, fixed-degree prune, capped reverse
construction, and deterministic forward/reverse interleave. This is a
construction-stage acceleration, not a claim that ovVS beats hnswlib overall.

The later
[`nndescent-cooperative-minima-v1`](nndescent-cooperative-minima-v1.md)
checkpoint supersedes the complete-build timing: the current SIFT1M median is
61.000 seconds, while the optimizer implemented here remains promoted and
accounts for a 0.655-second median in that cohort. The measurements below stay
canonical for this isolated optimizer decision.

The implementation uses two bounded allocation phases, validates input and
output graph invariants on-device, and publishes only after successful staged
readback. `AUTO`, `GPU_IF_FASTER`, and `HETERO` use it when the GPU and shape are
supported. Capacity or device absence may fall back to the CPU implementation;
a selected GPU runtime or correctness failure does not. `FORCE_GPU` completes
the full NN-Descent CAGRA build only when both GPU stages are available, while
other initializers remain unavailable under that policy. `FORCE_CPU` retains
the CPU optimizer and `FORCE_NPU` remains unavailable.

Raw JSON, file hashes, and producer identity are in
[`evidence/cagra-gpu-optimize-v1/README.md`](evidence/cagra-gpu-optimize-v1/README.md).

## Exactness and verification

The native oracle compares CPU and GPU graph bytes across heavy reverse-edge
skew; repeated unordered scatter; final degrees equal to and below the input;
and input/output degree pairs `2/1`, `9/5`, `9/9`, `16/8`, `32/16`, `64/32`,
and `64/64`. Duplicate, self, negative, out-of-range, and degree-65 inputs fail
without publishing output. The committed checkpoint passed 95/95 native tests
with zero skips. CTest passed eight configured product lanes; the separate
root-group capability canary skipped as designed after its retained unsupported
capacity result.

A post-commit diagnostic strengthened the production-scale check. Separate
processes loaded the frozen product-code parent and committed candidate, rebuilt
the same checksum-pinned dataset with AUTO NN-Descent 16/32, and serialized the
graph without the dataset outside the timed benchmark wall.

| Rows | Serialized bytes | Whole-file SHA-256 | Graph-payload SHA-256 | Result |
|---:|---:|---|---|---|
| 100,000 | 6,400,036 | `3DDBC493…FE32` | `86430690…77A4` | exact parent/candidate match |
| 1,000,000 | 64,000,036 | `0A792378…15BE` | `59CADBE6…B315` | exact parent/candidate match |

The diagnostic timings are excluded from performance claims. Its purpose is
only to prove identical graph bytes while final-device attribution changes
from CPU to GPU.

## Measured result

The SIFT100K admission used three frozen-parent and three FORCE_GPU candidate
processes. All corresponding CAGRA recall points were exactly `0.9718` and
`0.9946`.

| SIFT100K metric | Parent median | Candidate median | Change |
|---|---:|---:|---:|
| Complete build | 8.676 s | 7.907 s | -8.86% |
| Optimizer/prune/merge | 0.672 s | 0.0477 s | -92.91%, 14.10× faster |
| GPU NN-Descent initializer | 7.847 s | 7.848 s | +0.01% |
| Peak process RSS | 397.9 MiB | 393.7 MiB | -1.07% |

Three final AUTO SIFT1M candidate processes then passed the matched recall gate
with two successful lanes and no failed, skipped, timed-out, or unavailable
lane. They are compared with the existing three-run clean canonical baseline;
the processes were not interleaved. One candidate initializer took 99.069
seconds, so complete-build ranges overlap narrowly, while optimizer ranges are
strongly disjoint.

| SIFT1M metric | CPU-optimizer baseline | GPU-optimizer candidate | Change |
|---|---:|---:|---:|
| Complete Python/API build | 100.093 s (99.874–100.525) | 91.987 s (91.791–100.018) | **-8.10%, 1.088× faster** |
| Optimizer/prune/merge | 9.034 s (8.995–9.096) | 0.641 s (0.638–0.795) | **-92.90%, 14.08× faster** |
| GPU NN-Descent initializer | 90.444 s | 90.883 s | +0.49% |
| Peak process RSS | 2,343.2 MiB | 2,269.8 MiB | **-3.13%, -73.4 MiB** |
| CAGRA recall@10 | 0.9036 | 0.9036 | unchanged |

All SIFT1M runs retained six iterations, 5,911,309 final changed edges,
5,949,466 pending-NEW edges, 1,098 initializer kernels, 2,458 initializer
submissions, 1,820 initializer waits, and 459 initializer readbacks. These are
initializer-only lifecycle counters; the frozen telemetry times the complete
optimizer stage but does not expose optimizer-specific allocations, transfers,
submissions, or waits.

The unchanged hnswlib 0.8.0 comparator retained `M=16`,
`ef_construction=200`, 20 threads, and seed 7. Its candidate-cohort median build
was 77.821 seconds. ovVS therefore remains 1.182×, or 18.2%, slower to build.

## Promotion disposition

The earlier generic construction gate required at least 10% complete-build
improvement. An isolated optimizer that occupied only 9.03% of the baseline
wall cannot meet that criterion even at zero cost. This checkpoint therefore
uses a scoped exception: it removes 92.90% of the named stage, captures about
89.7% of the maximum possible whole-build reduction, produces an 8.10%
repeated complete-wall improvement, exactly matches graph bytes at SIFT100K and
SIFT1M, and does not regress recall or peak memory. The generic 10% gate remains
in force for future initializer or multi-stage changes.

Energy was disabled and the SIFT1M run is a two-point quality gate, not a full
B1 curve. Search timings are not compared with the older construction baseline
because later promoted search-kernel changes intervene; exact graph bytes make
an optimizer-induced search-topology change inapplicable. No product or energy
acceleration claim is made.

The remaining construction barrier is the 90.883-second GPU NN-Descent
initializer, about 98.8% of candidate build wall. Matching the current-cohort
hnswlib build requires removing another 14.17 seconds, approximately 15.6% of
that initializer. Further work starts from device-side phase profiling and a
separately measured device-resident initializer-to-optimizer handoff, not from
the four rejected synchronization-counter proxies.
