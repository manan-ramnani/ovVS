# Arrow Lake 265K CAGRA build telemetry V1

## Scope and provenance

Six sequential isolated processes measured the fixed CAGRA path on 2026-08-29: three checksum-pinned SIFT100K diagnostics followed by three full SIFT1M matched gates. Each process built one ovVS CAGRA index with AUTO NN-Descent, then searched it with FORCE_GPU; the unchanged hnswlib 0.8.0 comparator used `M=16`, `ef_construction=200`, `ef=32`, and 20 threads. Every lane used one warmup and five measured search passes.

All artifacts record clean commit `5dc0ec7b4b901b62c5c90f6b26ed10ba5e8cd448` and explicitly loaded `build-icpx/bin/ovvs.dll`, SHA-256 `c4ee36a1f7540726e7686c138190d201bfb825112ede9fdd6249048fab319f8f`, size 1,131,520 bytes. That commit changes only the benchmark and canonical documentation relative to native checkpoint `725f27f`; a clean rebuild reported no work. This binds the measured binary operationally, but the DLL does not embed a reproducible-build attestation.

The host was Windows 11 build 26200 with Python 3.13.14, NumPy 2.2.6, FAISS-CPU 1.15.0, and hnswlib 0.8.0. The pre-run load sample was not retained in the raw artifacts; there is no in-run clock, thermal, or occupancy trace, and energy was disabled. These are clean revision-bound current-lab diagnostics, not the complete B1 curve/energy report.

Raw immutable JSON and hashes are in [`evidence/cagra-build-v1/README.md`](evidence/cagra-build-v1/README.md).

## SIFT100K admission diagnostic

All three prefix preflights completed with two successful lanes, zero validation issues, exact recomputed truth, the expected byte geometry, one instrumented GPU initializer, and identical structural counters. The harness correctly labels these artifacts partial and noncanonical because SIFT100K and energy-disabled search cannot close B1.

| Lane | Recall@10 | Build wall, median (range) | Search QPS, median (range) | Peak RSS, median |
|---|---:|---:|---:|---:|
| ovVS CAGRA | 0.9718 in all runs | 9.084 s (8.904–9.087) | 3,946 (3,938–3,952) | 398.8 MiB |
| hnswlib | 0.9533 (0.9503–0.9538) | 1.632 s (1.555–1.633) | 23,003 (21,984–23,302) | 202.9 MiB |

The diagnostic admitted SIFT1M, but did not show acceleration: ovVS was 5.566× slower to build, delivered 17.15% of hnswlib throughput, and used about 1.97× the process RSS.

## SIFT1M matched gate

Each full-base process passed the plan's recall-closeness gate with zero validation issues. All 192 CAGRA calls per process were attributed to direct GPU walks with zero explicit index uploads and no NPU compile/fallback delta.

| Lane | Recall@10 | Build wall, median (range) | Search QPS, median (range) | Amortized p99, median | Peak RSS, median |
|---|---:|---:|---:|---:|---:|
| ovVS CAGRA | 0.9036 in all runs | 100.093 s (99.874–100.525) | 3,465 (3,459–3,475) | 1.090 ms/query | 2,343.2 MiB |
| hnswlib | 0.8911 (0.8899–0.8915) | 76.869 s (76.566–80.493) | 11,417 (11,093–11,464) | 0.101 ms/query | 1,387.0 MiB |

The median recall difference is `hnswlib-CAGRA=-0.0125`; every process remained within the maximum 0.0200 gap. This is a quality-gate pass only. CAGRA remains 0.0464 below the separate 0.95 product target, took 1.302× the build wall, delivered 30.35% of hnswlib throughput, had 10.76× the amortized p99, and used 1.689× the peak RSS. The three ovVS build walls span only 0.65%, so the earlier 185.3-second load-contaminated wall is superseded for current performance status.

## Measured attribution

The V1 stages cover the SIFT1M inner wall to within 1.7–5.1 microseconds. The complete Python/API wrapper adds a median 7.09 ms.

| SIFT1M stage | Median wall | Share of median complete build |
|---|---:|---:|
| Dataset copy and finite validation | 0.527 s | 0.53% |
| GPU NN-Descent initializer | 90.444 s | **90.36%** |
| Host optimizer/prune/merge | 9.034 s | 9.03% |
| Persistent graph materialization | 0.0077 s | 0.01% |

Structural counters were identical across all three processes at each scale:

| Counter | SIFT100K | SIFT1M |
|---|---:|---:|
| Iterations / converged calls | 6 / 0 | 6 / 0 |
| Final changed / pending-NEW edges | 168,506 / 168,506 | 5,911,309 / 5,949,466 |
| Allocations / peak owned USM | 12 / 97.80 MiB | 12 / 653.98 MiB |
| H2D calls / bytes | 0 / 0 | 0 / 0 |
| D2H calls / bytes | 69 / 16.79 MiB | 459 / 167.85 MiB |
| Kernels / all submissions / literal waits | 136 / 326 / 260 | 1,098 / 2,458 / 1,820 |
| Dataset / initializer / published logical bytes | 51.2 / 12.8 / 6.4 MB | 512 / 128 / 64 MB |

Both sizes exhausted the six-iteration budget. At SIFT1M, changed and pending-NEW edges remain about 18.47% and 18.59% of the 32-million-edge intermediate graph; this is iteration-cap exhaustion, not convergence. Zero explicit H2D is consistent with shared USM, but does not prove physical residency or exclude driver-managed migration.

## Disposition

The first measured construction target is the GPU NN-Descent initializer, which owns 90.36% of complete SIFT1M build wall and exposes 2,458 submissions, 1,820 waits, and 459 readbacks. Begin with the repeated reverse/proposal loop in `src/prim/gpu/backend_gpu.cpp`; the telemetry does not time that sub-loop separately. The host optimizer is the second target at 9.03 s. No NPU work is justified by this evidence.

Promote a construction change only after interleaved repeated complete-wall measurements preserve recall and structural correctness, improve median SIFT1M build by at least 10%, keep search p99 within 5%, and avoid more than 5% peak-memory growth. Product acceleration still requires recall at least 0.95, faster construction and search than full-strength hnswlib, competitive batch-one tails, and complete curves plus energy.
