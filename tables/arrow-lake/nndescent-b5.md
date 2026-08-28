# Arrow Lake 265K NN-Descent B5 bounded evidence

Source: native `ovvs_tests` against `build-icpx` (`OVVS_WITH_SYCL=ON`) on 2026-08-28. Inputs are deterministic `make_data` fixtures; overlap is against independent exact L2 neighbors on sampled rows. CTest runs the threshold and bounded-scale cases in separate processes with an Intel-GPU resource lock and 45/120-second timeouts.

| n | dim | degree | iterations | Build wall, isolated processes | Sampled exact overlap | Validation |
|---:|---:|---:|---:|---:|---:|---|
| 4,097 | 8 | 8 | 6 | median 107.1 ms; 106.8–1079.5 ms (5 runs) | 0.7266 (64 rows) | all rows valid/unique/non-self; repeat graph identical; last device GPU |
| 16,384 | 16 | 16 | 4 | median 832.6 ms; 826.7–1333.2 ms (21 runs) | 0.2832 (32 rows) | all rows valid/unique/non-self; last device GPU |

The kernel uses two device graph buffers plus a dataset copy only when the input is not device-accessible. Candidate storage and the duplicate Bloom are per-work-group SLM, so transient graph memory is linear in `n×degree`; these runs did not measure peak RSS. The wide upper range retains first-use SYCL/driver initialization and occasional scheduling outliers rather than presenting only warm timings.

This table validates the first scale/correctness checkpoint only. It is not SIFT1M evidence, not a comparison with cuVS, and not the downstream CAGRA/hnswlib quality gate. Peak RSS was not measured. The B1 harness now requests one independent AUTO construction policy for all search lanes and keeps CAGRA above 4,096 rows opt-in until a full measured build completes. AUTO CAGRA can use the GPU initializer but finishes with CPU prune; full FORCE_GPU CAGRA build remains unavailable until T13.3.

Checkpoint verification: 63/63 native tests and 6/6 CTest lanes passed; after build/search policy separation, the Python benchmark/fetcher suites pass 25/25 and 6/6. GPU scale lanes use explicit CTest SKIP on device absence.
