# Arrow Lake 265K NN-Descent B5 bounded evidence

Source: native `ovvs_tests` against `build-icpx` (`OVVS_WITH_SYCL=ON`) on 2026-08-28. Inputs are deterministic `make_data` fixtures; overlap is against independent exact L2 neighbors on sampled rows. CTest runs the threshold and bounded-scale cases in separate processes with an Intel-GPU resource lock and 45/120-second timeouts.

| n | dim | degree | iterations | Build wall, isolated processes | Sampled exact overlap | Validation |
|---:|---:|---:|---:|---:|---:|---|
| 4,097 | 8 | 8 | 6 | median 107.1 ms; 106.8–1079.5 ms (5 runs) | 0.7266 (64 rows) | all rows valid/unique/non-self; repeat graph identical; last device GPU |
| 16,384 | 16 | 16 | 4 | median 832.6 ms; 826.7–1333.2 ms (21 runs) | 0.2832 (32 rows) | all rows valid/unique/non-self; last device GPU |

The kernel uses two device graph buffers plus a dataset copy only when the input is not device-accessible. Candidate storage and the duplicate Bloom are per-work-group SLM, so transient graph memory is linear in `n×degree`; these runs did not measure peak RSS. The wide upper range retains first-use SYCL/driver initialization and occasional scheduling outliers rather than presenting only warm timings.

The first downstream SIFT1M gate has now run. AUTO CAGRA construction completed in an observed 830.0 s and ended with CPU prune, but CAGRA recall@10 was 0.4057 versus hnswlib 0.8903 at matched M=16/effort=32. The build time was collected under substantial unrelated iGPU load and is not an isolated performance baseline; the quality failure is valid. This confirms scale execution but rejects the current sampled initializer/prune/search combination for product quality. Convergence/new-edge/reverse-candidate work and a search-effort diagnostic remain before attributing the full gap to one stage. Peak RSS was not measured, and full FORCE_GPU CAGRA build remains unavailable until T13.3.

Checkpoint verification: 63/63 native tests and 6/6 CTest lanes passed; after the strict SIFT1M gate landed, the Python benchmark/fetcher suites pass 45/45 and 6/6. GPU scale lanes use explicit CTest SKIP on device absence.
