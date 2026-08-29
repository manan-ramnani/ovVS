# Arrow Lake 265K recall–QPS (icx SYCL + oneMKL)

Source: `tools/bench/bench.py` against `build-icpx` `ovvs.dll` (`OVVS_WITH_SYCL=ON`, `OVVS_WITH_MKL=ON`).
Dataset: checksum-pinned `data/sift-128-euclidean.hdf5`; each section states its selected base/query geometry.
Comparators: faiss-cpu 1.15.0, hnswlib 0.8.0.

## SIFT1M promoted cooperative-pick search-effort curve

Three clean candidate processes at commit `c9bb312` built fresh CAGRA
degree-16/intermediate-32 indexes and reused each index across the normal
batch-32 effort curve plus the high-effort batch-one point. The candidate adds
exact subgroup min-location selection for `search_width > 1` to the previously
promoted prefix-sort/frontier path; its repeated gate includes three frozen
serial-pick baseline processes under the same clean runner.
The hnswlib lanes retained `M=16`, `ef_construction=200`, 20 threads, and the
complete `ef=32/64/128/256` curve. Every point used one warmup, five measured
passes, exact validated HDF5 truth, and successful whole-package energy
sampling. The table reports the median of the three candidate process medians.

| Lane / effort | Recall@10 | Median QPS | Batch p99 | Package µJ/query |
|---|---:|---:|---:|---:|
| CAGRA 32/1, batch 32 | 0.9036 | 6,961.3 | 5.177 ms | 4,521.4 |
| CAGRA 64/2, batch 32 | 0.9609 | 3,243.5 | 10.420 ms | 11,775.1 |
| CAGRA 128/4, batch 32 | 0.9889 | 1,211.6 | 26.937 ms | 24,994.5 |
| CAGRA 128/4, batch 1 | 0.9889 | 48.167 | 24.813 ms | 527,187.5 |
| hnswlib ef32, batch 32 | 0.8910 | 11,670.4 | 3.093 ms | 2,996.5 |
| hnswlib ef64, batch 32 | 0.9593 | 6,784.5 | 5.318 ms | 4,664.5 |
| hnswlib ef128, batch 32 | 0.9882 | 3,788.1 | 9.468 ms | 8,301.5 |
| hnswlib ef256, batch 32 | 0.9972 | 2,079.8 | 17.289 ms | 14,898.6 |
| hnswlib ef256, batch 1 | 0.9972 | 2,049.3 | 0.644 ms | 14,342.8 |

The graph crosses the ≥0.95 target at `64/2`, but this still does not establish
topology parity or isolate `itopk_size` from `search_width`. At the closest
batch-32 recall points, hnswlib remains 1.68×, 2.09×, and 3.13× faster as effort
rises. The batch-one rows are not recall matched; the stronger-quality hnswlib
point remains 42.55× faster. Candidate package energy changed by -8.5%, +4.5%,
-4.8%, and -17.6% versus the serial baseline; the whole-package ranges and
scope support no blanket or isolated-iGPU energy claim. Cooperative selection
is promoted against its frozen A/B gate, not as an hnswlib or FAISS win.
Width one retains the serial-selection semantics, so its +2.0% A/B movement is
non-attributable and treated as neutral. One candidate hnswlib `ef=64` process
had a retained late CPU-tail outlier; the table uses the robust three-process
median, and the evidence report discloses the process range.

The six-process comparison, process ranges, exactness fixture, limitations,
and immutable hashes are in `tables/arrow-lake/cagra-cooperative-pick-v1.md`.
The prior frontier, cached-worst, and original diagnostic curves remain in
`tables/arrow-lake/cagra-frontier-v1.md`,
`tables/arrow-lake/cagra-cached-worst-v1.md`, and
`tables/arrow-lake/cagra-search-v1.md`.

## Historical SIFT100K scale preflight (`5dc0ec7`)

Noncanonical intermediate run after the query-seed, host graph-optimizer, and bounded NEW/OLD reverse-join changes: checksum-pinned SIFT prefix `n=100,000`, `dim=128`, `nq=1,000`, `k=10`; exact FAISS `IndexFlatL2` truth recomputed against the prefix; M=16, seed 7, one warmup and five measured passes. ovVS used AUTO construction followed by FORCE_GPU search at `itopk=32`, width=1, batch=32. hnswlib 0.8.0 used `ef_construction=200`, `ef=32`, batch=32, and 20 explicit threads.

The table reports medians across three independent full preflight processes; parentheses give the observed range.

| Lane | Recall@10 | Build wall | Median QPS | Isolated peak RSS | Validity |
|---|---:|---:|---:|---:|---|
| ovVS CAGRA | 0.9718 (same in all runs) | 9.084 s (8.904–9.087); final primitive CPU | 3,946.1 (3,937.8–3,952.0) | 398.8 MiB (398.8–398.9) | 192/192 GPU attributions per run; `192/192/0/0` transfer |
| hnswlib | 0.9533 (0.9503–0.9538) | 1.632 s (1.555–1.633) | 23,003.4 (21,984.3–23,301.6) | 202.9 MiB (200.9–203.5) | 20 explicit threads; finite unique output |

CAGRA returned 185 more exact top-10 hits than the median hnswlib run across the 10,000-result oracle, an absolute recall advantage of 0.0185. Recall is reported rather than thresholded because this prefix is not the plan's SIFT1M quality gate. By the three-run medians, hnswlib remained 5.83× faster in search and 5.57× faster to build, while CAGRA used about 1.97× the isolated peak process RSS. The CAGRA peak includes the dataset, runtime, and index; it is not device-allocation telemetry. AUTO construction ends in the CPU host optimizer, so the reported final build primitive does not describe the complete mixed build pipeline.

The preflight contract completed in all three runs at clean commit `5dc0ec7` with both lanes successful, exact truth, five QPS samples per lane, all 192 CAGRA searches per run attributed to GPU, zero NPU fallbacks, and zero explicit index uploads. Each artifact fingerprints the explicitly loaded DLL. Energy was disabled and no in-run device-occupancy trace was captured. These walls are therefore retained as revision-bound historical diagnostics, not a publishable isolated performance baseline or current throughput status. Every artifact remains partial and noncanonical; the promoted cooperative-pick SIFT1M section above is the current full-scale search status. Build-stage evidence and raw hashes: `tables/arrow-lake/cagra-build-v1.md`.

## Historical SIFT1M matched-quality gate (`5dc0ec7`)

### Recall-closeness pass retained for gate history

Checksum-pinned full SIFT1M (`n=1,000,000`, `dim=128`, `nq=1,000`, `k=10`), exact validated HDF5 truth, M=16, seed 7, one warmup and five measured passes. ovVS used AUTO construction followed by FORCE_GPU search at `itopk=32`, width=1, batch=32. hnswlib 0.8.0 used `ef_construction=200`, `ef=32`, batch=32, and 20 explicit threads.

| Lane | Recall@10 | Build wall | Median QPS (five-pass range) | Isolated peak RSS | Validity |
|---|---:|---:|---:|---:|---|
| ovVS CAGRA | 0.9036 (same in all runs) | 100.093 s (99.874–100.525); final primitive CPU | 3,465.0 (3,458.7–3,474.5) | 2,343.2 MiB | 192/192 GPU attributions per run; `192/192/0/0` transfer |
| hnswlib | 0.8911 (0.8899–0.8915) | 76.869 s (76.566–80.493) | 11,417.3 (11,093.3–11,463.8) | 1,387.0 MiB | 20 explicit threads; finite unique output |

The T13.4 recall-closeness gate **passed in all three processes**. The median difference is `0.8911 - 0.9036 = -0.0125`, within the maximum allowed gap of 0.0200. CAGRA returned 125 more exact top-10 hits than the median hnswlib run across the 10,000-result oracle. This is a closeness verdict for the fixed low-effort point, which remains 0.0464 below the separate ≥0.95 target. The later same-index curve above reaches that target by increasing traversal effort; it does not change this gate's narrow verdict. All 192 CAGRA searches per process reported GPU with zero attribution failures and zero explicit index uploads. AUTO construction ends in the CPU host optimizer, so its final-primitive label does not describe the complete mixed pipeline.

At this checkpoint, the performance result was negative: hnswlib was 3.30× faster in median search throughput, 1.302× faster to build, 10.76× better in median amortized p99, and 1.689× lighter by isolated peak process RSS. Energy was disabled, the pre-run load sample was not retained, and the artifacts contain no in-run clock, thermal, or occupancy trace. The runs are clean revision-bound diagnostics rather than the full publishable B1 curve/energy baseline. Their build walls span only 0.65%. The promoted cooperative-pick section above supersedes the 3,465.0-QPS search value for current throughput while preserving this run's recall-closeness verdict.

Compared with the prior pre-change failure below, CAGRA recall improved by 0.4979 and observed build wall fell from 830.0 to a clean 100.093-second median. Those are end-to-end mixed-change/current-environment differences across the query seed, host graph optimizer, NN-Descent rewrite, and machine load; they do not isolate causality or constitute a controlled speedup claim. The earlier 185.3-second then-current-code run is retained in history but superseded for performance status. Full build attribution and raw artifact hashes: `tables/arrow-lake/cagra-build-v1.md`.

### Prior pre-change failure

Commit `93e688b`; pinned full SIFT1M (`n=1,000,000`, `dim=128`, `nq=1,000`, `k=10`), exact HDF5 truth, M=16, seed 7, one warmup and five measured passes. ovVS used AUTO construction followed by FORCE_GPU search at `itopk=32`, width=1, batch=32. hnswlib 0.8.0 used `ef_construction=200`, `ef=32`, batch=32, and 20 explicit threads.

| Lane | Recall@10 | Observed build | Median QPS | Validity |
|---|---:|---:|---:|---|
| ovVS CAGRA | 0.4057 | 830.0 s; final primitive CPU | 1,549.7 | FORCE_GPU search; valid finite unique output; `192/192/0/0` transfer |
| hnswlib | 0.8903 | 61.7 s | 10,568.7 | valid finite unique output |

The quality gate **failed**: `0.8903 - 0.4057 = 0.4846`, versus a maximum allowed absolute gap of 0.0200. Exact truth was 1,000×10 with 10,000 valid, duplicate-free IDs against the full 1M-vector index. Both lanes succeeded; ovVS recorded zero NPU fallbacks and no explicit dataset/graph uploads. This was a valid algorithm-quality failure, not a fixture, policy, or transfer failure. It predates the query-content seed fix, host detour-rank optimizer, and bounded NEW/OLD reverse-join rewrite; the later recall-closeness rerun above supersedes it for quality-gate history.

The timing columns are diagnostic only. Preflight sampled about 62.7% unrelated iGPU 3D load (WUDFHost, ChatGPT, and DWM), so neither QPS nor build wall is a publishable isolated-performance baseline. The recall gap is far from the threshold; the near-boundary hnswlib rerun rule did not apply. The artifact remains intentionally partial because this single pair omits the full curves and package energy.

A bounded post-gate audit found no CPU/SYCL walk divergence on the same graph. Before the seed fix, batching was not invariant because both paths mixed the query ordinal within the native call: at n=4,200, 30/32 queries changed between tested partitions (mean top-10 overlap 0.675); at n=16,384, batch 32 versus batch 1 changed 1.77% of IDs. The same 16,384-row graph improved from 0.90625 recall at `32/1` to 0.953125 at `64/2`, supporting traversal effort as a secondary limiter. Query-content hashing now removes the batching/order defect and passes exact CPU/GPU ID regressions; these bounded diagnostics do not apportion the SIFT1M gap or claim improved full-corpus recall.

## B2 work-group checkpoint

Final bounded rerun after the B2 regression fixes: `bench.py --profile smoke --algorithms cagra --policies gpu --no-energy --repeats 3`. Energy was intentionally disabled, so this is a correctness and relative-throughput smoke, not B1 full-profile evidence.

| Lane | Search point | Recall@10 vs exact | Median QPS | Batch p50 ms |
|---|---|---:|---:|---:|
| ovVS CAGRA FORCE_GPU | itopk=32, width=1, batch=32 | 0.990625 | 7364.6 | 4.340 |
| ovVS CAGRA FORCE_GPU | itopk=64, width=2, batch=32 | 1.000000 | 2901.5 | 11.012 |
| ovVS CAGRA FORCE_GPU | itopk=64, width=2, batch=1 | 1.000000 | 86.8 | 10.487 |
| hnswlib | ef=32, batch=32 | 0.993750 | 71684.6 | 0.437 |
| hnswlib | ef=64, batch=32 | 1.000000 | 49162.7 | 0.648 |
| hnswlib | ef=64, batch=1 | 1.000000 | 39662.9 | 0.021 |

At this historical checkpoint, the recall held at this scale, but ovVS was roughly 10× slower at the lower-effort batch point and much farther behind at higher effort or batch size one. Later resource-local counters showed direct shared-USM index access and zero explicit dataset/graph uploads for every point in this bounded run. Leader-serial selection/sort was still open here; the promoted cooperative-pick section above supersedes that status. This is not a SIFT1M result.

## Legacy pre-B2 comparison

Validation correction (2026-08-28): the legacy FORCE_NPU brute rows below returned non-finite distances. Their wall times are retained only to document the failed attempt; they are not valid QPS, recall, or energy evidence. The range guard now returns `DEVICE_UNAVAILABLE` for this unsafe FP16-range workload until scaled execution is implemented.

| Run | ovVS brute CPU ms / QPS | ovVS brute NPU | ovVS brute GPU | FAISS brute ms | ovVS vs FAISS recall | FAISS IVF-Flat ms | ovVS IVF-Flat ms / vs-faiss-recall | FAISS IVF-PQ ms | ovVS IVF-PQ ms / vs-faiss-recall | hnswlib ms | ovVS CAGRA ms / vs-hnswlib-recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.895 / 35750 | **invalid** (607.301 ms wall; non-finite distances) | 82.072 / 389.9 (last=GPU) | 3.845 | 1.000 | 0.306 | 1.414 / 0.981 | 0.339 | 2.069 / 0.637 | 0.400 | 1.956 / 0.984 |
| 2 | 0.871 / 36722 | **invalid** (626.270 ms wall; non-finite distances) | 74.270 / 430.9 (last=GPU) | 2.300 | 1.000 | 0.141 | 1.119 / 0.981 | 0.362 | 2.057 / 0.637 | 0.428 | 2.071 / 0.984 |

The legacy NPU call reported last_device=2, but device attribution alone did not validate its numeric output. GPU last_device=3 matches FORCE_GPU (cpu=1, npu=2, gpu=3).
FAISS IVF-PQ train warns that 2000 points is below 256-centroid advice; recall is still reported.
ovVS IVF-PQ vs-faiss 0.637 is competitor-overlap (FAISS IVF-PQ vs-brute on this slice is 0.637); ovVS IVF-PQ nprobe=8 krefine=32 vs-brute is 0.947.
CAGRA (graph_degree=16, itopk=32, search_width=2) vs-brute is 0.991 vs hnswlib M=16 ef=32 vs-brute 0.994. vs-hnswlib-recall is ID overlap with hnswlib, not vs-brute.
The bench does **not** emit `FAISS comparator parked`.
Python must load `build-icpx/bin/ovvs.dll` (`OVVS_LIBRARY`); `python/ovvs/__init__.py` honors that env var and prefers `build-icpx` over `build/bin`.

Package RAPL µJ/query (`Resources.energy_uj`, EMI `RAPL_Package0_PKG`, ~150 ms repeat window) on the same slice:

| Path | last_device | uj/query |
|---|---|---|
| brute CPU | 1 | 1049 |
| brute NPU | 2 | **invalid output** (4638 measured during the failed attempt) |
| brute GPU | 3 | 17893 |
| IVF-Flat CPU | 1 | 1239 |
| IVF-PQ CPU | 1 | 2186 |
| CAGRA CPU | 1 | 2990 |

These are **package** joules (CPU+iGPU+uncore), not NPU-isolated. Short CPU searches are noisy. The legacy NPU energy delta is not a valid per-query result because the corresponding search output was invalid.
