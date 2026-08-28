# Arrow Lake 265K recall–QPS (icx SYCL + oneMKL)

Source: `tools/bench/bench.py` against `build-icpx` `ovvs.dll` (`OVVS_WITH_SYCL=ON`, `OVVS_WITH_MKL=ON`).
Dataset: checksum-pinned `data/sift-128-euclidean.hdf5`; each section states its selected base/query geometry.
Comparators: faiss-cpu 1.15.0, hnswlib 0.8.0.

## SIFT100K current-code scale preflight

Noncanonical intermediate run after the query-seed, host graph-optimizer, and bounded NEW/OLD reverse-join changes: checksum-pinned SIFT prefix `n=100,000`, `dim=128`, `nq=1,000`, `k=10`; exact FAISS `IndexFlatL2` truth recomputed against the prefix; M=16, seed 7, one warmup and five measured passes. ovVS used AUTO construction followed by FORCE_GPU search at `itopk=32`, width=1, batch=32. hnswlib 0.8.0 used `ef_construction=200`, `ef=32`, batch=32, and 20 explicit threads.

The table reports medians across three independent full preflight processes; parentheses give the observed range.

| Lane | Recall@10 | Build wall | Median QPS | Isolated peak RSS | Validity |
|---|---:|---:|---:|---:|---|
| ovVS CAGRA | 0.9718 (same in all runs) | 8.686 s (8.672–8.800); final primitive CPU | 3,923.6 (3,874.5–3,941.5) | 398.4 MiB (398.2–398.5) | 192/192 GPU attributions per run; `192/192/0/0` transfer |
| hnswlib | 0.9510 (0.9504–0.9525) | 1.682 s (1.643–1.695) | 22,142.6 (22,104.8–22,978.9) | 201.2 MiB (201.0–202.7) | 20 explicit threads; finite unique output |

CAGRA returned 208 more exact top-10 hits than the median hnswlib run across the 10,000-result oracle, an absolute recall advantage of 0.0208. Recall is reported rather than thresholded because this prefix is not the plan's SIFT1M quality gate. By the three-run medians, hnswlib remained 5.64× faster in search and 5.16× faster to build, while CAGRA used 1.98× the isolated peak process RSS. The CAGRA peak includes the dataset, runtime, and index; it is not device-allocation telemetry. AUTO construction ends in the CPU host optimizer, so the reported final build primitive does not describe the complete mixed build pipeline.

The preflight contract completed in all three runs with both lanes successful, exact truth, five QPS samples per lane, all 192 CAGRA searches per run attributed to GPU, zero NPU fallbacks, and zero explicit index uploads. Energy was disabled and no in-run device-occupancy trace was captured. These walls are therefore retained as diagnostic current-code measurements, not a publishable isolated performance baseline. Every artifact remains partial and noncanonical; the separate full SIFT1M section below contains the current full-scale gate.

## SIFT1M matched quality gate

### Current-code rerun: recall-closeness pass

Checksum-pinned full SIFT1M (`n=1,000,000`, `dim=128`, `nq=1,000`, `k=10`), exact validated HDF5 truth, M=16, seed 7, one warmup and five measured passes. ovVS used AUTO construction followed by FORCE_GPU search at `itopk=32`, width=1, batch=32. hnswlib 0.8.0 used `ef_construction=200`, `ef=32`, batch=32, and 20 explicit threads.

| Lane | Recall@10 | Build wall | Median QPS (five-pass range) | Isolated peak RSS | Validity |
|---|---:|---:|---:|---:|---|
| ovVS CAGRA | 0.9036 | 185.3 s; final primitive CPU | 1,839.1 (1,692.2–1,853.8) | 2,342.5 MiB | 192/192 GPU attributions; `192/192/0/0` transfer |
| hnswlib | 0.8915 | 96.7 s | 11,069.0 (9,814.8–13,485.6) | 1,383.0 MiB | 20 explicit threads; finite unique output |

The T13.4 recall-closeness gate **passed**: `0.8915 - 0.9036 = -0.0121`, within the maximum allowed gap of 0.0200. CAGRA returned 121 more exact top-10 hits across the 10,000-result oracle. This is a closeness verdict, not the plan's separate ≥0.95 product target: CAGRA remains 0.0464 below that target. All 192 CAGRA searches reported GPU with zero attribution failures and zero explicit index uploads. AUTO construction ends in the CPU host optimizer, so its final-primitive label does not describe the complete mixed pipeline.

The performance result remains negative in this diagnostic run: hnswlib was 6.02× faster in search, 1.92× faster to build, and 1.69× lighter by isolated peak process RSS. Energy was disabled. A contemporaneous counter sample showed the Intel compute engine saturated during CAGRA construction while unrelated 3D processes consumed roughly 50–60%, so these walls are not an isolated publishable baseline. The recall/completion verdict is unaffected. The artifact is intentionally partial and noncanonical because it contains one matched point rather than the full B1 curves and package energy.

Compared with the prior result below, CAGRA recall improved by 0.4979 and observed build wall fell from 830.0 to 185.3 s. Those are end-to-end mixed-change/current-environment differences across the query seed, host graph optimizer, and NN-Descent rewrite; they do not isolate causality or constitute a controlled speedup claim.

### Prior pre-change failure

Commit `93e688b`; pinned full SIFT1M (`n=1,000,000`, `dim=128`, `nq=1,000`, `k=10`), exact HDF5 truth, M=16, seed 7, one warmup and five measured passes. ovVS used AUTO construction followed by FORCE_GPU search at `itopk=32`, width=1, batch=32. hnswlib 0.8.0 used `ef_construction=200`, `ef=32`, batch=32, and 20 explicit threads.

| Lane | Recall@10 | Observed build | Median QPS | Validity |
|---|---:|---:|---:|---|
| ovVS CAGRA | 0.4057 | 830.0 s; final primitive CPU | 1,549.7 | FORCE_GPU search; valid finite unique output; `192/192/0/0` transfer |
| hnswlib | 0.8903 | 61.7 s | 10,568.7 | valid finite unique output |

The quality gate **failed**: `0.8903 - 0.4057 = 0.4846`, versus a maximum allowed absolute gap of 0.0200. Exact truth was 1,000×10 with 10,000 valid, duplicate-free IDs against the full 1M-vector index. Both lanes succeeded; ovVS recorded zero NPU fallbacks and no explicit dataset/graph uploads. This was a valid algorithm-quality failure, not a fixture, policy, or transfer failure. It predates the query-content seed fix, host detour-rank optimizer, and bounded NEW/OLD reverse-join rewrite; the current-code rerun above supersedes it for present status.

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

The recall checkpoint holds at this scale, but ovVS remains roughly 10× slower at the lower-effort batch point and much farther behind at higher effort or batch size one. Later resource-local counters showed direct shared-USM index access and zero explicit dataset/graph uploads for every point in this bounded run. Leader-serial selection/sort remains open. This is not the SIFT1M result above.

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
