# Arrow Lake HNSW export evidence

Status: **stock hnswlib interoperability validated; acceleration and promotion rejected at this checkpoint**. The ovVS CAGRA exporter produced a structurally valid, fully reachable single-layer hnswlib index in three clean SIFT100K processes. Stock hnswlib loaded and searched every exported index, but the exported path lost to native hnswlib at the nearest measured comparable-recall points and was substantially slower to construct. This is retained as B24 partial evidence, not as a performance claim.

## Measurement contract

The retained artifacts are the ignored local files `out/bench/hnsw-export-sift100k-r1.json` through `r3.json` and their Markdown renderings. All three were produced from clean `main` commit `edbf656a21d00248f5029d16b1e0ef4bd3473ba8` with ovVS 0.2.0, hnswlib 0.8.0, and FAISS 1.15.0 for exact truth. The command was:

```text
tools/bench/bench.py --profile sift-100k --algorithms cagra --policies gpu --build-policy auto --include-hnsw-export --allow-unscalable-cagra --hnsw-threads 20 --no-energy --output out/bench/hnsw-export-sift100k-rN.json
```

The checksum-pinned source was `data/sift-128-euclidean.hdf5`, SHA-256 `dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984`. Each process selected the first 100,000 base vectors and first 1,000 queries at dimension 128, used squared L2 and `k=10`, and recomputed exact prefix truth with FAISS `IndexFlatL2`. Each curve point used batch 32, one warmup, and five measured passes.

Native hnswlib retained its full settings: `M=16`, `ef_construction=200`, seed 7, 20 explicit threads, and search `ef={32,40,48,56,64}`. The hybrid lane built ovVS CAGRA with AUTO, exported graph degree 16, released the producer resources, loaded the file into stock hnswlib, and searched the same `ef` curve with the same 20 threads. Direct CAGRA search used FORCE_GPU at `(itopk_size=32, search_width=1)` and `(64,2)`. No comparator setting was weakened.

Values below are the median of the three independent process medians. Parentheses are the minimum and maximum process medians. All 9 requested lanes completed successfully: direct CAGRA, native hnswlib, and exported hnswlib in each process.

## Export contract and graph validation

Every process produced the same file and topology:

| Property | Retained value |
|---|---:|
| Format | stock hnswlib native, single layer |
| Rows / maximum elements | 100,000 / 100,000 |
| Dimension / metric | 128 / squared L2 |
| Entry point / maximum level | 0 / 0 |
| `M` / `maxM` / `maxM0` | 16 / 16 / 32 |
| Active out-degree, min / mean / max | 16 / 16.000 / 16 |
| Entry-reachable nodes | 100,000 / 100,000 (100%) |
| Identity labels / deletion-marked nodes | yes / 0 |
| Upper-layer bytes | 0 |
| File size | 65,600,096 bytes |
| SHA-256 | `97fb9c9b6fc2fb25aa6b66c8467be77ba484daddc104171b7e976f9c1234dd6d` |

The strict reader checked the sequential native header, dimensions, identity labels, deletion and reserved bits, every active neighbor ID, absence of upper blocks, exact end-of-file, hash, and reachability before stock hnswlib search. Focused interoperability coverage also loads, queries, saves, reloads, and queries the exported index. The file intentionally advertises `maxlevel=0`; this proves safe base-layer interoperability, not hierarchical-HNSW construction parity. Although native `M=16` reserves 32 base-layer slots, the exporter currently populates the 16 CAGRA graph-degree links only. Serialized `ef_construction=200` is compatibility metadata, not an ovVS construction parameter.

## Construction wall and memory

| Construction path | Complete wall, ms | Isolated peak RSS, MiB |
|---|---:|---:|
| Direct ovVS CAGRA build | 8,906.172 (8,889.753–9,125.720) | 398.777 (398.730–399.102) |
| Native hnswlib build | 1,634.989 (1,599.235–1,831.551) | 200.977 (200.266–202.750) |
| CAGRA → export → loaded stock hnswlib | 9,448.863 (9,427.063–9,453.759) | 545.672 (540.598–546.219) |

The complete hybrid scope begins with benchmark resource creation and ends with a loaded stock hnswlib index. Its instrumentation-adjusted median was 9,448.596 ms (9,426.814–9,453.485), while the sum of production stages was 9,447.320 ms (9,425.585–9,452.194). The hybrid build was **5.779× slower** than native hnswlib by the outer medians.

| Hybrid stage | Median ms (range) |
|---|---:|
| Resource creation | 191.193 (182.648–196.721) |
| ovVS CAGRA build | 8,882.315 (8,869.125–8,899.239) |
| CAGRA-to-HNSW conversion | 6.683 (6.364–6.810) |
| Serialization | 189.446 (186.391–196.968) |
| Producer release | 0.112 (0.110–0.117) |
| Stock hnswlib load | 170.363 (168.450–172.045) |

AUTO construction reported CPU only for its final primitive; that attribution does not characterize the complete mixed construction pipeline. Peak RSS is isolated process peak working set including the dataset, runtime, and index. The hybrid peak additionally covers graph validation, file mapping, and hashing, so it is a conservative diagnostic peak rather than a clean production-memory comparison.

## Stock hnswlib search curves

### Native hnswlib construction

| `ef` | Recall@10 | Median QPS | Batch p50 ms | Batch p99 ms |
|---:|---:|---:|---:|---:|
| 32 | 0.9512 (0.9510–0.9525) | 22,128.1 | 1.408 | 1.744 |
| 40 | 0.9665 (0.9664–0.9683) | 19,540.3 | 1.637 | 2.123 |
| 48 | 0.9762 (0.9757–0.9776) | 16,979.4 | 1.873 | 2.708 |
| 56 | 0.9825 (0.9820–0.9841) | 12,159.2 | 2.479 | 3.203 |
| 64 | 0.9878 (0.9874–0.9889) | 10,581.5 | 3.030 | 3.511 |

### ovVS CAGRA export loaded by stock hnswlib

| `ef` | Recall@10 | Median QPS | Batch p50 ms | Batch p99 ms |
|---:|---:|---:|---:|---:|
| 32 | 0.9020 | 19,065.7 | 1.660 | 1.963 |
| 40 | 0.9292 | 16,332.7 | 1.960 | 2.219 |
| 48 | 0.9462 | 14,193.3 | 2.227 | 2.872 |
| 56 | 0.9583 | 12,879.1 | 2.456 | 2.855 |
| 64 | 0.9657 | 11,769.5 | 2.725 | 3.234 |

The closest practical comparisons remain negative:

- Native `ef=32` reached 0.9512 recall at 22,128.1 QPS. The hybrid needed `ef=56` to reach 0.9583 and delivered 12,879.1 QPS, or **0.582× native throughput** (41.8% lower).
- Native `ef=40` reached 0.9665 recall at 19,540.3 QPS. Hybrid `ef=64` reached 0.9657 at 11,769.5 QPS, or **0.602× native throughput** (39.8% lower) with a 0.0008 recall deficit.

The export therefore did not turn CAGRA construction into a faster CPU-searching HNSW index. The result is useful negative evidence: the format and connectivity are sound, while the current graph geometry requires more search work than native hnswlib at comparable quality.

## Direct CAGRA diagnostic

| FORCE_GPU point | Recall@10 | Median QPS (range) | Batch p50 ms | Batch p99 ms |
|---|---:|---:|---:|---:|
| `itopk_size=32`, width 1 | 0.9718 | 3,950.6 (3,941.0–3,952.1) | 7.854 | 8.686 |
| `itopk_size=64`, width 2 | 0.9946 | 1,316.7 (1,314.7–1,317.4) | 23.661 | 25.384 |

Each direct point recorded 192/192 GPU-attributed walks per process with zero explicit index uploads. The lower-effort direct CAGRA point exceeded the best hybrid point by 0.0061 absolute recall (0.9718 versus 0.9657), but these use different traversal algorithms and knobs; the difference is only a bounded diagnostic, not a B24 quality proof. Direct CAGRA also remained much slower than stock hnswlib in this prefix run.

## Disposition and limits

The single-layer exporter is safe enough for stock hnswlib load, query, save, and reload, and the benchmark now measures its complete construction-to-loaded-index scope. That completes the interoperability portion of the work. It does **not** justify promotion:

- SIFT100K is a noncanonical prefix diagnostic; the full SIFT1M quality/throughput gate remains open.
- Package energy was disabled, so all artifacts correctly remain `partial`.
- No hierarchical layers are constructed, and the populated base degree differs from native HNSW's available base capacity.
- The hybrid loses both complete build wall and nearest comparable-recall search throughput to unmodified hnswlib.
- The hybrid RSS includes diagnostic mapping and hashing and cannot be treated as a production footprint.

No acceleration claim is made. Further tuning of this exported-search branch is parked unless a new graph-construction or optimizer change materially alters the measured topology/quality tradeoff; active CAGRA work should first improve and attribute the underlying GPU construction and search kernels, then rerun the same complete end-to-end gate.
