# ovVS benchmark harness

`bench.py` measures recall, steady-state latency/QPS, index build time, and package energy for ovVS, FAISS-CPU, and hnswlib. Each native lane runs in its own process with a timeout, so a crash or resource failure remains an explicit lane result instead of erasing the run.

## Profiles

```powershell
# Bounded default: at most 2,000 base vectors, 32 queries, three repeats.
python tools/bench/bench.py

# Fixed noncanonical resource/quality preflight on the first 100,000 SIFT vectors.
python tools/bench/bench.py --preflight-only cagra-sift100k --hnsw-threads 20

# Full SIFT base; no fallback if the HDF5 file or dependencies are absent.
python tools/bench/bench.py --profile sift1m

# Provisional deterministic 100,000 x 768 workload.
python tools/bench/bench.py --profile embedding-100k

# Real/custom 100,000 x 768 arrays when B20 supplies the corpus.
python tools/bench/bench.py --profile embedding-100k --base data/base.npy --queries data/queries.npy
```

Install the benchmark-only dependencies in the Python environment used to launch the harness:

```powershell
python -m pip install numpy h5py faiss-cpu hnswlib
```

For a source-pinned Windows AVX2 comparator candidate, keep the ambient wheel as a control and build the candidate in ignored `out/` storage. Visual Studio 2022 C++ Build Tools and `llvm-objdump.exe` must be available from the selected developer environment. Run the harness with the same Python ABI used by the builder, and use the exact `module=` and `provenance=` paths it prints:

```powershell
$python = 'C:\path\to\python.exe'
pwsh tools/bench/build_hnswlib_avx2.ps1 -Python $python -StockModule C:\path\to\stock\hnswlib.pyd
& $python tools/bench/bench.py --profile sift1m --hnsw-threads 20 `
  --hnsw-module '<builder-emitted module path>' `
  --hnsw-provenance '<builder-emitted provenance path>'
```

The builder pins hnswlib v0.8.0 source and dependency versions, records downloaded wheel hashes, injects MSVC `/arch:AVX2`, and retains hash-bound candidate disassembly plus stock disassembly when `-StockModule` is supplied. It never installs into ambient Python. The harness rejects an explicit module unless its manifest matches the binary and certifies the exact source commit/tree, clean tracked source, MSVC AVX2 options, bound positive SIMD evidence, and finite import/add/query smoke. It records the loaded binary path/hash/bytes and infers the pinned 0.8.0 binding's effective search threads per actual batch. SIFT1M includes targeted comparable-recall batch-cap 256 and 1024 throughput points; because the fixture has 1,000 queries, the latter is one actual 1,000-row call. Static SIMD evidence is not active-cycle attribution or a performance result; tracked build evidence is in `tables/arrow-lake/evidence/hnswlib-avx2-build-v1/`.

Use `--algorithms` and `--policies` for a subset, for example `--algorithms brute,ivf-flat --policies auto,cpu,gpu`. Defaults include AUTO, FORCE_CPU, FORCE_NPU, FORCE_GPU, and HETERO. `--build-policy` independently selects one construction policy for every ovVS lane and defaults to AUTO; for example, `--build-policy force-cpu` restores CPU-only construction while leaving search lanes unchanged. HETERO is recorded honestly as equivalent to AUTO until B6 lands. `--allow-unscalable-cagra` is an explicit opt-in to CAGRA above 4,096 vectors while full-scale build quality, CPU-prune cost, and peak-resource evidence remain open; otherwise those lanes stay resource-gated. CAGRA FORCE_NPU remains visible but skipped because there is no NPU graph-walk path. IVF-PQ FORCE_GPU is runnable through the fused iGPU scan/select path; AUTO remains on the measured CPU/NPU route until matched end-to-end evidence promotes the GPU.

`--include-hnsw-export` adds an experimental hybrid lane: build CAGRA under the selected ovVS build policy, convert and serialize the base graph, release the producer indexes, load the file in stock hnswlib, then run the same hnswlib search curve and explicit thread count as the native comparator. The lane records each construction stage, the construction-pipeline wall from ovVS resource creation through a loaded stock hnswlib index, its instrumentation-adjusted value, file hash/bytes, entry reachability, recall, latency/QPS, and peak worker RSS. Dataset, module, and truth loading occur before that construction timer. It does not alter native hnswlib `M`, `ef_construction`, `ef`, or threads. The serialized `ef_construction` value is file metadata, not an ovVS construction knob, and the base-only graph is not claimed to be a full HNSW hierarchy. Peak RSS includes the bounded graph diagnostic.

`sift-100k` reads the checksum-pinned SIFT HDF5 but selects only the first 100,000 base vectors and 1,000 queries. Its exact oracle is recomputed with FAISS because the file's official neighbors apply to the complete 1M base. The dedicated preflight fixes AUTO construction, FORCE_GPU CAGRA search, one matched CAGRA/hnswlib effort point, one warmup, five measured passes, seed 7, and disables energy. It records isolated-worker peak RSS, build/search timing, recall, QPS, device attribution, and the derived `192/192/0/0` direct-walk/zero-upload transfer delta. Peak RSS includes the dataset, runtime, and index; it is not device-only allocation telemetry.

The preflight exits zero only when its dataset, truth, two lanes, fixed settings, measurements, transfer counters, and memory evidence validate. Recall is reported but has no pass threshold. The artifact remains `partial`, `canonical_b1_evidence=false`, and noncanonical even when the preflight completes; the full SIFT1M quality gate remains required. A timeout or resource failure is retained explicitly and exits nonzero. When `--hnsw-threads` is omitted, all logical CPUs reported by `os.cpu_count()` are used and recorded.

The narrow CAGRA quality checkpoint has a dedicated mode:

```powershell
python tools/bench/bench.py --gate-only cagra-recall --hnsw-threads 20
```

This mode deterministically selects the checksum-pinned full SIFT1M fixture, AUTO construction, FORCE_GPU CAGRA search, and one matched pair: CAGRA `itopk_size=32,search_width=1,query_batch_size=32` versus hnswlib `ef=32,query_batch_size=32`. It fixes one warmup, five measured passes, seed 7, and disables energy. The gate passes only when `hnswlib recall - CAGRA recall <= 0.0200` with valid exact truth, device attribution, and an exact CAGRA transfer delta of `192/192/0/0`. QPS is reported but never enters the verdict. The artifact and Markdown remain `partial` and noncanonical even on a passing exit, because one pair cannot close B1's full recall-QPS curves. A passing gate exits zero without `--allow-partial`; a failed or invalid gate exits nonzero. When `--hnsw-threads` is omitted, the mode pins the recorded `os.cpu_count()` value. Near-boundary recall should be rerun because parallel hnswlib insertion can vary despite a fixed seed and thread count.

## Evidence contract

- Official HDF5 neighbors are accepted only when the complete HDF5 base is selected. Prefix, synthetic, and custom bases use `faiss.IndexFlatL2`; without FAISS, exact recall is explicitly unavailable.
- Recall uses unique-ID set intersection, so duplicate returned IDs cannot inflate the score. Neighbor IDs, distances, shapes, and duplicates are validated before a point can succeed.
- Every curve point has configured warmups, repeated whole-query passes, batch and amortized-per-query p50/p95/p99 statistics, and QPS samples. Only the explicit batch-size-one points represent individual-query latency. IVF lanes sweep `nprobe`; CAGRA/hnswlib lanes sweep search effort. A normal full profile is incomplete unless each CAGRA/HNSW lane contains the exact configured point multiset; gate/preflight selections remain explicit partial subsets. Full-profile hnswlib evidence also requires a loaded-binary fingerprint, explicit threads, and binding-effective thread provenance.
- CAGRA points record cumulative transfer counters immediately before warmups and after timed repeats plus any package-energy passes. The Markdown `Wcalls/Dcalls/Ucalls/Ubytes` delta proves whether every batched walk call bound GPU-accessible dataset/graph pointers directly and avoided explicit index `memcpy`; it does not measure driver-managed shared-memory page migration. A successful full-profile FORCE_GPU point is incomplete without monotonic counter evidence, at least one walk call, equal walk/direct-call deltas, and zero upload calls and bytes.
- Index construction and search policies are independent. All ovVS lanes request the same AUTO construction policy by default, so CAGRA can route scalable NN-Descent initialization to the iGPU before CPU pruning. Each lane applies its requested search policy only after construction completes.
- Each ovVS build record includes its status/reason, requested policy, elapsed time, resource telemetry before and after construction, counter deltas for NPU compile failures/fallbacks, and final-primitive `last_device`. Build `last_device` is not a pipeline trace and cannot prove every CAGRA construction stage.
- Energy is package µJ/query from ovVS telemetry when available. It includes CPU, iGPU, and uncore and is not isolated NPU/GPU energy. Missing or disabled energy keeps a full profile partial.
- `last_device` is final-primitive evidence, not a pipeline trace. Forced search lanes fail their policy contract when the final primitive is missing or attributed to another device; numeric/device failures remain explicit rather than being reclassified as mixed execution.

Each lane has one of `success`, `unavailable`, `failed`, `timeout`, or `skipped`. Full profiles return nonzero for incomplete evidence unless `--allow-partial` is explicit. Known B5 CAGRA skips keep a full run partial. The synthetic 100K x 768 profile is also marked provisional and cannot close the real-corpus B20 requirement. Smoke remains exit-zero for development convenience, but its artifact still says `partial` when evidence is missing.

JSON and a concise Markdown table are written together under ignored `out/bench/` by default. `--output path.json` selects another destination. Artifacts record Git/machine/dependency metadata, SIFT/custom SHA-256 identities (or a deterministic generation-spec fingerprint for synthetic data), exact-ground-truth provenance, lane stderr tails, and every failed/skipped comparator rather than silently omitting it.

## Tests

The helper and CLI-contract tests do not require the dataset, native library, or accelerator hardware:

```powershell
python -m unittest discover -s tools/bench -p "test_*.py" -v
```
