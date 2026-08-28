# ovVS benchmark harness

`bench.py` measures recall, steady-state latency/QPS, index build time, and package energy for ovVS, FAISS-CPU, and hnswlib. Each native lane runs in its own process with a timeout, so a crash or resource failure remains an explicit lane result instead of erasing the run.

## Profiles

```powershell
# Bounded default: at most 2,000 base vectors, 32 queries, three repeats.
python tools/bench/bench.py

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

Use `--algorithms` and `--policies` for a subset, for example `--algorithms brute,ivf-flat --policies auto,cpu,gpu`. Defaults include AUTO, FORCE_CPU, FORCE_NPU, FORCE_GPU, and HETERO. `--build-policy` independently selects one construction policy for every ovVS lane and defaults to AUTO; for example, `--build-policy force-cpu` restores CPU-only construction while leaving search lanes unchanged. HETERO is recorded honestly as equivalent to AUTO until B6 lands. `--allow-unscalable-cagra` is an explicit opt-in to CAGRA above 4,096 vectors while full-scale build quality, CPU-prune cost, and peak-resource evidence remain open; otherwise those lanes stay resource-gated. CAGRA FORCE_NPU remains visible but skipped because there is no NPU graph-walk path. IVF-PQ FORCE_GPU is likewise a visible nonblocking skip: ADC deliberately has no iGPU backend and now fails closed.

## Evidence contract

- Official HDF5 neighbors are accepted only when the complete HDF5 base is selected. Prefix, synthetic, and custom bases use `faiss.IndexFlatL2`; without FAISS, exact recall is explicitly unavailable.
- Recall uses unique-ID set intersection, so duplicate returned IDs cannot inflate the score. Neighbor IDs, distances, shapes, and duplicates are validated before a point can succeed.
- Every curve point has configured warmups, repeated whole-query passes, batch and amortized-per-query p50/p95/p99 statistics, and QPS samples. Only the explicit batch-size-one points represent individual-query latency. IVF lanes sweep `nprobe`; CAGRA/hnswlib lanes sweep search effort.
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
