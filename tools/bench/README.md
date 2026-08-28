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

Use `--algorithms` and `--policies` for a subset, for example `--algorithms brute,ivf-flat --policies auto,cpu,gpu`. Defaults include AUTO, FORCE_CPU, FORCE_NPU, FORCE_GPU, and HETERO. HETERO is recorded honestly as equivalent to AUTO until B6 lands. `--allow-unscalable-cagra` is an explicit opt-in to the current host CAGRA build above 4,096 vectors; otherwise those lanes remain resource-gated. CAGRA FORCE_NPU remains visible but skipped because there is no NPU graph-walk path.

## Evidence contract

- Official HDF5 neighbors are accepted only when the complete HDF5 base is selected. Prefix, synthetic, and custom bases use `faiss.IndexFlatL2`; without FAISS, exact recall is explicitly unavailable.
- Recall uses unique-ID set intersection, so duplicate returned IDs cannot inflate the score. Neighbor IDs, distances, shapes, and duplicates are validated before a point can succeed.
- Every curve point has configured warmups, repeated whole-query passes, batch and per-query p50/p95/p99 statistics, and QPS samples. IVF lanes sweep `nprobe`; CAGRA/hnswlib lanes sweep search effort and include a batch-size-one latency point.
- Index build time is separate. ovVS indexes are built under FORCE_CPU for identical construction across search-policy lanes; the requested policy applies before warmup/search.
- Energy is package µJ/query from ovVS telemetry when available. It includes CPU, iGPU, and uncore and is not isolated NPU/GPU energy.
- `last_device` is final-primitive evidence, not a pipeline trace. In particular, IVF-PQ FORCE_GPU may execute ADC on NPU; the artifact flags this policy contract as nonconforming rather than claiming a pure-GPU result.

Each lane has one of `success`, `unavailable`, `failed`, `timeout`, or `skipped`. Full profiles return nonzero for incomplete evidence unless `--allow-partial` is explicit. Known B5 CAGRA skips keep a full run partial. The synthetic 100K x 768 profile is also marked provisional and cannot close the real-corpus B20 requirement. Smoke remains exit-zero for development convenience, but its artifact still says `partial` when evidence is missing.

JSON and a concise Markdown table are written together under ignored `out/bench/` by default. `--output path.json` selects another destination. Artifacts record Git/machine/dependency metadata, SIFT/custom SHA-256 identities (or a deterministic generation-spec fingerprint for synthetic data), exact-ground-truth provenance, lane stderr tails, and every failed/skipped comparator rather than silently omitting it.

## Tests

The helper and CLI-contract tests do not require the dataset, native library, or accelerator hardware:

```powershell
python -m unittest discover -s tools/bench -p "test_*.py" -v
```
