# Environment knobs

The complete inventory of `OVVS_*` environment variables, by class. Anything not
listed here was removed deliberately; do not resurrect a knob without a measurement
that says so. All knobs are read by `std::getenv` at call time unless noted.

## Operational

| Knob | Default | Effect |
| --- | --- | --- |
| `OVVS_CAGRA_F16` | unset (fp32) | Storage override consulted only by `OVVS_CAGRA_STORAGE_AUTO` builds and by file loads. `1`: builds convert to fp16; fp32 files convert on load only when every value round-trips exactly. `force`: fp32 files convert lossily on load. For builds, prefer the `ovvsCagraBuildEx2` `storage=` parameter — it ignores this env. |
| `OVVS_CAGRA_PATIENCE` | `0` (off) | Early-stop patience for the walk (CPU and GPU): stop after N consecutive zero-admission iterations. Output-changing; frontier-positive at several operating points, judged per workload. |
| `OVVS_CPU_WALK_THREADS` | hardware concurrency | CPU walk worker count. |
| `OVVS_CAGRA_MUTATE_GPU` | on | `0` disables GPU-assisted relink in fp16 mode (mutation repair falls back to the CPU pool). fp32 mode always uses the CPU. |
| `OVVS_CACHE_DIR` | `~/.cache/ovvs` | Runtime cache home for probe results. |
| `OVVS_TABLES` | unset | Extra search root for probe table files (`<dir>/<sku>/gemm_large.json`). |

## Escape hatches

| Knob | Effect |
| --- | --- |
| `OVVS_CAGRA_SERIAL_MUTATE` | Forces the pre-batched serial insert/update path (fp32 storage only). Diagnostic. |
| `OVVS_CPU_INT8=0` | Disables the int8 CPU walk leg (exact for integer data; on by default for L2). |
| `OVVS_GPU_INT8=0` | Disables the GPU int8 dataset mirror. |

## Sweep instruments

Defaults are the swept winners on Arrow Lake-S + Xe-LPG; the portability pass re-sweeps
these per SKU. Not for production configuration.

| Knob | Default | Sweeps |
| --- | --- | --- |
| `OVVS_CAGRA_RELINK_ITOPK` / `OVVS_CAGRA_RELINK_WIDTH` | `degree` / `1` | Mutation-repair search effort. |
| `OVVS_CAGRA_MUTATE_CHUNK` | `clamp(n/64, 256, 4096)` | Mutation chunk size (graph staleness bound). |
| `OVVS_CAGRA_MUTATE_THREADS` | hardware concurrency | Back-link prune worker cap. |
| `OVVS_HYBRID_WALK` | off | Forces a CPU+GPU split of one batch at the given GPU fraction. Splits measured losing at every fraction on BOTH lanes of this box (package power/bandwidth sharing); the knob exists to re-ask on dGPU-class hardware. |
| `OVVS_HYBRID_MIN_NQ` | 8 at ≥320K rows, ∞ below | HETERO's batch-to-GPU crossover override (R2 boundary work). |
| `OVVS_CAGRA_TILE` | kernel default | GPU frontier tile size. |
| `OVVS_CAGRA_VISITED_MULT` | kernel default | Visited-table headroom multiplier. |
| `OVVS_CAGRA_SEEDS` | `cagra_seed_count` | Seed count override (CPU/GPU parity requires both engines see the same value). |

## Experiment lane (recorded losers on Xe-LPG, kept for other SKUs)

| Knob | Verdict here |
| --- | --- |
| `OVVS_CAGRA_QPW=8` | Sub-group-per-query mapping: 0.82× at b1024 vs the classic path. Kept for SKUs with more thread slots / SLM per core. |
| `OVVS_CAGRA_BEAM_DEDUP` | Beam-membership dedup companion for the QPW=8 lane (bit-identical output; 14% regression at one-work-group-per-query). |
| `OVVS_CAGRA_SEEN_RING` | Rejected-id cache for the QPW=8 lane; any size loses residency here (measured default 0). |

## Removed

| Knob | Why |
| --- | --- |
| `OVVS_CAGRA_STOP_EF` | Convergence-stop rule lost on the recall/throughput frontier; feature and kernel blocks deleted. |
| `OVVS_GPU_SUBGROUP_GEOM` | A/B isolation for the sub-group-per-candidate geometry; the geometry is the recorded winner (2.4–5.7×) and is now unconditional. |
