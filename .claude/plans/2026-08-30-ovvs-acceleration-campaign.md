# ovVS acceleration campaign — north star

**Purpose of this file.** Durable objective + state anchor for the current campaign. Re-read this
first after any context loss, compaction, or realignment tick. If this file and a memory of the
conversation disagree, **this file wins**. Update the Status log at the bottom every time
something real lands or is ruled out.

Related canon: `AGENTS.md` (rules), `.claude/plans/2026-08-20-ovvs-intel-cuvs-equivalent.md` (spec),
`.claude/backlog.md` (B1–B32 status).

---

## 1. The objective

Make on-device vector search actually fast on ordinary Intel laptops/desktops — the majority of
enterprise machines, which have no NVIDIA or Apple accelerator. The product this unlocks is local
RAG / on-device AI memory: embeddings stored and retrieved on the user's own machine, alongside a
running LLM.

That means ovVS must behave like a **living store**, not a static benchmark artifact.

## 2. Hard gates (all four must hold — no trading one for another)

| # | Gate | Bar |
|---|---|---|
| G1 | **Query** beats hnswlib CPU | Higher QPS at equal-or-better recall@10, hnswlib never weakened |
| G2 | **Balanced throughput** | query / insert / update / delete all practical; no operation is a stall |
| G3 | **RAM not over-leveraged** | Resident footprint ≤ hnswlib for the same corpus. Winning by spending RAM is cheating |
| G4 | **Quality holds under churn** | Recall must not decay as inserts/updates/deletes accumulate |

hnswlib comparator settings are **frozen**: `M=16, ef_construction=200, ef=64, 20 threads`.
Never weaken them. Use the source-pinned `/arch:AVX2` build as the fair comparator, not the
packaged legacy-SSE wheel.

## 3. Scale ladder — win at each rung before climbing

Do **not** jump to the next rung until the current one passes G1–G4.

| Rung | Corpus | Status | Notes |
|---|---|---|---|
| R1 | **SIFT100K** (first 100k of SIFT1M) | ← **current rung** | Fast iteration. Exact truth must be **recomputed** — the HDF5 `neighbors` field is only valid for the full 1M base |
| R2 | **SIFT1M** | not started | HDF5 `neighbors` truth valid as-is |
| R3 | **10M** | not started | Needs a corpus that does not exist yet. 10M×128×f32 = 5.1 GB raw — G3 forces quantization at this rung |

## 4. Measured baselines on this machine (Arrow Lake-S, Core Ultra 7 265K)

SIFT1M, 1,000 queries, hnswlib AVX2 `ef=64` (medians of 3 processes):

| Lane | batch | threads used | recall@10 | QPS |
|---|---:|---:|---:|---:|
| ovVS CAGRA `itopk=64/width=2` | 32 | iGPU | 0.9609 | 3,243 |
| hnswlib packaged (legacy SSE) | 32 | 1 | 0.9593 | 6,785 |
| hnswlib AVX2 | 32 | **1** | 0.9584 | 9,716 |
| hnswlib AVX2 | 256 | 20 | 0.9586 | 26,558 |
| **hnswlib AVX2** | **1024** | **20** | **0.9586** | **27,721** |

### ⚠ The two bars — do not confuse them

**hnswlib's Python binding silently collapses to ONE thread when `rows <= num_threads*4`**
(`bindings.cpp:631` for `knn_query`, `:264` for `add_items`; verified in source). With
`num_threads=20` that means **every batch-1 and batch-32 point runs hnswlib on a single core**.

- **Bar A (single-core hnswlib):** 9,716 QPS @ batch 32. Beating this is beating *one CPU core*
  with the whole iGPU. Necessary, but a hollow headline on its own.
- **Bar B (fully-threaded hnswlib) — THE REAL TARGET:** **27,721 QPS @ batch 1024, recall 0.9586.**
  This is "max practical throughput" with hnswlib using the whole machine.

The user's goal is max practical throughput with hnswlib never handicapped, so **G1 is judged
against Bar B**. Report both, always labelled with the thread count. Currently ~8.5× short of Bar B.
ovVS must also be measured at batch 256/1024 — the kernel is one work-group per query, so it has
never been measured in a regime where the iGPU is actually saturated.

Other measured facts:
- ovVS SIFT1M build 61.0 s (median, post GPU-optimizer + cooperative-minima) vs hnswlib AVX2 78.2 s.
  Build is the one thing we already win: ~22% faster.
- Resident ovVS index = 576,000,036 B (512 MB f32 vectors + 64 MB degree-16 graph).
  2,343 MiB is **build peak**, not resident — do not confuse the two.
- Batch-1 is catastrophic: ~48 QPS vs hnswlib ~2,982 (AVX2). Batch 1 is the *product* case.

## 4b. R1 (SIFT100K) CLEAN BASELINE — quiet machine, 1,000 queries, 1 warmup + 5 passes

ovVS CAGRA (FORCE_GPU) vs hnswlib AVX2 (`M=16, ef_construction=200`, seed 7, 20 threads requested).
hnswlib's binding collapses to 1 thread at batch ≤80 — `threads` column is what actually ran.

| ovVS point | recall | b32 | b256 | b1024 | b1 |
|---|---:|---:|---:|---:|---:|
| `32/1` | 0.9718 | 7,590 | 14,785 | **15,696** | — |
| `64/2` | 0.9946 | 3,559 | 7,061 | 6,594 | 139 |
| `128/4` | 0.9993 | 1,351 | — | — | — |

| hnswlib | recall | b32 (1t) | b256 (20t) | b1024 (20t) | b1 (1t) |
|---|---:|---:|---:|---:|---:|
| `ef=32` | 0.9516 | 21,018 | 66,001 | **105,699** | 21,192 |
| `ef=64` | 0.9885 | 8,210 | 35,182 | 44,127 | 7,692 |
| `ef=128` | 0.9983 | 4,733 | 33,465 | 39,993 | 4,524 |
| `ef=256` | 0.9998 | 2,790 | 21,656 | 23,653 | 2,662 |

Build at 100K: hnswlib **1.82 s** vs ovVS **5.48 s** — hnswlib is 3× faster here. (Our build win is
scale-dependent: at 1M we are 61 s vs 78 s, i.e. 22% faster. Do not generalise the 1M build win.)

**Honest gap at R1:** ovVS `32/1` (0.9718) sits between hnswlib `ef=32` (0.9516) and `ef=64`
(0.9885). Interpolating hnswlib to our recall gives roughly 70–80k QPS at batch 1024 against our
15,696 → **~5× behind**. At batch 32 we are ~3.5× behind. **At batch 1 we are ~32× behind**
(139 vs 4,524). The gap is ~5× at both R1 and R2, so it is scale-stable.

### The saturation claim — measured, and the inference was wrong

The CRUD judge argued the kernel workstream is "optimizing a path that has already hit the hardware
wall", citing CAGRA plateauing across batch. The plateau is **real and reproduced on a quiet
machine**: `32/1` goes 7,590 → 14,785 → 15,696 (plateau), and `64/2` goes 3,559 → 7,061 → 6,594
(**regresses** at 1024). But the inference does not follow:

- At the plateau we are at **0.35% of FP32 peak and ~10% of the bandwidth roofline**. That is not a
  hardware wall.
- The plateau is explained by the **visited table**: 128 KiB/query at `itopk=64`. At batch 32 that
  is 4 MiB (fits the 4 MiB L2); at batch 256 it is 32 MiB (thrashes). The regression at batch 1024
  is the `queries_per_launch = 64 MiB / visited_bytes` split into two launches.
- **Step 3 targets exactly this.** `max_iters` is 3·SW× oversized, and it is the multiplier on
  `visited_capacity`; right-sizing it drops the table ~4× (to ~32 KiB/query at `itopk=64`).

Correct reading: **more parallelism will not help; less work and a smaller working set will.**
That is what steps 2–6 do. The kernel plan stands.

## 5. Root-cause diagnosis driving the work

The iGPU CAGRA search kernel (`src/prim/gpu/backend_gpu.cpp`, `gpu_graph_walk`, ~L2555–2960) runs
at roughly **1% of the iGPU's FP32 peak**:

- `cooperative_distance()` gives each of 128 work-items **one dimension** of one 128-dim vector,
  then does a full work-group `reduce_over_group`. Candidates are processed **one at a time**.
- Lane 0 alone does beam insertion, worst-slot rescan on eviction, neighbour gather, and visited-hash
  probing while 127 work-items idle.
- `nseeds = max(itopk*16, sw*32, 512)` → 512/1024/2048 **random** seeds, each a full 128-dim
  evaluation. This is **40–45% of ALL distance work** and contributes almost nothing to recall.
- Visited set is a global-memory hash (32–131 KiB/query), zeroed on every kernel entry, at a
  3–8% load factor.
- Scalar FP32 loads, no vectorization, no `reqd_sub_group_size` anywhere in the codebase.
- Per `ovvsCagraSearch` call: 4 `malloc_device`/`free`/`wait_and_throw` round trips. Pure tax at batch 1.
- The query vector is re-read from global memory for every single candidate.

**Measured achieved throughput: ~0.15% of the 2.05 TFLOP/s FP32 peak** (~3.1 GFLOP/s).
Roofline: a bandwidth-bound implementation tops out near ~58,000 QPS; Bar B (27,721) needs ~48% of
that roofline, Bar A (9,716) needs ~17%. **The hardware is not the obstacle.**

### Corrections to the original diagnosis (verified in the map phase)

- ❌ **`max_iters = itopk*6` is NOT the cost driver.** The `npicks == 0` break fires at
  ≈ `2*itopk/SW` = 64 iterations at all three benchmark points. `max_iters` only inflates
  `visited_capacity`. **However two map agents disagreed** on whether the eviction path
  (`consider()` resets `candidate_expanded[worst]=0`, L2790) defeats the break. **Unresolved —
  the distance-eval counter settles it empirically. Do not act on either claim until measured.**
- ✅ Everything else confirmed, and understated: ~5,300 work-group barriers per query at itopk=64;
  384 FLOPs bought per candidate with 2 barriers and ~150 SLM accesses.

## 6. Mutation surface — the other half

There is **no delete and no update anywhere in the C ABI**. Only `Extend` (append) for
IvfFlat / IvfPq / IvfRabitq / Cagra. `ovvsCagraExtend` is a host-side insert + `robust_prune`,
tested only at n=40. CAGRA is build-once by design.

G2/G4 therefore need net-new architecture, not tuning. A tombstone/filter primitive already exists
(`ovvsBitsetFromAllowList` + `allowed_id` honoured in the GPU walk) and is the obvious foundation.

hnswlib's own delete is weak — `markDelete` is a tombstone that never reclaims memory and degrades
graph quality under churn. That is the genuine opening. Do not claim it before measuring it.

## 7. Anti-rabbit-hole rules

1. **Measure on a quiet machine.** No agents, no builds, no other GPU work during a timed run.
   A prior cohort was invalidated by 13.7% background GPU load. Recall is load-independent; QPS is not.
1b. **THERMAL DISCIPLINE (learned the hard way, 2026-08-30).** Sustained iGPU load degrades this
   box badly: after ~40 min of back-to-back runs the SAME binary+config produced 7,794–15,864 QPS
   (>50% spread) where it had produced 22,510 earlier. The iGPU shares package power with 20 CPU
   cores. Rules:
   - **Prefer work counters over QPS** for accept/reject. `evals/query` is load- and
     thermal-independent and is the honest measure of a work-reduction change.
   - One point per process; **interleave** baseline/candidate; **≥3 alternations**.
   - Check `LoadPercentage` and `CurrentClockSpeed` vs `MaxClockSpeed` before a timed run.
   - Cool down between cohorts. Never compare a run at sequence position 7 against one at
     position 1 — later points in a multi-point process are measurably throttled.
   - Terminal output is itself load: WindowsTerminal accumulated 42,111 s CPU in this session.
     Keep tool output small during measurement.
2. **One variable per experiment.** If two things change, the measurement is worthless.
3. **Timebox.** Any single experiment that has not produced a number in ~60 minutes gets parked
   with a written reason. Park it, do not abandon it silently.
4. **No smoke-test spirals.** A smoke test proves the harness runs. It never proves a speedup.
   Get to a real recall+QPS number as fast as possible.
5. **Negative results are results.** Record them here and move on. Do not re-litigate a parked idea
   without a materially changed capability.
6. **Never weaken the competitor** to manufacture a win.
7. **Resident vs peak memory** are different numbers. Always say which one.
8. Prefer the search-only A/B loop (`tools/bench/quick_cagra.py`) over full harness runs while
   iterating. Full harness only for the final result at a rung.

## 7b. Hardware ground truth (verified — supersedes docs/devices.md)

Arrow Lake-S iGPU, Core Ultra 7 265K:

- Plain **Xe-LPG** (not Xe-LPG+), 1 render slice, **4 Xe-cores = 64 XVE = 512 FP32 lanes**,
  8 HW threads/XVE → 512 HW threads device-wide. 2.0 GHz boost.
- **FP32 peak ≈ 2.05 TFLOP/s**, FP16 ≈ 4.1, INT8 DP4a ≈ 8.2 TOPS.
- **NO XMX / DPAS matrix engines — DP4a only.** `docs/devices.md` and `docs/hw-split.md` are
  **factually wrong** where they claim iGPU "XMX" paths. Independent confirmation from the repo's
  own bakeoffs: `gemm_i8.json` (170.5 ms) is 1.94× *slower* than `gemm_f16.json` (87.8 ms) on the
  same shape — the opposite of what real XMX would give. **Fix these docs.**
- SLM: 128 KB per Xe-core, **64 KB max per work-group** (probed `local_mem_size=65536`).
  Max work-group size **1024** (probed). Sub-group sizes **8 / 16 / 32**.
- Memory: shared DDR5-6400, ~102 GB/s theoretical, **shared with the 20 CPU cores**.
- Level-Zero submission cost ≤50 µs vs a 9.6–20.9 ms kernel → **<1%. Launch overhead is NOT the
  batch-1 bottleneck**; under-parallelisation is.

### The root-group canary was a misread — direction un-parked

`max_num_work_groups` lowers onto `zeKernelSuggestMaxCooperativeGroupCount` and, per the SYCL
root_group spec, applies **only to kernels launched with `use_root_sync`**. It must be ≥1 on every
device and **does not constrain ordinary `nd_range` launches or sub-groups**. The repo already
launches 521 work-groups successfully elsewhere (`pq-gpu-fused-131072-final.json`) and up to 512
in the walk itself. Only **device-scope root barriers** are unavailable. Multi-work-group and
multi-sub-group designs are therefore **available**, and a batch-1 fix needs no cross-work-group
synchronisation at all. `.claude/backlog.md` items B2/B9 park this direction on a false premise.

## 7c. The bitwise-identity conflict — needs a decision

No **test** requires bitwise-identical output across a rewrite. The binding constraints are:
`cagra_query_seed_is_batch_invariant` (bit-identical distances across batch size and query order —
self-consistency, not cross-version), `cagra_force_gpu_cosine_near_zero_matches_oracle` (exact
oracle order, load-bearing on the `1e-12` clamp), transfer counters incrementing exactly once per
call, bitset semantics, and recall floors.

But the project's **promotion gate** (`.claude/backlog.md:104`) demands "exact old/new outputs".
**A real rewrite cannot satisfy that** — reassociating a float reduction changes low bits.
Resolution adopted: hold **recall-equivalence + self-consistency + CPU/GPU ID parity**, not
cross-version bitwise identity. Update the promotion gate wording rather than cripple the design.

`prim_graph_walk` has **five** callers: `ovvsCagraSearch`, `ovvsVamanaSearch`, `ovvsHnswSearch`
(via `graph_search`), and two build-path nq=1 walks (iterative init, extend). A rewrite touches
Vamana and HNSW search too.

## 8. Tooling (established, working)

- **Build env:** oneAPI `setvars.bat` is **broken on this box** (every component `vars.bat` lookup
  fails). Use `build_env.cmd` in the repo root, which calls compiler/mkl/tbb/ocloc `vars.bat`
  directly after VsDevCmd. Build: `cmd /c "C:\Personal\ioVS\build_env.cmd cmake --build build-icpx"`.
- **SYCL is JIT**, not AOT. First launch pays kernel compilation — the warmup pass matters.
- **Iteration harness:** `tools/bench/quick_cagra.py`
  - `build --scale {100k,1m} --graph <path>` — build once, serialize.
  - `search --scale {100k,1m} --graph <path>` — search-only against whatever `OVVS_LIBRARY` points at.
  - recall@10 vs exact truth, 1,000 queries, 1 warmup + 5 passes, batch 32 and batch 1.
  - Deliberately simple: no hashing/provenance machinery (explicit user instruction).
- **Prebuilt graphs:** `out/quick/sift1m-base.ovvs` (576 MB, degree 16 / intermediate 32).
- Full harness for final numbers: `python tools/bench/bench.py --profile sift1m --hnsw-threads 20`.

## 9. Status log

Append newest last. One line per real event: what landed, what number it produced, or what was ruled out.

- **2026-08-30** — Campaign opened. Fixed the broken oneAPI build environment (`build_env.cmd`).
  Wrote `tools/bench/quick_cagra.py`. Built + serialized the SIFT1M graph. Harness validated:
  reproduces published recall exactly (0.9036 @ 32/1, 0.9609 @ 64/2).
- **2026-08-30** — Root-caused the search deficit to kernel work-mapping (see §5) by reading the
  kernel, not the benchmark tables. Two design workflows launched: search-kernel inversion, and
  CRUD + memory architecture.
- **2026-08-30** — Scope widened by user: query is the floor, not the goal. Balanced
  query/insert/update/delete + bounded RAM (G2/G3/G4 added). Scale ladder R1→R2→R3 adopted;
  **start at 100K**, not 1M.
- **2026-08-30** — R1 fixture ready. `quick_cagra.py` gained `--scale {100k,1m}` with exact truth
  **recomputed** for 100K (HDF5 `neighbors` is 1M-only). SIFT100K graph builds in **5.5 s**, so an
  R1 A/B iteration is seconds, not minutes. Prebuilt: `out/quick/sift100k-base.ovvs`.
- **2026-08-30** — **R1 recall baseline (untimed metric, valid under load):**
  `32/1` → 0.9718, `64/2` → 0.9946, `128/4` → 0.9993. Matches the published SIFT100K evidence
  (0.9718) exactly, confirming the recomputed truth. **Key finding: at 100K, ovVS graph quality
  already beats hnswlib (0.9718 vs its 0.9510 at ef=32) at the CHEAPEST search point.** At R1 the
  problem is purely speed — we do not need to buy recall with traversal effort. The right G1
  comparison at R1 is ovVS `32/1` vs hnswlib at whichever `ef` reaches ~0.97.
- **2026-08-30** — Hourly realignment cron `203cc5c9` armed (session-only, 7-day expiry) pointing
  at this file.
  **R1 QPS baseline still OUTSTANDING** — deferred under rule §7.1 because design agents were
  saturating the CPU. Must be taken on a quiet machine before any kernel change lands.
- **2026-08-30 (realign tick 1)** — Map phase of the kernel workflow landed (4/4 agents). Material
  changes recorded above:
  1. **⚠ TARGET MOVED.** hnswlib's binding runs **single-threaded** at batch ≤80
     (`bindings.cpp:631`, verified in source). Bar A = 9,716 QPS (1 core); **Bar B = 27,721 QPS
     (20 cores, batch 1024) is the real "max practical throughput" target.** ~8.5× away, not 3×.
  2. **Hardware truth established** (§7b): 4 Xe-cores, 2.05 TFLOP/s FP32, **no XMX** — docs are
     wrong. Achieved throughput is **0.15% of peak**, not ~1%. Roofline says Bar B needs ~48% of
     the memory roofline; hardware is not the obstacle.
  3. **Root-group canary was a misread** — cooperative-launch-only; ordinary multi-WG launches are
     fine. B2/B9 parked on a false premise; direction un-parked.
  4. **Diagnosis corrected:** `max_iters` is NOT the driver (break fires at ~64 iters); **seeding is
     40–45% of all distance work**. Two agents disagree on the eviction/break interaction —
     unresolved, the counter settles it.
  5. Fixed a real bug in `quick_cagra.py`: `POLICY_FORCE_CPU` was 1 (`NPU_IF_FASTER`), now 6.
     Prior GPU measurements unaffected (policy 5 was correct).
  6. Added batch 256/1024 points — ovVS has never been measured where the iGPU is saturated.
- **2026-08-30** — **STEP 1 LANDED: kernel work counters.** Opt-in (`ovvsResourcesSetCagraWalkCounters`),
  off by default, one atomic per work-group, and the counting pass in `quick_cagra.py --counters` is
  a separate **untimed** pass, so timed numbers are never perturbed. Counters: queries, evals,
  seed_evals, iterations, max_iterations, table_full, admits.
  **First reading, SIFT100K (measured, not estimated):**

  | point | evals/query | seeds/query | seed share | iters/query | max_iters seen | table_full |
  |---|---:|---:|---:|---:|---:|---:|
  | `32/1` | 1,190.9 | 512 | **43.0%** | 66.6 | 79 | 0 |
  | `64/2` | 2,212.3 | 1,024 | **46.3%** | 65.9 | 71 | 0 |
  | `128/4` | 4,099.0 | 2,048 | **50.0%** | 65.5 | 69 | 0 |

  **Resolved:** the `npicks==0` break fires at ~65 iterations at every point. `max_iters`
  (192/384/768) **never binds** — dead slack, exactly as the judge argued and contrary to my
  original diagnosis and to the MAP:contract agent. Do not tune it for speed; it only oversizes
  `visited_capacity`.
  **Bigger than predicted:** random seeding is **43–50% of ALL distance work**. At `itopk=128`
  literally half the kernel's work is scoring 2,048 random vectors. This makes Step 4 (seed cut,
  ~30 lines) the highest gain-to-effort change in the whole plan.
- **2026-08-30** — **R1 CLEAN BASELINE TAKEN** (both lanes, quiet machine) — see §4b. Wrote
  `tools/bench/quick_hnsw.py` (frozen settings, reports effective thread count). Gap at R1 is ~5×
  at large batch and **~32× at batch 1**. Measured and refuted the CRUD judge's "already at the
  hardware wall" inference: the batch plateau is real but is an L2-thrashing artifact of the
  oversized visited table, at 0.35% of FP32 peak. Kernel plan stands.
- **2026-08-30** — **CRUD workflow landed (7/7).** Two findings that outrank the architecture choice:
  1. **ovVS cannot build any IVF index at SIFT1M today.** `std::vector<float> scores(n * nlist)`
     at `indexes.cpp:291`, `:502`, `graphs.cpp:735` = **16.4 GB** at nlist=4096; and
     `scores(n * n)` at `graphs.cpp:203`, `:1935` = **4 TB** at n=1e6. This is why every IVF
     artifact in the repo stops at n=2,000. Any IVF-based CRUD design is blocked behind this.
  2. **`ovvsCagraExtend` does `CagraIndex staged = *ix;`** (`graphs.cpp:1298`) — a full copy of
     dataset + graph (**+576 MB at SIFT1M**) before inserting a single vector, then geometric
     realloc on top. Peak ≈1.8 GB to insert **one** vector, and the over-allocated capacity is
     retained. Directly violates G3.
  3. **IDs are positional row offsets with no indirection layer** anywhere (no id map in any struct
     or any of the 5 serialization formats), so delete-by-compaction would invalidate every
     caller-held id. Forces tombstone + slot-reuse, not compaction-renumbering.
  4. **G3 gate derived from pinned hnswlib source:** resident at SIFT1M = **842,553,000 B
     (803.5 MiB)**, including 80 MB of per-element mutexes, ~49 MB label_lookup, 40 MB visited
     pools. ovVS CAGRA resident today is 576 MB = **0.68×** — we are *already* well inside G3 at
     rest. The 2,343 MiB is build peak. **We are not over-leveraging RAM; construction is fat.**
- **2026-08-30** — **STEP 3a LANDED: visited-table right-sizing.** `max_iters` changed from
  `max(24, itopk*6)` to `max(24, BEAM/SW + 32)`. Measured maxima are 69–79 against a cap of 96, so
  **the cap never binds and traversal is unchanged** — every work counter and recall value is
  identical to baseline. Shrinks the visited table 2–8× (128 KiB → 32 KiB per query at `itopk=64`).
  Clean measurement (quiet machine, same 8-point ordering as baseline):
  `64/2` b32 3,559→3,731 (+4.8%), b256 7,061→7,449 (+5.5%), **b1024 6,594→8,069 (+22.4%)**;
  `32/1` b1024 15,696→17,020 (+8.4%). **The batch-1024 regression is gone**, confirming the
  L2-thrashing diagnosis. `table_full=0` everywhere.
- **2026-08-30** — **STEP 4 LANDED: seed cut, 512–2048 → 32.** GPU now calls the shared
  `cagra_seed_count` instead of re-deriving it (they had drifted apart in structure); a temporary
  `OVVS_CAGRA_SEEDS` env override exists for sweeping and **must be removed** once fixed.
  Swept 512→16 with everything else held: **recall@10 is flat to four decimals** across a 64×
  reduction (`32/1`: 0.9718→0.9721; `64/2`: 0.9947→0.9946). The code's "score a larger sample"
  premise is simply wrong for this graph. New default `max(32, search_width)`.
  **Load-independent result (this is the certified one):**

  | point | evals/query before | after | reduction | recall before → after |
  |---|---:|---:|---:|---|
  | `32/1` | 1,190.9 | **736.5** | **−38.2%** | 0.9718 → 0.9721 |
  | `64/2` | 2,212.3 | **1,276.1** | **−42.3%** | 0.9946 → 0.9946 |
  | `128/4` | 4,099.0 | **2,208.6** | **−46.1%** | 0.9993 → 0.9992 |

  **All 9 CTest lanes pass**, including batch-invariance, CPU/GPU ID parity, recall floors,
  Vamana and HNSW interop.
  ⚠ **QPS for step 4 is NOT certified.** The thermal collapse in §7.1b happened during this
  measurement. Sweep-era readings (+36–46%) are plausible but must be re-taken cold, interleaved.
  **Open action: re-measure steps 3a+4 QPS on a cold, quiet machine.**
  ⚠ **Also open: re-validate the seed cut at R2 (1M).** Random entry points may matter more on a
  10× larger graph; this was swept only at R1.
- **2026-08-30 (realign tick 2)** — **STEP 4 QPS NOW CERTIFIED by ratio methodology.**
  Absolute QPS on this box drifts ~2× run-to-run, so absolutes were abandoned in favour of
  **interleaved pairs with the median of per-pair ratios** — adjacent runs share thermal state,
  so the ratio survives drift that destroys absolutes. Five pairs at `64/2` b256, legacy 1024
  seeds vs new 32: ratios 1.389 / 1.307 / 1.368 / 1.500 / 1.033 → **median 1.368 (+36.8%)**.
  Consistent with the −42.3% work reduction (a −42.3% work cut gives at most 1.73×; seeds are the
  most latency-tolerant part of the walk, so sub-proportional gain is expected).
  **Adopt this as the standard method for any timed accept/reject on this machine.**
  Environment note: live sampling shows Windows Defender (`msmpeng`) at ~4.2% sustained plus WMI
  noise; each harness process also re-reads the fixture and deserializes the graph. A Defender
  exclusion for the repo and build dir would likely stabilise absolutes — **needs the user's
  approval, do not change security settings unilaterally.**
  **Steps 3a + 4 combined ≈ 1.44× at b256** (1.055 × 1.368), at unchanged recall, tests green.
- **2026-08-30** — **STEP 3-full (persistent per-Resources GPU workspace): NULL RESULT.**
  Implemented — one persistent `malloc_device` buffer holds the visited table plus query/output/
  bitset/counter staging, removing 4 device allocations, 4 frees and 4 queue drains per search
  call. Correct (recall identical at b32/b256/b1024), all 9 CTest lanes green.
  **Measured with the interleaved-ratio method, 5 pairs each: batch 1 → 1.002, batch 32 → 1.012.**
  The plan predicted 1.5–2.5× at batch 1. **It is worth nothing.**
  **Why it matters:** this independently confirms the hardware map's claim that Level-Zero launch
  and allocation overhead is <1% of the kernel wall. **Batch-1 latency is pure kernel time**, not
  per-call overhead — one work-group leaves 3 of 4 Xe-cores idle for ~7 ms. The only levers for
  the 32× batch-1 deficit are **Step 2** (sub-group per candidate) and **Step 7** (width widening).
  Do not re-attempt allocation/launch-overhead work for batch 1.
  Change **kept** (neutral, correct, strictly less per-call churn); the temporary
  `OVVS_GPU_WALK_WORKSPACE` A/B toggle has been removed.
- **2026-08-30** — **Cold-machine absolutes after steps 1/3a/4** (positions 1–3, cooled box):
  `32/1` b256 **20,775 QPS @ 0.9721**; `64/2` b1024 **11,628 @ 0.9946**; `128/4` b32 **1,978 @ 0.9992**.
  Against hnswlib interpolated to our recall at b256 (~48,900 QPS at 0.9721), we are now
  **~2.4× behind at b256** and ~3.4× behind at b1024 — improved from ~5× at the R1 baseline.
  G1 still NOT met. Remaining gap must come from Step 2.
- **2026-08-30** — **GEOMETRY GATE (`out/sg_gate.cpp`) — REORDERS THE PLAN.** Standalone SYCL
  micro-benchmark: 32 work-groups x 2,048 candidates, D=128, per-group candidate ids so ~33 MB is
  touched (past the 4 MB L2) — i.e. DRAM-realistic, not cache-resident.

  | variant | vs OLD |
  |---|---:|
  | OLD+int8 (current geometry, rows int8) | **1.11×** |
  | NEW (sub-group/candidate, fp32) | **1.18–1.67×** |
  | **NEW+int8 (both)** | **4.53–5.89×** |

  **Neither change is worth much alone; together they are ~5×. They are SYNERGISTIC, not
  additive.** Landing them as the plan's separate "independently landable" steps 2 and 6 would
  have produced two disappointing results and risked abandoning both.
  **First cut of the gate was misleading:** with all groups sharing one 2,048-id list (~1 MB,
  L2-resident) it reported 7.45× for geometry alone. Making the gather DRAM-realistic dropped it
  to 1.5×. *Always make a micro-benchmark's working set realistic before believing it.*

  **Mechanism:** the kernel is **transaction/latency-bound, not bandwidth- or compute-bound.**
  NEW moves 33.5 MB in 1.15 ms = 29 GB/s and NEW8 8.4 MB in 0.32 ms = 26 GB/s — both far under
  the ~102 GB/s roofline. What matters is **cache-line transactions per candidate**: fp32 needs 8
  lines/row, int8 needs 2. OLD can only ever have one candidate outstanding, so shrinking its rows
  buys almost nothing (1.11×); NEW has 8 candidates in flight, and only once the rows are int8 do
  they all actually overlap (3.85×).

  **BIT-EXACTNESS PROVEN, not argued.** For integer-valued data in [0,255] every partial sum is
  <= 128·255² = 8,323,200 < 2²⁴, so fp32 holds the exact integer whatever the reduction order.
  Measured: **OLD(fp32) vs NEW8(int8) bitwise-equal 2048/2048**, and OLD vs NEW max relative
  difference exactly 0.000e+00. **This dissolves the §7c promotion-gate conflict** — on
  integer-valued datasets the combined rewrite can land under even the strictest reading.

  ⚠ **BUT THE WIN IS DATASET-CONDITIONAL — DO NOT OVERCLAIM.** SIFT vectors are natively uint8.
  Real RAG embeddings are arbitrary floats, where int8 means *lossy* scalar quantization: recall
  drops and an exact-rerank pass is needed to recover it. So:
  - On SIFT (our benchmark) the 4.5–5.9× is exact and free.
  - On real embeddings it becomes SQ8 + rerank, with a recall cost yet to be measured.
  This makes **B20 (a real 768-d float corpus in `data/`) a blocker for the product claim**, not a
  nice-to-have. Implement with a predicate: integer-valued -> exact int8 path; otherwise fp32, with
  SQ8+rerank as a separate measured decision.

  **REVISED PLAN ORDER:** merge old steps 2 and 6 into ONE change — sub-group-per-candidate
  geometry *with* an int8 row mirror, landed and measured together. Old step 5 (parallel frontier
  gather) follows, since lane-0 serial work becomes the next bottleneck once the distance engine
  is ~5× faster.
- **2026-08-30** — **STEP 2+6+5 LANDED as one kernel rewrite. Recall bit-identical, all 9 CTest
  lanes green.** Three changes, each measured as it went in:
  1. **Sub-group per candidate** (`subgroup_distance`) — one sub-group owns a whole candidate, so
     a 128-item work-group has 8 rows in flight instead of 1. `reqd_sub_group_size(16)` is applied
     only when the device advertises 16; the body is width-agnostic, so other devices keep the GPU
     path rather than falling back to CPU.
  2. **uint8 dataset mirror**, cached per Resources, built once per dataset, keyed on pointer +
     rows + dim + a sampled content fingerprint. Guarded by `dim <= 256` and an on-device
     integrality scan, so it is EXACT (bitwise) where it applies and simply absent otherwise.
     Queries are quantized per call and the whole int8 path is dropped for that call if any query
     element is out of range. Kill switch: `OVVS_GPU_INT8=0`.
  3. **Parallel frontier gather** — adjacency + bounds + bitset reads now issued by the whole
     work-group into local memory; only the visited-set insert stays serial.
  Interleaved A/B (3 alternations, `64/2` b32, baseline DLL vs new): **1.225 / 1.229 / 1.237,
  median 1.229**. Recall identical at all three points.
  ⚠ **The isolated 4.5-5.9x geometry gate did NOT translate** — it measured the distance engine in
  a vacuum, and the distance engine turned out to be only ~25-30% of the kernel. *A micro-benchmark
  bounds a component, never the system.*
- **2026-08-30** — **THE REAL BOTTLENECK WAS `consider()`'s EVICTION RESCAN.** On every admission
  that evicted, lane 0 rescanned all BEAM slots to find the new worst: O(2*itopk) dependent local
  reads, ~2,200 times per query at `128/4`, with 127 work-items parked. Replaced with a
  **sub-group argmax on sub-group 0** that returns the identical slot (strict `>` keeps the lowest
  index in a lane; ties across lanes resolve to the lowest index), in beam/16 steps plus a
  butterfly. Admissions still run in the original tile order, so the beam is unchanged.
  **Recall bit-identical (0.9721 / 0.9946 / 0.9992). All 9 CTest lanes pass.**

  | point | R1 baseline | now | gain |
  |---|---:|---:|---:|
  | `32/1` b32 | 7,590 | **16,704** | 2.20x |
  | `64/2` b32 | 3,559 | **12,464** | 3.50x |
  | `128/4` b32 | 1,351 | **6,233** | 4.61x |
  | `32/1` b256 | 14,785 | **35,418** | 2.40x |
  | `32/1` b1024 | 15,696 | **37,777** | 2.41x |
  | `64/2` b256 | 7,061 | **25,482** | 3.61x |
  | `64/2` b1024 | 6,594 | **27,507** | 4.17x |
  | `64/2` b1 | 139 | **476** | 3.42x |

  **G1 status. Against Bar A (hnswlib single-threaded, batch 32) we now WIN at every recall:**
  `32/1` 16,704 vs ~13,900 interpolated at 0.9721; `64/2` 12,464 vs ~6,050 at 0.9946;
  `128/4` 6,233 vs ~3,570 at 0.9992.
  **Against Bar B (hnswlib 20 threads, batch 1024) we are still behind:** `32/1` 37,777 vs
  ~71,500 interpolated at 0.9721 (**1.9x**); `64/2` 27,507 vs ~41,600 at 0.9946 (**1.51x**).
  Batch 1: 476 vs ~5,700 (**12x**, was 32x at baseline).
  Remaining serial terms in the walk, in suspected order: the serial `visit_once` hash probe
  chain (~1 per eval), then the SW>1 pick loop, then the final k-selection sort.
- **2026-08-30** — **Parallel `visit_once` + order-exact frontier commit.** The visited-set insert
  was the last lane-0 serial chain: up to 64 dependent global probes per tile. Replaced by
  `commit_frontier`, three parallel local passes — intra-tile duplicate removal, then a CAS-based
  probe/insert (safe because every id reaching it is distinct within the tile, so membership is
  order-independent and only the physical slot layout can differ), then a stable compaction.
  Also parallelized the seed tile the same way. Gain **+6% to +14%** — smaller than expected, so
  the probe chain was ~10% of the kernel, not the ~50% estimated.
  **Verification that the whole rewrite is behaviour-preserving: the work counters are byte-for-byte
  the pre-rewrite values** — `evals/q` 736.5 / 1276.1 / 2208.6, `seeds/q` 32.0, `table_full=0` —
  and recall is bit-identical at 0.9721 / 0.9946 / 0.9992. All 9 CTest lanes pass.
- **2026-08-30** — **Removed the 3 per-call queue drains the query mirror had added** (memset +
  quantize + readback). The verdict now lives in device memory and the walk reads it as one
  uniform broadcast; the quantize kernel is chained by event onto the existing pre-launch wait.
  Also replaced the 1024-sample content fingerprint with a **USM free-generation counter** plus a
  64-sample probe: a cached pointer key cannot go stale without a free happening first.
  **Measured effect: none (within noise).** Third independent confirmation that Level-Zero launch
  and sync overhead is <1% here. Change kept — strictly fewer syncs and a sounder cache guard.
- **2026-08-30** — **EARLY STOP (`OVVS_CAGRA_PATIENCE`, default 0 = off).** Stop expanding after
  `patience` consecutive iterations that admit nothing to the beam. This is the **one change in
  this batch that is NOT output-preserving** — it is a recall/throughput trade, so it is off by
  default and must be judged on the frontier, not in isolation. At `64/2` b256:
  p0 → 0.9946 @ 28,462 (1,276 evals); p8 → 0.9944 @ 29,183; p4 → 0.9909 @ 32,083;
  p2 → 0.9788 @ 38,130 (813 evals); p1 → 0.9589 @ 32,823.
  **It genuinely moves the frontier, it does not just slide along it:** `64/2 p2` gives
  0.9788 @ 38,595 where `32/1 p0` gives only 0.9721 @ 36,926 — better recall *and* better
  throughput. New best points at b1024: `64/4 p2` → 0.9923 @ 35,815; `96/4 p3` → 0.9980 @ 24,283.
  ⚠ Tuned only at R1; patience is a heuristic and must be re-swept at R2.
- **2026-08-30** — ⚠ **THE §4b hnswlib BASELINE WAS TOO PESSIMISTIC — DO NOT USE IT FOR G1.**
  Re-measured on the current machine state, hnswlib b1024 gives ef=32 → 79,911 @ 0.9511 and
  ef=128 → 35,113 @ 0.9981, where §4b recorded 105,699 and 39,993. Worse still, **linear
  interpolation between the frozen ef points misstates the curve badly** — measured intermediates:
  ef=48 → 0.9769 @ 80,277; ef=56 → 0.9835 @ 72,214; ef=72 → 0.9910 @ 60,454;
  ef=80 → 0.9928 @ 56,136; ef=96 → 0.9958 @ 48,734. **Never interpolate the comparator again;
  measure hnswlib at an ef that actually matches our recall, interleaved with our run.**
  `quick_hnsw.py` gained `--ef` and `--batches` for exactly this.
- **2026-08-30** — **G1 SCORED HONESTLY: interleaved, matched-recall, batch 1024, 20 threads
  (Bar B).** Three alternations each, one point per process.

  | recall | ovVS (QPS) | hnswlib (QPS) | hnswlib ahead by |
  |---|---|---|---|
  | ~0.992 | `64/4 p2` 35,165 / 35,140 / 34,839 | `ef=80` 40,932 / 39,652 / 35,300 | **1.13x** (median of pair ratios) |
  | ~0.998 | `96/4 p3` 24,101 / 24,386 / 24,173 | `ef=128` 28,625 / 36,816 / 36,247 | **1.50x** |

  **G1 still NOT met, but the gap is 1.13-1.50x, down from ~5x at the R1 baseline.**
  Worth noting: **ovVS is far more stable than the comparator** (spread <1% across the three
  alternations vs 14-29% for hnswlib, which degrades as the CPU heats). By the third alternation
  at 0.992 the two are level (34,839 vs 35,300).
  Batch 1 remains the weak case and is now structural: one query = one work-group = 8 hardware
  threads on 1 of 4 Xe-cores, i.e. ~1.5% of the machine. Allocation and launch overhead have been
  ruled out three times; the only levers left are a wider work-group at small nq, or splitting one
  query across work-groups.
- **2026-08-30 (realign tick 3)** — Corrected my own plan: the previous message listed "R2 (SIFT1M)"
  as the next action. **§3 forbids that — R1 has not passed G1.** Staying on R1.
  Remaining serial term identified: the admit loop runs one candidate at a time on sub-group 0,
  each with a `group_barrier` + broadcast (+ argmax on eviction), for **every** scored candidate —
  including the large majority that late in the walk cannot possibly be admitted.
  Next experiment: **pre-filter the tile against the beam's worst distance.** `consider()` is a
  strict no-op when the beam is full and `distance >= worst`, and `worst` only ever decreases
  within a tile, so a candidate failing the tile-entry threshold also fails every later one.
  Filtering them out is therefore **exactly output-preserving**, not a trade.
- **2026-08-30** — **ADMIT PRE-FILTER LANDED (output-preserving).** Once the beam is full,
  `consider` is a strict no-op for any candidate at or beyond the current worst, and the worst
  only falls as a tile is admitted -- so a candidate failing the threshold at tile entry fails at
  every later point too. Marking those in parallel and skipping them keeps the sub-group barrier,
  broadcast and eviction argmax off the majority of candidates, which late in a walk are all
  rejects. **Exactness re-confirmed: recall 0.9721 / 0.9946 / 0.9992 and evals/q 736.5 / 1276.1 /
  2208.6, identical to before.** All 9 CTest lanes pass. Gain **+4.6% to +6.0%** at b32.
  ⚠ **A DATA RACE WAS CAUGHT BY THE WORK COUNTERS, NOT BY RECALL.** The first cut of this change
  consumed the barrier that separated the scoring writes from the threshold reads, so
  `frontier_scores` was read across sub-groups unsynchronized. Recall moved 0.9946 -> 0.9941,
  which reads exactly like a semantics bug and would have sent me hunting the wrong thing; the
  giveaway was `evals/q` drifting 1276.1 -> 1275.9, which no *semantic* change of mine could
  produce. **Keep taking the counters on every exactness check -- recall alone cannot tell a race
  from a trade.**
- **2026-08-30** — **MEASUREMENT METHOD REPLACED: `tools/bench/ab_g1.py`.** Separate-process
  interleaving was not good enough. hnswlib alone produced **35,300 to 61,287 QPS at one identical
  setting** across sessions -- a 1.74x spread, far larger than the gap being measured -- because
  every hnswlib process rebuilt its index on 20 threads first, so its search always ran on a
  package heated differently, and the two lanes never shared a thermal window.
  The new harness builds both indexes ONCE and alternates timed rounds **inside one process**,
  reporting the median of per-round ratios. Both lanes immediately became stable (ovVS 4.4%
  spread, hnswlib 6.9%). **This supersedes the separate-process G1 numbers above.**
- **2026-08-30** — **G1 RE-SCORED with `ab_g1.py`, batch 1024, hnswlib 20 threads, 5 rounds:**

  | recall | ovVS | hnswlib | hnswlib ahead |
  |---|---:|---:|---:|
  | ovVS 0.9923 vs hnsw 0.9929 | `64/4 p2` **36,060** | `ef=80` **49,977** | **1.43x** |
  | ovVS 0.9980 vs hnsw 0.9978 | `96/4 p3` **24,593** | `ef=128` **40,079** | **1.62x** |

  ⚠ **This corrects the 1.13x / 1.50x recorded in the entry above** -- that pair was taken when
  the CPU had been degraded by a long GPU run, which flattered us. **The honest gap is 1.43-1.62x**
  (still down from ~5x at the R1 baseline). G1 NOT met.
- **2026-08-30** — **Barrier count RULED OUT.** Fused the visited dedup pass into the probe/insert
  pass, and folded the admit pre-filter into the scoring pass (the threshold comes from state that
  is stable since the previous barrier, so the sub-group that computes a score can record its own
  verdict). **7 work-group barriers per tile -> 5.** Exact across 3 repeat runs (recall
  0.9946 / 0.9992, evals/q 1276.1 / 2208.6 every time). **QPS unchanged** (14,438 / 14,489 / 14,495
  vs 14,511). Barriers are not the wall. Change kept -- same semantics, less synchronization.
- **2026-08-30** — **THE BOTTLENECK IS NOW THE ROW GATHER, and the int8 A/B proves it.**
  Re-ran the `OVVS_GPU_INT8` toggle at `64/4` b1024, interleaved: on 33,890 / 33,494, off
  24,092 / 23,668 -> **1.41x**. int8 changes nothing but cache lines per row (8 -> 2), so the
  kernel is transaction/latency bound on the dataset gather.
  **The same toggle was worth only 7% before the eviction-rescan fix.** The share of a component
  is not a property of the component -- re-measure every attribution after each landing.
  Roofline context: 5.8 GB/s of useful traffic against a ~102 GB/s ceiling, i.e. **5.7%**, at
  full thread occupancy (512 resident hardware threads). Not bandwidth bound; latency bound.
- **2026-08-30** — **NEGATIVE: 2-wide scoring loop (memory-level parallelism) is a 3% REGRESSION.**
  Split `subgroup_distance` into accumulate + reduce so a sub-group could keep two candidate rows
  in flight instead of stalling on one. Output exact (recall and counters identical).
  b1024 32,627 vs 33,890; b32 14,167 vs 14,489; `128/4` b32 7,602 vs 7,714. Eight live
  accumulators cost more in registers than the extra overlap buys. **Reverted.**
  Do not retry per-thread MLP widening on this kernel without evidence that register pressure has
  changed. The gather is still the bottleneck; the remaining ideas for it are **dataset reordering
  by graph locality** (exact, needs a permutation + output remap, and the int8 mirror is already
  ours to permute freely) and sub-byte quantization (lossy, needs rerank, blocked on B20).
- **2026-08-30** — **Session net at R1, all output-preserving unless noted, 9/9 CTest green:**
  `32/1` b32 7,590 -> 18,447 (2.4x); `64/2` b32 3,559 -> 14,631 (4.1x);
  `128/4` b32 1,351 -> 7,652 (5.7x); `64/2` b1 139 -> 526 (3.8x);
  `64/4` b1024 -> 33,542 @ 0.9946. G1 gap 1.43x (0.992) / 1.62x (0.998), from ~5x.
- **2026-08-30 (realign tick 4)** — Nothing in flight; both open experiments closed last tick
  (barriers null, MLP a 3% regression, reverted). Starting the gather attack:
  **reorder the uint8 mirror by graph locality.** Design constraint that makes it cheap: the
  mirror is built and owned by ovVS, so it can be stored in ANY row order without touching the
  serialized format, the caller's ids, or the fp32 dataset. The walk needs the permuted row for a
  neighbour id, so a permuted copy of the adjacency is stored alongside; the walk then never
  needs an indirection load in the hot path and output ids stay in the caller's numbering.
  Expected mechanism: a candidate's neighbours land in nearby rows, so the two cache lines a
  uint8 row costs are often already resident from a sibling fetch. Exact by construction --
  a permutation changes no distance.
- **2026-08-30** — **`commit_frontier`'s O(tile) passes RULED OUT, via a tile-size sweep.**
  Made `kFrontierTileSize` env-tunable (`OVVS_CAGRA_TILE`) because the duplicate scan and prefix
  compaction cost O(tile^2) while every other phase is O(tile) — so shrinking the tile is a clean
  way to weigh them. `64/4` b1024, output exact at every setting (recall 0.9946, evals/q 1333.6):
  tile 16 → 27,995; 32 → 31,012; 64 → 33,454; 128 → 33,645. **Smaller tiles are worse, not
  better**, so those passes are not the cost. Left at 64 (128 is identical because
  `width*degree` = 64 already fits one tile).
  ⚠ **Also corrects my own headline from the last tick.** Solving `4G + R = 1.41(G + R)` on the
  int8 A/B puts the row gather at only **~14%** of kernel time, not "the bottleneck". The int8 win
  is real but it removed ~30% of total time; what remains is elsewhere.
- **2026-08-30** — **BIG WIN: deleted the `search_width == 1` serial pick path.** It was a lane-0
  scan over every beam slot with 127 work-items idle — O(2*itopk) per iteration, and width 1 runs
  ~2x as many iterations to pay it on. The sub-group pick path already handled any width and
  returns the identical slot (strict `<` keeps the lowest index in a lane; ties across lanes
  resolve to the lowest index), so the special case was pure legacy. **Output exact** — recall
  0.9721 / 0.9946 and evals/q 736.5 / 1250.3 unchanged. All 9 CTest lanes pass.

  | point | before | after | gain |
  |---|---:|---:|---:|
  | `32/1` b1024 | 39,728 | **54,797** | +37.9% |
  | `64/1` b256 | 16,632 | **27,981** | +68.2% |
  | `32/1` b32 | 18,447 | **24,679** | +33.8% |
  | `64/4` b1024 | 33,542 | 33,856 | unchanged (never used that path) |

  This is what the tile sweep's "fixed per-iteration cost is ~90% of the width-1 config" pointed
  at. The sweep was worth running even though its own hypothesis was refuted.
- **2026-08-30** — **G1 re-scored with `ab_g1.py`, b1024, 20 threads, 5 interleaved rounds:**

  | recall | ovVS | hnswlib | hnswlib ahead |
  |---|---:|---:|---:|
  | ovVS 0.9721 vs hnsw 0.9733 | `32/1 p0` **52,071** | `ef=44` **70,592** | **1.37x** |
  | ovVS 0.9923 vs hnsw 0.9926 | `64/4 p2` **35,292** | `ef=80` **47,717** | **1.34x** |

  **Gap now 1.34-1.37x across the practical recall range** (was 1.43-1.62x last tick, ~5x at the
  R1 baseline). G1 NOT met. hnswlib holds a slightly higher recall in both pairs, so if anything
  these are generous to it.
- **2026-08-30** — **New counter: `OVVS_CAGRA_WALK_COUNTER_SURVIVORS` (slot 7).** Counts candidates
  that survive the beam-worst pre-filter, i.e. what the serial admit loop actually pays for.
  `ADMITS` counts every scored candidate, so the difference is what the filter removes.
  **Measured: survivors are 31.0% / 30.3% / 33.9% / 30.7% of scored candidates** at
  `32/1`, `64/2`, `64/4`, `128/4` -- 228.6 / 386.4 / 452.6 / 679.0 per query. The pre-filter
  removes about two thirds.
- **2026-08-30** — **RULED OUT ON PAPER (no build spent): exact bulk merge of a tile into the
  beam.** `consider` with capacity C is provably "keep the C smallest of the union", and ties
  resolve to the incumbent, so a stable bitonic merge would be exact. But the arithmetic kills it:
  a bitonic sort over beam+tile (up to 320 entries, padded 512) is ~45 stages x 256
  compare-exchanges = ~90 cooperative rounds per tile, against roughly 13 serial survivors per
  tile (452.6 / 35.9). **The serial loop is cheaper than the parallel sort that would replace it.**
  Do not revisit without a much larger beam.
- **2026-08-30** — **VISITED TABLE IS ON THE CRITICAL PATH, but its size is already near optimal.**
  Added `OVVS_CAGRA_VISITED_MULT` (headroom over the visit budget; default 2 = unchanged).
  Semantics are unaffected by capacity -- only the load factor moves -- so it is a clean sizing
  probe. Note there are **more visited probes than distance evaluations** (2,297 vs 1,334 per
  query at `64/4`), since rejected neighbours probe too.
  - **8x headroom: 1.72x / 2.33x SLOWER** (33,443 -> 14,350 on the clean pair). Working set blows
    past L2 and the walk feels it immediately. So the table matters.
  - **Halving it (headroom 1): median of 5 interleaved pair ratios 0.711, i.e. a regression** --
    but the per-run spread had grown to 1.58x (headroom 2) and 2.14x (headroom 1), so a 30%
    effect is below this box's noise floor by then. **PARKED under rule 3** with the number it
    produced; default left at 2.
  ⚠ **Thermal state after this cohort: p99/p50 was 12.7/1.3 ms at `32/1` b32** -- roughly a 10x
  intra-run spread, against ~1.2x when cool. **The box is spent for timing; the next timed cohort
  needs a genuine cool-down first.** Counters and recall stayed exact throughout, which is exactly
  why rule 1b says to accept/reject on counters.
- **2026-08-30** — Also considered and rejected without building: a **bit-per-node visited set**
  (12.5 KB/query at 100K vs the hash's 32 KB, and O(1) with no probe chain -- better on both axes
  at R1). Rejected because the hash's size is **independent of N** (it is sized by the visit
  budget, ~2,656) while a bitset grows with N: 125 KB/query at 1M, i.e. it *reverses* at R2.
  Optimising the current rung with something that inverts at the next one is exactly the trap
  the scale ladder exists to prevent.
- **2026-08-30** — **GRAPH DEGREE 32 DOMINATES DEGREE 16 (counter-based, so thermally immune).**
  The campaign has run on degree 16 throughout, which is low for CAGRA. Built SIFT100K at degree
  32 (intermediate 64) and 48 (intermediate 96) and compared on **recall vs evals/query**, both
  load-independent, which is what let this run at all on a thermally spent box.

  | graph / point | recall | evals/q | iters/q | survivors |
  |---|---:|---:|---:|---:|
  | d16 `64/2` | 0.9946 | 1,276 | 67.7 | 30.3% |
  | **d32 `32/1`** | **0.9959** | **1,255** | 67.3 | 20.9% |
  | d16 `64/4` | 0.9946 | 1,334 | 35.9 | 33.9% |
  | **d32 `32/2`** | **0.9962** | **1,304** | 35.4 | 24.4% |
  | d16 `128/4` | 0.9992 | 2,209 | 67.4 | 30.7% |
  | **d32 `64/2`** | **0.9995** | **2,120** | 67.0 | 21.0% |
  | d32 `32/4` | 0.9962 | 1,410 | **19.6** | 26.3% |

  **d32 wins on every axis at matched recall** — higher recall, fewer evals, same or fewer
  iterations — and additionally **halves the beam** (`BEAM = 2*itopk`, and d32 reaches the same
  recall at half the itopk), which makes the pick loop and the eviction argmax cheaper too.
  The pre-filter also rejects more at higher degree (survivors 21-26% vs 30-34%).
  **d48 is worse than d32**: 0.9975 needs 1,723 evals against d32's ~1,639 interpolated. Stop at 32.

  ⚠ **Two headwinds before this converts to QPS, and it has NOT been timed** (box is thermally
  spent, p99/p50 ~10x):
  1. **The visited table doubles.** `visit_budget = nseeds + max_iters*SW*DEG` scales with degree,
     so d32 `32/2` sizes to 64 KB/query against d16 `64/4`'s 32 KB — and the table is on the
     critical path (8x headroom measured 2.3x slower). **Likely fix already identified:** the
     `+32` slack in `max_iters = BEAM/SW + 32` is my own Step 3a constant, and measured maxima at
     d32 are 38-40 against a cap of 64. Tightening to `+16` gives a cap of 48, still clear of the
     observed max, and puts the d32 table back at 32 KB. `MAX_ITERATIONS` is the canary.
  2. **RAM.** d32 graph at 1M = 128 MB vs 64 MB. Resident becomes 512 + 128 + 128 (int8 mirror)
     = 768 MB against hnswlib's 803 MB. **Still inside G3, but the headroom is nearly gone** —
     record this before adopting d32 as a default.
  Build cost at 100K: 11.4 s (d32) / 19.5 s (d48) vs 5.5 s (d16).
- **2026-08-30 — CAMPAIGN STATE / FORK.** Kernel track is at diminishing returns: G1 gap
  1.34-1.37x with **no dominant term left** (gather ~14%, visited probes significant but not
  resizable in either direction, admit loop ~1/3 of candidates). Contrast with earlier in the
  session when the eviction rescan alone was ~60%. Two live directions:
  - **G1 / kernel:** land degree 32 + the `max_iters` slack tightening, then re-time cold.
    Counter evidence says the frontier improves; QPS unproven.
  - **G2 + G4 / mutation:** still **0% done and needs no GPU**. No delete or update exists in the
    C ABI at all, and `ovvsCagraExtend` costs ~1.8 GB peak to insert one vector (`CagraIndex
    staged = *ix;`, graphs.cpp:1298) which is a live G3 violation on the insert path.
  ⚠ **Nothing from this entire session is committed.** Working tree carries the whole kernel
  rewrite, the counters, `ab_g1.py`, and the harness changes.
- **2026-08-30** — **G2/G4 PHASE 1 LANDED. The mutation surface exists for the first time.**
  Design in `.claude/plans/2026-08-30-ovvs-cagra-mutation.md`. New ABI: `ovvsCagraDelete`,
  `ovvsCagraUpdate`, `ovvsCagraCounts`, plus Python bindings and `tools/bench/churn.py`.
  All 9 CTest lanes pass.
  - **Delete is a tombstone**, forced by the audit finding that ids ARE row offsets with no
    indirection: compaction would renumber and invalidate every caller-held id.
  - **A deleted row stays traversable but is not returnable.** This is the decision the gate turns
    on. Excluding deleted nodes from traversal fragments the graph -- every node is a routing hop
    for its neighbours -- so recall would collapse in proportion to the deletion rate, which is
    exactly hnswlib's `markDelete` weakness. Instead they are scored and expanded as normal and
    dropped only at k-selection, with the walk over-fetching (capped at `itopk_size`, so the
    caller's search effort is not silently raised) so a thinned result still yields k live rows.
  - **Update keeps the id**: overwrite the row, re-search its out-edges, re-prune its
    in-neighbours. Delete+insert would hand back a different id, which is not an update.
  - **Serialization fails closed**: the tombstone section bumps `ver` to 3 *only when tombstones
    exist*, so an un-deleted index still writes v2 and stays readable by existing builds, while a
    deleted one is refused by an old build rather than silently resurrecting deleted rows.

  **G4 measured -- recall against the LIVE set, recomputed every round** (comparing against the
  original truth would be meaningless once rows are deleted or overwritten). 5 rounds of
  20,000 deletes (20% of the corpus), 10,000 updates, 10,000 inserts:
  recall **0.9733 -> 0.968 -> 0.964 -> 0.9637 -> 0.9663 -> 0.968**. It dips ~1% and then
  **recovers and stabilises rather than decaying**, and `returned_frac` stays 1.0 throughout.
  **G4 passes at 20% deletion.**

  **G2 measured (per operation):** delete **0.01-0.03 us** (a bit set), update **150-212 us**,
  insert **151-216 us**. No operation is a stall.
- **2026-08-30** — **`ovvsCagraExtend`'s 1.8 GB-per-insert defect is FIXED, and measured against
  the old binary.** The whole-index copy (`CagraIndex staged = *ix;`) existed for all-or-nothing
  semantics; those are kept, via an in-place insert plus a rollback journal of only the <= degree
  neighbour rows `robust_prune` overwrites -- about 1 KB at degree 16.
  Then a second bug behind it: growth. Reserving the *exact* final size makes every single-vector
  Extend reallocate and copy the whole dataset, O(n) per insert, and one vector at a time is the
  on-device-memory workload. Growing by at least half again keeps incremental insert amortised
  O(1) while a batch still pays one reallocation.

  | metric, SIFT100K | old | new | |
  |---|---:|---:|---:|
  | peak working-set delta, 1-vector extend | +99.6 MB | **+13.2 MB** | 7.5x |
  | single-vector insert, median of 200 | 10,622 us | **187 us** | **56.7x** |
  | same, p90 | 12,482 us | **230 us** | 54x |

  The 99.6 - 41.9 = 57.7 MB removed by dropping the copy matches the dataset + graph size
  (51.2 + 6.4 MB) almost exactly, which is the check that the fix did what it claimed.
- **2026-08-30** — **G2 wrinkle worth recording: mutation is refused under `FORCE_GPU`.**
  Delete/update/extend all have no GPU path and return `DEVICE_UNAVAILABLE` under
  `OVVS_POLICY_FORCE_GPU`, so a caller that pins the GPU for search cannot mutate at all without
  flipping policy. `churn.py` flips it around every mutating call. Defensible, but it is a real
  usability edge for a "living store" and should either be documented or made to fall back.
- **2026-08-30** — **NOT done in phase 1, stated so it is not mistaken for done:** no slot reuse,
  so deleted rows do not return their storage and sustained churn grows the footprint (a G3 risk).
  The fix is `(slot, generation)` packed into the `int64_t` the ABI already returns, so a stale id
  is *detectably* stale; a freshly built index has all generations 0 and behaves identically.
  Also no graph repair on delete: tombstoned nodes keep their edges, and the quality curve is
  measured rather than prevented.
- **2026-08-30 (realign tick 5)** — **Degree 32 confirmed faster AND more accurate than degree 16,
  but the G1 re-score is PARKED as untrustworthy.**
  Interleaved d16 `64/4` vs d32 `32/2` at b1024, 5 pairs: ratios 1.375 / 0.980 / 1.918 / 0.760 /
  1.286, **median 1.286**; the least-throttled reading of each (33,652 vs 42,205) agrees at 1.25x.
  d32 delivers that at **higher** recall (0.9962 vs 0.9946), so it dominates d16 outright rather
  than trading along the frontier. Direction is solid; the magnitude is soft.
  ⚠ **The follow-up G1 run against hnswlib is NOT a usable number and is not recorded as one.**
  ovVS spanned **14,951-35,976 QPS (2.4x) across five rounds of one `ab_g1.py` invocation**, and
  the per-round ratio ranged 1.65-2.99. Standalone d32 runs minutes earlier gave ~42,000, so the
  whole box was degraded, not the configuration. Reporting the 2.479 median would have been
  worse than reporting nothing. **PARKED under rule 3 pending a genuine cool-down.**
  Note the GPU had been idle through the entire CRUD stretch and still had not recovered — the
  CPU-heavy mutation work heats the same package. A real cool-down means idle, not "different work".
  **Carry forward:** the last G1 number measured under conditions that passed the noise check
  remains **1.34-1.37x behind at d16**. d32 should improve it; that is a prediction, not a result.
  Also unmeasured and required before d32 becomes a default: **G3 at 1M** — the d32 graph is
  128 MB against d16's 64 MB, putting resident at ~768 MB versus hnswlib's 803 MB. Inside the
  gate, but with almost no headroom left.
- **2026-08-30** — **MUTATION PHASE 2 LANDED: slot reuse with generations. The G3-under-churn hole
  is closed.** New ABI `ovvsCagraExtendEx(..., int64_t* out_ids)`. All 9 CTest lanes pass and the
  search path is unchanged where nothing has been reused (recall 0.9946 / 0.9992, evals/q
  1276.1 / 2208.6 — identical).
  - A public id is now `slot | (generation << 32)`. **Generation 0 packs to the bare slot**, so an
    index that has never reused a row returns byte-identical ids and both new vectors stay empty:
    the feature costs nothing until it is used.
  - Reuse bumps the row's generation, so every id previously handed out for that row becomes
    **detectably stale** and is rejected, instead of silently resolving to whoever now occupies it.
    That was the whole reason phase 1 refused to reuse slots.
  - **Reuse is offered only through `ExtendEx`.** Plain `ovvsCagraExtend` still appends, because
    without `out_ids` the caller cannot learn that a row was recycled. The API shape enforces this
    rather than documenting it.
  - Generations are serialized alongside the tombstones under the v3 section, and the free list is
    rebuilt from the tombstones on load rather than stored.

  **Verified behaviour** (`tools/bench/churn.py --reuse`, plus a direct test):
  plain `extend` leaves `deleted` untouched; `extend_ex` recycles rows and drops `deleted` 5 -> 2;
  a stale bare id is **rejected**; the freshly packed id is accepted; search-returned ids round-trip
  through delete.

  **G3 under churn, 4,000 deletes + 4,000 inserts per round, footprint in slots:**

  | round | append-only | with reuse |
  |---|---:|---:|
  | 1 | 104,000 | **100,000** |
  | 2 | 108,000 | **100,000** |
  | 3 | 112,000 | **100,000** |
  | 4 | 116,000 | **100,000** |

  Append-only grows linearly with churn forever; reuse is **flat**, with `deleted` returning to 0
  every round and recall unchanged (0.9630 vs 0.9607 at round 4).
  **All four gates now have a real measurement at R1: G1 fails at 1.34-1.37x, G2 passes, G3 passes
  including under churn, G4 passes at 20% deletion.**
- **2026-08-30** — **CORRECTION: step 2 (sub-group per candidate) was under-credited, and I never
  re-measured it.** Prompted by a direct question about whether step 2 was ever really done.
  It was: it is `subgroup_distance` in the kernel, shipped in `d30e995`. But it was only ever
  measured **bundled** with the int8 mirror and the parallel gather (1.229x), and I recorded that
  as "the isolated 4.5-5.9x geometry gate did NOT translate" — then never revisited it after
  removing the eviction rescan that was masking everything.
  Re-added the pre-rewrite whole-work-group geometry behind `OVVS_GPU_SUBGROUP_GEOM=0`, with the
  int8 fast path mirrored into it so the toggle changes **geometry and nothing else**. Both
  geometries produce identical recall and identical work counters (0.9946, evals/q 1276.1),
  confirming a clean single-variable test.

  | point | old geometry | new geometry | median of 5 pair ratios |
  |---|---|---|---:|
  | `64/2` b32 | 9855 10048 10078 8245 10076 | 14888 14944 11559 14890 14869 | **1.487** |
  | `64/4` b1024 | 17678 12977 13001 12853 17695 | 26768 17713 18755 26818 26806 | **1.514** |

  **Step 2 is worth ~1.5x on its own.** Neither-vs-both (`geom=0,int8=0` against `geom=1,int8=1`)
  at b1024 gives a median 1.79x, though that pairing was noisy (0.92-2.69) and the two isolated
  numbers are the trustworthy ones.
  **The lesson is one this file already states and I failed to apply to my own work:** "the share
  of a component is not a property of the component -- re-measure every attribution after each
  landing." The 1.229x verdict was true on the day and wrong a few hours later, because the rescan
  was consuming 60% of the kernel when step 2 landed. The gate's 4.5-5.9x was still an
  overestimate — it isolated the distance engine — but "did not translate" under-sold a 1.5x.
  `OVVS_GPU_SUBGROUP_GEOM` is temporary and must be removed with the other knobs.
- **2026-08-30** — **SECOND-OPINION CONSULT on the performance wall (full-plan + kernel read).**
  Key reframe, derived from our own numbers: b1 latency 1.92 ms vs b1024 per-slot latency 2.33 ms
  (1024/27,507 QPS ÷ 16 queries/slot) — co-residency of 64 WGs costs only +21%, so **QPS = 64 /
  per-query-latency and the device is concurrency-capped, not compute/bandwidth/barrier-bound.**
  Little's law on 5.8 GB/s @ ~600 ns ⇒ ~50 lines in flight ≈ 0.1 outstanding requests/thread vs
  hnswlib's ~200 machine-wide: we are losing the concurrency war ~4:1 with 512 thread slots.
  ≥99.5% of thread-cycles are stall/idle. Residual mystery: bottom-up iteration model predicts
  10-18 µs vs ~51 µs measured at nominal 2 GHz — suspects: GT clock ≪ 2 GHz during runs (prime,
  given thermal history), fabric latency ≫ 600 ns, or spills (never checked).
  **Plan adopted:** (1) d32 + max_iters +16 (already queued); (2) register-hoist the query row
  (re-read from global per eval today, IGC can't hoist across visited-table aliasing) + vec/uint64
  row loads instead of 8 scalar byte loads; (3) **delete the global visited hash** — intra-tile
  dedup + beam-membership scan + the ≥worst pre-filter are PROVABLY output-identical (evicted ⇒
  ≥ non-increasing beam-max forever; pre-full ⇒ everything admits ⇒ in-beam), counters gate:
  recall/iters/survivors unchanged, evals ↑ ≤ probes−evals (963/q); kills 2,297 CAS/q + the
  32 KB/q state; (4) **queries_per_wg ∈ {1,2,4,8}** remap — same 128-item WG, 8 independent
  queries, zero WG barriers, exactly output-preserving (seeds are per-query), lifts queries in
  flight 64 → up to 512; the sweep is itself the phase-wait-vs-memory-wait discriminator. d32
  halves the beam and hence per-query SLM, which is what keeps 16 WGs/Xe-core resident under the
  remap — adopt d32 first. Diagnostics queued for a quiet window: VTune gpu-hotspots (XVE
  Stalled-vs-Idle split + real GT frequency), zeKernelGetProperties spillMemSize log.
  **IGC shader-dump env route PARKED** (rule 3): release driver ignores the env flags; registry
  route needs admin consent. Verdict: G1 winnable with margin (conservative d32 1.25 × remap 1.8
  ≈ 80K vs hnswlib 48-50K at 0.992); if the qpw sweep is flat, that falsifies the diagnosis in
  one day and the "OoO cores win graph walks" conclusion gets taken seriously.
- **2026-08-30** — **USER DECISIONS: admin diagnostics granted; d32 adoption + kernel changes
  approved.** VTune 2025.3 confirmed installed (`oneAPI\vtune\latest`, not on PATH).
- **2026-08-30** — **QUERY-ROW REGISTER HOIST + PACKED ROW LOADS LANDED: +5.0% at `64/2` b256.**
  The query row was re-read from global memory for every candidate (~1,300 redundant row fetches
  per query — IGC cannot hoist it across the visited-table writes). Now each lane loads its slice
  once into registers; the int8 data row becomes two dword loads per lane instead of eight byte
  loads; L2 uses the integer identity sum(a-b)² = Σa² + Σb² − 2Σab (every term < 2²⁴ so int32 is
  exact and per-lane values are bit-identical to the subtractive form). Specialised to
  D == 8*subgroup_size so the register buffers are statically indexed and cannot demote to
  scratch; any other shape keeps the original path. fp32 path gets the same hoist at the same
  shape, iteration order unchanged (bit-exact).
  **Exactness: recall 0.9721/0.9946/0.9992 and evals/q 736.5/1276.1/2208.6 identical; 9/9 CTest.**
  Interleaved pairs vs saved pre-hoist DLL, `64/2` b256: 1.052/1.027/1.066/1.050/1.043 →
  **median 1.050**. (Baseline DLL must live in build-icpx/bin — its runtime deps resolve from the
  DLL's own directory; `out/ab/` copies fail to load.)
- **2026-08-30** — **`max_iters` SLACK TIGHTENED +32 → +24 (d32-adoption prerequisite): combined
  with the hoist, +8.8% at `64/4` b1024.** Measured maxima with 32 seeds: d16 76/75/71 at
  BEAM/SW=64 (caps now 88, margin ≥12) and 41 at BEAM/SW=32 (cap 56, margin 15); d32 74/70 at 64
  and 40/23 at 32/16 (margins ≥14). **The excess over BEAM/SW does not grow with the ratio**, so
  +24 clears everything observed while moving d32 `32/2` AND d16 `64/4` back across a pow2
  boundary: visited tables halve 64→32 KiB/query. Recall and every counter bit-identical on both
  graphs (d32 values match the adoption table digit-for-digit); 9/9 CTest.
  Interleaved pairs at `64/4` b1024 (hoist + halved table vs pre-hoist): 1.081/1.080/1.088/
  1.099/1.090 → **median 1.088**; net of the hoist's 1.050 the table halving is worth ~+3.5-4%
  at b1024, consistent with the L2-thrash mechanism. **d32 G1 re-score queued behind a 12-min
  cool-down** (box warmed by ~25 min of builds + A/B pairs): `ab_g1.py --graph sift100k-d32
  --itopk 32 --width 2 --ef 96` b1024, 20 threads, 5 rounds, p0 (patience stays off for the
  first d32 score — keep it output-preserving-comparable).
- **2026-08-30** — **d32 G1 SCORED 3×: gap ≈ 1.10-1.17× at ~0.996 recall (was 1.34-1.37×).
  NOT a G1 pass — do not claim one.** Three back-to-back `ab_g1.py` runs (d32 `32/2` p0 vs
  `ef=96`, b1024, 20t, after a 12-min cool-down; third saved to `out/quick/g1-d32-32x2-ef96.json`):
  run1 median 0.923 (hnswlib BEHIND) but its rounds degraded 40.3K→33.0K; run2: ovVS stable
  44.6-46.2K (spread 3.5%) vs hnswlib 50.2-53.1K → ~1.17; run3 median 0.943 but BOTH lanes
  unstable (ovVS 33.8-45.1K; hnswlib rising 29.3K→45.8K).
  ⚠ **New methodology trap found: a sub-1.0 median from `ab_g1.py` is NOT a win when one lane's
  rounds are still cold** — round-alignment artifacts produce ratios spanning 0.76-1.37 inside
  one run. Judge a run only if each lane's own rounds are stable (≤~10% spread); otherwise
  compare healthy plateaus across runs. Healthy plateaus here: **ovVS 45.4K @ 0.9962 vs hnswlib
  50-53K @ 0.9955** (ovVS at slightly higher recall). ovVS is now inside the comparator's own
  run-to-run noise band — the remaining gap is smaller than hnswlib's variance, which is itself
  the argument for landing beam-dedup + the qpw remap to make G1 unambiguous rather than
  re-rolling the dice on scoring runs.
  Cumulative today at `32/2`-class configs: d32 adoption (+~1.25×) + hoist (+5%) + table halving
  (+~4%) took the b1024 flagship from ~36K to ~45.4K at higher recall.
- **2026-08-30** — **BEAM-DEDUP LANDED AND GATED (env `OVVS_CAGRA_BEAM_DEDUP`, default = only
  with qpw>1).** The visited hash is replaced by intra-tile dedup + an SLM beam-membership scan
  + the existing ≥worst pre-filter, **provably output-identical when `nseeds <= itopk`** (a seed
  phase that cannot overfill the beam ⇒ every previously-scored id is in-beam or forever above
  the threshold). **Measured exactly as proved:** recall, iters/q and survivors/q bit-identical
  at every point on both graphs; evals/q rises by the revisit rate — +24-41% at d16,
  **+43-46% at d32** (graph-overlap property, width-independent: 32/1 +45.6%, 32/2 +43%).
  ⚠ **At one-WG-per-query it is a 14% REGRESSION** (3 toggle pairs, d32 32/2 b1024: 0.676/
  0.862/0.908) — extra evals cost more than 2,297 CAS probes save at 64 resident queries.
  Kept as the enabler of the sub-group mapping only; hash stays the default.
- **2026-08-30** — **SUB-GROUP-PER-QUERY (qpw=8) BUILT, BIT-EXACT, AND PARKED AT 0.84×.
  The concurrency diagnosis was half right and the counters say precisely which half.**
  Implementation: `OVVS_CAGRA_QPW=8` — one sub-group owns one query end to end, zero WG
  barriers (tail-guard `return` is legal because none exist), per-query state in registers +
  a <1 KiB SLM slot; v2 fuses score+admit against the frozen tile threshold (provably counter-
  identical) and compacts the tile in place. **Gates: recall and every counter bit-identical
  to the WG-dedup path at every config tried; 9/9 CTest in both modes; classic path untouched
  (hash counters re-certified).**
  **Timed verdicts (interleaved pairs, d32 32/2 b1024, same binary):**
  | config | ratio vs qpw1 | note |
  |---|---:|---|
  | v1 (72 q/Xe-core) | **0.823** | qpw1 42K, qpw8 34.4K stable ±2.4% |
  | v1 + seen-cache 256 (40 q/Xe) | **0.459** | throughput tracked residency, not evals |
  | v1 + seen-cache 512 (32 q/Xe) | **0.495** | ditto — SLM/query is the binding resource |
  | **v2 fused, ≤1 KiB/query (128 q/Xe)** | **0.843** | qpw8 38.0K ±1% vs qpw1 45.1K |
  **What was confirmed:** the mapping raises machine eval throughput **58.8M → 70.8M evals/s
  (+20%)** — queries-in-flight do convert to work, and throughput tracked SLM residency in
  both directions. **What was refuted:** (1) an exact SLM dedup with hash-grade coverage does
  not exist under the residency constraint — recency ring caught 16-47% (revisits are NOT
  temporally local), direct-mapped cache 41-74% but every KiB/query costs more residency than
  it recovers; (2) **evals/s saturates ≈70M beyond ~300 resident queries** — NOT bandwidth
  (9.1 GB/s of ~102), NOT issue (~2.5%), NOT thread slots. Something serves random 128-B
  reads at a fixed rate: DRAM row-activate service or L2 miss-handling concurrency are the
  suspects, and **counters/A-B cannot distinguish them — this is exactly the VTune
  memory-access question** for the admin-granted quiet window. Net ceiling if the tax were
  zero at the saturated rate: 70.8M/1304 ≈ 54K ≈ 1.20× qpw1 — real but modest until the
  70M wall itself is understood.
  **Verdict: qpw stays default 1; qpw=8 + beam-dedup + seen-cache kept in-tree behind their
  TEMPORARY knobs as the vehicle for the post-VTune revisit. Do not re-tune blind — measure
  the wall first.** qpw1 flagship after today's landings: **~45-47K @ 0.9962** (was ~36K at
  tick 5), vs healthy hnswlib ~50-53K @ 0.9955 → gap ~1.10-1.17×.
- **2026-08-30 (realign tick 6)** — **VTUNE SESSION RUN (no admin needed after all — the env
  route works when launched via a cmd.exe wrapper; Store-python can't be VTune's target
  directly). Three collections + two follow-up experiments; the qpw story is now CLOSED with
  a named cause and a shared wall.**
  1. **qpw1 overview:** XVE Stalled/Idle 74.7%, occupancy **84.4%**, "L3 bandwidth bound"
     3.5% of peak. Confirms the consult's stall accounting; byte-bandwidth is nowhere near
     the limit.
  2. **qpw8 v2 overview:** Stalled/Idle 78.2%, occupancy **29.0%** — the remap never held its
     queries resident. Hypothesis: dispatch raggedness (128 coarse WGs, each pinned open by
     its slowest of 8 queries; ~2 queries per claimant at b1024).
  3. **v3 BUILT: persistent sub-groups + global atomic query queue** (`off_queue` in the
     workspace; each sub-group claims the next query, serves it, claims again). Bit-identical
     counters and recall; 9/9 CTest both modes. **Result: 0.840 median vs qpw1 — unchanged.**
     VTune on v3: occupancy 31-33% still, but **Peak XVE Threads Occupancy = 100%** — no
     resource cap; pure fill/drain.
  4. **b4096 probe (8 queries/claimant, truth recomputed for 4096):** ratios 0.870/0.873/
     0.874 — larger batches barely help. **Fill/drain falsified as the dominant term too.**
  5. **The remaining explanation, consistent with every number:** both mappings converge on a
     **shared request-rate wall ≈ 120-150M random 64-B line-requests/s (~9 GB/s, i.e. <10% of
     byte bandwidth)** — qpw1 60.5M evals/s, qpw8 75.6M evals/s (+25% — the remap's true
     concurrency yield), but qpw8's +43% dedup-eval tax eats it: net 0.84-0.87 at every
     batch size and every dedup variant tried. Not occupancy, not barriers, not L3 bytes,
     not dispatch. **This wall also owns qpw1's ceiling and therefore the remaining G1 gap.**
  **PARKED (rule 3): the qpw=8 mapping**, three variants measured, verdict stable. Kept
  in-tree behind knobs; revisit only if the wall itself moves.
  **NEXT: characterize the wall directly with a standalone microbenchmark** (sg_gate-style,
  ~30 lines: achievable random-line request rate vs outstanding-requests-per-thread and
  vs footprint) — it yields the true roofline for ANY mapping, decides whether qpw1's 60M
  evals/s is already at it (→ G1 must come from FEWER requests: graph-locality reorder of
  the uint8 mirror + permuted adjacency, the tick-4 direction) or short of it (→ keep tuning
  qpw1). CLI VTune cannot decompose LSC vs GTI vs DRAM (GUI-only grid); the microbenchmark
  can, without admin.
- **2026-08-30 (realign tick 7)** — **`out/req_gate.cpp` BUILT AND RUN. ⚠ IT REFUTES THE
  "SHARED REQUEST-RATE WALL" RECORDED LAST TICK — that conclusion is RETRACTED.** The gate
  measures random row reads in the walk's exact eval shape (one sub-group per row, lanes 8 B
  apart, dependent chase = 1 outstanding/chain), quiet box, best of 3:

  | sub-groups | fp | row | burst | rows/s | GB/s | chain ns |
  |---:|---|---|---:|---:|---:|---:|
  | 32 | 16 MiB | 128 B | 1 | 74.2M | 9.5 | 431 |
  | 64 | 16 MiB | 128 B | 1 | 145.3M | 18.6 | 440 |
  | 128 | 16 MiB | 128 B | 1 | 275.2M | 35.2 | 465 |
  | 256 | 16 MiB | 128 B | 1 | 465.6M | 59.6 | 550 |
  | **512** | **16 MiB** | **128 B** | **1** | **586.2M** | **75.0** | 873 |
  | 128 | 16 MiB | 128 B | 2 | 401.5M | 51.4 | — |
  | 512 | 16 MiB | 128 B | 2 | 476.0M | 60.9 | — (queue-saturated; B>1 hurts at 512) |
  | 512 | **2 MiB** | 128 B | 1 | **2,256M** | **288.8** | 227 |
  | 512 | 128 MiB | 128 B | 1 | 317.4M | 40.6 | 1,613 |
  | 512 | 8 MiB | 64 B | 1 | 1,044.6M | 66.9 | 490 |

  **Readings:** (1) at the walk's own concurrency and shape the fabric serves 145-586M
  rows/s — the walk's 60-75M evals/s is **4-8× BELOW the memory roofline; the walk is
  bookkeeping-bound, not memory-bound.** Estimated split at qpw8's effective ~130-170
  resident sub-groups: fetches ≈ 27%, bookkeeping (commit dup-scan + beam scan + pick +
  admit, all SLM-serial per candidate) ≈ **~73% of sub-group time**. This also reframes the
  beam-dedup −14%: the cost was the O(count) membership SCANS, not the extra evals.
  (2) **L2-resident footprint runs 3.8-15×** — the graph-locality reorder (tick-4 direction)
  has a much higher ceiling than assumed; it helps both mappings and R2.
  (3) 128 MiB (R2-like) still serves 317M rows/s — memory is not the R2 blocker either.
  (4) 64-B rows double the rate — sub-byte/half-row packing has real headroom when B20 lands.
  **REVISED ATTACK (in order):** (a) qpw8 bookkeeping diet — replace the per-candidate
  O(count) beam scan with an exact SLM open-addressed hash of beam contents (insert on
  admit, tombstone on evict, amortized rebuild), and the O(r) dup scan with a sub-group
  ballot/shuffle pass; the mapping's +25% concurrency yield is real and currently buried
  under self-inflicted SLM serialization. (b) locality reorder of the uint8 mirror +
  permuted adjacency. (c) VTune GUI session for the LSC grid stays optional — the gate
  answered the question CLI VTune could not.
- **2026-08-30** — **Committed and pushed everything to origin/main** (`0adefa3` kernel,
  `4c87eaf` gates incl. req_gate + sg_gate force-added under out/, `53e7386` docs).
- **2026-08-30** — **BEAM-HASH DIET: NEUTRAL (median 1.017 of 3 noisy pairs, d32 32/2
  b1024).** Exact SLM open-addressed hash of beam membership (insert on admit, tombstone
  on evict, amortized rebuild) replacing the O(beam) commit scan — bit-identical counters,
  9/9 CTest. Verdict: the beam scan was NOT the dominant bookkeeping term (pipelined SLM
  reads are cheap); kept for its asymptotics at larger beams. Attribution lesson: "~73%
  bookkeeping" was right in total but wrong about the largest single term.
- **2026-08-30** — **QPW8 + GLOBAL VISITED HASH: THE CROSSOVER. 0.955 at b1024, 1.014 at
  b4096 — the first configuration where the sub-group mapping beats the classic one.**
  req_gate's data un-parked the idea: the fabric has ~8× request headroom and achieved
  residency is ~150-170 queries (~5 MiB of tables, not the feared 16 MiB), so the sgq gate
  no longer requires beam_dedup; hash mode (`OVVS_CAGRA_QPW=8 OVVS_CAGRA_BEAM_DEDUP=0`)
  wires per-query claimed visited regions (cleared at claim time) into sg_commit.
  **evals/q back to 1303.7 — bit-identical to the classic path; 9/9 CTest in all three
  modes.** Interleaved pairs vs qpw1, d32 32/2: b1024 0.957/0.947/0.955/0.961/0.951 →
  **median 0.955** (was 0.840 with dedup — the +43% eval tax was the dominant qpw8 loss,
  not the scans); b4096 1.009/1.014/1.019 → **median 1.014, rising with batch** while qpw1
  is flat. **Defaults unchanged (qpw1 remains the Bar-B champion); qpw8's preferred
  companion is now the global hash, with dedup mode retained behind its knob as the
  SLM-resident fallback.** The sgq mapping is the batch-scaling asset for R2/R3, where
  practical batch sizes grow with the corpus.
- **2026-08-31** — **GRAPH-LOCALITY REORDER (tick-4 direction): REFUTED BY PROBE, PARKED
  BEFORE IMPLEMENTATION.** req_gate gained `--window`/`--cluster` modes to bound the
  reorder's ceiling at UNCHANGED footprint. Two traps and a verdict, all measured:
  1. ⚠ A bounded ±W chase first showed "3.7×" (446M → 1,640M rows/s) — **artifact**: a
     bounded 1-D walk is recurrent, so the gain was L2 revisit hits, which the dedup'd
     walk can never have. Same lesson as sg_gate: make the probe's working set honest.
  2. The honest probe — uniform-jumping dependent base + G−1 clustered reads per step
     (burst locality, zero revisits): **clustering HURTS. cluster=8: uniform 513M vs
     ±1024-row window 352M (−31%); cluster=32: 676M vs 313M (−54%).** Uniformly random
     concurrent requests spread across DRAM channels/banks; clustered bursts serialize on
     few of them. On this shared-DDR5 SoC, co-locating co-accessed rows is anti-optimal
     at fixed footprint.
  3. Bonus: cluster=32 uniform reached **676M rows/s** — more independent reads per
     dependent hop beats the plain chase's 586M; the walk's batched gather shape is right.
  **Do not revisit row reordering without a mechanism that SHRINKS the working set**
  (that is what the 2-MiB 2.26B rows/s ceiling rewards — quantization/B20, not permutation).
  **NEXT: d32 patience sweep** — never swept at d32 (tuned only at d16, flagged in §9 at
  the time); pure env knob. p∈{1..4} at d32 `32/2`, recall/evals by counters first, then
  matched-recall ab_g1 pairs vs hnswlib at the measured-ef curve. The fastest remaining
  path to a G1 verdict with zero new code.
- **2026-08-31** — **d32 PATIENCE FRONTIER SWEPT + G1 RE-SCORED AT MATCHED RECALL: gap
  ≈ 1.11× at both practical recall points.** Frontier (counters, d32 `32/2`): p4 0.9922 @
  1,064 evals / 25.5 iters; p5 0.9942 @ 1,131; p6 0.9953 @ 1,176; p0 0.9962 @ 1,304.
  ab_g1 matched-recall pairs (b1024, 20t, in-process; judge lane spreads):
  - `32/2 p4` (0.9922) vs `ef=80` (0.9923-0.9927 — equal/higher, fair): run 1 median 1.369
    but ovVS lane unstable (39-51K, 30% spread) → discarded per the tick-5 rule; run 2 with
    stable ovVS lane (44.8-51.2K): ratios 0.973/1.109/1.123/1.199/1.042 → **median 1.109**.
  - `32/2 p6` (0.9953) vs `ef=96` (0.9957): **median 1.114** (both lanes wobbly ~30%).
  Note both engines read HIGH tonight (ovVS p4 51-53K in ovVS-only pairs; hnswlib ef=80
  49-57K) — box in a strong state; the shared-window ratios are the trustworthy quantity.
  **G1 NOT passed; the honest gap is ~1.11×, down from 1.34-1.37× at yesterday's morning
  baseline.** JSONs: `out/quick/g1-d32-p4-ef80{,-r2}.json`, `g1-d32-p6-ef96.json`.
- **2026-08-31** — **NEGATIVE: width 4 at matched recall loses to width 2 (`d32`).**
  Hypothesis: `32/4` pays per-iteration overhead 45% less often (14.5-17.5 iters vs
  25.5-29.4) for +10-13% evals → should win if fixed-per-iteration cost dominates.
  Measured (interleaved pairs, b1024): `32/4 p2` vs `32/2 p4` → 0.945/0.708/0.931, median
  **0.931**; `32/4 p3` vs `32/2 p6` → 0.929/0.928/0.740, median **0.928**. Refuted — in the
  classic mapping the per-iteration cost tracks work, not a fixed floor. `32/2` remains the
  scored configuration (now with p4/p6 as the matched-recall points).
  **NEXT CANDIDATE (design task, not a knob): a convergence stop rule.** The walk always
  runs to beam exhaustion (~2·itopk/SW iterations); patience approximates convergence
  crudely via zero-admission streaks. hnswlib's own termination — stop when the best
  unexpanded candidate is worse than the current result set's worst — is sharper and is
  the likely source of its evals-per-recall edge. Needs a cheap running top-k(=10) worst
  tracker in the walk; output-changing (a trade like patience), so it must be judged on
  the frontier, swept via env before any default flips.
- **2026-08-31** — **CONVERGENCE STOP RULE: BUILT, SWEPT, AND REFUTED — patience dominates
  at every matched recall.** Implemented hnswlib's termination adapted to the merged beam
  (`OVVS_CAGRA_STOP_EF`, both mappings: stop when ≥ stop_ef beam entries are strictly
  better than the best unexpanded entry; one strided count fused after the pick argmin).
  Default off; default path bit-identical (1303.7/35.36/317.9 re-certified); 9/9 CTest.
  Frontier at d32 `32/2` (counters): ef=12→0.9096@508 … ef=32→0.9794@837 … ef=48→
  0.9917@1081, ef=56→0.9943@1197, ef=60→0.9955@1254. Versus patience: p4 0.9922@1064,
  p5 0.9942@1131, p6 0.9953@1176 — **patience wins by 2-7% evals at equal recall,
  everywhere.** Reading: the beam IS the ef-set; a zero-admission streak detects "top set
  stable" earlier than the one-hop distance test, which only fires once the tail stops
  churning. Knob kept solely to ride along in the R2 patience re-sweep, then both get a
  single winner and the loser's knob is removed.
  **R1 kernel campaign state after this: every remaining single-change idea on the list
  has now been landed or refuted with a number. G1 stands at ~1.11× at matched recall
  (p4/p6 configs). The honest options on the table: (a) accept 1.11× at R1 and climb to
  R2 measurement anyway is FORBIDDEN by §3 — so rather: (b) grind compound micro-gains
  (each ruled-out term was 1-5%; none dominant), or (c) attack evals-per-recall at the
  GRAPH level (better construction: more build effort at fixed degree buys recall per
  eval — build is our strong side, 22% faster at 1M with GPU idle headroom), or (d) the
  B20 float-corpus + SQ8 lane, which changes the byte economics 4× and is required for
  the product claim anyway. (c) and (d) are the two with headroom left.**
- **2026-08-31** — **THE HYBRID UNLOCK. User reframed the premise ("CPU + GPU combined
  should beat raw hnswlib") and exposed the campaign's blind spot: the entire G1 fight had
  been iGPU-alone vs 20 cores — parity-class silicon fighting one-handed. ovVS's CPU walk
  had NEVER been benchmarked; it was a serial single-thread loop (`mixer.cpp
  prim_graph_walk`, `for q < nq`) doing 6.4K QPS @ 0.9962 on ONE core — already ~1.4×
  hnswlib per-core at matched recall, courtesy of the d32 graph's evals-per-recall edge.**
  1. **Threaded the CPU walk** (work-stealing atomic query claim, per-query scratch was
     already loop-local, outputs disjoint → bit-identical at any thread count; TEMPORARY
     env `OVVS_CPU_WALK_THREADS`, default = all hardware threads, 1 = old serial). Serial
     re-measured identical (6.2K); **threaded: 38.5K QPS @ 0.9962** (6.2× on 20 cores,
     bandwidth-capped like hnswlib). 9/9 CTest.
  2. **⚡ Batch-1 flipped: 0.175 ms p50 (5.5K QPS) on CPU vs 1.9 ms on GPU — now FASTER
     than hnswlib's own single query at higher recall.** The product story becomes
     routing: small nq → CPU (or GPU when the CPU is busy with the LLM); large → both.
  3. **Concurrent two-process probe (both engines, same corpus, same config, overlapping
     measure windows, contended p50s confirm true overlap): GPU 32.7K + CPU 30.7K =
     ~63.4K QPS @ 0.9962** (each side −20-29% from shared power/bandwidth) **vs hnswlib
     45-53K @ 0.9957 → ~1.2-1.4× AHEAD. First G1-passing number of the campaign** — under
     the hybrid reading of G1, which is the product premise (use the whole SoC; hnswlib
     unweakened at its full 20 threads).
  ⚠ Provisional until: (a) a single-process router (`OVVS_POLICY_HYBRID`: split nq across
  a GPU submit thread + CPU workers, ~60-80 lines in the search dispatch) replaces the
  two-process estimate; (b) an ab_g1-style shared-window score with a hybrid lane;
  (c) the thread default gets an ABI story instead of an env knob. Also noted: the CPU
  walk reads fp32 rows — pointing it at the int8 mirror (AVX2) would cut its bandwidth
  4× and should lift the contended sum further. **NEXT: implement OVVS_POLICY_HYBRID and
  score it properly.**
- **2026-08-31** — **SINGLE-PROCESS HYBRID ROUTER LANDED (`OVVS_HYBRID_WALK=<frac>`,
  TEMPORARY until OVVS_POLICY_HETERO adopts it).** GPU leg [0, f·nq) on an
  exception-guarded worker thread + CPU work-stealing pool on [f·nq, nq); GPU-leg failure
  falls back to CPU for its share; forced policies stay exclusive. Per-query CPU/GPU
  parity makes the merged output bit-identical at any fraction — **recall 0.9962 at every
  f tested.** `quick_cagra --search-policy auto` and an ab_g1 hybrid lane added. 9/9 CTest.
  **Solo-window fraction sweep (b1024, d32 32/2 p0): f=0.4→53.9K, 0.45→55.9K, 0.5→
  53.8-55.8K, `f=0.55`→61.4K, 0.6→57.3K.** Best 61.4K = 1.31× the best single engine,
  ~73% of the naive engine sum (the rest is shared package power/bandwidth).
  **Shared-window ab_g1 vs hnswlib ef=96 (0.9962 vs 0.9955-6), 2×5 rounds:** run 1 median
  **0.808 (ovVS ahead 1.24×, 4/5 rounds)**; run 2 median 1.043 (hnswlib climbed through
  the run as the package heated). **Pooled 10-round median 0.932 → ovVS ahead ~7% at
  higher recall — the first time ovVS has led ANY shared window, but a statistical tie,
  not a certified pass.** Both lanes fail the ≤10% stability bar on the hours-hot box;
  rule 1b: no further scoring tonight. JSONs `out/quick/g1-hybrid-ef96{,-r2}.json`.
  **Margin sitting on the table for the certified pass (queued):** (1) CPU leg reads fp32
  — port it to the int8 mirror (AVX2 dot: 4× less bandwidth in the contended regime);
  (2) port patience to the CPU walk (unlocks the p-frontier configs for hybrid, ~+10%);
  (3) persistent CPU worker pool (thread spawn per call ≈ 6-10% at b1024);
  (4) adaptive split fraction. Then a COLD-BOX certified G1 session with lane-stability
  discipline. Known-good pieces, each measured-adjacent, all additive.
