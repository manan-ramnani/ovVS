# Consult prompt — iGPU graph-search throughput wall

Paste everything below the line into a fresh session (model: Fable). It is self-contained;
that session will have none of this conversation's context.

---

I need a second opinion on a performance wall. Please read
`C:\Personal\ioVS\.claude\plans\2026-08-30-ovvs-acceleration-campaign.md` first — it is the
campaign anchor and contains every measurement referenced here, with the methodology.
The kernel is `gpu_cagra_walk` in `C:\Personal\ioVS\src\prim\gpu\backend_gpu.cpp`.

## The goal

ovVS is a cuVS-shaped vector search library for Intel client silicon. The point of the project:
almost nobody outside the NVIDIA/Apple world has a hardware-accelerated local vector store, and
the enterprise majority is on Intel. Local RAG and on-device AI memory need one.

The gate: **beat hnswlib on CPU at equal-or-better recall@10, without weakening hnswlib.**

## Hardware ground truth (verified by probe, not datasheet)

- Intel Core Ultra 7 265K, Arrow Lake-S desktop. 20 CPU cores.
- iGPU is plain **Xe-LPG**: 1 render slice, **4 Xe-cores = 64 XVE = 512 FP32 lanes**, 8 HW threads
  per XVE → **512 hardware threads device-wide**, ~2.0 GHz.
- **FP32 peak ≈ 2.05 TFLOP/s. INT8 DP4a ≈ 8.2 TOPS. NO XMX / DPAS matrix engines.**
- SLM 128 KB per Xe-core, **64 KB max per work-group**. Sub-group sizes 8 / 16 / 32. Max
  work-group 1024. **L2 is 4 MB.**
- Memory: shared DDR5-6400, ~102 GB/s theoretical, **shared with the 20 CPU cores**.
- SYCL via Intel icx, Level Zero backend, JIT (no AOT).

## The competitor

hnswlib, source-pinned `/arch:AVX2` build, frozen at `M=16, ef_construction=200, 20 threads`.
At SIFT100K, batch 1024 it does roughly 48,000–70,000 QPS depending on `ef`/recall.
(Its Python binding silently collapses to one thread when `rows <= num_threads*4`, so batch ≤ 80
measurements are single-core. All numbers below are batch 1024, genuinely 20-threaded.)

## Where we are

CAGRA graph walk on the iGPU. Best current points at SIFT100K, batch 1024:
~35,000 QPS at recall 0.992, ~42,000 at 0.996. **hnswlib is 1.34–1.37× ahead at matched recall.**
Started the campaign ~5× behind, so the rewrite worked — it has just stalled.

## What the kernel does now

One work-group per query: 128 work-items = 8 sub-groups of 16 (`reqd_sub_group_size(16)`).
Per iteration (~36 of them per query at search_width 4, ~1,334 distance evals per query total):

1. **Pick**: sub-group 0 selects the `search_width` best unexpanded beam slots (sub-group argmin,
   order-exact).
2. **Gather**: the whole work-group reads `search_width × degree` adjacency entries plus the
   optional filter bitset, in parallel, into local memory.
3. **Commit**: intra-tile duplicate removal, then a CAS probe-insert into a **per-query
   open-addressed hash** (the visited set), then a stable order-preserving compaction.
4. **Score**: one sub-group per candidate. Rows are **uint8** (exact for this data: every partial
   sum < 2^24, so fp32 holds the exact integer), so a 128-dim row is **2 cache lines**.
5. **Admit**: serial on sub-group 0, in tile order, with a pre-filter that drops any candidate at
   or beyond the beam's worst distance (exactly equivalent, skips ~2/3 of candidates).

5 work-group barriers per tile. Beam = 2 × itopk.

## Measured attributions — all interleaved, median of per-pair ratios, output bit-identical

| change | worth |
|---|---:|
| removing an O(beam) serial eviction rescan | **2.5–3×** |
| sub-group per candidate (vs one work-item per dimension) | **1.49–1.51×** |
| uint8 rows instead of fp32 (8 → 2 cache lines/row) | **1.41×** |
| removing a `search_width==1` serial pick path | **1.3–1.7×** |
| parallelising the visited-set probe | 1.06–1.14× |

## Already ruled out — with numbers. Please do not re-propose these.

- **Barrier count**: 7 → 5 barriers per tile changed throughput by **zero**.
- **Launch/allocation overhead**: persistent device workspace, removing per-call queue drains —
  **null result three separate times**. Level Zero submission is <1% of kernel wall.
- **Memory-level parallelism via unrolling**: keeping 2 candidate rows in flight per sub-group was
  a **3% regression** — register pressure outweighed the overlap.
- **Bitonic bulk-merge of a tile into the beam**: ~90 cooperative rounds per tile against ~13
  serial survivors. Rejected on arithmetic before building it.
- **The O(tile²) dedup/compaction passes**: tile sweep 16/32/64/128 → 27,995 / 31,012 / 33,454 /
  33,645 QPS. Smaller tiles are *worse*, so those passes are not the cost.
- **Resizing the visited hash**: 8× larger is **2.3× slower**; halving it measured 0.711 (also a
  regression). Current size is near a local optimum.
- **Bit-per-node visited set**: better at 100K (12.5 KB vs 32 KB per query, no probe chain) but it
  *inverts* at 1M, because the hash is sized by the visit budget (~2,656, independent of N) while a
  bitset is N/8. Rejected to avoid optimising one rung and regressing the next.
- **Cooperative/root-group launches**: unavailable on this device, and not needed.

## The actual puzzle — this is what I want your help on

Simultaneously true, all measured:

- **Full thread occupancy**: 16 work-groups per Xe-core × 8 threads = 128 = the hardware maximum.
  SLM use is ~2 KB per work-group, so SLM is not limiting occupancy. **512 resident threads.**
- **~0.9% of FP32 peak.**
- **Memory traffic 5.8 GB/s against a ~102 GB/s roofline = 5.7%.** Not bandwidth-bound.
- **The row gather is only ~14% of kernel time.** Derived from the int8 A/B: uint8 cuts lines per
  row 4× and buys 1.41×, so solving `4G + R = 1.41(G + R)` gives `G ≈ 0.14`.
- **Barriers ≈ 0%** (measured, above).

So: full occupancy, not bandwidth-bound, not compute-bound, not barrier-bound, and the gather is
only a seventh of it. **Where is the time actually going?**

Supporting facts that may or may not matter:

- There are **more visited-set probes than distance evaluations**: 2,297 vs 1,334 per query. Every
  rejected neighbour still probes. Probes are CAS operations into a per-query 32 KB region; at
  batch 1024 the total visited allocation is 32 MB, though only ~64 work-groups are resident at
  once so the *active* footprint is ~2 MB.
- The admit pre-filter leaves **31–34%** of scored candidates for the serial admit loop
  (~450 per query at width 4), each costing a sub-group barrier + broadcast, and an argmax on
  actual eviction.
- **Per-iteration cost is non-linear in tile size**: normalised per-query cost of one iteration is
  4.57e-7 / 5.03e-7 / 9.06e-7 / 1.80e-6 for tiles of 16 / 32 / 64 / 128. Nearly flat from 16 to 32,
  then linear. Implies a large fixed component at small tiles that I have not identified.
- **Batch 1 is 522 QPS.** One query = one work-group = 8 threads on 1 of 4 Xe-cores ≈ 1.5% of the
  machine. This is the actual product case (local RAG asks one question at a time).
- **Graph degree 32 beats degree 16** on the recall/eval frontier (higher recall at fewer evals,
  and ~1.29× faster), not yet adopted as default.
- **I have never checked whether the kernel spills registers.** The MLP regression is circumstantial
  evidence that it is near the limit. I do not know how to get IGC to report spill counts here and
  have not tried.
- The distance code uses scalar `uint8` loads and an int32 accumulator. **DP4a is never used
  explicitly**, and I do not know whether IGC contracts the pattern into it.

## Measurement environment — important if you suggest an experiment

This box drifts violently. hnswlib alone spanned **35,300–61,287 QPS at one identical setting**
across sessions. The iGPU shares package power with 20 CPU cores and degrades under sustained load
(p99/p50 reached 10× after a long run). So:

- Accept/reject on **work counters** (evals/query, iterations/query) where possible — they are
  load- and thermal-independent.
- For timing: one point per process, interleave baseline and candidate, ≥3 alternations,
  **median of per-pair ratios**, never compare absolutes across sessions.
- `tools/bench/quick_cagra.py` (ovVS, with `--counters`), `tools/bench/ab_g1.py` (both engines in
  ONE process, alternating rounds — separate processes were unusable).

## What I am asking

1. **Where is the missing time?** Given the five constraints above, what mechanism am I not seeing?
   Concrete candidates I cannot rule out: register spilling to scratch, SLM bank conflicts on the
   beam arrays, the CAS probes serialising in L3, sub-group 16 being the wrong width, IGC failing
   to vectorise the uint8 inner loop, or thread-launch/EU-scheduling overhead per work-group.
   Tell me how to *distinguish* these — the diagnostic matters more than the guess.
2. **Is there a fundamentally better mapping of graph search onto this hardware?** The current
   design is one-work-group-per-query with a sequential frontier. Alternatives I have not costed:
   multiple queries per work-group for latency hiding, a two-phase gather/score split across kernel
   launches, restructuring toward wide batched BFS, or exploiting DP4a explicitly.
3. **The honest question: is this winnable?** A 4-Xe-core iGPU at 2.05 TFLOP/s against 20 AVX2
   cores at ~1.25 TFLOP/s, on a pointer-chasing, latency-bound workload where the CPU's
   out-of-order execution and large caches are exactly the right tool. hnswlib itself only reaches
   ~4.6% of its own peak, so neither side is efficient — but it is 1.35× ahead. If the answer is
   "graph traversal structurally favours the big out-of-order core, and you should change the
   algorithm rather than the kernel", say so plainly and tell me what the alternative algorithm is.
   I would rather hear that than grind out another 10%.

Please be concrete and quantitative, and challenge any of my measurements or inferences that look
wrong — especially the 14% gather figure, which rests on assuming fp32 costs exactly 4× the int8
gather time.
