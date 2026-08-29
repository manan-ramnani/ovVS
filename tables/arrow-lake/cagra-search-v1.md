# Arrow Lake 265K CAGRA same-index search-effort evidence

## Scope and provenance

One clean revision-bound process on 2026-08-29 built a single checksum-pinned SIFT1M CAGRA index, then reused that exact index for four search points. The ovVS build used AUTO NN-Descent with final/intermediate degrees 16/32; search used FORCE_GPU. The unchanged hnswlib 0.8.0 lane used `M=16`, `ef_construction=200`, 20 threads, and `ef={32,64,128,256}`. Every point used one warmup and five measured passes against exact validated HDF5 top-10 truth. Package energy includes CPU, iGPU, and uncore; it is not isolated device energy.

The run came from clean commit `acd67e231e8993ca0b2e041f7998a5430d3a3139` and loaded the explicitly requested DLL at `C:\Personal\ioVS\build-icpx\bin\ovvs.dll`, SHA-256 `C4EE36A1F7540726E7686C138190D201BFB825112EDE9FDD6249048FAB319F8F`. Both lanes succeeded, completion was strict and complete, and validation reported no invalid IDs, duplicate rows, or non-finite distances. The raw artifact and its hash are retained in [`evidence/cagra-search-v1/`](evidence/cagra-search-v1/README.md).

## Measured curve

| Lane / effort | Recall@10 | Median QPS | Batch p50 / p99 | Package µJ/query |
|---|---:|---:|---:|---:|
| CAGRA `itopk=32`, width 1, batch 32 | 0.9036 | 3,480.2 | 8.883 / 10.213 ms | 6,702.1 |
| CAGRA `itopk=64`, width 2, batch 32 | 0.9609 | 1,169.1 | 26.569 / 29.126 ms | 18,297.5 |
| CAGRA `itopk=128`, width 4, batch 32 | 0.9889 | 347.9 | 89.891 / 94.798 ms | 116,004.4 |
| CAGRA `itopk=128`, width 4, batch 1 | 0.9889 | 13.0 | 78.553 / 89.459 ms | 1,710,529.2 |
| hnswlib `ef=32`, batch 32 | 0.8911 | 11,112.6 | 2.873 / 4.570 ms | 2,981.3 |
| hnswlib `ef=64`, batch 32 | 0.9589 | 6,675.0 | 4.876 / 5.482 ms | 4,891.6 |
| hnswlib `ef=128`, batch 32 | 0.9879 | 3,765.6 | 8.689 / 9.813 ms | 8,638.5 |
| hnswlib `ef=256`, batch 32 | 0.9976 | 2,061.7 | 15.773 / 17.675 ms | 13,464.8 |
| hnswlib `ef=256`, batch 1 | 0.9976 | 2,040.2 | 0.507 / 0.635 ms | 14,713.9 |

The same CAGRA graph crosses the separate 0.95 product-quality threshold at `64/2`. That point reaches 0.9609 recall versus hnswlib's 0.9589 at `ef=64`, so the prior 0.9036 result is not a hard topology ceiling below 0.95. This does not establish topology parity or prove that construction quality is irrelevant: more traversal may compensate for a weaker graph, and `itopk_size` and `search_width` rise together. Construction changes are not the first response to this recall gap; the measured first search target is the lowest-effort GPU walk that reaches the required quality.

Performance remains negative at every measured point:

- At the nearest ≥0.95 comparison, CAGRA has 0.0020 higher recall but delivers 17.51% of hnswlib throughput, or 5.71× lower QPS. Its batch p99 is 5.31× higher and package energy/query is 3.74× higher.
- At approximately 0.989 recall, CAGRA has 0.0010 higher recall but delivers 9.24% of hnswlib throughput, or 10.82× lower QPS. Its batch p99 is 9.66× higher and package energy/query is 13.43× higher.
- The batch-one rows are not recall matched: hnswlib has 0.0087 higher recall. Even at that higher quality, hnswlib is 156.37× faster by QPS, has 140.97× lower batch p99, and uses 116.25× less measured package energy/query. This exposes a severe small-batch throughput/latency deficit consistent with under-utilization and/or submission and serial overhead; it is not a matched-recall speed ratio.

The CAGRA build took 99.647 seconds versus 75.337 seconds for hnswlib, a 1.323× deficit. Isolated peak process RSS was 2,343.2 versus 1,459.3 MiB, a 1.606× ratio. CAGRA build telemetry again attributes 90.331 seconds to GPU NN-Descent and 9.204 seconds to host optimize/prune/merge; these single-run values agree with the separate three-process build analysis but do not replace it.

## Route and measurement invariants

All 7,704 ovVS search calls were direct GPU walks with successful device attribution and zero explicit index-upload calls or bytes. Per-point direct-walk counts were 256, 224, 224, and 7,000; they include warmup, timed, and energy passes. The index therefore stayed on the direct shared-USM route visible to the library. The counters do not measure driver-managed shared-memory migration.

This is one benchmark invocation with two sequential isolated lane processes. CAGRA points were measured before hnswlib and in increasing effort order within each lane. The invocation has no in-run clock, temperature, utilization, or background-load trace. The performance and energy ratios are diagnostic, not promotion evidence; a winning change still requires at least three clean interleaved complete invocations. B1 remains open for the other algorithms and policies plus the real 100K×768 corpus.

This curve is the old-source baseline for the later cached-worst promotion. The
promoted cached-worst result, three-process ranges, and unchanged-comparator
status are in [`cagra-cached-worst-v1.md`](cagra-cached-worst-v1.md). The raw
baseline values above remain historical evidence and are not rewritten.

## Engineering disposition

The exact cached-worst candidate passed its repeated complete-wall gate and is
retained. The later prefix-sort/frontier and cooperative multi-pick checkpoints
also passed; the stable one-pass pick was exact but rejected. A measured
four-workgroup/query small-batch capacity gate later reported one cooperative
workgroup versus four required and parked that route on the measured Arrow Lake
stack. hnswlib settings remain unchanged, and no kernel-only timing can
establish a product win. Current results:
[`cagra-cooperative-pick-v1.md`](cagra-cooperative-pick-v1.md) and
[`gpu-root-group-canary-v1.md`](gpu-root-group-canary-v1.md).
