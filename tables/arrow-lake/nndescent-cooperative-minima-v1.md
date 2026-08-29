# Arrow Lake NN-Descent cooperative-minima V1

Status: **promoted for the existing iGPU NN-Descent path**. This is a
construction improvement, not a CAGRA search acceleration claim.

## Why this changed

VTune on the current SIFT1M build attributed 82.113 seconds of 91.034 seconds
of GPU time (90.2%) to the proposal-join kernel. That kernel ran SIMD16 at
22.8% occupancy, with 10.8% XVE active time and 86.1% stalled time; the run was
not materially L3-bandwidth-bound. The high-value serial region was the work-
group leader scanning up to 96 row/column minima after distance matrices were
already complete.

The promoted kernel assigns each of the `2 * new_count + old_count` minima to
one work-item. Every work-item scans its row or column in the same ascending
candidate order with the existing `(distance, ID)` comparator. After a uniform
barrier, lane 0 retains the prior reciprocal-emission order and atomic order.
Distance production and floating-point association are unchanged. The two
local result arrays add 768 bytes at the maximum bounded join, raising local
memory from 8,456 to 9,224 bytes on the measured shape.

There is no public ABI, serialized-index, algorithm-parameter, allocation,
submission, wait, transfer, or telemetry-counter change. A GPU with less than
9,224 bytes of local memory may now reject this join with `OOM`; the recorded
Arrow Lake device exposes 65,536 bytes per work-group.

## Admission evidence

The parent is the pushed exact-GPU-optimizer product code. Candidate processes
used DLL SHA-256
`6B90289980D01A69189FAC4B35F7F92CBD68BC2163C8B6908AA75A5F7C5D4FAB`.
Every benchmark lane loaded the requested DLL, validated outputs, used exact
SIFT truth, one warmup and five measured search passes, and retained hnswlib
`M=16`, `ef_construction=200`, seed 7, and 20 requested build threads. Energy
was disabled, so these remain matched quality gates rather than complete B1
energy curves.

| Scope | Product parent | Candidate | Change |
|---|---:|---:|---:|
| SIFT100K complete build, 3-process median | 7.913 s | 5.346 s | **-32.44%** |
| SIFT100K initializer, 3-process median | 7.851 s | 5.286 s | **-32.67%** |
| SIFT1M complete build, 3-process median | 91.978 s | 61.000 s | **-33.68%** |
| SIFT1M initializer, 3-process median | 90.883 s | 60.164 s | **-33.80%** |
| SIFT1M CAGRA QPS median | 6,929.0 | 6,931.8 | +0.04% |
| SIFT1M batch-latency p99 median | 4.976 ms | 4.978 ms | +0.03% |
| SIFT1M process peak RSS median | 2,269.8 MiB | 2,272.9 MiB | +0.14% |

All six SIFT100K builds retained recall@10 0.9718 and final
`168,506/168,506` changed/pending edges. All six SIFT1M product-parent/candidate
builds retained recall@10 0.9036, six iterations,
`5,911,309/5,949,466` changed/pending edges, 1,098 kernels, 2,458 submissions,
1,820 waits, and 459 D2H calls. The SIFT100K serialization matches the frozen
parent byte-for-byte; the SIFT1M serialization is 64,000,036 bytes and matches
the canonical production SHA-256
`0A792378EFD0652A4410894BF7ABAF888EEC314B5C4CAE7F7463532002AF15BE`.

In the three candidate processes, unchanged stock hnswlib 0.8.0 built in a
77.086-second median versus 61.000 seconds for ovVS. This is a 20.87% build-
wall win over the currently packaged comparator, not yet a claim against full-
speed hnswlib: the installed Windows extension uses legacy SSE distance code,
and a separately pinned `/arch:AVX2` build remains required. The same stock
comparator still wins low-effort search, 11,611 versus 6,932 median QPS. The
existing comparable-recall CAGRA search deficit is unchanged by this build
optimization.

## Rejected coordinate-staging controls

Two one-process SIFT100K screens preserved recall and all structural counters
but failed before repeated admission:

| Candidate | Complete build | Parent | Result |
|---|---:|---:|---:|
| Bank-safe 64-dimension NEW+OLD tiles, physical stride 65 | 19.922 s | 7.924 s | 2.51x slower |
| Full 128-dimension NEW-only staging, physical stride 129 | 17.655 s | 7.924 s | 2.23x slower |

They are retained as bounded negative evidence. The measured join is not a
simple global-memory bandwidth problem; further coordinate tiling is parked
unless a materially different pair-mapping kernel changes the compute and SLM
access pattern.

## Verification and limits

- 95/95 native tests passed with zero skips.
- Eight CTest product/device/interoperability lanes passed; the separate
  unsupported root-group capability canary skipped as designed.
- Independent source audit found the scan order, tie order, barriers, bounds,
  zero-count behavior, fail-closed status, and emission order equivalent.
- Benchmark processes were fresh, but the system persistent SYCL cache was not
  cleared. Install-cold JIT is outside this performance claim.
- NN-Descent still exhausts six iterations and is not a standalone convergence
  claim. T12.5 IVF-PQ-initializer comparison and K64 scale remain open.

Raw immutable JSON and hashes:
[`evidence/nndescent-cooperative-minima-v1/`](evidence/nndescent-cooperative-minima-v1/README.md).
