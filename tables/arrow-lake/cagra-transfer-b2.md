# Arrow Lake 265K CAGRA index-transfer evidence

Source: `build-icpx` with `OVVS_WITH_SYCL=ON` on 2026-08-28. Counters are cumulative per `ovvsResources` and commit only after a batched GPU walk call and output copy complete. `Wcalls/Dcalls/Ucalls/Ubytes` means completed walk calls, calls with both index pointers GPU-accessible, calls requiring an explicit dataset and/or graph upload, and uploaded index bytes. Query, output, and bitset transfers are excluded.

| Workload | Search points | Transfer deltas | Result |
|---|---:|---|---|
| Native CAGRA regression, n=4,200 | 2 FORCE_GPU searches | `2/2/0/0` | pass |
| SIFT prefix, n=2,000, batch=32, itopk=32/width=1 | 1 warmup + 3 measured | `4/4/0/0` | pass |
| SIFT prefix, n=2,000, batch=32, itopk=64/width=2 | 1 warmup + 3 measured | `4/4/0/0` | pass |
| SIFT prefix, n=2,000, batch=1, itopk=64/width=2 | 32 queries × 4 passes | `128/128/0/0` | pass |
| SIFT prefix, n=100,000, nq=1,000, batch=32, itopk=32/width=1 | 3 runs × (1 warmup + 5 measured) | `192/192/0/0` each | pass; all 192 device attributions per run reported GPU |
| SIFT1M current code, nq=1,000, batch=32, itopk=32/width=1 | 1 warmup + 5 measured | `192/192/0/0` | transfer and attribution pass; recall-closeness gate passed |
| SIFT1M pre-change, nq=1,000, batch=32, itopk=32/width=1 | 1 warmup + 5 measured | `192/192/0/0` | transfer pass; prior recall gate failed |

This proves that all listed runs bound GPU-accessible shared-USM dataset and graph pointers directly and performed no explicit index `memcpy`. It does not prove physical residency or zero driver-managed page migration. At SIFT1M degree 16, a heap-fallback walk would explicitly upload 512,000,000 dataset bytes plus 64,000,000 graph bytes; across the 192-call gate that would have been 110.592 GB. The zero-upload result isolates the failed SIFT1M recall gate from explicit index-transfer overhead.

Verification at the latest checkpoint: focused transfer regression 1/1; full native suite 69/69 including CPU/GPU batch-invariant seed parity; Python benchmark suite 57/57. Benchmark tests cover rejection logic using synthetic upload deltas and require a successful GPU attribution for every fixed-preflight walk; the native forced-fallback/upload counter path is not yet exercised.
