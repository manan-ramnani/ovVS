# Arrow Lake NPU + iGPU escape-route evidence

Status: **public custom-ActShave toolchain blocked; direct Level Zero and every
measured nonzero NPU partition negative; bounded residual-PQ Q projection also
negative; a peer-reported production FORCE_GPU IVF-PQ run was correct but
13.65-22.69x slower than CPU end to end. NPU capacity remains zero for this Arrow Lake experimental PQ-ADC
portfolio while the active optimization focus moves to iGPU and size-gated
CPU+iGPU execution**.

The target is the complete client accelerator system, not NPU utilization as an
end in itself. NPU capacity zero is a valid measured choice when the iGPU or a
CPU+iGPU partition wins. CPU remains the exact oracle, setup/refinement engine,
and matched performance control. No BIOS, firmware, production driver, or
working OpenVINO installation was changed.

Hardware and runtime for these runs: Core Ultra 7 265K, Intel(R) AI Boost,
Intel(R) Graphics, and OpenVINO 2025.3.0 build
`2025.3.0-19807-44526285f24-releases/2025/3`. The independently tracked
`../../docs/toolchain.md` inventory records NPU driver `32.0.100.4841`; a
contemporaneous host inspection reported Level Zero loader 1.32.0. The retained
raw artifacts do not self-record those two values, so they are machine context,
not artifact-contained attribution. The fused/Q-factor SYCL artifacts report
driver `1.15.39183+1`; the composition probe reports `32.0.101.8974`. Builds
and caches were worktree-local.

## Existing Gather + ReduceSum wall

The historical old-base `ovvs_pq_adc_bench` diagnostic uses deterministic
`M=8, Ks=256` affine-scale-0.588 tables, compressed u8 codes, an independent
scalar f32 oracle, one warmup, and five complete-call measurements. Its
telemetry ABI is superseded by the main task and the tool is intentionally not
part of this cherry-pickable checkpoint. Its raw JSON is not retained here, so
the following values are diagnostic context rather than promotion-grade raw
evidence. Every successful lane preserved canaries and all CPU top-32 IDs; the
NPU maximum absolute score error was 27.203125.

| Candidates | CPU p50/p95/p99 ms | NPU p50/p95/p99 ms | NPU/CPU p50 | NPU requests/call |
|---:|---:|---:|---:|---:|
| 131,072 | 0.4895 / 0.5041 / 0.5059 | 46.9289 / 47.3461 / 47.4014 | 95.9x | 16 |
| 524,288 | 2.9246 / 3.2222 / 3.2604 | 187.1894 / 188.9497 / 189.1291 | 64.0x | 64 |

The current-run ratio differs from the earlier retained 106x/73.3x medians but
does not change the decision. The new telemetry attributes the five-call totals
without pretending that host clocks are device profiling:

| Candidates | Preflight | Fill + u8-to-i32 expansion | Blocking infer | Validate/read/restore | Publication | Values are total ms for five calls |
|---:|---:|---:|---:|---:|---:|---|
| 131,072 | 7.045 | 5.286 | 219.768 | 1.659 | 0.527 | yes |
| 524,288 | 20.315 | 22.566 | 884.646 | 7.581 | 2.051 | yes |

Per NPU call, the 131,072/524,288 fixtures read 1,048,576/4,194,304 compressed
code bytes, expand 4,194,304/16,777,216 i32-index bytes, explicitly write
9,437,184/37,748,736 request bytes, and consume 524,288/2,097,152 output bytes.
`publication_bytes` includes both atomic staging boundaries and is therefore
1,048,576/4,194,304 bytes per call. These are application-visible copies, not
claims about driver DMA or scratchpad traffic.

## Direct Level Zero graph extension

`ovvs_npu_l0_escape` imports the exact OpenVINO native blob through public
`ZE_extension_graph` 1.18. It compares persistent OpenVINO async-request pools
with one persistent direct graph/queue, four slot-owned command lists/events,
depth-owned fences, and page-aligned `zeMemAllocHost` buffers. The corrected
fixture gives every slot/refill epoch distinct LUTs, codes, and expected output;
every output buffer is NaN-poisoned before submission and every row is validated
after completion. Each cell below has one warmup and five measured repetitions.

| Candidates | Scope | OpenVINO depth 1/2/4 p50 ms | Direct L0 depth 1/2/4 p50 ms |
|---:|---|---:|---:|
| 131,072 | prefilled orchestration | 3.2394 / 5.6281 / 10.9549 | 4.7344 / 7.1006 / 11.0799 |
| 131,072 | refill + execute + consume | 3.5827 / 7.2921 / 13.9737 | 5.1606 / 7.7378 / 14.7300 |
| 524,288 | prefilled orchestration | 11.0652 / 21.8185 / 43.9175 | 15.0815 / 21.3930 / 43.5084 |
| 524,288 | refill + execute + consume | 13.2304 / 27.1379 / 52.7402 | 14.8623 / 29.9803 / 57.5308 |

Direct L0 reduces depth-2/4 prefilled submission count from 2/4 to one, but its
small 524K prefilled differences are not an end-to-end gain. With application
refill and output consumption included, direct L0 is slower at every depth and
shape. The compiled model and OpenVINO request pool are destroyed before direct
setup, so the direct lane does not carry the other lane's request buffers.
Process peak working set was 88,252,416 bytes at 131K and 235,016,192
bytes at 524K. OpenVINO request payload was 18,907,136/75,530,240 bytes; direct
owned buffers were 19,042,304/75,767,808 bytes. Separate-process compile calls
were 4.2695/11.6802 ms with one successful
`LATENCY+NPU_TURBO+optimization-level=2` attempt. Driver cache was uncontrolled,
so neither is labeled a controlled cold compile. OpenVINO is always measured
first; that fixed order remains a documented confounder.

Tracked raw artifacts are
`evidence/npu-l0-131072-final-v3.json` and
`evidence/npu-l0-524288-final-v3.json`, SHA-256
`1FD38DB8E8BD0542FF06B9C35EAB648BEB47011D3AB318713B03B455A03DD912` and
`0C9EB67447EA884B0353B2F4A52E918BFF835413C9699AABE4B2894BFC53DFB7`.
The producing source/executable hashes are
`4709899950D39A3A1E2C48C22593D19695598503AA0E01FEB915CFA9415C9B79` and
`27B1F77C1C807A078B3418A98346FBCE09E41B9FC8825F8F959E5E93CA558086`.

Conclusion: in this fixed-order experiment, bypassing OpenVINO does not remove graph execution, expanded
indices, full score production, or readback. Preserve this lane as a negative
experiment; do not integrate it.

## Fused Intel GPU scan/select and CPU+GPU partition

After separately reported one-time code/block setup and upload,
`ovvs_pq_gpu_fused_bench` consumes persistent compressed u8 codes and computes
ADC inside 256-row work-groups. Each block emits 32 slots: the exact top
`min(32, live_rows)` pairs plus invalid sentinel padding for short tail blocks.
A bounded host union ignores those sentinels. The tool neither expands i32
Gather indices nor reads a full score vector. Device selection fails closed
unless there is one Intel Level Zero GPU with the required work-group,
local-memory, profiling, and device-USM capabilities. The tool proves one Intel
Level Zero GPU; its integrated classification is inferred from the recorded
Arrow Lake identity. The steady-state search-call wall includes table upload,
kernel, partial readback, host union, validation, and fail-closed output
publication; it excludes persistent setup/upload and is not complete IVF search.

| Candidates | Lane | p50/p95/p99 ms | Speedup vs scalar fused CPU |
|---:|---|---:|---:|
| 131,072 | CPU oracle | 0.4442 / 0.4552 / 0.4573 | 1.000x |
| 131,072 | iGPU 100% | 0.8271 / 0.9244 / 0.9312 | 0.537x |
| 131,072 | CPU+iGPU 75/25 | 0.5663 / 0.7926 / 0.8224 | 0.784x |
| 131,072 | CPU+iGPU 50/50 | 0.6270 / 0.7415 / 0.7577 | 0.708x |
| 131,072 | CPU+iGPU 25/75 | 0.7227 / 0.7661 / 0.7674 | 0.615x |
| 524,288 | CPU oracle | 1.7944 / 1.8469 / 1.8561 | 1.000x |
| 524,288 | iGPU 100% | 2.3596 / 9.5464 / 10.9490 | 0.760x |
| 524,288 | CPU+iGPU 75/25 | 1.6843 / 8.6186 / 9.9293 | 1.065x |
| 524,288 | CPU+iGPU 50/50 | 1.3197 / 21.9529 / 26.0749 | 1.360x |
| 524,288 | CPU+iGPU 25/75 | 1.8796 / 1.9204 / 1.9285 | 0.955x |

All five measured repetitions in every lane had exact final ID order, 32/32
overlap, recall 1.0, finite values, and scores within the producer's 1e-5
absolute-error gate. Full-GPU canaries were intact; for mixed lanes the canaries
guard only the GPU partition staging, not the final CPU/GPU union. The raw
artifact does not emit the mixed-lane maximum score error. At 131K every mixed
lane loses. The 524K 50/50 lane has a bounded p50 lead but a 21.95/26.07 ms
p95/p99 tail, so it is not a promotion result; routing must be size-, tail-, and
SKU-gated.

For the 524K full-iGPU lane, persistent upload is 4,243,672 bytes, per-search
input/output are 131,076/526,596 bytes, owned device memory is 4,901,340 bytes,
16,777,216 expanded-index bytes are avoided, and 1,570,556 full-score-readback
bytes are avoided. The 50% iGPU partition owns 2,459,004 device bytes and moves
73,732 input plus 263,428 output bytes per search.

This is a synthetic primitive result, not an IVF-PQ acceleration claim. A peer
reported that the production FORCE_GPU fused path at checkpoint `1ca14e2`
retained exact bounded recall but lost the clean complete search by 22.69x at
`nprobe=2` and 13.65x at `nprobe=8`. That raw run is not part of this checkpoint,
so the ratios are non-canonical context rather than independently auditable
evidence here. This
experimental portfolio therefore assigns no capacity from the standalone p50
lead and does not itself modify production AUTO/HETERO routing.

Tracked raw artifacts are
`evidence/pq-gpu-fused-131072-final.json` and
`evidence/pq-gpu-fused-524288-final.json`, SHA-256
`891E8F5CE9769171EBED8AA2063CDA3F958858040C6ACF2EC3C91A01B6A8C077` and
`7442BFE298B21BBC221647A28B1AB546A6CDD78C632DC89A58702CD309B77158`.
The producing source/executable hashes are
`8E0DE1B2FC77944C4C3E837DAA75499CC04BD56A87EC194387499F5122316B54` and
`A89C692596AAB6C20F3D7843052D1062495F2A899FBDB726AC23370926FF8E2B`.

## NPU + iGPU composition probe

`ovvs_hetero_ivfpq_bench` separately exercises disjoint NPU-prefix and Intel
GPU-suffix work with explicit host-fence handoff. The artifact proves selection
of an Intel GPU; integrated classification is inferred from the Arrow Lake SKU.
It does not claim an
OpenVINO cross-device RemoteTensor or an NPU external semaphore. The simple
GPU kernel is a scheduling/correctness probe, not the fused local-top-32
kernel above.

Directly merging restored affine NPU scores with exact iGPU scores failed the
global oracle for NPU shares 25/50/75% at both required sizes. Those failed
lanes remain in the artifacts and have no claimed warm latency. The corrected
lane takes each device's local top-32 IDs, exactly recomputes the 64-ID union
from the original compact codes/LUTs on CPU, and then selects the final top-32.
All five repetitions preserve exact ordered IDs and scores within the recorded
0.001 absolute tolerance:

| Candidates | NPU share | Exact-union-rerank p50 ms | Searches/s |
|---:|---:|---:|---:|
| 131,072 | 25% | 26.3367 | 37.97 |
| 131,072 | 50% | 40.6784 | 24.58 |
| 131,072 | 75% | 51.7782 | 19.31 |
| 524,288 | 25% | 69.5642 | 14.38 |
| 524,288 | 50% | 107.1275 | 9.33 |
| 524,288 | 75% | 153.1978 | 6.53 |

The 100%-NPU-row lane, which still requires iGPU staging/select, was
64.8887/210.0621 ms p50. The exact rerank repairs the score-domain boundary but
cannot repair NPU wall time, so every measured nonzero NPU share is negative in
this complete experimental scope. This Arrow Lake experimental portfolio keeps
zero NPU capacity for this workload.

Tracked raw artifacts are `evidence/hetero-131072-final-v2.json` and
`evidence/hetero-524288-final-v2.json`, SHA-256
`2E802DCAF83A8F887F9120169A4A833CF57521C8C148EE2243C247828537D26D` and
`144823B63B2ED5C97611AB49E40E09E10757EF6B7EA99D941F55CCB071204243`.
The producing source/executable hashes are
`544AB80F3571B49744DF2554725F5BB8FE131909383025C2429AF9D942FD0345` and
`4F1AEE1AF18C4C25F29FF793B0E15AABD23F72228FE433C6D7670D7721242E82`.
The source-matched run loaded `ovvs.dll` SHA-256
`58308B9130A6E3AAE97AC54F55A5005FCF860C438C96E6C26CD35864A1F6B083`;
the old-base telemetry changes that produced this DLL remain excluded from the
cherry-pickable checkpoint.

## Public `npu_compiler` / ActShave gate

The fail-closed gate pins public `npu_compiler` commit
`6761af885b8ff54ddf0da5bf8ad44e30746b2f62` and its required OpenVINO commit
`4089686065a245d648cdd2b99c31884f53cb7a5e`. The exact clean clone inventories
770 Intel prebuilt ELFs, zero tracked custom-kernel sources, and zero descriptors.
Both public MoviTools manifest paths are absent; `moviCompile`, `moviLLD`, and
the `37xxxx` runtime archives are unavailable. There is no pinned isolated
OpenVINO developer build or local Compiler-In-Plugin build on this machine.

The retained result is `BASELINE_BLOCKED` (exit 10). A stock
ActShave ELF, host-linked `compiler/shave/*.c`, `add_extension()`, or caller
markers cannot make the gate pass. There is no separate public signing command
in this source path: MoviTools would build an ELF that CMake embeds into the
Compiler-In-Plugin graph. Stop at this missing public boundary rather than
claiming a custom ovVS NPU kernel.

## Residual-PQ Q/T factorization gate

The current IVF-PQ index trains one shared residual codebook `W[M,K,dsub]` on
`x-c_l`. Its squared-L2 ADC score has the exact real-arithmetic decomposition

```text
||q-c_l||^2 + sum_m[-2 q_m.W[m,k_m]
                     + ||W[m,k_m]||^2 + 2 c_l,m.W[m,k_m]]
```

This makes `Q(q,m,k)=-2 q_m.W[m,k]` query-only and
`T(l,m,k)=||W[m,k]||^2+2 c_l,m.W[m,k]` persistent. `T` can be derived
transactionally at build/load without changing `IPQ1` v1; extend does not alter
centroids or codebooks. At `nlist=1024,K=256`, `T` is 8 MiB for `M=8` and
16 MiB for `M=16`.

The identity is not an unconditional f32 ordering proof. Signed Q/T
cancellation and reassociation can perturb the cutoff even when every value is
finite. A production path therefore needs a deployable analytic score interval
and must require the factorized k/k+1 gap to exceed twice that interval. A tied,
overlapping, non-finite, or uncertified cutoff executes the direct residual-LUT
path and publishes only its validated result. Fixed oversampling is not used as
the correctness proof.

`ovvs_pq_lut_factor_bench` implements the bounded gate with a persistent Intel
Level Zero GPU
`T`/compressed codes and emits `top min(k+1, block_rows)` from every 256-row
block. Blocks larger than k provide the exact k/k+1 boundary witness; tail
blocks at or below k emit every candidate. The tool then directly f32-rescores
the selected set and has a full direct-LUT fallback. It was compiled with
precise floating-point semantics and contraction/reassociation disabled.
The dyadic fixture is exactly representable: 262,144 and 1,048,576 scores were
bit equal at 131,072 and 524,288 candidates, respectively. Its global cutoff
gaps were 4,151 and 3,995 for the two queries while `E=0`. Every normal joined
scan/select lane was certified with exact ordered IDs/distances and recall@32
1.0; an intentional
rank-32/33 tie activated the exact fallback without partial publication. This
fixture-specific exhaustive/analytic band is evidence for this fixture only,
not a deployable bound for arbitrary non-dyadic IVF-PQ data. The optional
anchored form `Q=-2(q-a).W`, `T=||W||^2+2(c-a).W` remained dormant because raw
bounded-dyadic-fixture Q/T/coarse passed the <=0.001 parity gate.

Warm timing uses one warmup and five measured repetitions. Times below include
Q production, visible handoff, persistent iGPU scan/local top-k, bounded partial
readback, final selection, direct selected-set rescore, and synchronization.
They exclude coarse assignment and exact-vector refinement, so this is a
synthetic joined-stage result, not end-to-end IVF search.

| Candidates | Lane | p50 / p95 / p99 (ms) | QPS at p50 | Candidate/s at p50 | Visible copied bytes/call | NPU requests | GPU launches / submissions / total host waits |
|---:|---|---:|---:|---:|---:|---:|---:|
| 131,072 | scalar direct-LUT oracle | 0.4479 / 0.4527 / 0.4527 | 2,232.64 | 292.64 M | n/a | 0 | 0 / 0 / 0 |
| 131,072 | CPU-Q + iGPU | 0.8631 / 0.9034 / 0.9034 | 1,158.61 | 151.86 M | 155,644 | 0 | 1 / 6 / 1 |
| 131,072 | iGPU-Q + iGPU | 0.9629 / 12.3384 / 12.3384 | 1,038.53 | 136.12 M | 139,772 | 0 | 2 / 7 / 1 |
| 131,072 | NPU-Q + iGPU | 1.3384 / 1.5072 / 1.5072 | 747.16 | 97.93 M | 156,156 | 1 | 1 / 6 / 2 |
| 524,288 | scalar direct-LUT oracle | 1.5405 / 1.6474 / 1.6474 | 649.14 | 340.34 M | n/a | 0 | 0 / 0 / 0 |
| 524,288 | CPU-Q + iGPU | 2.7617 / 2.8037 / 2.8037 | 362.10 | 189.84 M | 561,148 | 0 | 1 / 6 / 1 |
| 524,288 | iGPU-Q + iGPU | 2.7867 / 2.8189 / 2.8189 | 358.85 | 188.14 M | 545,276 | 0 | 2 / 7 / 1 |
| 524,288 | NPU-Q + iGPU | 3.0663 / 11.3985 / 11.3985 | 326.13 | 170.98 M | 561,660 | 1 | 1 / 6 / 2 |

Projection-only warm p50/p95/p99 was 0.0035/0.0112/0.0112 ms (CPU),
0.1591/0.2055/0.2055 ms (iGPU), and 0.3330/0.3737/0.3737 ms (NPU) at
131,072 candidates. At 524,288 it was 0.0036/0.0157/0.0157,
0.1425/0.1853/0.1853, and 0.4439/0.4625/0.4625 ms. NPU cold compile was
263.1036/261.7605 ms and its first call was 285.0604/282.1062 ms. Peak owned
GPU memory was 2,921,316/9,655,140 bytes; process peak working set was
128,892,928/142,020,608 bytes. Devices were Intel Core Ultra 7 265K, Intel
Graphics through SYCL Level Zero driver 1.15.39183+1, and Intel AI Boost through
OpenVINO 2025.3.0. GPU integration is inferred from the recorded Arrow Lake
identity; the generic selector proves only one Intel Level Zero GPU. Tracked
artifacts are `evidence/pq-lut-factor-131072-final-v2.json` and
`evidence/pq-lut-factor-524288-final-v2.json`. The five-sample p95/p99 values
are the retained maximum-ranked samples; joined lanes contain real tail outliers
and are not filtered. Their SHA-256 digests are
`FB7C483CF95097D3A5D7DD1AB2D17047344CB37B247A88B191A19E4B40C5CF04` and
`6D223089EBA8281D8FA371079837455AC4ACC89F6DDD533B5AC5C08A9D4ED9B5`.
The producing source/executable digests are
`5322969BD8476578D3411AF6FC3071CBCF74B17A1D51272A1CF2CA9B0E2609DF` and
`A10183C4813BFB74B56B245C6B21D9FE9488615F36BA9389450EAC1C6C650EFE`;
the producing binary is `build-npu-escape/bin/ovvs_pq_lut_factor_bench.exe`.
Earlier Q-factor JSON files are superseded diagnostics.

This factorization is native to squared L2. No lane is promoted: the best
factorized p50 used 1.93x and 1.79x the wall time of the scalar direct-LUT oracle
at the two sizes, and NPU-Q used 1.55x and 1.11x the wall time of the best
factorized CPU/GPU producer in the complete joined scope. The 131,072 iGPU-Q
and 524,288 NPU-Q lanes had sharp tail regressions. Production still requires
the existing direct path for all metrics and all uncertified inputs.

## Active iGPU experiment plan

Q/T factorization is retained negative evidence, not an acceleration candidate.
The Arrow Lake optimization track proceeds in this order:

1. **GPU construction, merge, and CPU-searchable export.** Treat bulk CAGRA and
   NN-Descent construction as a primary product lane. Move graph optimization,
   prune/merge, and HNSW export onto the iGPU where the complete pipeline wins,
   then benchmark CPU search over the exported graph. Report complete build and
   merge wall, export wall/bytes, peak host/device memory, graph quality and
   connectivity, and resulting CPU-search recall/QPS against honest hnswlib and
   optimized CPU construction controls.
2. **CAGRA search-kernel portfolio.** Retain one workgroup/query on this Arrow
   Lake stack: the separate four-workgroup cooperative capacity gate reported
   one group versus four required. Continue the SIMD16 portfolio with
   one-workgroup and bounded multi-kernel/persistent-ring designs that do not
   require unsupported cross-workgroup synchronization. Reopen a cooperative
   multi-workgroup/query path only on a materially different measured
   capability result. Replace remaining leader-serial insertion, expansion,
   and selection work only behind exact output and complete-wall gates.
3. **Hierarchical PQ selection.** Replace the 256-element bitonic sort, three
   readbacks, and CPU merge with SIMD16 register top-k, subgroup merge,
   work-group local top-k, a second GPU global top-k, and one final k-row
   readback. Specialize fused selection for small k and measure a non-fused path
   above the crossover. Preserve deterministic `(score, ID)` order and atomic,
   fail-closed publication.
4. **Packed SIMD16-friendly PQ storage.** Bake off true 4/6/8-bit codes,
   16-byte aligned/padded records, and an AoSoA layout grouping one subspace
   across 16 candidates. Report copied bytes, occupancy, bandwidth, peak memory,
   complete wall, and recall; kernel time alone cannot select the layout.
5. **Direct residual-LUT placement.** Compare FP32 LUT in SLM, bounded FP16 LUT
   in SLM with a conservative error-band/cutoff-gap certificate plus exact
   refinement/direct fallback, and global-memory fallback. Select by `pq_m`,
   `pq_bits`, available SLM, measured occupancy, and complete wall. Do not revive
   Q/T factorization in this lane.
6. **Dynamic CAGRA/IVF batching.** Provide latency mode with a short bounded
   microbatch window and persistent request ring, throughput mode over large
   `queries*probes` work, and deadline-aware closure by size or SLA. Report queue
   delay, device execution, and complete response latency separately.
7. **Squared-L2 early abandonment.** Use only non-negative direct-LUT
   contributions, a validated current kth threshold, and subspace ordering by
   measured discrimination. Record SIMD divergence and discard the lane when
   full wall loses even if executed distance work falls.
8. **Held-out multi-objective SKU tuner.** Search CPU/iGPU/CPU+iGPU policy,
   CAGRA kernel variant, `nlist/nprobe`, PQ bits/dim/refinement, LUT
   precision/location, batch size, and CPU+iGPU partition. Optimize build time
   plus latency/QPS subject to correctness/recall, then validate the chosen
   configuration on held-out queries and datasets.

Every lane retains failed/unsupported results. Promotion requires at least five
repeated complete end-to-end measurements, exact correctness, unchanged recall,
QPS and p50/p95/p99 response latency, full copied bytes/allocations/submissions/
waits, peak memory, and device attribution. Construction claims additionally
require complete build/merge/export timing and downstream CPU-search quality.

Architecture references: [Meta/Faiss cuVS build and search integration](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/),
[Elasticsearch GPU index construction](https://www.elastic.co/search-labs/blog/elasticsearch-gpu-accelerated-vector-indexing-nvidia),
[cuVS IVF-PQ internals](https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/),
[cuVS IVF-PQ tuning](https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-performance-tuning-part-2/),
[cuVS CAGRA C API](https://docs.rapids.ai/api/cuvs/nightly/c_api/neighbors_cagra_c/),
and [cuVS tuning guide](https://docs.rapids.ai/api/cuvs/nightly/tuning_guide/).

## Decision

- Do not integrate the direct Level Zero path.
- Keep the public compiler gate. Do not resume Arrow Lake work merely because
  MoviTools or compiler inputs appear; reopening requires a new SKU or explicit
  direction first, then an authorized isolated pinned OpenVINO/npu_compiler
  build plus the required MoviTools and source/descriptor inputs.
- Do not promote the fused iGPU or CPU+iGPU primitive result. The newer clean
  production FORCE_GPU result is correct but loses CPU end to end.
- Stop pursuing a nonzero NPU share in this Arrow Lake experimental track.
  Preserve the
  compiler, OpenVINO, direct-Level-Zero, partition, and Q-projection lanes as
  negative evidence; experimental portfolio capacity is zero. This checkpoint
  does not silently rewrite production runtime policy.
- Continue only evidence-driven iGPU work: Intel SIMD16 subgroups, persistent
  compressed-code residency, fused scan/local top-k, fewer allocations/copies/
  submissions/synchronizations, and size-gated CPU+iGPU cooperation. Promote
  nothing without repeated complete end-to-end IVF timing, unchanged exact
  correctness and recall, and honest optimized CPU/SIMD controls.
