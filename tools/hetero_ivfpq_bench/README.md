# Standalone IVF-PQ NPU + iGPU experiment

This tool is an experimental, same-fixture pipeline probe. It does not change
`OVVS_POLICY_HETERO`, the production mixer, or the canonical benchmark harness.
It first sweeps CPU+iGPU partitions at requested iGPU shares of 0, 25, 50, 75,
and 100 percent. It then runs requested NPU shares of 0, 25, 50, 75, and 100
percent, with the remainder scored and locally selected on the iGPU. Zero NPU
capacity is a first-class measured lane, not a missing result.

For nonzero NPU shares, current production `ovvsPqAdcBatch` produces full scores.
Only after the OpenVINO request has completed does the tool explicitly copy those
scores into an iGPU buffer and select them. This is intentionally the negative
NPU-to-iGPU staging lane: there is no supported OpenVINO NPU/GPU RemoteTensor
zero-copy pair, no NPU external semaphore on the current stack, and no claim that
an NPU tensor is directly SYCL-addressable. Real overlap occurs only between the
NPU prefix and direct iGPU scoring of a distinct suffix.

Two NPU/GPU result families are retained in schema version 2:

- `npu_gpu_raw_direct_merge_lanes` is the intentionally invalid negative lane.
  It merges restored affine NPU scores with exact GPU scores and must fail the
  oracle gate when their global score scales do not preserve the boundary.
- `npu_gpu_exact_union_rerank_lanes` takes the local top-k IDs from each disjoint
  partition, recomputes only their union (at most 64 IDs for the default k=32)
  from the original compact codes and LUTs on the CPU, then applies the final
  exact top-k. This fixes cross-device score calibration, but cannot recover a
  candidate omitted by a device-local shortlist; the ordered oracle gate detects
  that case.

The deterministic `M=8, Ks=256` fixture is the existing scale-about-0.588 affine
case. Every lane is checked against one scalar CPU oracle, including the merged
ordered top-k IDs and scores, finite and fully published NPU scores, and output
canaries. The scaled NPU output has an explicit 128-point absolute score tolerance;
the exact-union rerank uses 0.001. A lane cannot report `success` unless its oracle
gate passes. Failed and unavailable shares remain in the JSON. Candidate counts
131,072 and 524,288 are separate invocations:

```powershell
ovvs_hetero_ivfpq_bench.exe 131072 5 32 hetero-131072.json
ovvs_hetero_ivfpq_bench.exe 524288 5 32 hetero-524288.json
```

Use the output-path argument on hardware runs. OpenVINO or the NPU driver may log
diagnostics to standard output; the explicitly written artifact remains pure JSON.

The root CMake graph builds `ovvs_hetero_ivfpq_bench` when SYCL tools are
enabled with IntelLLVM/Clang. Configure and build it in the canonical initialized
oneAPI environment. Keep the matching `ovvs.dll` and OpenVINO runtime DLLs on
`PATH` when running.

Interpretation limits:

- The first call is reported, but all share lanes run in one process and can reuse
  the process-wide OpenVINO graph cache. It is not an isolated cold-compile study.
- The iGPU kernel is an intentionally small standalone correctness/overlap probe:
  one work-item scans each 2,048-row tile, emits a partial top-k, and the CPU merges
  partials. It is not the production subgroup/SLM fused scan/select kernel.
- Mixed CPU+iGPU lanes keep one CPU worker alive across first-call, warmup, and
  measured passes, run disjoint candidate partitions concurrently, and merge the
  two exact partial top-k lists. CPU controls still materialize their partition's
  scores; the corresponding byte count is explicit.
- The CPU path is a scalar fixture oracle/partition probe. It is not the matched
  production `FORCE_CPU` competitor and must not be used to weaken that baseline.
- The selected SYCL GPU name, vendor, vendor ID, and driver are recorded. The tool
  requires an Intel GPU, but the generic selector does not itself prove integrated
  topology; correlate the device with the reported Arrow Lake SKU.
- `npu_public_api_wall_ms` is the complete synchronous ovVS call. GPU pipeline
  walls include submission, kernels, readback, and CPU tile-partial merging. They
  are not device-execution measurements; the full wall is the promotion-relevant
  number.
- Explicit byte counts distinguish compact codes, current logical expanded i32
  indices, full NPU scores, cross-device staging, and partial readback. They do not
  claim to measure driver-managed migration or NPU scratchpad traffic.
- Depth is one because the public production API currently owns one synchronous
  request per shape. Depth 2/4 must be measured in the bounded request-pool work;
  this tool does not fake concurrency by creating unrelated resources.
- Five raw samples satisfy this diagnostic fixture request, but interpolated
  p95/p99 from five points are not claim-grade. Promotion requires randomized
  paired process runs and a substantially larger sample count.
- No lane is promotable from this probe alone. Promotion still requires repeated
  end-to-end IVF-PQ wall, p50/p95/p99, recall, peak memory, energy, device-stage
  attribution, and matched CPU/iGPU/NPU competitor evidence.
