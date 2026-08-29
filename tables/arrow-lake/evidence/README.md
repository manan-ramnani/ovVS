# Arrow Lake evidence manifest

The table below covers the tracked raw inputs for
`../npu-igpu-escape-routes.md`. Every escape-route timing artifact is
standalone negative evidence: none authorizes production routing or an
acceleration claim.

| Experiment | 131,072 SHA-256 | 524,288 SHA-256 |
|---|---|---|
| Corrected direct Level Zero v3 | `1FD38DB8E8BD0542FF06B9C35EAB648BEB47011D3AB318713B03B455A03DD912` | `0C9EB67447EA884B0353B2F4A52E918BFF835413C9699AABE4B2894BFC53DFB7` |
| Fused GPU/CPU+GPU | `891E8F5CE9769171EBED8AA2063CDA3F958858040C6ACF2EC3C91A01B6A8C077` | `7442BFE298B21BBC221647A28B1AB546A6CDD78C632DC89A58702CD309B77158` |
| NPU+GPU exact-union rerank v2 | `2E802DCAF83A8F887F9120169A4A833CF57521C8C148EE2243C247828537D26D` | `144823B63B2ED5C97611AB49E40E09E10757EF6B7EA99D941F55CCB071204243` |
| Bounded Q/T factor gate v2 | `FB7C483CF95097D3A5D7DD1AB2D17047344CB37B247A88B191A19E4B40C5CF04` | `6D223089EBA8281D8FA371079837455AC4ACC89F6DDD533B5AC5C08A9D4ED9B5` |

Producer provenance:

| Experiment | Source SHA-256 | Executable SHA-256 |
|---|---|---|
| Corrected direct Level Zero v3 | `4709899950D39A3A1E2C48C22593D19695598503AA0E01FEB915CFA9415C9B79` | `27B1F77C1C807A078B3418A98346FBCE09E41B9FC8825F8F959E5E93CA558086` |
| Fused GPU/CPU+GPU | `8E0DE1B2FC77944C4C3E837DAA75499CC04BD56A87EC194387499F5122316B54` | `A89C692596AAB6C20F3D7843052D1062495F2A899FBDB726AC23370926FF8E2B` |
| NPU+GPU exact-union rerank v2 | `544AB80F3571B49744DF2554725F5BB8FE131909383025C2429AF9D942FD0345` | `4F1AEE1AF18C4C25F29FF793B0E15AABD23F72228FE433C6D7670D7721242E82` |
| Bounded Q/T factor gate v2 | `5322969BD8476578D3411AF6FC3071CBCF74B17A1D51272A1CF2CA9B0E2609DF` | `A10183C4813BFB74B56B245C6B21D9FE9488615F36BA9389450EAC1C6C650EFE` |

The first three binaries were built under `build-escape-final/bin`. The Q/T
artifacts were produced by
`build-npu-escape/bin/ovvs_pq_lut_factor_bench.exe`; a later rebuild has a
different hash and is not their producer. These local, ignored build outputs
are not retained by this checkpoint; their digests preserve producer identity.
The tracked sources and CMake targets are the reproducible inputs, for example
`cmake --build build-npu-escape --config Release --target
ovvs_pq_lut_factor_bench` in the documented oneAPI/OpenVINO environment.

Absolute `out/bench/...`, `out/npu-escape-cache/...`, and build paths embedded
in the immutable JSON are historical producer-time paths. The canonical copies
are the files in this directory; the JSON was not rewritten because doing so
would invalidate its recorded digest. The Q/T JSON records the selected Intel
Level Zero GPU name/vendor/driver but no PCI ID or UUID, and it records the NPU
through OpenVINO rather than a raw loader/driver identity. The table therefore
keeps GPU integration qualified and does not infer finer device provenance.

The NPU+GPU executable dynamically loaded `build-escape-final/bin/ovvs.dll`
SHA-256
`58308B9130A6E3AAE97AC54F55A5005FCF860C438C96E6C26CD35864A1F6B083`.
That DLL contains old-base telemetry work intentionally excluded from this
checkpoint; the tracked JSON binds the completed experiment, while a downstream
cherry-pick must rerun against its own current library before integration.

Direct Level Zero uses separate processes, one warmup, five repetitions, and
depths 1/2/4. Its v3 fixture replaced invalid earlier diagnostics: slot/epoch
signatures are distinct, outputs are NaN-poisoned before every submission, and
every completion is validated. OpenVINO is measured first with uncontrolled
driver-cache state, but its compiled model/request pool is released before the
direct graph is created.

The historical Gather+ReduceSum telemetry JSON and the peer production
FORCE_GPU IVF-PQ run are not present in this checkpoint; their prose values are
explicitly non-canonical context. The public compiler boundary is retained in
`../../../compiler/shave/PUBLIC_TOOLCHAIN_GATE.md`.

Other evidence manifests: [`cagra-build-v1/`](cagra-build-v1/README.md),
[`cagra-search-v1/`](cagra-search-v1/README.md),
[`cagra-cached-worst-v1/`](cagra-cached-worst-v1/README.md),
[`cagra-frontier-v1/`](cagra-frontier-v1/README.md),
[`cagra-cooperative-pick-v1/`](cagra-cooperative-pick-v1/README.md),
[`gpu-root-group-canary-v1/`](gpu-root-group-canary-v1/README.md),
[`nndescent-active-count-v1/`](nndescent-active-count-v1/README.md),
[`nndescent-convergence-reduction-v1/`](nndescent-convergence-reduction-v1/README.md),
and [`nndescent-heads-reset-v1/`](nndescent-heads-reset-v1/README.md).
