# HostCompile GEMM tiles

`npu_gemm` tries a full-shape OpenVINO MatMul on NPU. If that compile/infer fails,
it loops the leading dimension in static tiles of **M=256** (`src/prim/npu/backend_npu.cpp`).
`npu_topk` tiles rows of **32**; `npu_gather_rows` tiles indices of **128**. Each tile is a
cached static graph — the HostCompile idea without a vendored `npu_compiler` interpreter.

PQ ADC on NPU is OpenVINO Gather+ReduceSum tiles; those ops already execute as
ActShave + DPU (see `compiler/shave/README.md`). Host-linked `adc.c` is the
correctness oracle. A custom `iovs_pq_adc` SHAVE ELF still needs MoviTools.
