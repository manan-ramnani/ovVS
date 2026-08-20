# HostCompile GEMM tiles

`npu_gemm` tries a full-shape OpenVINO MatMul on NPU. If that compile/infer fails,
it loops the leading dimension in static tiles of **M=256** (`src/prim/npu/backend_npu.cpp`).
`npu_topk` tiles rows of **32**; `npu_gather_rows` tiles indices of **128**. Each tile is a
cached static graph — the HostCompile idea without a vendored `npu_compiler` interpreter.

PQ ADC remains host-linked `compiler/shave/adc.c` until unsigned SHAVE ELF can load on this
NPU firmware.

SHAVE kernels in `compiler/shave/` remain the sources to lower into `npu_compiler`
`sw_runtime_kernels` when that compiler can load unsigned ELF on this NPU firmware.
