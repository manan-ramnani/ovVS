# npu_compiler / SHAVE track (Phase 23)

This tree holds HostCompile tile loops and SHAVE kernel sources that plug into
[openvinotoolkit/npu_compiler](https://github.com/openvinotoolkit/npu_compiler)
`sw_runtime_kernels` when that compiler is available.

Pin: https://github.com/openvinotoolkit/npu_compiler @ `6761af885b8ff54ddf0da5bf8ad44e30746b2f62` (local clone in `compiler/npu_compiler/`, gitignored; `sw_runtime_kernels/` is present). SHAVE device ELF still needs the compiler's SHAVE toolchain + signed load on this NPU firmware.

Until that compiler can load **unsigned SHAVE ELF** on this NPU, the running path is:

- OpenVINO NPU MatMul / TopK / Gather (and HostCompile M=256 GEMM tiles in `npu_gemm`)
- Host-linked `shave/topk.c` and `shave/adc.c` (correctness oracles + future lowering)

Do not commit compiled NPU ELF blobs.
