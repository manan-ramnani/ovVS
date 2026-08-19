# npu_compiler / SHAVE track (Phase 23)

This tree holds HostCompile tile loops and SHAVE kernel sources that plug into
[openvinotoolkit/npu_compiler](https://github.com/openvinotoolkit/npu_compiler)
`sw_runtime_kernels` when that compiler is available.

Until `npu_compiler` is vendored as a submodule, these kernels are the
source-of-truth implementations we will lower:

- `shave/topk.c` — per-row k-selection for NPU SHAVE
- `shave/adc.c` — IVF-PQ asymmetric distance compute
- `hostcompile/tile_gemm.md` — how the host SCF loop tiles dynamic N onto static GEMM ELF blobs

Do not commit compiled NPU ELF blobs.
