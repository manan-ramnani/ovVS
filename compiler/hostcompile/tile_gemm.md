# HostCompile GEMM tiling

Dynamic dimension `N` (dataset rows, IVF list length, candidate slab) is unknown
at NPU compile time. HostCompile_Interpreter (npu_compiler) emits CPU `scf.for`
that calls a static ELF blob of shape `[M, T, K]` for tile size `T`.

ioVS `npu_gemm` is the static blob. The host loop lives in
`src/prim/npu/backend_npu.cpp` (`npu_gemm` today; when OpenVINO is present it
compiles one static MatMul per shape and caches the blob). Larger `N` is split
by the mixer into tiles that match a cached shape family `{32,64,128,256,...}`.
