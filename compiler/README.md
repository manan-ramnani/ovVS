# npu_compiler / SHAVE track (Phase 23)

This tree holds HostCompile tile loops and SHAVE kernel sources that plug into
[openvinotoolkit/npu_compiler](https://github.com/openvinotoolkit/npu_compiler)
`sw_runtime_kernels` when that compiler is available.

Pin: https://github.com/openvinotoolkit/npu_compiler @ `6761af885b8ff54ddf0da5bf8ad44e30746b2f62` (local clone in `compiler/npu_compiler/`, gitignored; `sw_runtime_kernels/` is present). SHAVE device ELF still needs the compiler's SHAVE toolchain + signed load on this NPU firmware.

In-driver VCL (`openvino_intel_npu_compiler_loader.dll`) compiles **IR xml+weights**
to a graph ELF64. SHAVE kernels from `sw_runtime_kernels` are copied into
`.text.KernelText` / `.text.ActKernelInvocations`. Firmware loads that graph blob.
PQ ADC Gather+ReduceSum already executes ActShave+DPU on silicon (`PERF_COUNT`
exec_type `Shave` / `DPU`). Probe: `shave_silicon_load: compiler_actshave`.

A loose unsigned SHAVE ELF32 is still not a loadable object: VCL IR is xml+weights
(`invalid_ir`); Level Zero native format is the graph ELF64 (`invalid_native_binary`
for ELF32). `ze_activation_kernel_desc_t` is extra kernel data at graph compile,
not an unsigned injector. MoviTools/`moviCompile` is not public on this host.

Until a new kernel can be compiled with MoviTools and registered as `VPU.SW.Kernel`,
the running path is:

- OpenVINO NPU MatMul / TopK / Gather (and HostCompile M=256 GEMM tiles in `npu_gemm`)
- Host-linked `shave/topk.c` and `shave/adc.c` (correctness oracles + future lowering)

Do not commit compiled NPU ELF blobs.
