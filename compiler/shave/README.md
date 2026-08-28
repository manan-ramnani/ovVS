# SHAVE on this NPU

Intel documents SHAVE as the DSP next to the MAC array. Custom unsigned ELF is
**not** a public load path. What actually works on Arrow Lake 265K:

## What runs on silicon today

OpenVINO NPU compile of Gather+ReduceSum (ovVS PQ ADC) lowers to Intel's
prebuilt ActShave kernels (`gather.3720xx.elf`, `reduce_sum.3720xx.elf` in
`npu_compiler/sw_runtime_kernels`). Profiling (`PERF_COUNT`):

- `lut` → `Shave`
- `Gather` → `Shave`
- `ReduceSum` → `DPU`
- `Result` → `Shave`

The driver does **not** load those ELF32 files as graphs. The compiler copies
`.text` into a graph ELF64 blob (`.text.KernelText`, `.text.ActKernelInvocations`)
and firmware runs that blob. `ovvs_probe` reports `shave_silicon_load: compiler_actshave`.

## What does not work (tried)

- VCL `vclExecutableCreate` with ELF as IR → `invalid_ir` (IR is xml+weights).
- Level Zero `ZE_GRAPH_FORMAT_NATIVE` with a SHAVE ELF32 → not a graph container
  (`invalid_native_binary` expected).
- `ze_activation_kernel_desc_t` (`ZE_STRUCTURE_TYPE_GRAPH_ACTIVATION_KERNEL`)
  chained onto a graph desc: extra kernel bytes at compile time, not an unsigned
  ELF loader. Probe records `shave_l0_actkernel_unsigned`.
- MoviTools / `moviCompile`: downloaded from Intel artifactory via
  `artifacts/vpuip_2/revisions.json` when `ENABLE_SHAVE_BINARIES_BUILD=ON`. That
  JSON is not in the public clone; `moviCompile` is not on PATH.

OpenVINO issue [31847](https://github.com/openvinotoolkit/openvino/issues/31847):
NPU kernels are closed relative to GPU OpenCL custom ops.

OpenVINO 2026 does not change that public boundary. Its custom-operation extension
defines graph semantics, shape inference, serialization, and optional CPU
`evaluate`; it is not a target-specific NPU kernel registration API. The 2026
compiler-in-plugin path may shorten the firmware/compiler release loop, but it
still compiles supported OpenVINO graphs to the proprietary NPU format. Treat an
NPU custom op as unsupported until the compiler accepts it and profiling proves
device execution. References: [custom operations](https://docs.openvino.ai/2026/documentation/openvino-extensibility/custom-openvino-operations.html),
[NPU plugin design](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md).

## How a new SHAVE kernel would land

1. SHAVE C in this directory (`adc.c`, `topk.c`) — host-linked oracles today.
2. `npu_compiler` descrip + `moviCompile -mcpu=3720xx` → ELF32 with `.text` at
   `0x1d000000` and `.arg.data` at `0x1e000000` (see `shave_kernel.ld`).
3. Register as `VPU.SW.Kernel` (see `npu_compiler/src/vpux_compiler/docs/sw_layer_enabling.md`).
4. Compiler embeds `.text` into a graph ELF64; firmware loads the **graph**, not
   a loose ELF.

Until MoviTools is present, custom work stays on Intel’s existing SHAVE library
via OpenVINO graphs:

- PQ ADC: Gather + ReduceSum
- Hamming: GreaterEqual + LogicalXor + ReduceSum
- Lp: Subtract + Abs + Power + ReduceSum

`PERF_COUNT` shows ActShave + DPU on those graphs. A new SHAVE C ELF still needs
MoviTools.
