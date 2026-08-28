# Toolchain pins

| Component | Windows | Linux |
|---|---|---|
| Host compiler | MSVC 14.44 (VS 2022 Build Tools) or clang-cl | GCC 11+ / clang 16+ |
| CMake | 3.20+ (VS bundle or distro) | 3.20+ |
| SYCL / iGPU | Intel oneAPI DPC++ **2025.1.1** after elevated Base Toolkit 2025.1.3.8. On Windows Ninja, use **`icx`** for both C and CXX (`icpx` is GNU-like and CMake feeds it MSVC flags). In one `cmd.exe` session, call `VsDevCmd.bat -arch=amd64`, then `setvars.bat intel64`, then run `cmake -DOVVS_WITH_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx`. Both environments are required for Intel runtime and Windows SDK link libraries. Fallback: `intel/llvm` nightly `clang++ -fsycl` (`%USERPROFILE%\intel\sycl-nightly`). `sycl-ls` / ovVS tests see `Intel(R) Graphics` via Level Zero. | same + intel-compute-runtime |
| NPU | Measured build: OpenVINO **2025.3.0** runtime/plugin + Intel NPU driver **32.0.100.4841**. The installed 2025.3 headers expose `ZeroContext`, `create_l0_host_tensor`, shared-buffer wrapping, and mapped-file tensors; use a current-stack capability probe before an isolated 2026 upgrade. Compiler-in-plugin, CPU-address import, and strided IO remain upgrade/support probes, not current performance evidence. | OpenVINO + linux-npu-driver; pin exact versions when device CI lands. |
| Python | 3.10–3.13 | 3.10–3.13 |

Optional CMake flags:

```
-DOVVS_WITH_OPENVINO=ON
-DOVVS_WITH_SYCL=ON
-DOVVS_WITH_MKL=ON   # oneAPI MKL 2025.1 GPU GEMM + LAPACKE gesvd/syev; Windows Ninja uses icx
```

CPU-only builds are supported. Device-sensitive cases use the `ovvs_tests` executable and feature-detect hardware; three bounded GPU scale cases now have separate `gpu;scale` CTest entries with resource locks and device-absence skips. Dedicated full NPU, GPU, and HETERO device CI, especially on Linux, remains backlog B18. A device lane may skip only when its device is absent, not when compilation fails.
