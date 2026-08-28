# Toolchain pins

| Component | Windows | Linux |
|---|---|---|
| Host compiler | MSVC 14.44 (VS 2022 Build Tools) or clang-cl | GCC 11+ / clang 16+ |
| CMake | 3.20+ (VS bundle or distro) | 3.20+ |
| SYCL / iGPU | Intel oneAPI DPC++ **2025.1.1** after elevated Base Toolkit 2025.1.3.8. On Windows Ninja, use **`icx`** for both C and CXX (`icpx` is GNU-like and CMake feeds it MSVC flags). In one `cmd.exe` session, call `VsDevCmd.bat -arch=amd64`, then `setvars.bat intel64`, then run `cmake -DOVVS_WITH_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx`. Both environments are required for Intel runtime and Windows SDK link libraries. Fallback: `intel/llvm` nightly `clang++ -fsycl` (`%USERPROFILE%\intel\sycl-nightly`). `sycl-ls` / ovVS tests see `Intel(R) Graphics` via Level Zero. | same + intel-compute-runtime |
| NPU | Measured build: OpenVINO **2025.3.0** runtime/plugin + Intel NPU driver. OpenVINO 2026 remote-tensor/compiler-in-plugin features are an isolated upgrade experiment, not current evidence. | OpenVINO + linux-npu-driver; pin exact versions when device CI lands. |
| Python | 3.10–3.13 | 3.10–3.13 |

Optional CMake flags:

```
-DOVVS_WITH_OPENVINO=ON
-DOVVS_WITH_SYCL=ON
-DOVVS_WITH_MKL=ON   # oneAPI MKL 2025.1 GPU GEMM + LAPACKE gesvd/syev; Windows Ninja uses icx
```

CPU-only builds are supported. Device-sensitive cases currently live inside the CPU-labelled `ovvs_tests` executable and feature-detect hardware; separate `npu` / `gpu` / `hetero` CTest lanes remain backlog B18. Once split, a lane may skip only when its device is absent, not when compilation fails.
