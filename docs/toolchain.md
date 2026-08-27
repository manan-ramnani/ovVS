# Toolchain pins

| Component | Windows | Linux |
|---|---|---|
| Host compiler | MSVC 14.44 (VS 2022 Build Tools) or clang-cl | GCC 11+ / clang 16+ |
| CMake | 3.20+ (VS bundle or distro) | 3.20+ |
| SYCL / iGPU | Intel oneAPI DPC++ **2025.1.1** after elevated Base Toolkit 2025.1.3.8. On Windows Ninja, use **`icx`** for both C and CXX (`icpx` is GNU-like and CMake feeds it MSVC flags). `cmake -DOVVS_WITH_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx` after `setvars.bat`. Fallback: `intel/llvm` nightly `clang++ -fsycl` (`%USERPROFILE%\intel\sycl-nightly`). `sycl-ls` / ovVS tests see `Intel(R) Graphics` via Level Zero. | same + intel-compute-runtime |
| NPU | OpenVINO with NPU plugin + Intel NPU driver | OpenVINO + linux-npu-driver |
| Python | 3.10–3.13 | 3.10–3.13 |

Optional CMake flags:

```
-DOVVS_WITH_OPENVINO=ON
-DOVVS_WITH_SYCL=ON
-DOVVS_WITH_MKL=ON   # oneAPI MKL 2025.1 GPU GEMM + LAPACKE gesvd/syev; Windows Ninja uses icx
```

CPU-only builds are supported and run the full correctness suite. Device tests are labeled `npu` / `gpu` and skip only when the device is absent.
