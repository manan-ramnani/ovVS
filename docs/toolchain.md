# Toolchain pins

| Component | Windows | Linux |
|---|---|---|
| Host compiler | MSVC 14.44 (VS 2022 Build Tools) or clang-cl | GCC 11+ / clang 16+ |
| CMake | 3.20+ (VS bundle or distro) | 3.20+ |
| SYCL / iGPU | Intel oneAPI DPC++ (`icpx`) **or** `intel/llvm` nightly `clang++ -fsycl`. oneAPI Base Toolkit 2025.1.3.8 winget needs admin (UAC `0x800704c7`); user-scope winget has no installer; silent install to a user dir still requires elevation. Working path on this host: `gh release download nightly-2026-08-18 --repo intel/llvm --pattern sycl_windows.tar.gz` then `cmake -DIOVS_WITH_SYCL=ON -DCMAKE_CXX_COMPILER=<nightly>/bin/clang++.exe`. `sycl_hello` and ioVS tests see `Intel(R) Graphics` via Level Zero. | same + intel-compute-runtime |
| NPU | OpenVINO with NPU plugin + Intel NPU driver | OpenVINO + linux-npu-driver |
| Python | 3.10–3.13 | 3.10–3.13 |

Optional CMake flags:

```
-DIOVS_WITH_OPENVINO=ON
-DIOVS_WITH_SYCL=ON
```

CPU-only builds are supported and run the full correctness suite. Device tests are labeled `npu` / `gpu` and skip only when the device is absent.
