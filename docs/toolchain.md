# Toolchain pins

| Component | Windows | Linux |
|---|---|---|
| Host compiler | MSVC 14.44 (VS 2022 Build Tools) or clang-cl | GCC 11+ / clang 16+ |
| CMake | 3.20+ (VS bundle or distro) | 3.20+ |
| SYCL / iGPU | Intel oneAPI DPC++ (`icpx`), Level Zero GPU | same + intel-compute-runtime |
| NPU | OpenVINO with NPU plugin + Intel NPU driver | OpenVINO + linux-npu-driver |
| Python | 3.10–3.13 | 3.10–3.13 |

Optional CMake flags:

```
-DIOVS_WITH_OPENVINO=ON
-DIOVS_WITH_SYCL=ON
```

CPU-only builds are supported and run the full correctness suite. Device tests are labeled `npu` / `gpu` and skip only when the device is absent.
