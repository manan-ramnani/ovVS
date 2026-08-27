# ovVS

GitHub remote for **ioVS**: a [NVIDIA cuVS](https://github.com/NVIDIA/cuvs)-shaped vector search and clustering library targeting Intel client SoCs (NPU + Arc iGPU).

```text
git clone https://github.com/manan-ramnani/ovVS.git
```

The C ABI, headers, and shared library are still named `iovs` (`include/iovs/`, `libiovs` / `iovs.dll`). Spec: `AGENTS.md` and `.claude/plans/2026-08-20-iovs-intel-cuvs-equivalent.md`.

Build (Windows Ninja + oneAPI): `icx` for both C and C++, `-DIOVS_WITH_SYCL=ON`. Set `IOVS_LIBRARY` to the built `iovs.dll` before importing the Python package.
