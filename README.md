# ovVS

GitHub remote for **ovVS**: a [NVIDIA cuVS](https://github.com/NVIDIA/cuvs)-shaped vector search and clustering library targeting Intel client SoCs (NPU + Arc iGPU).

```text
git clone https://github.com/manan-ramnani/ovVS.git
```

The C ABI, headers, and shared library are still named `ovvs` (`include/ovvs/`, `libovvs` / `ovvs.dll`). Spec: `AGENTS.md` and `.claude/plans/2026-08-20-ovvs-intel-cuvs-equivalent.md`.

Build (Windows Ninja + oneAPI): `icx` for both C and C++, `-DOVVS_WITH_SYCL=ON`. Set `OVVS_LIBRARY` to the built `ovvs.dll` before importing the Python package.
