# ovVS

GitHub remote for **ovVS**: a [NVIDIA cuVS](https://github.com/NVIDIA/cuvs)-shaped vector search and clustering library targeting Intel client SoCs (NPU + Arc iGPU).

```text
git clone https://github.com/manan-ramnani/ovVS.git
```

The C ABI, headers, and shared library are still named `ovvs` (`include/ovvs/`, `libovvs` / `ovvs.dll`). **0.2.0** ships the cuVS-shaped API surface and Arrow Lake bakeoffs. It is not yet a hardware-accelerated cuVS equivalent: AUTO dense primitives are CPU-dominated on this SKU, while the wired CAGRA/IVF accelerator paths do not beat hnswlib/FAISS at published sizes.

- Spec: `AGENTS.md` and `.claude/plans/2026-08-20-ovvs-intel-cuvs-equivalent.md`
- Remaining work: `.claude/backlog.md`
- Device split: `docs/hw-split.md`

Build (Windows Ninja + oneAPI): `icx` for both C and C++, `-DOVVS_WITH_SYCL=ON`. Set `OVVS_LIBRARY` to the built `ovvs.dll` before importing the Python package.
