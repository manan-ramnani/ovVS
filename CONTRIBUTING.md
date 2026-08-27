# Contributing

Read `AGENTS.md` and `.claude/plans/2026-08-20-ovvs-intel-cuvs-equivalent.md`.

- Algorithms call `ovvs::prim`, never OpenVINO or SYCL directly.
- Do not drop a cuVS feature. Use the punch-through ladder.
- Add a test that calls the shipped C ABI, not a reimplementation.
- CPU oracles must keep working without NPU/iGPU.
