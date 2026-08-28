# Contributing

Read `AGENTS.md`, `.claude/plans/2026-08-20-ovvs-intel-cuvs-equivalent.md`, and `.claude/backlog.md` (what is still open).

- Algorithms call `ovvs::prim`, never OpenVINO or SYCL directly.
- Do not drop a cuVS feature. Use the punch-through ladder.
- Do not add more C ABI algorithm names; v0.2 already has them. Work the backlog critical path (SIFT1M harness, CAGRA T13.4, IVF-PQ/RaBitQ search rewrite).
- Add a test that calls the shipped C ABI, not a reimplementation.
- CPU oracles must keep working without NPU/iGPU.
