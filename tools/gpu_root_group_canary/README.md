# GPU root-group capability canary

This standalone, default-off experiment checks whether the selected Intel
integrated Level Zero GPU can cooperatively run the four workgroups needed by
the proposed small-batch CAGRA owner/helper route. It is not linked into
`libovvs` and does not route or benchmark vector search.

The named kernel is queried at four workgroups, 128 work-items per group, and
2,640 bytes of local memory per group. This is the conservative admission
footprint of the current `128/4` CAGRA walk. A capacity below four exits before
launch. An admitted kernel must then pass five launches with converged
device-scope root barriers and exact host validation.

Build with `OVVS_WITH_SYCL=ON` and `OVVS_BUILD_ESCAPE_EXPERIMENTS=ON`, then run:

```text
cmake --build build-icpx --target ovvs_gpu_root_group_canary
ctest --test-dir build-icpx -R ^gpu_root_group_canary$ -V
```

Exit code 0 means the canary passed. Exit code 77 means the compile-time API or
selected device resources are absent, or the named-kernel query reports fewer
than four workgroups. Exit code 1 means bundle/query execution, launch, or
validation failed. Even a pass would not admit production routing:
the eventual CAGRA kernel must repeat the named-kernel query with its real
register and local-memory footprint.
