# CAGRA search-effort V1 evidence manifest

The JSON file in this directory is the immutable raw input for [`../../cagra-search-v1.md`](../../cagra-search-v1.md) and the current curve section in [`../../bench-recall-qps.md`](../../bench-recall-qps.md). It is retained byte-for-byte from the benchmark output.

Producer identity:

- Git commit: `acd67e231e8993ca0b2e041f7998a5430d3a3139`, branch `main`, clean worktree.
- Loaded DLL: `C:\Personal\ioVS\build-icpx\bin\ovvs.dll`.
- DLL SHA-256: `C4EE36A1F7540726E7686C138190D201BFB825112EDE9FDD6249048FAB319F8F`.
- DLL size: 1,131,520 bytes.
- Dataset SHA-256: `DD6F0A6ED6B7EBB8934680F861A33ED01FF33991EAEE4FD60914D854A0CA5984`.

| Artifact | SHA-256 |
|---|---|
| `cagra-effort-sift1m-acd67e231e89-r1.json` (61,941 bytes) | `F1D7B043BC4E574779AF5A22C1D58D7624E58EBE43D091C79E878F7425C66BE1` |

The benchmark invocation completed with exact validated SIFT1M truth, two sequential isolated successful lane processes, zero failed/skipped/timed-out/unavailable lanes, one warmup and five measured passes per point, and successful whole-package energy sampling. It has no in-run clock, thermal, utilization, or background-load trace. It is sufficient to diagnose same-index recall versus traversal effort, but it is not a repeated performance-promotion result and makes no acceleration claim.
