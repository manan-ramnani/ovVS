# GPU root-group canary V1 evidence manifest

This immutable raw file supports
[`../../gpu-root-group-canary-v1.md`](../../gpu-root-group-canary-v1.md).

Producer identity:

- Clean source and runner revision:
  `6deb17341995be18a0d2ff0d273f3989a6c05e8b`.
- Executable: `ovvs_gpu_root_group_canary.exe`, 112,640 bytes, SHA-256
  `1BB0C51DE5F2527EBB81ED9E23C5520BB3A20DD6FAA44C7FAD2212AB249604AD`.
- oneAPI compiler: 2025.1.1; selected device/runtime identity is embedded in
  the JSON.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `gpu-root-group-canary-6deb173.json` | 506 | `C0937BCD52764E1D4218582761D654105CB8B738BF0FB2FF59F77937312A7BB1` |

The producer exited 77 after the named-kernel capacity query returned one
cooperative workgroup versus four required. It did not submit the root-barrier
kernel and did not fall back. This is capability-only negative evidence, not a
search result or acceleration claim.
