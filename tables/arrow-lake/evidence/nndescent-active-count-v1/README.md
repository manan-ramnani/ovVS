# NN-Descent active-count experiment evidence manifest

These files are the immutable raw inputs for
[`../../nndescent-active-count-v1.md`](../../nndescent-active-count-v1.md).
They were copied byte-for-byte from `out/bench/nnd-sync-active-count-v1/`.
The experiment was rejected and its production-source changes were restored.

| Artifact | Scope | Bytes | SHA-256 |
|---|---|---:|---|
| `baseline-sift100k-r1.json` | Invalid setup: ovVS unavailable because the frozen DLL dependency closure was incomplete | 15,474 | `4DBC952D687E5A8B40C6B4716EBFF4A6F8755A40AFBD637B648504E78A21A72E` |
| `baseline-sift100k-r2.json` | Invalid setup: same unavailable baseline lane after a second incomplete closure attempt | 15,455 | `DF29BCC8A0C1C0EA62689068E5477DEEA7470162D713AFD8AAFB2CF5EEDE2024` |
| `baseline-sift100k-r3.json` | Valid current-parent SIFT100K admission run | 36,666 | `CF2468F1D1FA3C897315FC1F3774A0AA62A46676CF80041D064B2313BB640DE9` |
| `candidate-sift100k-r1.json` | Valid candidate SIFT100K admission run | 36,696 | `23A7AC4884468BE379C5C258D0693B1532670D59825FA4E5D404DEA37A1DD0BC` |
| `baseline-sift1m-r1.json` | Valid current-parent SIFT1M gate | 36,395 | `11CB6CA4F4285BF7F96258D63B50124634A28DED45BCA1B532087F6C2BD7D3E9` |
| `candidate-sift1m-r1.json` | Valid candidate SIFT1M gate | 36,434 | `05C16A04B4EBB1832B8E4B09B645AC7543C79B900C1C6E5E004B309981E6E137` |
| `candidate-dirty.patch` | Rejected diagnostic source patch; not production code | 53,719 | `7DF21A7C37FA3F94E789AE80AF6A50C908795E31D2B24B264FDE2BA961EFC417` |

Producer identity:

- Source base: `e2f8a04`, branch `main`. The immutable harness metadata marks
  every run `dirty:true`; the loaded-DLL fingerprint, not the Git field alone,
  distinguishes the clean-parent build from the dirty candidate.
- Parent DLL: 1,134,592 bytes, SHA-256
  `1828C050A447B47E3690511A8CB2DC88C57D3095A2A20408DD13B6CF6F95C4C5`.
- Candidate DLL: 1,138,176 bytes, SHA-256
  `19D2EDE4A37FFCAD7875725CA3656CFA80E43591835B55C0CEA0E5628BC89702`.
- Both DLLs were loaded from the explicit path recorded in each valid JSON and
  passed the harness path/fingerprint check. The local binaries and runtime
  closure are intentionally not tracked; their digests retain identity.

The two `baseline-sift100k-r1/r2` files are retained because unavailable lanes
are part of the experiment history. They are not included in any comparison.
The valid SIFT100K and SIFT1M files each contain two successful lanes and zero
failed, skipped, timed-out, or unavailable lanes. Their overall completion is
still `partial`: package energy was disabled and the prefix/gate-only modes are
not the full B1 curve.

The SIFT1M candidate ran before the fresh parent process; this is one process
per DLL, not an interleaved or repeated promotion study.
It has unchanged recall and exact iteration/final-count telemetry, but no
cross-DLL raw graph fingerprint. That limitation does not weaken the rejection:
the candidate missed the construction admission gate by regressing complete
build wall. Absolute `out/bench/...` paths in the JSON are historical
producer-time paths and were not rewritten because that would invalidate the
digests.

The retained patch touches only `backend_gpu.cpp` and `test_neighbors.cpp` and
applies to the `e2f8a04` index, but it contains substantial formatting churn.
It is a diagnostic snapshot, not an integration-ready or cherry-pickable
change. The contemporaneous patch and DLL hashes preserve provenance; they are
not a reproducible-build proof linking source bytes cryptographically to the
binary.
