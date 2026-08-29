# NN-Descent producer-to-copy event evidence manifest

These files are the immutable raw inputs for
[`../../nndescent-producer-copy-event-v1.md`](../../nndescent-producer-copy-event-v1.md).
Both event-chain variants were rejected and production source/tests were
restored.

| Artifact | Scope | Bytes | SHA-256 |
|---|---|---:|---|
| `handler-baseline-sift100k-r1.json` | Valid production SIFT100K admission parent for the handler-copy diagnostic | 36,584 | `E714C31305B0DAC30676B362E169662D1ABC44ABD21CD1DB1DD60FA47DB3AD26` |
| `handler-candidate-sift100k-r1.json` | Valid handler-copy SIFT100K diagnostic; rejected before SIFT1M | 36,765 | `1A329FBCD0CBD8FEF9084833765385BB0A0F2E463DFAD0E58D132EED901966AA` |
| `direct-baseline-sift100k-r1.json` | Valid production SIFT100K admission parent for the direct-copy candidate | 36,708 | `F0FA9EDA57FF384577BDC247D8D08D6D5A8C5C2BEE463C2471665BFD14F6AFE4` |
| `direct-candidate-sift100k-r1.json` | Valid direct-copy SIFT100K admission candidate | 36,794 | `6FDC3ECFB19B9EB088DFBA0DF16AAE86B0B3915FE5F278FB15FA628687390CC2` |
| `direct-baseline-sift1m-r1.json` | Valid production SIFT1M gate | 36,378 | `ABFFDA6CBF6FE74F8A62D19620A97F47A753ABB94CA7684EBC283B839DD0F0B8` |
| `direct-candidate-sift1m-r1.json` | Valid direct-copy SIFT1M gate | 36,481 | `6E23B2AEF5F7ABA2038B09CE978F0CC57BBD9E35ECB7C451742CE9225BC6719E` |
| `candidate-direct-dirty.patch` | Rejected direct-copy source/test patch; not production code | 4,045 | `5EAD8779F2712FB9260AC72B247BF6907AC4EE0BBD232B22EC2A7A4BA4A54BF1` |

Producer identity:

- Source base: `7a7ee57`, branch `main`. The retained direct-copy patch applies
  cleanly and touches only `backend_gpu.cpp` and `test_neighbors.cpp`.
- Parent DLL: source-equivalent frozen production build, 1,134,592 bytes,
  SHA-256
  `1828C050A447B47E3690511A8CB2DC88C57D3095A2A20408DD13B6CF6F95C4C5`.
  Commits after its `e2f8a04` label changed documentation/evidence only; both
  product files were verified identical to `7a7ee57` before the experiment.
- Handler-copy candidate DLL: 1,137,664 bytes, SHA-256
  `78955CFA6FA031B77027EEDB4C1DE7E46E0AC14C258A3A86488A5A643B187DD9`.
- Direct-copy candidate DLL: 1,135,616 bytes, SHA-256
  `3E7C41346D4F89FCE3C8120575255564F75AE72593B65EE143E4173CC819B91E`.
- Every artifact recorded `matches_requested_path=true` for its ovVS lane and
  the expected DLL hash and size. Binaries and runtime closure remain local and
  untracked.

All six JSON files contain two successful lanes and zero failed, skipped,
timed-out, or unavailable lanes. Completion remains `partial` because package
energy was disabled and the prefix/gate-only modes are not the full B1 curve.
The candidate ran before the parent in all three pairs; each file contains one
build sample and five search timing passes, not a repeated promotion study.

The handler-copy variant used an explicit `q.submit` / `h.depends_on` /
`h.memcpy` command group. The direct-copy patch uses the standard
`queue::memcpy(..., depEvent)` overload and is the only variant retained as a
source patch because it superseded the handler form before the full-scale run.

The JSON has no raw NN-Descent or serialized native CAGRA graph digest.
Matching recall, iterations, final changed/pending counts, and successful
output validation are claimed; bitwise cross-DLL graph identity is not.
Absolute producer-time paths were not rewritten because that would invalidate
the artifact digests.
