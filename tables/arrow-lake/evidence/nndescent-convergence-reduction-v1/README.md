# NN-Descent convergence-reduction evidence manifest

These files are the immutable raw inputs for
[`../../nndescent-convergence-reduction-v1.md`](../../nndescent-convergence-reduction-v1.md).
They were copied byte-for-byte from
`out/bench/nnd-convergence-reduction-v1/`. The candidate was rejected and its
production-source changes were restored.

| Artifact | Scope | Bytes | SHA-256 |
|---|---|---:|---|
| `baseline-sift100k-r1.json` | Valid current-production SIFT100K admission run | 36,703 | `6473FF8AD79927EC00FF994F22639301A3EF93F0E4EEC23653A747F9B7082AA6` |
| `candidate-sift100k-r1.json` | Valid candidate SIFT100K admission run | 36,767 | `C2019EFAAEFF8A0C0DA92742AA091884B5CD6806F7134EE3BA6F4FACAE0927E6` |
| `baseline-sift1m-r1.json` | Valid current-production SIFT1M gate | 36,412 | `4F9C9CF65432DBB7E9C75C47E7C2791435E044A0F2B7926DC69B6E6DEF269C53` |
| `candidate-sift1m-r1.json` | Valid candidate SIFT1M gate | 36,492 | `6F8327C27A729B8619543EE99434F5FA63D6D132B38E7AEE49E46A2147810522` |
| `candidate-dirty.patch` | Rejected diagnostic source patch; not production code | 17,847 | `A04BEADBA5E3A3B7B46345111390108E4945F6D8E4BF9A3D381E1E380509701B` |

Producer identity:

- Source base: `b1a777c`, branch `main`; the candidate patch applies cleanly
  to that index and touches only `backend_gpu.cpp` and `test_neighbors.cpp`.
- Parent DLL: the source-equivalent frozen `e2f8a04` production build,
  1,134,592 bytes, SHA-256
  `1828C050A447B47E3690511A8CB2DC88C57D3095A2A20408DD13B6CF6F95C4C5`.
  Commits `e2f8a04..b1a777c` changed documentation/evidence only; the two
  production files were verified identical before and after the experiment.
- Candidate DLL: 1,144,320 bytes, SHA-256
  `E904A8688BC98AAC8F8E6080AB9C3CF4FD15766ADBB9D337D7E0D7766516045E`.
- Every valid artifact loaded its explicit DLL path and recorded the matching
  SHA-256 and size. Binaries/runtime closure remain local and untracked.

All four JSON files contain two successful lanes and zero failed, skipped,
timed-out, or unavailable lanes. Completion remains `partial` because package
energy was disabled and the prefix/gate-only modes are not the full B1 curve.
The candidate ran before the parent at both sizes; each artifact contains one
build sample and five search timing passes, not a repeated promotion study.

The JSON has no raw NN-Descent or serialized CAGRA graph digest. Matching
recall, iterations, final changed/pending counts, and successful output
validation are claimed; bitwise cross-DLL graph identity is not. Absolute
`out/bench/...` paths are retained as producer-time provenance and were not
rewritten because that would invalidate the digests.
