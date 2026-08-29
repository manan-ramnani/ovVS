# NN-Descent target-owned head-reset evidence manifest

These files are the immutable raw inputs for
[`../../nndescent-heads-reset-v1.md`](../../nndescent-heads-reset-v1.md). The
candidate was rejected and its production-source and test changes were
restored.

| Artifact | Scope | Bytes | SHA-256 |
|---|---|---:|---|
| `baseline-sift100k-r1.json` | Valid current-production SIFT100K admission run | 36,588 | `07587EEC1CC866013AC154013B69D2C65DF39770094FB5EC812FD8F3D93C2B6E` |
| `candidate-sift100k-r1.json` | Valid candidate SIFT100K admission run | 36,736 | `9ED55C53B8396740BCFFD68A22ED9D44CDF309C01BBD578CB58CDB43D4C33C27` |
| `baseline-sift1m-r1.json` | Valid current-production SIFT1M gate | 36,387 | `A3EBA1161CB83924F7E67393FF78DF06AE79970F7CDA5218D5908849BC00B524` |
| `candidate-sift1m-r1.json` | Valid candidate SIFT1M gate | 36,426 | `3A7322E94DD328461D7DF5324BDEB9EDFD9C15FB21EDDE190B858C17D54BF011` |
| `candidate-sift100k-invalid-runtime-closure.json` | Excluded first attempt: ovVS unavailable because the frozen DLL directory lacked its Windows runtime dependencies | 15,460 | `8EB99D4AC6DF975F6E9B0133E2B43B68CDB260790C3BE9FCBC6A91338FDA4CE9` |
| `candidate-dirty.patch` | Rejected diagnostic source/test patch; not production code | 3,586 | `E596BCF0AFFFE33DD4AE787F0B52731F622E60CDDB49CA2E42C093973BA39BAD` |

Producer identity:

- Source base: `6bce58e`, branch `main`. The patch applies cleanly and touches
  only `backend_gpu.cpp` and `test_neighbors.cpp`.
- Parent DLL: source-equivalent frozen production build, 1,134,592 bytes,
  SHA-256
  `1828C050A447B47E3690511A8CB2DC88C57D3095A2A20408DD13B6CF6F95C4C5`.
  Commits after its `e2f8a04` label changed documentation/evidence only; both
  product files were verified identical to `6bce58e` before the experiment.
- Candidate DLL: 1,134,080 bytes, SHA-256
  `EF86BE07668B01C81D3C4ED218A3C68570D9159E9F9F64F20C4ABADE1E135F58`.
- The candidate directory was repaired with the same local runtime closure
  before the valid run. Every valid artifact recorded
  `matches_requested_path=true` and the expected DLL hash and size. Binaries
  and runtime dependencies remain local and untracked.

All four valid JSON files contain two successful lanes and zero failed,
skipped, timed-out, or unavailable lanes. Completion remains `partial` because
package energy was disabled and the prefix/gate-only modes are not the full B1
curve. The candidate ran before the parent at both sizes; each file contains
one build sample and five search timing passes, not a repeated promotion study.

The excluded first candidate attempt is retained rather than silently removed.
Its hnswlib lane succeeded, but the ovVS lane was unavailable before execution
because a copied DLL did not yet have a loadable dependency closure. It is not
used in any metric.

The JSON has no raw NN-Descent or serialized native CAGRA graph digest.
Matching recall, iterations, final changed/pending counts, and successful
output validation are claimed; bitwise cross-DLL graph identity is not.
Absolute producer-time paths were not rewritten because that would invalidate
the artifact digests.
