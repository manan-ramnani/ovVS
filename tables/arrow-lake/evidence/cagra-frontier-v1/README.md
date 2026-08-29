# CAGRA frontier V1 evidence manifest

These files are the immutable raw inputs for
[`../../cagra-frontier-v1.md`](../../cagra-frontier-v1.md). Five new benchmark
JSON files and the final exactness fingerprint were copied byte-for-byte from
their producer outputs. The first baseline process and retained graph already
live in [`../cagra-cached-worst-v1/`](../cagra-cached-worst-v1/README.md) and
are referenced instead of duplicated.

Producer identity:

- Baseline implementation: clean `c7e9c04753a20a14802266513ac24bd00752c7ff`,
  DLL SHA-256
  `7999DF05575A7CAB252BB09B1A5536FF9439F608CADE4EC9644B421A878E45C0`.
- Candidate implementation: clean `892252bf0811504d6c617904c5d8034e616d7b5d`,
  DLL SHA-256
  `68A0C89172FD583142858892C570BD0CE4ADD89713C94410C3F1D96B771AB67F`.
- Dataset SHA-256:
  `DD6F0A6ED6B7EBB8934680F861A33ED01FF33991EAEE4FD60914D854A0CA5984`.

The two fresh baseline artifacts were produced by loading the frozen baseline
DLL while the runner worktree was clean at the candidate revision. Their Git
metadata therefore names the runner checkout; `implementation_metadata` binds
the executed baseline binary. The retained first baseline artifact is clean at
the baseline revision itself.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Retained baseline: `../cagra-cached-worst-v1/cagra-cached-worst-candidate-c7e9c04753a2-i3.json` | 61,903 | `03C5F65D8598C6CEAFCBDE1356C16B03A0C64CA2C9066C21F63741C713B1CC5C` |
| `cagra-frontier-baseline-c7e9c04753a2-i1.json` | 62,064 | `78BC81C70F7CEB67B5B5C7A8298C04C665C16688B79486FC7FF1CF536DFB9C68` |
| `cagra-frontier-baseline-c7e9c04753a2-i2.json` | 62,067 | `69DF38A8EA50D4195AC345F6B4E8C591B8EEBBFD5EC808F15FA4DC2EB3F1D78B` |
| `cagra-frontier-candidate-892252bf0811-r1.json` | 61,832 | `33C31C4FB59194B2A0019F649DE67D16A4DF7830EC2E96C972FD162ADCAC9E93` |
| `cagra-frontier-candidate-892252bf0811-r2.json` | 61,840 | `3046B8250D324277BC28DDB55DE0FC4267D8408436EE96F00076056AC31766D2` |
| `cagra-frontier-candidate-892252bf0811-r3.json` | 61,958 | `271B40276CD37C86E2A51CC80E84D23F33502BBC7A9F5BACACA7487D1F52E9E9` |
| `cagra-frontier-exactness-candidate.json` | 1,898 | `60CB3703498C854CB6F64F61B1618C3B45BFC195759300638C77A5C805D79E97` |

The chronological sequence was the retained baseline followed by five new
alternating processes: candidate, baseline, candidate, baseline, candidate.
Every process completed two isolated lanes and all nine requested points with
exact HDF5 truth, one warmup, five measured passes, package-energy sampling,
and no failed, skipped, timed-out, or unavailable cell. hnswlib remained at
`M=16`, `ef_construction=200`, 20 threads, and
`ef={32,64,128,256}`. Every CAGRA point reported direct GPU walks and zero
explicit index uploads.

The final candidate loaded the retained 268,836-byte graph, SHA-256
`43E9000EC667BE2D3B4A1075F4116EED1F30B512556C83FC07587135C42F6D25`.
Its IDs and float-distance bytes match the frozen baseline fingerprint at
`32/1`, `64/2`, and `128/4`; it reports three direct walks and zero uploads.
This bounded bitwise seam complements, but does not replace, the SIFT1M recall
and complete-wall evidence.

Absolute `C:\Personal\ioVS\out\...` paths embedded in the immutable files are
producer-time paths. The files were not rewritten because doing so would
invalidate their digests.
