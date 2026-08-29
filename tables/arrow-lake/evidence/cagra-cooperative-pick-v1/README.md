# CAGRA cooperative-pick V1 evidence manifest

These immutable raw files support
[`../../cagra-cooperative-pick-v1.md`](../../cagra-cooperative-pick-v1.md).
They were copied byte-for-byte from their producer outputs.

Producer identity:

- Runner checkout: clean `c9bb312e11cd5f626273ad5f86fdd88dc6682516`.
- Frozen serial-baseline source: clean parent
  `f340eafa597beabb188536ebc2afd7c537a931c0`; this is the pre-change
  serial-selection source used to build the retained baseline DLL.
- Frozen serial-baseline DLL SHA-256:
  `0BA0067B2920C1833E0F856D1B3998A81245F87671B275BF4D11B76A21C8B6A5`.
- Candidate DLL SHA-256:
  `228A0E49AA40C4345094E8DC54E136C7A4B8894CF723CC445DA535A234BC8061`.
- SIFT1M dataset SHA-256:
  `DD6F0A6ED6B7EBB8934680F861A33ED01FF33991EAEE4FD60914D854A0CA5984`.

All six benchmark processes completed two isolated lanes and all nine points
with exact HDF5 truth, one warmup, five measured passes, package-energy
sampling, and no failed, skipped, timed-out, or unavailable cell. The sequence
was candidate/baseline/candidate/baseline/candidate/baseline. The baseline
runs use the clean candidate checkout as the runner but bind the executed
serial implementation by its loaded-library fingerprint.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `cagra-cooperative-pick-baseline-c9bb312-r1.json` | 61,984 | `E0FB5CDCFBE4F7E6EA9287705414017B78130063DC9A4FCC90772DD82CF5A909` |
| `cagra-cooperative-pick-baseline-c9bb312-r2.json` | 62,078 | `3E1266013EF8CB68E2F312EDE649265FB95567015776394AC2A75E16E5142183` |
| `cagra-cooperative-pick-baseline-c9bb312-r3.json` | 62,045 | `408ABBB838482D3CC2AD2EFF5C3DAE11C3CF0E395F1CA7A5DBB7214986F55C3D` |
| `cagra-cooperative-pick-candidate-c9bb312-r1.json` | 62,087 | `91BEDAD2FBB273E938E3A804E825BB9CB116F96D37D86B0C7728FD4E4E81162D` |
| `cagra-cooperative-pick-candidate-c9bb312-r2.json` | 62,010 | `57EBFE347E735744B006DBF76DE55166C3544056CC47936A3C708616C0D6C0A9` |
| `cagra-cooperative-pick-candidate-c9bb312-r3.json` | 62,018 | `1D5D2968C84FD08A6B194E7AACE8EDDAC2B6320D3DFD4DF69B8B47621568E912` |
| `cagra-cooperative-pick-exactness-candidate.json` | 1,898 | `D93FF83186F802F4513000D21574D171AC2D0B3CBA78F793E392375DFC26CAB3` |

hnswlib remained at version 0.8.0, `M=16`, `ef_construction=200`, 20
threads, seed 7, and `ef={32,64,128,256}`. Every CAGRA point reported direct
GPU walks and zero explicit index-upload calls or bytes.

The exactness artifact loads the retained 268,836-byte graph, SHA-256
`43E9000EC667BE2D3B4A1075F4116EED1F30B512556C83FC07587135C42F6D25`.
Its ID and float-distance hashes match the frozen frontier fingerprint at
`32/1`, `64/2`, and `128/4`; it reports three direct walks and zero uploads.
The reference is
[`../cagra-frontier-v1/cagra-frontier-exactness-candidate.json`](../cagra-frontier-v1/cagra-frontier-exactness-candidate.json),
SHA-256
`60CB3703498C854CB6F64F61B1618C3B45BFC195759300638C77A5C805D79E97`.

Absolute `C:\Personal\ioVS\out\...` paths embedded in the JSON files are
producer-time paths. The files were not rewritten because doing so would
invalidate their digests.
