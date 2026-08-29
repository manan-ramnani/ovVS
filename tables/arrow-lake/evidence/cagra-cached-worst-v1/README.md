# CAGRA cached-worst V1 evidence manifest

These files are the immutable raw inputs for
[`../../cagra-cached-worst-v1.md`](../../cagra-cached-worst-v1.md). The five
new benchmark JSON files and three exactness fingerprints were copied
byte-for-byte from their producer outputs. The first old-source process remains
in [`../cagra-search-v1/`](../cagra-search-v1/README.md) and is referenced rather
than duplicated.

Benchmark producer identity:

- Original old-source process: clean `acd67e231e8993ca0b2e041f7998a5430d3a3139`,
  DLL SHA-256 `C4EE36A1F7540726E7686C138190D201BFB825112EDE9FDD6249048FAB319F8F`.
- Rebuilt old-source processes: clean detached
  `615dd971b8c88a75cbd0f50fdb3e9f87f3cbbefd`, DLL SHA-256
  `BACD5BC9D6071E70518993CFDB52A2C78EBF4C4DFA7C55999EC08715A7150E64`.
  That revision differs from the original only in documentation/evidence files
  relevant to this experiment; the compiled search source is unchanged.
- Candidate processes: clean `c7e9c04753a20a14802266513ac24bd00752c7ff`,
  DLL SHA-256 `7999DF05575A7CAB252BB09B1A5536FF9439F608CADE4EC9644B421A878E45C0`.
- Dataset SHA-256:
  `DD6F0A6ED6B7EBB8934680F861A33ED01FF33991EAEE4FD60914D854A0CA5984`.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Original old-source process: `../cagra-search-v1/cagra-effort-sift1m-acd67e231e89-r1.json` | 61,941 | `F1D7B043BC4E574779AF5A22C1D58D7624E58EBE43D091C79E878F7425C66BE1` |
| `cagra-cached-worst-baseline-615dd971b8c-i1.json` | 62,219 | `3CB24F734675AFC0756939C4C86645545A597BB460559985D531A668AD5FBB5E` |
| `cagra-cached-worst-baseline-615dd971b8c-i2.json` | 62,162 | `22370764E4F756529A87B5D9942F82BC43EBCBEDE2F42C0A8266680E398E95D4` |
| `cagra-cached-worst-candidate-c7e9c04753a2-i1.json` | 61,988 | `203030C11921041AE8EDE225A13B3E320ACDD222438DD19660D3C0F28FDD436C` |
| `cagra-cached-worst-candidate-c7e9c04753a2-i2.json` | 61,931 | `FAFBE621DDBB1D61F878D8E0A53CF0EE9680CD2A82264AAEF1F5861CBBD95451` |
| `cagra-cached-worst-candidate-c7e9c04753a2-i3.json` | 61,903 | `03C5F65D8598C6CEAFCBDE1356C16B03A0C64CA2C9066C21F63741C713B1CC5C` |
| `cagra-cached-worst-baseline-graph.ovvs` | 268,836 | `43E9000EC667BE2D3B4A1075F4116EED1F30B512556C83FC07587135C42F6D25` |
| `cagra-cached-worst-exactness-original.json` | 1,845 | `B769FD3E690457F4CBE61AEFC142B469186FA01F6D925B34C5EB131EC13B9BD2` |
| `cagra-cached-worst-exactness-rebuilt-baseline.json` | 1,867 | `AE86A0D0047072495F2C3C92D89707B9294F7E73C2C7A2E4EC8618983C0E6995` |
| `cagra-cached-worst-exactness-candidate.json` | 1,844 | `D08C635E297EEC7F0D574C5253D2B8743A630A66DBC59F5F7297BB56F226A347` |

Every benchmark process completed both isolated lanes, all nine requested
points, exact HDF5 truth validation, one warmup, five measured passes, and
whole-package energy sampling with no failed, skipped, timed-out, or unavailable
cell. hnswlib remained at `M=16`, `ef_construction=200`, 20 threads, and
`ef={32,64,128,256}`.

The retained 4,200-row graph was produced by the original DLL. All three DLLs
loaded that same graph and emitted the same ID SHA-256 and float-distance-byte
SHA-256 at CAGRA `32/1`, `64/2`, and `128/4`. Each fingerprint reports three
direct GPU walks and zero explicit index uploads. The fixture is a bounded
bitwise seam, not a substitute for the full SIFT1M performance and recall gate.

Absolute `C:\Personal\ioVS\out\...` paths embedded in the immutable files are
producer-time paths. The files were not rewritten because doing so would
invalidate their digests.
