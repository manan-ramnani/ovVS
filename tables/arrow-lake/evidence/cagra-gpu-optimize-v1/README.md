# CAGRA GPU optimizer V1 evidence manifest

These immutable artifacts support
[`../../cagra-gpu-optimize-v1.md`](../../cagra-gpu-optimize-v1.md). The ten
benchmark JSON files were copied byte-for-byte from `out/bench/`; the exactness
record was produced separately after the implementation commit.

Producer identity:

- SIFT dataset SHA-256:
  `DD6F0A6ED6B7EBB8934680F861A33ED01FF33991EAEE4FD60914D854A0CA5984`.
- Final AUTO candidate DLL: 1,204,736 bytes, SHA-256
  `C8448624DDED7F1490AAAB623C96A3B811E8C93DF682A7EB0C03924D3125880D`.
  The benchmark metadata truthfully records dirty base `ab35c24`. The unchanged
  implementation was then committed as `28f2dbd`; a no-work clean-source build
  left `build-icpx/bin/ovvs.dll` byte-identical to the measured DLL. This binds
  the binary to the committed source but is not a reproducible-build attestation.
- Bounded FORCE_GPU-only candidate DLL: 1,204,736 bytes, SHA-256
  `6E10E9C6CCB47DCB4933A507681E39101F9C9FFF2E05D6AA2444BDABA72428BB`.
  It is used only for the repeated SIFT100K admission comparison.
- Frozen product-code parent DLL: 1,134,592 bytes, SHA-256
  `1828C050A447B47E3690511A8CB2DC88C57D3095A2A20408DD13B6CF6F95C4C5`.
- The clean three-run SIFT1M CPU-optimizer baseline remains in
  [`../cagra-build-v1/`](../cagra-build-v1/README.md); it is referenced rather
  than duplicated here.

Every retained benchmark artifact has two successful lanes, zero failed,
timed-out, or unavailable lanes, successful output validation, and an explicitly
matched loaded-library fingerprint. Energy was disabled, so completion remains
partial. SIFT100K is a prefix diagnostic and the SIFT1M files are matched
quality gates, not full B1 curves.

The exactness record compares dataset-free CAGRA v2 serializations in separate
processes. The frozen parent reports CPU as its final build primitive and the
candidate reports GPU. Parent and candidate match both whole-file and graph
payload SHA-256 at 100K and 1M rows. Its sequential current-load build times are
diagnostic only and are excluded from performance claims.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `cagra-gpu-optimize-graph-digest.json` | 2,493 | `C9FB69C3957CBC0804126B07EE686AAF047AF7B4CE2F0938DD93E0D24D3AD502` |
| `cagra-gpu-optimize-sift100k-auto-r1.json` | 36,656 | `C95C3C00765079CEC6EE93E0E630A1E5A5F845029ECE38D5070C50DC5C9C27CF` |
| `cagra-gpu-optimize-sift100k-force-r1.json` | 52,179 | `A8CD0F71A08040580FFDEC0EBDC1D479E2563DC6E8AFC0EFD3D377869D292765` |
| `cagra-gpu-optimize-sift100k-force-r2.json` | 52,239 | `219C54C53154B8A230F26DF0FEBC2DA035C0E96178E31C1B634635354B5DEE81` |
| `cagra-gpu-optimize-sift100k-force-r3.json` | 52,154 | `740E5DCAE0825F905161AA3D2F8D52AE8CC85E217204500AA3BE965625AA29EB` |
| `cagra-gpu-optimize-sift100k-parent-r1.json` | 52,052 | `794E05AEB80F17D9EA7271A3EF02C45518AC3FA424680DA07F887E95C484CC60` |
| `cagra-gpu-optimize-sift100k-parent-r2.json` | 52,109 | `98AC8DD20FFB652741D291FA9363CA804D8F6493D46237F811AA7FBB28047C22` |
| `cagra-gpu-optimize-sift100k-parent-r3.json` | 51,988 | `85CF795137CA3D0F7FEB1453895A0A57FFF42A4FD6E93D5796CD4366B3D199AC` |
| `cagra-gpu-optimize-sift1m-auto-r1.json` | 36,402 | `7BA2A9990F7BF7610F4171BB07B5538D7B62D151E8F4405B44E95E793EEFE2DD` |
| `cagra-gpu-optimize-sift1m-auto-r2.json` | 36,398 | `23D025C5D84484449EADA0811F877DD47BAEE98D370C0BD2A1C7F6C86ADA53B6` |
| `cagra-gpu-optimize-sift1m-auto-r3.json` | 36,367 | `52DF481E68F7922DFA64E0D828263D79A7497517441111E579A271D27A406874` |

Two producer files are deliberately excluded. `force-gpu-sift100k-r1.json`
failed because its copied DLL lacked the runtime dependency closure.
`force-gpu-sift100k-r2.json` has a misleading filename: fixed preflight mode
normalized construction to AUTO and the parent CPU optimizer ran.
