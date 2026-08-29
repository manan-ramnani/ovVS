# NN-Descent cooperative-minima V1 evidence manifest

These immutable artifacts support
[`../../nndescent-cooperative-minima-v1.md`](../../nndescent-cooperative-minima-v1.md).
The benchmark JSON files were copied byte-for-byte from `out/bench/`; embedded
absolute paths and dirty-worktree metadata are producer-time facts and were not
rewritten.

Producer identity:

- Dataset SHA-256:
  `DD6F0A6ED6B7EBB8934680F861A33ED01FF33991EAEE4FD60914D854A0CA5984`.
- Candidate DLL: 1,208,832 bytes, SHA-256
  `6B90289980D01A69189FAC4B35F7F92CBD68BC2163C8B6908AA75A5F7C5D4FAB`.
  The benchmark metadata records dirty base `2897f35`; the measured source was
  committed unchanged as `1ec94ad`. This binds the evidence to source but is
  not a reproducible-build attestation.
- Frozen 100K parent DLL: 1,204,736 bytes, SHA-256
  `6F4E5FBEA1F0E5E3FFE431A4F3CE88A84BE1260661D1A79C11C5D66DB1A9A45C`.
- The three current-product SIFT1M parent runs remain under
  [`../cagra-gpu-optimize-v1/`](../cagra-gpu-optimize-v1/README.md) and are
  referenced rather than duplicated.

Every retained benchmark artifact has two successful lanes, zero failed,
skipped, timed-out, or unavailable lanes, successful output validation, and a
matched loaded-library fingerprint. Energy was disabled. SIFT100K is a prefix
admission diagnostic; SIFT1M is the repeated matched quality/search/RSS gate.
The two `rejected-*` files are one-process screens and support only the stated
negative disposition.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `baseline-sift100k-r1.json` | 36,518 | `8665F2B4E2D233A4274857F90FED38214FE981A65122921536E280437C007368` |
| `baseline-sift100k-r2.json` | 36,518 | `E024A7A4D782BCD46C83867D80049C20300842A7715E63CC7A679001223CA79E` |
| `baseline-sift100k-r3.json` | 36,548 | `120C951090DD75153E4352026F3680964F33EA8B9BB1E8AC99CE9529E8CF28B1` |
| `candidate-sift100k-r1.json` | 36,298 | `8E79DE07D8BC6426C6356C606EB225B41FCDF281B8B76FED003C6CF6CB1AC03D` |
| `candidate-sift100k-r2.json` | 36,362 | `CB75D15F526E712F913F84C629B6C6DC434112F55AC270D8A281B6C0436206E5` |
| `candidate-sift100k-r3.json` | 36,340 | `BBE5AD329FFDC4B8CF7C5CDACF52E474957A08E9CD9DA34433263FD95C283186` |
| `candidate-sift1m-r1.json` | 36,191 | `5D2405C0DD34CC4901B4323741ADC122053C71EBF5ADC8445CAA54F6941ABD57` |
| `candidate-sift1m-r2.json` | 36,143 | `2A47A1CAA3CC8C3EA50F4A9885121B54E476C34E44BC3DC5F6AA09F9F396255E` |
| `candidate-sift1m-r3.json` | 36,168 | `33EF89A7897034BB1A48CA44BD0D98EA357F1132D2B4AE613AEEBCA99546B940` |
| `graph-digest.json` | 2,072 | `B6C88A8BC1766A65212B4C26BC1D512722D59C8045BB8E345561D3FEDFE9F1F8` |
| `rejected-new-only-sift100k.json` | 36,418 | `EBFBDA8C113F24028AFFF097383FC3361D3C81FB7566410941E4819A2F6496D1` |
| `rejected-padded-t64-sift100k.json` | 36,378 | `A092A30CFE631F196CECFA606439E94DDFF23A8D0F075847588DC72EA8FCF47A` |

The graph digest was generated separately from dataset-free CAGRA v2
serializations. Its build times are diagnostic and excluded from the promotion
timing claim.
