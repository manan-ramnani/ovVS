# CAGRA build telemetry V1 evidence manifest

These JSON files are the immutable raw inputs for [`../../cagra-build-v1.md`](../../cagra-build-v1.md). They were copied without rewriting from `out/bench/`; the hashes below therefore identify both the original and canonical copies.

Producer identity shared by all six artifacts:

- Git commit: `5dc0ec7b4b901b62c5c90f6b26ed10ba5e8cd448`, branch `main`, clean worktree.
- Loaded DLL: `C:\Personal\ioVS\build-icpx\bin\ovvs.dll`.
- DLL SHA-256: `C4EE36A1F7540726E7686C138190D201BFB825112EDE9FDD6249048FAB319F8F`.
- DLL size: 1,131,520 bytes.
- Dataset SHA-256: `DD6F0A6ED6B7EBB8934680F861A33ED01FF33991EAEE4FD60914D854A0CA5984`.

| Artifact | SHA-256 |
|---|---|
| `cagra-build-v1-sift100k-5dc0ec7b4b90-r1.json` | `C34F98888864861753174D59AA33D6D200306BC3F07C3990EF0507F4C6430B94` |
| `cagra-build-v1-sift100k-5dc0ec7b4b90-r2.json` | `D271E95F80667B06C8545D2C65887B0724C5194CEF97F82D9C33251B4F9D5535` |
| `cagra-build-v1-sift100k-5dc0ec7b4b90-r3.json` | `234AF9F8E8B2E3ED7A4E3782E593EE0C10F741B54F3D8630736D33CE9DFF34DA` |
| `cagra-build-v1-sift1m-5dc0ec7b4b90-r1.json` | `56E2FB587B0BB9F1C34CEEC568E6CDE586E635FFEF25416BC8A474ED4B359D17` |
| `cagra-build-v1-sift1m-5dc0ec7b4b90-r2.json` | `A4CC0E982833C22ED0D0590158B667852A111F665899DAE80D1D305BE7EAE638` |
| `cagra-build-v1-sift1m-5dc0ec7b4b90-r3.json` | `453255E99F12A07D7740DEE0A6963B89380A8B0B67EBA514C87FB4FA1DA48769` |

SIFT100K is an admission diagnostic and is intentionally noncanonical. SIFT1M is a matched recall-closeness gate, not the full B1 curve/energy report. Every JSON retains its partial-completion caveats; none is an acceleration claim. The later same-index effort/energy manifest is [`../cagra-search-v1/`](../cagra-search-v1/README.md).
