# hnswlib AVX2 comparator build evidence v1

Status: **reproducible static candidate; end-to-end performance unmeasured**.

On 2026-08-29, the finalized Windows builder ran from two fresh ignored roots.
Both runs checked out hnswlib v0.8.0 commit
`3f3429661187e4c24a490a0f148fc6bc89042b3d` and tree
`fa3b6d92bd6b0c3a81a7236731c7ad7905c949a1`, kept tracked source files clean,
used Python 3.13.14 and MSVC 19.44 with `/O2 /openmp /EHsc /arch:AVX2`, and
used the same nine pinned build dependencies.

The independent builds produced the same 155,130-byte wheel, SHA-256
`023CD8C0AC0D67F0193AC00C513AC549C67CA4DCA01BA7BF1991491DB68EBE5C`,
and the same 340,480-byte module, SHA-256
`7D88F1F633766113FC2CDA0F65379FB63231A9F98A3096E61809BA82F9EEE6F5`.
Both import/add/query smokes succeeded with finite distances. Static
disassembly counted 303 YMM operands and positive `vsubps`/`vmulps`/`vaddps`/
`vmovups` instructions. The packaged control module, SHA-256
`EE278018B71E7814D1D11F3501C5AC3EC90F5CACB229278E0E1231AC0EEC2ADB`,
had zero instances of those four VEX float instructions and retained legacy
`subps`/`mulps` instructions.

The canonical extracted record is [`build-evidence.json`](build-evidence.json).
Its two full-manifest hashes bind the ignored producer outputs. Those manifests
and disassemblies retain absolute producer-root paths, so they are deliberately
not presented as byte-identical. The wheel and module are byte-identical.

This is static and import-time evidence only. It does not establish that AVX2
instructions dominate active search cycles, and it contains no build-time,
recall, latency, throughput, energy, or peak-memory result. Any comparator or
ovVS performance statement still requires repeated complete SIFT1M artifacts
with unchanged `M`, `ef_construction`, `ef`, seed, and thread settings.
