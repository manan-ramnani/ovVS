# ovVS CAGRA mutation surface — delete, update, and a non-catastrophic insert

Companion to `.claude/plans/2026-08-30-ovvs-acceleration-campaign.md`, which stays the
authoritative campaign anchor. This document covers gates **G2** (balanced query / insert /
update / delete) and **G4** (recall holds under churn), which are at **0%**.

## 1. What exists today — verified in source, not assumed

| Claim | Evidence |
|---|---|
| No delete, no update anywhere in the C ABI | `include/ovvs/ovvs.h:293-314` — CAGRA has Build, BuildEx, Search, Quantize, Detach/AttachDataset, Serialize(Ex), Deserialize, Extend, Destroy. Nothing else. |
| IDs are positional row offsets, with no indirection layer | `struct CagraIndex` (`graphs.cpp:113`) holds only `Dataset ds` + `UsmI32Vec graph`. `struct Dataset` (`:81`) is `x, n, dim, metric`. No id map, in the struct or in the serialized format (`:1195`). |
| `ovvsCagraExtend` copies the whole index to insert one vector | `CagraIndex staged = *ix;` (`:1298`) deep-copies `ds.x` (512 MB at SIFT1M) and `graph` (64 MB), mutates the copy, then `*ix = std::move(staged)`. **~1.8 GB peak to insert a single vector.** |
| Insert only fails inside the search | `cagra_insert_one` (`:493`) calls `graph_search` and returns on failure; everything after that is `insert` / `resize` / `robust_prune`, which fail only on allocation. |
| A per-search filter primitive already reaches the walk | `ovvsBitsetFromAllowList` (`ovvs.h:349`), and `ovvsCagraSearch` takes `const uint8_t* bitset` which becomes `allowed_id` inside the GPU walk. |
| The serialized format can carry new sections safely | `graphs.cpp:1200-1206` writes magic, `ver`, then a `flags` bitfield. |

## 2. Design decisions, and why

### D1. Tombstones, not compaction
Compaction renumbers rows. Because ids **are** row offsets with no indirection, renumbering
invalidates every id a caller holds — including ids returned by an earlier search. So delete is a
tombstone bit. O(1), no data movement.

### D2. A deleted node stays traversable, but is not returnable
This is the decision that determines whether G4 is achievable at all.

If deleted nodes are excluded from traversal, the graph **fragments**: every deleted node was a
routing hop for its neighbours, and removing it disconnects regions. Recall then collapses in
proportion to the deletion rate — which is precisely hnswlib's `markDelete` weakness and the
opening §6 of the campaign anchor identifies.

So: deleted nodes are scored, admitted to the beam, and expanded exactly as before. They are
skipped only at the final k-selection. Cost: the beam carries dead entries, so the effective
result set shrinks; `BEAM = 2*itopk` gives slack, and the recall-vs-deletion-rate curve is exactly
what G4 must measure.

### D3. Update keeps the id — overwrite plus relink
`update = delete + insert` would hand the caller a new id, which is not an update. Instead
overwrite the row in `ds.x` and then repair the graph: the node's out-edges are re-searched, and
its in-neighbours are re-pruned, the same work one insert does. The id is stable, which is what
matters when ids are the caller's only handle.

### D4. NO slot reuse in phase 1 — deliberately deferred
Reusing a tombstoned row recycles its id, so a caller holding the old id would silently read a
different vector. Without an id map there is no way to detect that.

The fix is to make an id a `(slot, generation)` pair packed into the `int64_t` the ABI already
returns — `id = slot | (generation << 32)` — so a stale id is *detectably* stale. For a freshly
built index every generation is 0, so ids equal slots and current behaviour is bit-identical.
That is a real design commitment and it is **phase 2**.

Consequence to state plainly: **in phase 1, sustained churn grows the footprint**, because delete
frees nothing. That is a G3 risk, and it is why phase 2 is not optional — it just is not first.

### D5. Serialization: bump `ver` only when tombstones exist
Tombstones are written as a trailing section under a new `flags` bit. A file with no deletions
stays `ver = 2` and remains readable by existing builds. A file **with** deletions is written as
`ver = 3`, so an old build refuses it instead of silently resurrecting deleted rows. Failing
closed matters more than compatibility here.

### D6. Insert becomes in-place with a rollback journal
The whole-index copy exists to give Extend all-or-nothing semantics. Keep the semantics, drop the
cost: mutate in place, and journal what changed. Per insert that is the ≤ `degree` neighbour graph
rows that `robust_prune` rewrites — `degree * degree * 4 B` ≈ **1 KB at degree 16**, against
576 MB today. On failure, restore the journal, truncate `ds.x`, restore `ds.n`.

Sequential semantics are preserved (insert *i+1* still sees insert *i*), so results are unchanged.

## 3. ABI additions

```c
/* Tombstone rows. Deleted rows still route traffic through the graph, so recall degrades
   gracefully rather than fragmenting, but they are never returned by a search. Idempotent:
   deleting an already-deleted id succeeds. */
OVVS_API ovvsStatus ovvsCagraDelete(ovvsResources_t res, ovvsCagraIndex_t index,
                                    const int64_t* ids, int64_t nids);

/* Overwrite the vectors at `ids` and repair their graph neighbourhoods. Ids are unchanged. */
OVVS_API ovvsStatus ovvsCagraUpdate(ovvsResources_t res, ovvsCagraIndex_t index,
                                    const int64_t* ids, const float* vectors, int64_t nids);

/* Rows currently live, and rows tombstoned. For churn tests and for deciding when a rebuild
   is worth it. */
OVVS_API ovvsStatus ovvsCagraCounts(ovvsCagraIndex_t index, int64_t* live, int64_t* deleted);
```

## 4. Tasks

1. `CagraIndex` gains `std::vector<uint8_t> deleted` (bit per row) and `int64_t deleted_count`.
   Empty vector == nothing deleted, so the common path allocates nothing.
2. `ovvsCagraDelete` — bounds-check, set bits, maintain `deleted_count`, idempotent.
3. Search honours tombstones at k-selection only (D2), on **both** the CPU and GPU paths, and
   intersects with any caller-supplied bitset rather than replacing it.
4. `ovvsCagraUpdate` — overwrite row, re-search out-edges, re-prune in-neighbours.
5. `ovvsCagraCounts`.
6. Rewrite `ovvsCagraExtend` per D6; delete the whole-index copy.
7. Serialization per D5, plus a round-trip test with deletions.
8. Python bindings + a churn harness: interleave query / insert / update / delete and track
   recall against the live set, which is what G4 actually asks.

## 5. What this does NOT do

- No slot reuse and no id generations (D4) — phase 2.
- No graph repair on delete. Tombstoned nodes keep their edges; quality decay under heavy
  deletion is measured, not prevented. Repair is a phase-2 decision informed by the G4 curve.
- Does not touch IVF. The campaign audit found ovVS cannot build any IVF index at SIFT1M
  (`std::vector<float> scores(n * nlist)` = 16.4 GB at nlist=4096, and `scores(n*n)` = 4 TB at
  n=1e6). Any IVF-based CRUD is blocked behind that, and CAGRA is the lane that actually runs.
