# Hostile-but-Fair VLDB Review (from peer-review subagent, 2026-05-10)

This is the strongest set of reviewer attacks against the current thesis,
along with what would defuse each. Use as ammunition to either fix things
or address head-on in the paper.

---

## Top 6 reviewer attacks (in order of severity)

### 1. "You re-implemented hnswlib's tombstone-rebuild outside the index"

The biggest e25 wins (e.g., 1M sequential -47%) come from periodic full
rebuild, but the `hnswlib direct` baseline does **no** rebuild. So the
comparison is `gamma+rebuild` vs `hnswlib+no-rebuild` — equivalent to
`rebuild on vs off`, not `architecture vs architecture`.

**Status check**: hnswlib Python *does not* have a built-in
tombstone-rebuild flag (only `mark_deleted`, `unmark_deleted`,
`allow_replace_deleted`). However the spirit of the critique stands —
we should add an `hnswlib direct + external periodic rebuild` baseline.

**Defuse**: e29 (running) — adds variant B: `hnswlib_direct + periodic
rebuild`. If gamma_v2+rebuild (D) > hnswlib_direct+rebuild (B), the
buffer/absorb pathway carries independent value. If D ≈ B, the rebuild
is the whole story.

### 2. "e25 cluster/partial_reset 1M wins came from runs where rebuild fired 0 times"

In `output_sift_cluster_1M.json`, `gamma_rebuild(0.5)` reports
`rebuild_count=0` yet runs 400.9s vs gamma_v2's 473.0s (-15%). Same in
`glove_cluster_1M.json` (rebuild_count=0; rebuild row 472s vs no-rebuild
row 439s — *worse* with rebuild logic enabled). With e23 multi-seed
showing ±15s std on 120s runs, this is run-to-run noise being mis-claimed
as a module win.

**Defuse**: e29 includes 3 seeds per cell. Mark e25 1M cluster/partial
results as "within-noise" instead of "wins" in FINDINGS.md.

### 3. "Single-thread fairness is methodology-shopping"

Real production deployments (Qdrant, Chroma, Milvus) all let hnswlib use
all cores. The "fair single-thread" defense (CLAUDE.md, PAPER_OUTLINE.md)
is exactly what reviewers want to see *in addition to* multi-thread.

**Defuse**: add a multi-thread comparison even if just for one pattern at
1M. Honest if gamma loses there — argue the workload regime where
buffer+absorb wins (high-tombstone, modest concurrency).

### 4. "The 4 delete patterns are cherry-picked"

`cluster` (delete k-NN of a random pivot) is maximally adversarial to
HNSW; `partial_reset` (8× burst every 8 batches) is adversarial. The
`sequential` baseline that gamma_v2 loses on at 1M is the closest match
to real production patterns (TTL cleanup is timestamp-bucketed → roughly
sequential).

**Defuse**: either add a real trace replay (Milvus/Pinecone log,
Wikipedia edits) or cite published statistics that argue cluster-delete
is common (content takedowns, GDPR purges).

### 5. "Recall trade-off is buried, not handled rigorously"

`glove 1M sequential`: gamma+rebuild 567s @ rec=0.842 vs hnswlib direct
1387s @ rec=0.880. **-59% time, -4 recall pts.** That's a different
operating point, not a Pareto improvement. The headline summary
"-9% to -82% time at recall ≥ 0.99" is incorrect for glove.

**Defuse**: iso-recall comparison. For each pattern, find the hnswlib
direct config (varying efSearch) that matches gamma+rebuild's recall,
then compare time. e19 has the data shape; need to extend to glove.

### 6. "No novel algorithmic contribution beyond a buffer + LSM-style flush"

`gamma_v2` is essentially "LSM L0 over hnswlib." Lehman-Yao defer-the-split
predates this by 40+ years. The C++ multi-partition machinery (the only
thing that *would be* novel) is broken (e21 partial_reset rec=0.006).

**Defuse options**:
- (a) Reframe as systems paper (CIDR, EuroSys industry).
- (b) Fix and validate the C++ multi-partition at 10M+ scale.
- (c) Demonstrate cost-model admit (e27) and heat-aware tiering (NEW e30)
  as adaptive-placement contributions on workloads that need them
  (mixed-lifetime, Zipfian queries).

---

## Suspicious numbers worth auditing

- **e15 1M sequential vs e25 1M sequential disagree by 30-40%** on the
  same configurations: e15 reports `gamma_v2+hnswlib=2231s` and
  `hnswlib direct=1602s`; e25 reports `gamma_v2=1488s` and
  `hnswlib direct=988s`. Same hardware, same workload code. Likely
  contention from concurrent jobs at e15's run time. Re-run on quiet
  system or add caveat in FINDINGS.

- **e22 partial_reset rec=0.0020/0.4820/0.0060** for the C++ tuned
  variants — these are broken-config measurements, not algorithmic
  performance. Pull them from the headline tables; keep them only in the
  "C++ default is broken" section.

- **e16 random-m cluster recall 0.50** for both gamma and direct — both
  are near random guessing on synthetic uniform data. Unclear what
  "win" means at this recall floor; consider dropping random-m from
  cluster comparisons.

- **e17 hnswlib direct ratio=1.5/2.0/3.0** report 991-2470s — verify
  hnswlib is not hitting the `RuntimeError fallback` in
  `gamma_py.py:HnswlibBackend.search` that re-runs with ef=800 and
  artificially inflates time.

---

## Bottom line from the reviewer

> The cleanest, most defensible paper is "delete-pattern characterization
> for incremental ANN" (e15+e16+e17 + iso-recall + a workload trace).
> Drop the algorithmic-contribution framing entirely; the rebuild module
> is engineering, not a contribution.

Counter-position (after the related-work subagent landed): the
**delete-absorption contract** between buffer and graph (vector dies in
buffer ⇒ never seen by graph) IS algorithmically novel relative to
FreshDiskANN's batched DeleteList, Milvus's segment compaction, and
SPFresh's IVF-LIRE. This needs its own ablation row showing what it
contributes ABOVE rebuild. e20 `no_absorb_deletes` already shows that
disabling absorption makes gamma *worse* than the direct backend (+25%);
this is the mechanism-isolation evidence to lead with.
