# e06 — Bulk delete

**Question:** When a large fraction of vectors gets deleted in a single bulk operation (GDPR purge, TTL expiry, batch cleanup), how does GammaFresh compare to HNSW and IVF?

**Hypothesis:** Bulk delete is GammaFresh's strongest scenario because:
- Many deleted IDs were still in the buffer → never touched the graph
- Of those that did reach the graph, many are co-located in partitions, allowing wholesale ghost-partition marks
- The hybrid query path doesn't suffer the tombstone tax that HNSW does
- Result: gamma 5× faster + perfect recall

**Setup:**
- Dataset: SIFT 1M (init 100K, stream 900K, then bulk-delete 30%)
- Single delete call removes 300K oldest vectors
- Final query for recall vs surviving slice

**What counts as confirmation:**
- gamma total time ≤ 1/3 HNSW total time
- gamma recall = 1.000 (perfect — buffer-resident vectors are deletes-only, no ghost links)
- HNSW recall < 0.99 (tombstone-degraded)

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e06_bulk_delete/run.py
```

**Output:** `experiments/e06_bulk_delete/output.json`
