# e02 — Recall and query latency under churn

**Question:** As delete fraction increases, how do recall and query latency change for gamma vs HNSW vs IVF?

**Hypothesis:** HNSW recall degrades and query latency rises monotonically with churn (tombstone pollution). GammaFresh's hybrid graph + buffer query path should keep both stable because the buffer absorbs short-lived inserts (delete-before-migration), so the graph sees fewer tombstones.

**Setup:**
- Dataset: SIFT 200K
- Workload: insert+delete interleaved with periodic query rounds
- Sweep: `delete_frac ∈ {0.0, 0.25, 0.5, 0.75, 0.9}`
- Measure: per-query latency avg/p95, recall after final maint
- 3 algorithms: gamma, faiss, ivf

**What counts as confirmation:**
- HNSW recall drops monotonically with delete_frac
- gamma recall stays ≥ 0.99 across all delete_frac
- HNSW query p95 jumps once deletes start; gamma query p95 stays flat

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e02_churn_recall_latency/run.py
```

**Output:** `experiments/e02_churn_recall_latency/output.json`

**Note:** This experiment is the foundation of the paper's Figure 1 + 2 (recall vs churn, query latency vs churn).
