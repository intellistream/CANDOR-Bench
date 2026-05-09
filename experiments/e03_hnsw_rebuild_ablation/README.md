# e03 — HNSW periodic rebuild ablation

**Question:** Can HNSW recover its recall under churn by adding periodic graph rebuild? If yes, at what cost in total time and query latency?

**Hypothesis:** With explicit `maintain(force=True)` triggering tombstone compaction every N batches, HNSW recall should match GammaFresh's. But because HNSW's graph contains all delete-touched nodes (no buffer to absorb short-lived ones), each rebuild is expensive. Total time should be 1.5-3× GammaFresh, and query latency between rebuilds will still be elevated (1-3ms vs gamma's 0.1ms).

**Setup:**
- Same as e02 (delete_frac sweep on SIFT 200K)
- 4 variants:
  - `gamma` (baseline)
  - `faiss` (HNSW, no rebuild)
  - `faiss + rebuild every 1 batch` (extreme)
  - `faiss + rebuild every 4 batches`
  - `faiss + rebuild every 16 batches`

**What counts as confirmation:**
- HNSW + rebuild16 recall recovers to ≥ 0.99
- HNSW + rebuild16 query p95 still ≥ 10× gamma's
- HNSW + rebuild16 total time ≥ 1.5× gamma at moderate churn

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e03_hnsw_rebuild_ablation/run.py
```

**Output:** `experiments/e03_hnsw_rebuild_ablation/output.json`
