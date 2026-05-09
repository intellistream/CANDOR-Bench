# e11 — hnswlib in streaming (does it suffer same tombstone tax as Faiss HNSW?)

**Question:** hnswlib outperforms gamma in offline-rebuild bulk_delete (e10).
But does hnswlib's streaming behavior (continuous insert + delete with no rebuild
window) also degrade like Faiss HNSW does?

**Hypothesis:** Yes. hnswlib's `mark_deleted` is a tombstone too. The graph
still has the deleted nodes; search visits them but filters at result time.
Recall should degrade and query latency should grow with churn — just like
Faiss HNSW. Gamma's structural advantage (buffer absorbs deletes) should
still hold.

**Setup:**
- SIFT 200K and 1M, streaming_sliding(lag=2) workload
- 4 algorithms:
  - gamma (our hybrid)
  - hnswlib (with mark_deleted as tombstone)
  - faiss HNSW (with mark_deleted as tombstone, same algorithm class)
  - hnswlib + periodic rebuild (the "fair" comparison — match gamma's maint)

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e11_hnswlib_streaming/run.py
```
