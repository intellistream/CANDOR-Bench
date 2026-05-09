# e15 — Full delete-pattern × backend × architecture matrix

**Premise (validated by e14):** hnswlib's mark_deleted handles sequential
delete fine but degrades on random/cluster patterns. We extend the matrix
to nail down the boundaries:

**Delete patterns tested:**
1. `sequential` (FIFO/LRU) — delete oldest IDs
2. `random` — uniform sample from alive
3. `cluster` — delete neighbors of random pivot (hot-spot)
4. `reinsert` — delete then reinsert same ID (stresses tombstone+reinsert)
5. `partial_reset` — every K cycles, delete a large fraction at once
6. `extreme_churn` — 95%+ deletion ratio (vs 50% baseline)

**Backends tested:**
- hnswlib (Qdrant/ChromaDB default)
- faiss_hnsw (Faiss IndexHNSWFlat)

**Architectures:**
- gamma_v2 + backend (hybrid)
- backend direct (mark_deleted only)

**Scale:** SIFT 200K and 1M

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e15_delete_pattern_matrix/run.py
```
