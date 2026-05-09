# e12 — Gamma architecture + hnswlib backend (POC)

**Question:** If gamma's structural advantage is real (not just due to weak Faiss backend), can a "buffer + hnswlib" hybrid beat plain hnswlib?

**Setup:**
- Python-level POC: numpy buffer + hnswlib (Index) graph
- On insert: append to buffer
- On maint: flush buffer → hnswlib.add_items, clear buffer
- On delete: if in buffer remove; else hnswlib.mark_deleted
- On query: hnswlib.knn_query + buffer scan

**Comparison:**
1. plain hnswlib (e11 baseline)
2. gamma_py_hnswlib = our POC (buffer + hnswlib)
3. gamma (FaissHNSW backend) for reference

**Hypothesis:** Buffer should still help when:
- Many short-lived vectors get deleted before promotion (saves graph build work)
- Bulk delete (graph stays smaller)

If POC beats hnswlib in some scenario, we know the buffer architecture has merit independent of backend.

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e12_gamma_with_hnswlib_backend/run.py
```
