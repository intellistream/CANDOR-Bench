# e08 — Where does gamma's recall come from?

**Question:** Is gamma's recall stability under churn coming from:
- A. Buffer scan acting as a heavily-tuned IVF (the "you're just running IVF on the buffer" interpretation)
- B. The graph staying clean because deletes get absorbed by buffer (structural advantage)

**Hypothesis:** B is correct. If A were correct, forcing gamma's buffer to drain (so query goes graph-only) should drop recall. If B is correct, recall should stay high regardless of buffer state, because the GRAPH already has fewer tombstones.

**Setup:**
- Same churn workload as e02 (sift 200K, delete_frac sweep)
- 4 variants:
  1. `gamma_full` — default (buffer + graph)
  2. `gamma_graph_only` — force `maintain` after every batch → buffer always empty at query time
  3. `faiss` — HNSW direct (no maint)
  4. `faiss_rebuild_every_batch` — HNSW with force rebuild every batch (matches gamma's maint frequency)

**What counts as confirmation:**
- B holds: `gamma_full` recall ≈ `gamma_graph_only` recall (within 0.5%)
- A would hold: `gamma_full` recall ≫ `gamma_graph_only` recall (≥ 5% gap)

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e08_recall_source_ablation/run.py
```
