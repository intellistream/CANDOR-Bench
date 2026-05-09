# e09 — Compare against more ANN baselines (NSG, IVFPQ)

**Question:** Is gamma's advantage over HNSW also an advantage over OTHER popular ANN algorithms (NSG, IVFPQ)?

**Hypothesis:** All graph-based ANN (HNSW, NSG, NSG2) have the same fundamental limitation under churn — tombstones pollute graph traversal. Quantized indexes (IVFPQ) lose recall by approximation. Gamma's hybrid architecture should outperform all on the recall+latency trade-off under churn.

**Setup:**
- Faiss-native indexes used directly (no PyCANDYAlgo build needed):
  - `IndexHNSWFlat(M=32)` — same as our existing HNSW
  - `IndexNSGFlat(R=32)` — Navigating Spreading-out Graph (Fu et al. VLDB'19)
  - `IndexIVFFlat(nlist=64)` — uncompressed IVF
  - `IndexIVFPQ(nlist=64, m=8)` — product-quantized IVF
- Workload: SIFT 200K with bulk_delete(30%) — gamma's strongest scenario
- Note: Faiss raw indexes don't support fine-grained delete; we use rebuild
  on delete (worst-case for them, but only fair option).

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e09_more_baselines/run.py
```

**Output:** `experiments/e09_more_baselines/output.json`
