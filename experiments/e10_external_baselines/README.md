# e10 — External ANN library baselines (hnswlib / scann / annoy)

**Question:** How does GammaFresh compare against widely-deployed external ANN libraries?

**Algorithms compared:**
- **hnswlib** — independent HNSW impl (ChromaDB, Qdrant default backend)
- **scann** (Google) — asymmetric hashing + quantization
- **annoy** (Spotify) — random projection trees
- **gamma** (our middle-routed hybrid)
- **faiss HNSW** (Faiss-native, also tested in e09)

**Setup:**
- SIFT 200K and 1M, bulk_delete(30%) scenario
- Build → bulk delete → query
- All non-gamma libraries simulate delete via REBUILD (most-favorable case)

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e10_external_baselines/run.py
```
