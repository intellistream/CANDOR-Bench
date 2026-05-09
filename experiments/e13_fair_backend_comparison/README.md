# e13 — Fair backend-matched comparison

**Question:** When comparing gamma vs algorithm X, swap gamma's graph backend
to also be X. This isolates the value of gamma's hybrid architecture from
the underlying graph backend's performance.

**Methodology:** For each backend X ∈ {hnswlib, FaissHNSW}:
- Run **gamma_py + X** (POC of gamma architecture using X as graph backend)
- Run **X direct** (just X with mark_deleted for delete handling)
- Report the architectural delta (positive = gamma's hybrid adds value)

**Three workloads:**
- bulk_delete (gamma's classical strong scenario)
- streaming_sliding lag=2 (high churn)
- streaming_sliding lag=4 (moderate churn)

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e13_fair_backend_comparison/run.py
```
