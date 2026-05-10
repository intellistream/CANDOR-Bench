# e14 — Optimized buffer (vectorized) + diverse delete patterns

**Two questions:**

1. Did Python overhead artificially inflate gamma's cost in e13? If we
   vectorize the buffer (`router`), does gamma+hnswlib catch up?

2. Does hnswlib's mark_deleted hold up under DIVERSE delete patterns?
   Maybe it's good at LRU/sequential but breaks on:
   - Random uniform delete
   - Cluster / hot-spot delete (delete one neighborhood at a time)
   - High frequency re-insertion (ID reuse)
   - Extreme churn (95%+ delete rate)
   - Mixed delete patterns

**Setup:**
- SIFT 200K
- 6 delete patterns × 2 backends × 2 architectures (gamma_v2 vs direct)
- Measure: total time, recall, query latency

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e14_optimized_buffer/run.py
```
