# e01 — Write throughput

**Question:** Does GammaFresh achieve higher write throughput than HNSW (and how does it compare to IVF)?

**Hypothesis:** GammaFresh's buffer-first design defers graph construction, so per-insert latency should be 3-5× lower than HNSW direct insert. IVF (no graph at all) should be the upper bound.

**Setup:**
- Dataset: SIFT 1M (slice configurable: 200K or full 1M)
- Workload: pure-insert stream (no query, no delete during stream)
- Measure: per-insert latency avg/p99, total stream throughput
- Final: one maint + recall query (separate timing)

**What counts as confirmation:**
- gamma per-insert ≤ 1/3 of HNSW per-insert
- gamma stream throughput ≥ 3× HNSW stream throughput
- Final recall ≥ 0.99 for both

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e01_write_throughput/run.py
```

**Output:** `experiments/e01_write_throughput/output.json`
