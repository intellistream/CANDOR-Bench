# e05 — Sliding-window streaming

**Question:** On realistic streaming workloads where the index size stays bounded by deleting old vectors (news feed, recent items, time-series), does GammaFresh win total time?

**Hypothesis:** Sliding-window streaming is GammaFresh's home turf. Most short-lived inserts get deleted from the buffer before ever migrating to the graph, so:
- Write throughput is high (4× HNSW)
- Graph stays small + clean (no tombstone pollution)
- Query stays fast
- Total time wins by 9-28% depending on lag (deletion delay)

**Setup:**
- Workload: `streaming_sliding(lag=N)` — every batch inserts new vectors AND deletes the oldest equal-volume after `lag` batches of warmup
- Sweep `lag ∈ {2, 4, 8}`
- Two scales: SIFT 200K and SIFT 1M
- Periodic queries every `qstride` inserts

**What counts as confirmation:**
- gamma total time < HNSW total time at lag=2,4
- gamma recall ≥ HNSW recall

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e05_sliding_window/run.py
```

**Output:** `experiments/e05_sliding_window/output.json`

**Note:** This is the strongest "real workload" win for the paper. lag=2 mimics tight retention (recent feed); lag=8 lets vectors settle in graph longer (less ideal for gamma).
