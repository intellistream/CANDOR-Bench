# e18 — Gamma buffer-capacity ablation

**Question**: How sensitive is gamma's win to buffer capacity?
- Too small → frequent maintenance flushes, high overhead.
- Too large → buffer scan dominates query latency.

Find the operating range where gamma's win is robust and the failure mode at
each extreme.

**Method**: SIFT 200K, cluster pattern, hnswlib backend. Sweep
`buf_capacity = batch * mult` for `mult ∈ {5, 10, 25, 50, 100, 200, 400}`.

**Run**:

```bash
OMP_NUM_THREADS=1 uv run python experiments/e18_buffer_capacity/run.py
```

Output: `output_cluster_200K.json`.
