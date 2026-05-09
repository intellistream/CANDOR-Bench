# e17 — Churn rate sweep on the cluster pattern

**Question**: As the deletes-per-batch ratio grows from light to heavy, does
gamma's lead grow, shrink, or invert?

**Method**: Cluster pattern, hnswlib backend (the relevant fair comparison),
SIFT 200K. Sweep `del_per_batch / insert_batch ∈ {0.25, 0.5, 1.0, 1.5, 2.0, 3.0}`.

**Run**:

```bash
OMP_NUM_THREADS=1 uv run python experiments/e17_churn_rate_sweep/run.py --pattern cluster
OMP_NUM_THREADS=1 uv run python experiments/e17_churn_rate_sweep/run.py --pattern random
```

Output: `output_<pattern>_<scale>.json`.
