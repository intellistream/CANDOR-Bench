# e19 — hnswlib parameter sensitivity

**Question**: Can hnswlib close the gap to gamma on the cluster pattern by
tuning M / efC / efS, or is the gap architectural?

**Method**: SIFT 200K, cluster pattern. Grid sweep:
- M ∈ {16, 32, 64}
- efC ∈ {120, 240}
- efS ∈ {80, 160}

Compare each (M, efC, efS) configuration of hnswlib direct against
gamma_v2+hnswlib at default M=32, efC=120, efS=80.

**Run**:

```bash
OMP_NUM_THREADS=1 uv run python experiments/e19_hnswlib_param_sensitivity/run.py
```

Output: `output_cluster_200K.json`.
