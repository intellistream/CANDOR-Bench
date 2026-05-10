# e34 — Cluster-buffer K-sweep

## Hypothesis

For `ClusterBuffer + rebuild`, sweep K ∈ {1, 4, 16, 64, 256}. Expect:
- K=1 ≈ flat+rebuild (sanity check; 1 cluster = no clustering)
- K=16-64 sweet spot (probe a small subset of clusters → faster
  buffer search than flat)
- K=256 hurts (k-means cost dominates; per-cluster overhead)

This is also a writable defense against reviewer concern "your choice
of K=16 is arbitrary".

## Run

```bash
OMP_NUM_THREADS=1 uv run python experiments/e34_cluster_K_sweep/run.py --pattern random
```

Output: `output_sift_random_200K.json`
