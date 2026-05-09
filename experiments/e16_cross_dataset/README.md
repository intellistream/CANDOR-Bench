# e16 — Cross-dataset replication of e15

**Question**: Does the e15 finding (gamma wins random/cluster/partial_reset
delete patterns by 25-30% on hnswlib backend) hold across datasets, or is
it SIFT-specific?

**Method**: Same 4-pattern × 2-backend × 2-architecture matrix as e15, but
across MSong (high-dim audio), GloVe (text embeddings), and random-m
(synthetic). SIFT lives in e15.

**Run**:

```bash
OMP_NUM_THREADS=1 uv run python experiments/e16_cross_dataset/run.py --dataset msong   --scale 200K
OMP_NUM_THREADS=1 uv run python experiments/e16_cross_dataset/run.py --dataset glove   --scale 200K
OMP_NUM_THREADS=1 uv run python experiments/e16_cross_dataset/run.py --dataset random-m --scale 100K
```

Output: `output_<dataset>_<scale>.json`.
