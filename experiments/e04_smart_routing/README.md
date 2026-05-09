# e04 — Smart partition routing ablation

**Question:** Does GammaFresh's smart partition routing (`SpatialPartitionStrategy::choose_owner_partition` — distance + scale-normalized scoring) actually pay off, vs cheap round-robin assignment?

**Hypothesis:** Smart routing creates spatially-coherent partitions (small radius), enabling effective `near_edge ≤ search_radius` geometric pruning during query. The benefit appears in workloads where query relies on partition candidate selection (i.e., buffer-resident queries with no maint).

**Setup:**
- A/B knob: `GAMMA_DUMB_ROUTING=1` env var swaps `choose_owner_partition` to round-robin
  (round-robin still respects target_partition_count, so partition count matches SMART)
- Workload: SIFT 200K, all 200K stay in buffer (no maint), 2000 queries
- Config: `candidate_reserve_size=999` so geometric pruning is the sole budget control
- Compare: SMART vs DUMB on insert time, query time, recall

**What counts as confirmation:**
- DUMB recall < SMART recall (or DUMB query much slower at same recall)
- SMART query speedup at same recall ≥ 1.5× DUMB

**Run:**
```bash
OMP_NUM_THREADS=1 uv run python experiments/e04_smart_routing/run.py
```

**Output:** `experiments/e04_smart_routing/output.json`

**Note:** The C++ side (`spatial_partition_strategy.cpp`) reads `GAMMA_DUMB_ROUTING` and bypasses the distance scan when set. This is a fair A/B because partition count and capacity logic stay identical.
