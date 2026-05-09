# e07 — Full scenario sweep

**Question:** Across the full set of canonical ANN-stream scenarios, where does each algorithm win and lose? What is the complete trade-off picture?

**Scenarios covered:**
1. `insert_only` — pure write throughput
2. `insdel_stream(50%)` — 50% delete-to-insert ratio, no query
3. `streaming` — periodic queries during insert
4. `streaming_sliding(lag=2)` — recent-items / sliding window
5. `streaming_sliding(lag=4)` — slightly longer retention
6. `burst` — large append batches then query
7. `drift` — multi-cycle insert+delete with growing index
8. `bulk_delete` — single 30% delete operation
9. `churn` — fixed-size index with continuous insert/delete cycles

**Setup:**
- Two scales: SIFT 200K (~10 min) and SIFT 1M (~130 min)
- 3 algorithms: gamma, faiss, ivf
- Per-scenario maintenance schedule documented in code

**What this delivers:**
- The "comprehensive table" reviewers expect
- Identifies which scenarios each algorithm wins
- Provides the data for the paper's main results table

**Run:**
```bash
# 200K only (~10 min)
OMP_NUM_THREADS=1 uv run python experiments/e07_full_scenario_sweep/run.py --scale 200K

# 1M only (~130 min)
OMP_NUM_THREADS=1 uv run python experiments/e07_full_scenario_sweep/run.py --scale 1M

# both
OMP_NUM_THREADS=1 uv run python experiments/e07_full_scenario_sweep/run.py --scale both
```

**Output:** `experiments/e07_full_scenario_sweep/output_200K.json` / `output_1M.json`
