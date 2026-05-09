# GammaFresh Experiments

Each subfolder is a **self-contained experiment**: one research question, one runnable script, one results file, one README.

Common code lives in `_shared/` (data loaders, algorithm builders, metric helpers). All experiments single-thread (`OMP_NUM_THREADS=1`) and use the same gamma/faiss/ivf configs from `_shared/builders.py`.

## Index of experiments

| # | Folder | Research question |
|---|---|---|
| 1 | `e01_write_throughput/` | Does GammaFresh achieve higher write throughput than HNSW? |
| 2 | `e02_churn_recall_latency/` | How does recall and query latency change with delete fraction? |
| 3 | `e03_hnsw_rebuild_ablation/` | Can HNSW recover recall via periodic rebuild — at what cost? |
| 4 | `e04_smart_routing/` | Does smart partition routing pay off vs round-robin? |
| 5 | `e05_sliding_window/` | Does GammaFresh win realistic sliding-window streaming? |
| 6 | `e06_bulk_delete/` | How much faster is GammaFresh on bulk delete + perfect recall? |
| 7 | `e07_full_scenario_sweep/` | All 9 scenarios × 3 algorithms — comprehensive picture. |

## Running

```bash
# from repo root
OMP_NUM_THREADS=1 uv run python experiments/e02_churn_recall_latency/run.py
```

Each `run.py` saves its raw results to `experiments/eXX_*/output.json`.

## Reproduce all

```bash
for d in experiments/e0*; do
  OMP_NUM_THREADS=1 uv run python "$d/run.py"
done
```

Total wall time ≈ 3-5 hours for 200K configs, ≈ 12 hours for 1M configs (single-thread).
