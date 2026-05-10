# e33 — Cluster-buffer × maintain-strategy ablation

## Hypothesis

If the buffer itself is structured (IVF-style K-bucket), does periodic
**full rebuild** still dominate per-cluster **in-place migration** (the
C++ GammaFresh design)?

We've already shown that for a **flat** buffer, rebuild beats lazy
maintenance (e25/e29). The interesting open question — relevant to the
SIGMOD story — is whether that finding *also* holds when the buffer
gets the structural advantages of clustering.

## Variants

| id | buffer | maintain strategy | corresponds to |
|---|---|---|---|
| V0 | none | n/a | hnswlib direct (reference) |
| V1 | flat | lazy_flush | `GammaRouter` |
| V2 | flat | rebuild | `GammaRouterWithRebuild` (★ recommended) |
| V3 | cluster | lazy_flush | "structured buffer alone" |
| V4 | cluster | rebuild | "structured buffer + rebuild" (the new combo) |
| V5 | cluster | in_place_migrate | C++ design (per-cluster γ-aware migration) |
| V6 | cluster | cluster_rebuild | rebuild + refit centroids on alive |

## Setup

- SIFT 200K (slice of 1M)
- `n_clusters = 16`, `n_probe = 4` (default)
- `rebuild_threshold = 0.5`
- 4 delete patterns: sequential, random, cluster, partial_reset
- `OMP_NUM_THREADS=1` (single-thread fair comparison; see CLAUDE.md)

## Run

```bash
# one pattern at a time (parallelizable)
OMP_NUM_THREADS=1 uv run python experiments/e33_cluster_buffer_main/run.py --pattern random
OMP_NUM_THREADS=1 uv run python experiments/e33_cluster_buffer_main/run.py --pattern sequential
OMP_NUM_THREADS=1 uv run python experiments/e33_cluster_buffer_main/run.py --pattern cluster
OMP_NUM_THREADS=1 uv run python experiments/e33_cluster_buffer_main/run.py --pattern partial_reset
```

Each output: `output_sift_{pattern}_K16_200K.json`

## Reading the results

The headline question is **V4 vs V5**:
- V4 wins → "rebuild beats incremental migration" generalizes from graph
  to cluster layer; F1+F3 mixed framing supported.
- V5 wins → C++'s per-cluster strategy actually pays off when wired
  correctly; we need to revisit our negative result on rich modules.
- Tie → both work; argue from simplicity (rebuild is fewer LoC).

Secondary checks:
- V3 vs V1 → does clustering alone help (no rebuild)?
- V4 vs V2 → does clustering add anything *on top* of rebuild?
- V6 vs V4 → does refitting centroids help when alive distribution shifts?
