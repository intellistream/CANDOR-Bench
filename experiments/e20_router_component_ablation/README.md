# e20 — Router-component ablation

**Question**: Which mechanism of the simplified hybrid router actually
contributes to gamma's win?

The router has four mechanisms; we toggle each independently:

| Variant | Disabled mechanism | What it tests |
|---|---|---|
| `full` | none (= gamma_v2 baseline) | reference |
| `no_buffer_scan` | search() merges only graph results | does buffer scan add recall under churn? |
| `no_buffer_inserts` | every add() goes straight to graph | does insert buffering pay off? |
| `no_absorb_deletes` | deletes always reach graph as mark_deleted | is delete absorption the key win? |
| `eager_maint` | maintenance after every add | is deferred/lazy maintenance important? |

If the gamma_v2 win disappears under one variant but not others, that
variant is the load-bearing component. This is the ablation reviewers
will ask for.

**Method**: SIFT (or msong/glove) at 200K, single delete pattern per run,
compared against hnswlib direct.

**Run** (4 patterns × 1 dataset = 4 processes):

```bash
for pat in sequential random cluster partial_reset; do
  OMP_NUM_THREADS=1 uv run python experiments/e20_router_component_ablation/run.py \
    --pattern $pat --scale 200K
done
```

Output: `output_<dataset>_<pattern>_<scale>.json`.
