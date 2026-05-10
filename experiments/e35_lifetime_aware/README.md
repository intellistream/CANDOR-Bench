# e35 — Lifetime-aware migration on cluster-correlated lifetime

## The question (PhD-level honest reframing)

In e27/e30 we tested per-vector lifetime-aware admission and saw it
hurt at 1M. But those workloads (mixed_lifetime in particular) had
**no ID↔cluster lifetime correlation** — every vector's lifetime was
sampled iid, so the EMA had no structure to learn.

In real systems, lifetime IS spatially structured: spam/throwaway
content vs. evergreen reference clusters. Per-cluster lifetime EMA
should be predictive *if and only if* the workload has cluster-level
lifetime structure.

This experiment **finally tests that**.

## Workload construction

```
1. kmeans(K=16) on the first 20K SIFT vectors → 16 ground-truth centroids
2. Assign every vector its ground-truth cluster_id by nearest centroid
3. Mark 8 of 16 clusters as LONG-lived: lifetime ~ Exp(1/50000 ops)
4. Mark the other 8 as SHORT-lived: lifetime ~ Exp(1/5000 ops)
5. At insert of vector v in cluster c: schedule death at op v.id + sample
6. Each step's deletes = vectors whose scheduled death has elapsed
```

The 10× lifetime asymmetry (50K vs 5K) is intentionally large: if
this can't beat rebuild, the negative result is unconditional.

## Variants

| id | strategy | description |
|---|---|---|
| V0 | hnswlib_direct | reference (no buffer/router) |
| V2 | flat + rebuild | strong baseline |
| V4 | cluster + rebuild | e33 cluster champion (no lifetime info) |
| V5 | cluster + in_place_migrate | C++-style w/o lifetime |
| **V7** | **cluster + lifetime_aware_migrate** | **the new mechanism** |
| **V8** | **cluster + lifetime_aware_migrate + rebuild** | **combined** |

## Possible outcomes

| outcome | paper interpretation |
|---|---|
| V7/V8 wins on `lifetime` workload AND ties on others | lifetime prediction works *when there's structure to predict* — paper has 2 positive contributions |
| V7/V8 ≈ V4 on every workload | strong negative: lifetime prediction adds no value even on the synthetic best case |
| V7/V8 wins lifetime but regresses on others | conditional optimization; needs detection mechanism |

All three outcomes are publishable.

## Run

```bash
# 5 workloads in parallel
for wl in lifetime sequential random cluster partial_reset; do
  OMP_NUM_THREADS=1 uv run python experiments/e35_lifetime_aware/run.py --workload $wl &
done
```

Output: `output_sift_{workload}_K16_200K.json`
