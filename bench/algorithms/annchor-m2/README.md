# ANNchor M2

This directory is a clean M1 fork plus the current M2 fresh-validation path.
M1 supplies the committed graph view. M2 handles the in-flight fresh horizon
with online graph-region rendezvous and exact admission.

The hot path is:

```text
committed query footprint regions
  join with
fresh insert-side graph regions
  through a temporary validation sidecar
```

The graph does not store live query lists or live insert lists. Graph nodes only
carry lightweight region metadata:

```text
node -> primary graph region + small overlap region set
```

The region metadata is an online graph-region cover, not a disjoint partition.
Regions may overlap. Dense graph neighborhoods open more regions only after
their evidence-supported local regions hit `ANNCHOR_M2_REGION_CAP`; sparse
neighborhoods keep reusing under-filled regions. The maximum region budget grows
with the committed graph size and defaults to a 65536 cap for compact postings.

Integrated path:

```text
1. M1 searches the committed, pruning-aware graph prefix.
2. M2 lazily builds graph-region metadata from committed base-layer edges.
3. Each committed node owns one primary region and a few overlap regions.
4. Each region keeps a few short edge anchors `(outside -> inside)`.
5. Query regions come from the committed competitive footprint.
6. Fresh regions come from inserted-node graph neighbors when available.
7. Large fresh windows build a temporary `region -> fresh rows` posting table.
8. Query regions union matching fresh postings into candidates.
9. Candidates are exact-scored before admission.
```

The region evidence is intentionally local. If `x` is closer to the inside
endpoint of a graph edge than to the outside endpoint, that edge supports the
inside endpoint's region for `x`. This follows the same practical signal used by
graph ANN search: useful graph steps move closer to the target. M2 does not try
to prove a full partition theorem; it uses the signal only to reduce candidate
fresh rows before exact scoring.

The implementation is wired through:

```text
annchor_m2.hpp::fresh_merge_graph_region
annchor_m2_cgo.cpp::annchor_m2_fresh_merge_float
```

Small fresh windows still use exact validation because region lookup can cost
more than it saves. Larger windows use region rendezvous, with optional exact
fallback if a candidate floor is configured. If a fresh vector has not published
graph-region metadata yet, M2 exact-scores that vector instead of using a
representative-point fallback.

The hot intersection path is layered:

```text
primary path: region posting table, no full fresh scan
fallback:     small-set bit test over inline region bitsets
SIMD path:    AVX2 word-wise bitset test when sets are dense enough
```

Default knobs:

```text
ANNCHOR_M2_REGION_CAP=512
ANNCHOR_M2_REGION_OVERLAP=1        # 2 when dim > 512
ANNCHOR_M2_EDGE_ANCHOR_K=2
ANNCHOR_M2_REGION_MAX unset        # adaptive: min 4096, capped at 65536
ANNCHOR_M2_POSTING_REGION_LIMIT=65536
```

Correctness contract:

```text
region hit -> candidate only
exact distance remains final
region miss -> allowed only under bounded-FN policy or covered by exact fallback
```
