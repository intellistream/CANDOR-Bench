# Module-by-module ablation: Python POC + C++-inspired additions

The user feedback after seeing e21/e22 (C++ default config broken on
partial_reset, tuned C++ no faster than Python POC) was: stick with the
Python POC, then incrementally add C++-inspired modules to see in which
scenarios each one actually pays off.

This file is the index of those module experiments.

---

## Base: gamma_v2 (`experiments/_shared/router.py`)

The simplest hybrid router: write-buffer + delete-absorption + lazy
flush on `maintain()`. e15-e23 already established this is the right
baseline at SIFT 200K (4 patterns) and at 1M scale (3 of 4 patterns).

---

## Module 1: Spatial routing (e24, `gamma_py_spatial.py`)

**C++ analogue**: PartitionAssignmentStrategy + per-partition graph.

**Mechanism**: K partitions, each holding its own backend graph + buffer.
Inserts routed by nearest-centroid; queries search top-M nearest-centroid
partitions and merge.

**Knobs**: K_partitions ∈ {1, 2, 4, 8, 16}; routing ∈ {centroid,
round_robin}; search_partitions = K/2.

**Hypothesis**:
- helps query latency by skipping irrelevant partitions
- helps cluster delete (deletes hit one partition's graph; others stay clean)
- hurts when M < K and queries straddle partitions (recall loss)

**File**: `experiments/e24_module_spatial_routing/`

---

## Module 2: Tombstone rebuild trigger (e25, `router_with_rebuild.py`)

**C++ analogue**: GammaFresh maintenance pass that rebuilds segments
when tombstone density crosses threshold; hnswlib's
`tombstone_rebuild_threshold`.

**Mechanism**: After each `maintain()`, check
`deleted_in_graph / total_inserted`. If it exceeds threshold, drop the
backend, rebuild from alive vectors.

**Knobs**: rebuild_threshold ∈ {0.25, 0.5, 0.75, 1.1=never}.

**Hypothesis**:
- helps gamma's 1M sequential loss (where tombstones accumulate faster
  than gamma's per-batch maintenance can prune)
- hurts low-churn workloads (rebuild is expensive amortized over few
  deletes)

**File**: `experiments/e25_module_tombstone_rebuild/`

---

## Module 3: Adaptive maintenance scheduling (e26, `gamma_py_adaptive_maint.py`)

**C++ analogue**: GammaFreshConfig.maintenance_interval +
maintenance_query_interval (currently fixed; the C++ runtime *could*
adapt but doesn't expose adaptivity through the config).

**Mechanism**: Auto-trigger `maintain()` on `add()` when buffer fullness
crosses a threshold. Caller-driven maint at qstride still fires but
becomes a no-op on empty buffer.

**Knobs**: auto_maint_fill_threshold ∈ {0.1, 0.25, 0.5, 0.8, 1.1=never}.

**Hypothesis**:
- lower threshold = more frequent maint = lower buffer-scan latency at
  query time but higher amortized maint cost
- helps partial_reset (where bursty deletes can leave a huge dead buffer)
- hurts when threshold is too small (too many small flushes)

**File**: `experiments/e26_module_adaptive_maint/`

---

## Module 4: Cost-model admit (e27, `gamma_py_cost_admit.py`)

**C++ analogue**: PlacementController.newcomer_epsilon_ms / migration_epsilon_ms.

**Mechanism**: Track rolling-mean of observed lifetimes (insert-op-index
minus delete-op-index for each deleted vector). For each new add(),
compare predicted lifetime against `admit_threshold_ops`. If short,
admit to buffer (likely to be canceled). If long, skip the buffer and
add directly to graph.

**Knobs**: admit_threshold_ops ∈ {0=always direct, 5K, 20K, 100K,
1e9=always buffer}.

**Hypothesis**:
- adaptive threshold should match the workload's actual churn rate
- helps mixed workloads (some short-lived + some long-lived); for our
  current uniform patterns the prediction is roughly stationary so the
  win could be limited
- the always-buffer case should match gamma_v2 (sanity check)

**File**: `experiments/e27_module_cost_admit/`

---

## Methodology

Each module experiment runs all 4 delete patterns (sequential, random,
cluster, partial_reset) at SIFT 200K with the module's knob swept.
Compared against:
- `gamma_v2` (the simplest router, no module)
- `hnswlib direct` (no router at all)

Output: `output_sift_<pattern>_200K.json` per pattern per experiment.

After each experiment lands, the meta-aggregator
(`experiments/_shared/aggregate.py`) picks up the rows and includes them
in `ALL_RESULTS_TABLE.md` and `ALL_DELTAS_TABLE.md`.

---

## Module composition (future)

After we know which modules matter where, the next step is to test
*combinations*: spatial routing + adaptive maint, etc. The
`gamma_py_*.py` files are intentionally separate classes (not a single
modular base) because that was the simplest way to keep the ablation
clear. If a module clearly wins, fold it into a new `gamma_py_v3_full.py`
that combines validated modules.

A useful next step would be `gamma_py_modular.py` with all four toggles
in one class, similar to how `gamma_py_v3.py` already exposes the
gamma_v2 component toggles for e20.
