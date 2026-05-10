# `ablations/` — per-experiment variant routers

**Don't import these in new code.** Each file is a copy of `gamma_py_v2.py`
with one mechanism added or replaced, used by exactly one experiment to
measure that single mechanism's contribution.

| file | used by | mechanism added |
|---|---|---|
| `gamma_py_v3.py` | e20 | 4 component toggles (use_buffer_scan / buffer_inserts / absorb_buffer_deletes / eager_maint) |
| `gamma_py_spatial.py` | e24 | K-partition + centroid-based routing |
| `gamma_py_adaptive_maint.py` | e26 | auto-trigger maint at buffer-fill threshold |
| `gamma_py_cost_admit.py` | e27 | per-batch lifetime EMA → buffer-or-direct admit decision |
| `gamma_py_combined.py` | e28 | rebuild + cost-admit combined |
| `gamma_py_rich.py` | e30 | full rich version: per-vector life-table + heat-tracker + hot tier + γ-aware migration + per-partition rebuild |

## Empirical verdict (see `experiments/FINDINGS.md`)

All of these were tested. None beat the canonical `gamma_py_v2 +
gamma_py_rebuild` design at our test scales (SIFT/MSong/GloVe at
200K-1M). Some actively hurt (per-vector cost-admit added +29%-36% at
1M mixed_lifetime; rich combined hurt +12% at 1M).

**The negative results are themselves a paper contribution**: they show
that "smarter" router designs (per-vector adaptive policies, multi-tier
caches, spatial routing) are over-engineering for the streaming-ANN
workloads we tested.

## How to add a new ablation

1. Copy `gamma_py_v2.py` → `ablations/gamma_py_<your_module>.py`
2. Change the relative import `from ..gamma_py import GraphBackend`
3. Modify just the one mechanism you want to test
4. In your experiment runner, `from _shared.ablations.gamma_py_<your_module> import ...`
