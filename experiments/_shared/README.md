# `_shared/` — what to use when

## ★ The paper's algorithm — these 3 files

```
backends.py            (~90 lines)  — backend wrappers (HnswlibBackend, FaissHnswBackend)
router.py         (~115 lines) — the simple router (buffer + delete-absorption + lazy maint)
router_with_rebuild.py    (~200 lines) — gamma_v2 + tombstone-rebuild trigger ★ recommended drop-in
```

Reading order to understand the paper algorithm:
1. `backends.py` — abstract `GraphBackend` + 2 implementations (hnswlib, Faiss HNSW). Both forced single-thread for fair comparison.
2. `router.py` — `GammaRouter`. The 3 mechanisms: buffer add, cancel-in-buffer on delete, lazy flush in maintain.
3. `router_with_rebuild.py` — `GammaRouterWithRebuild`. Adds the periodic full-rebuild trigger that is the dominant performance contributor (e25, e29).

## Use this for any new code

```python
from experiments._shared.backends import HnswlibBackend
from experiments._shared.router_with_rebuild import GammaRouterWithRebuild

backend = HnswlibBackend(dim=128, max_elements=200000)
rebuild_factory = lambda: HnswlibBackend(dim=128, max_elements=200000)
idx = GammaRouterWithRebuild(
    backend, rebuild_factory,
    dim=128, buf_capacity=125000,
    rebuild_threshold=0.5,
)
# then idx.add() / idx.delete() / idx.maintain() / idx.search() — same API as hnswlib
```

## Workload runners

| file | what it gives you |
|---|---|
| `workloads.py` | the **standard** `run_workload()` + 4 delete-pattern functions (sequential / random / cluster / partial_reset). Used by e15-e25, e29, e32. |
| `workloads_rich.py` | `run_mixed_lifetime()` + `run_zipfian_queries()` non-stationary workloads (used by e30 only). |
| `workloads_instrumented.py` | `run_instrumented()` — same shape as `run_workload` but records per-op latency (used by e31 only). |

## Baselines

| file | what it is |
|---|---|
| `baseline_with_rebuild.py` | `HnswlibDirectWithRebuild` — wraps hnswlib direct with an external periodic rebuild trigger. The "missing baseline" introduced for e29 to prove the buffer/absorb router contributes value beyond just the rebuild trigger. |

## Ablation variants (`ablations/`)

These files exist for the per-experiment ablations. **Do not use them for new code.** Each is a copy of `router.py` with one mechanism added or replaced, used to measure that single mechanism's contribution.

| file | used by | what it adds |
|---|---|---|
| `ablations/gamma_py_v3.py` | e20 | 4 component toggles (use_buffer_scan / buffer_inserts / absorb_buffer_deletes / eager_maint) |
| `ablations/gamma_py_spatial.py` | e24 | K-partition + centroid-routing |
| `ablations/gamma_py_adaptive_maint.py` | e26 | auto-trigger maint at buffer-fill threshold |
| `ablations/gamma_py_cost_admit.py` | e27 | per-batch lifetime EMA → buffer-or-direct admit |
| `ablations/gamma_py_combined.py` | e28 | rebuild + cost-admit combined |
| `ablations/gamma_py_rich.py` | e30 | full rich version: per-vector life-table + heat-tracker + hot tier + γ-aware migration + per-partition rebuild |

The ablation results (in `experiments/FINDINGS.md`) showed all of these except `router_with_rebuild.py` either don't add measurable value at our test scales or actively hurt. The simple router is sufficient.

## Other utilities

| file | what it gives you |
|---|---|
| `data.py` | dataset loaders (sift / msong / glove / random-m) + ground-truth + caching |
| `metrics.py` | `recall_at_k()`, percentile latency helpers |
| `aggregate.py` | scan all `experiments/eXX_*/output_*.json` → `ALL_RESULTS_TABLE.md` + `ALL_DELTAS_TABLE.md` |
| `plot.py` | generate `experiments/figures/fig_*.png` |
| `builders.py` | configs for the native C++ candy index (used only by e21/e22) |
| `__init__.py` | sets OMP/OpenBLAS/MKL thread limits + path setup; exports common helpers |
