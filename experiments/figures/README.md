# Generated figures

Auto-built by `_shared/plot.py` from the JSON outputs. Re-run after new
data lands.

| file | what it shows | source experiment |
|---|---|---|
| `fig_e15_main.png` | gamma vs hnswlib bars per pattern × scale | e15 |
| `fig_e16_cross_dataset.png` | same bars across MSong / GloVe / random-m | e16 |
| `fig_e17_churn_curve.png` | gamma's Δ% vs delete-rate / insert-batch ratio | e17 |
| `fig_e18_buffer_curve.png` | total time vs buffer-capacity multiplier | e18 |
| `fig_e19_param_grid.png` | hnswlib (recall, time) Pareto for 13 configs vs gamma | e19 |
| `fig_e20_ablation.png` | per-router-component time (full / no_buffer_scan / ...) | e20 |
| `fig_e21_cpp_ablation.png` | per-C++-component time (default / no_spatial / ...) | e21 |
| **`fig_e29_iso_rebuild.png`** ★ | grouped bars: A direct / B direct+rebuild / C gamma_v2 / D gamma_v2+rebuild | e29 |
| **`fig_e31_op_latency.png`** ★ | stacked bars: insert / delete / maintain / search time per design | e31 |
| **`fig_e32_iso_recall.png`** ★ | (recall, time) Pareto: hnswlib direct efS sweep + gamma+rebuild point | e32 |

Three star figures are the most paper-impactful (mechanism, contribution
decomposition, Pareto dominance).
