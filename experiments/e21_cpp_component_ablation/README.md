# e21 — C++ GammaFresh component ablation

**Question**: The C++ index ships with several mechanisms beyond the simplified
Python POC: spatial routing, adaptive split (γ + BIC), placement-controller
cost-model, scheduled maintenance. Which of these actually pays off?

**Method**: For each variant, build a `candy.Index("GammaFresh")` with one
mechanism toggled to a degenerate value (effectively disabled). Run the full
streaming workload and compare against the default.

| Variant | Disabled by | Tests |
|---|---|---|
| `default` | nothing | reference |
| `no_spatial` | `split_factor = 1e9` | spatial routing value |
| `no_split` | `gamma_split_threshold = 1e9` | adaptive split |
| `cost_off_admit` | `newcomer_epsilon_ms = 0` | cost-model: always admit to graph |
| `cost_off_buffer` | `newcomer_epsilon_ms = 1e9` | cost-model: always admit to buffer |
| `eager_maint` | `maintenance_interval = 1` | deferred maintenance |
| `lazy_maint` | `maintenance_interval = 1e6` | scheduled maintenance |

Run on each of the 4 patterns at SIFT 200K. Requires `candy` C++ binding
(GammaFresh must be built).

**Run**:

```bash
for pat in sequential random cluster partial_reset; do
  OMP_NUM_THREADS=1 uv run python experiments/e21_cpp_component_ablation/run.py \
    --pattern $pat --scale 200K
done
```

Output: `output_sift_<pattern>_<scale>.json`.
