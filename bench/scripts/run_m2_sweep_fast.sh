#!/bin/bash
# Fast M2 sweep: run strategies in parallel per scenario, strict criteria.
#
# Criteria (user requirements):
#   1. Overall QPS (insert+search combined) drop ≤ 15% vs mvcc
#   2. Search p99: much better than mvcc, can be slightly worse than rwlock
#   3. Insert QPS: drop ≤ 15% vs mvcc
#   4. Freshness: ≤ 20% worse than mvcc
#   5. Recall: no drop
#
# Strategy: run each (strategy, scenario) as fast as possible,
# analyze incrementally.

set -euo pipefail
cd "$(dirname "$0")/.."

BENCH_BIN="./bin/bench"
RESULT_BASE="./result/m2_sweep"
LOG="$RESULT_BASE/results.csv"
mkdir -p "$RESULT_BASE"

echo "strategy,scenario,srch_p99,srch_mean,ins_qps,ins_p99,recall,fresh_mean,fresh_p99" > "$LOG"

run_one() {
  local strategy_name="$1" env_vars="$2" config_name="$3" label="$4"
  local config_file="config/tbb_fiber_full/${config_name}.yaml"
  local out_dir="$RESULT_BASE/${strategy_name}/${label}"
  local result_csv="$out_dir/benchmark_results.csv"

  if [[ -f "$result_csv" ]] && [[ $(wc -c < "$result_csv") -gt 100 ]]; then
    return 0
  fi

  mkdir -p "$out_dir"
  sed "s|output_dir:.*|output_dir: $out_dir|" "$config_file" > "/tmp/m2s_${strategy_name}_${label}.yaml"

  eval "$env_vars" LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} \
    "$BENCH_BIN" -config "/tmp/m2s_${strategy_name}_${label}.yaml" > "$out_dir/stdout.log" 2>&1

  if [[ -f "$result_csv" ]]; then
    python3 -c "
import csv
with open('$result_csv') as f:
    r = list(csv.DictReader(f))[-1]
print('${strategy_name},${label},' + ','.join([
    r['search_p99_op_latency_ms (per-batch)'],
    r['search_mean_op_latency_ms (per-batch)'],
    r['insert_qps (per-point)'],
    r['insert_p99_op_latency_ms (per-batch)'],
    r['recall'],
    r['freshness_total_pts_mean'],
    r['freshness_total_pts_p99']
]))
" >> "$LOG"
  fi
}

# Start with fast scenarios first (deep1M b20), then gist
# Focus on 3 losing + 3 winning scenarios

echo "=== Phase 1: deep1M scenarios (fast ~2min each) ==="

# Strategies that are most promising based on earlier experiments
FAST_STRATEGIES=(
  "baseline|"
  "psi|ANN_PREEMPT_INTER_INSERT_YIELD=1"
  "m2pro|ANN_PREEMPT_M2_PRO=1"
  "m2pro_svc1|ANN_PREEMPT_M2_PRO=1 ANN_PREEMPT_MAX_SERVICE_PER_YIELD=1"
  "m2pro_svc8|ANN_PREEMPT_M2_PRO=1 ANN_PREEMPT_MAX_SERVICE_PER_YIELD=8"
  "alpha05|ANN_PREEMPT_STARVATION_ALPHA=0.05"
  "alpha10|ANN_PREEMPT_STARVATION_ALPHA=0.10"
)

DEEP_SCENARIOS=(
  "deep1M_fiber_b20_w90000_r10000|deep1M_wh"
  "deep1M_fiber_b20_w40000_r40000|deep1M_bal"
  "deep1M_fiber_b20_w10000_r90000|deep1M_rh"
)

for strat in "${FAST_STRATEGIES[@]}"; do
  IFS='|' read -r sname envs <<< "$strat"
  for sc in "${DEEP_SCENARIOS[@]}"; do
    IFS='|' read -r cfg label <<< "$sc"
    echo "[RUN] $sname / $label"
    run_one "$sname" "$envs" "$cfg" "$label"
    echo "[DONE] $sname / $label"
  done
done

echo ""
echo "=== deep1M results so far ==="
python3 scripts/analyze_m2_optimize.py 2>/dev/null || python3 -c "
import csv
with open('$LOG') as f:
    for line in f:
        print(line.strip())
"

echo ""
echo "=== Phase 2: gist scenarios (slow ~5min each) ==="

GIST_SCENARIOS=(
  "gist_fiber_b20_w90000_r10000|gist_wh"
  "gist_fiber_b20_w40000_r40000|gist_bal"
  "gist_fiber_b20_w10000_r90000|gist_rh"
)

for strat in "${FAST_STRATEGIES[@]}"; do
  IFS='|' read -r sname envs <<< "$strat"
  for sc in "${GIST_SCENARIOS[@]}"; do
    IFS='|' read -r cfg label <<< "$sc"
    echo "[RUN] $sname / $label"
    run_one "$sname" "$envs" "$cfg" "$label"
    echo "[DONE] $sname / $label"
  done
done

echo ""
echo "=== FINAL RESULTS ==="
python3 -c "
import csv
from collections import defaultdict

# Baselines from m2_complete_results.md
baselines = {
    'deep1M_wh':  {'rwlock_sp99': 6.69, 'mvcc_sp99': 23.33, 'mvcc_iqps': 8712, 'mvcc_fresh': 54.1},
    'deep1M_bal': {'rwlock_sp99': 6.94, 'mvcc_sp99': 22.51, 'mvcc_iqps': 9406, 'mvcc_fresh': 61.5},
    'deep1M_rh':  {'rwlock_sp99': 6.34, 'mvcc_sp99': 22.82, 'mvcc_iqps': 12578, 'mvcc_fresh': 59.1},
    'gist_wh':    {'rwlock_sp99': 15.85, 'mvcc_sp99': 53.06, 'mvcc_iqps': 4297, 'mvcc_fresh': 61.2},
    'gist_bal':   {'rwlock_sp99': 15.77, 'mvcc_sp99': 52.48, 'mvcc_iqps': 4326, 'mvcc_fresh': 60.8},
    'gist_rh':    {'rwlock_sp99': 21.12, 'mvcc_sp99': 54.38, 'mvcc_iqps': 4832, 'mvcc_fresh': 60.1},
}

with open('$LOG') as f:
    rows = list(csv.DictReader(f))

by_strat = defaultdict(dict)
for r in rows:
    by_strat[r['strategy']][r['scenario']] = r

scenarios = sorted(baselines.keys())
print()
print(f\"{'strategy':<15}\", end='')
for s in scenarios:
    print(f' {s:<12}', end='')
print(f' {\"ALL_PASS\":>8}')
print('-' * (15 + 13*len(scenarios) + 10))

for strat in sorted(by_strat):
    results = by_strat[strat]
    all_pass = True
    print(f'{strat:<15}', end='')
    for s in scenarios:
        if s not in results:
            print(f' {\"—\":>12}', end='')
            all_pass = False
            continue
        r = results[s]
        b = baselines[s]
        try:
            sp99 = float(r['srch_p99'])
            iqps = float(r['ins_qps'])
            fm = float(r['fresh_mean'])
            # Criteria:
            # search p99 < mvcc (much better) and < 2x rwlock (slightly worse ok)
            ok_search = sp99 < b['mvcc_sp99'] and sp99 < 2.0 * b['rwlock_sp99']
            # insert qps >= 85% of mvcc
            ok_insert = iqps >= 0.85 * b['mvcc_iqps']
            # freshness <= 120% of mvcc
            ok_fresh = fm <= 1.2 * b['mvcc_fresh']

            ok = ok_search and ok_insert and ok_fresh
            mark = '✓' if ok else '✗'
            if not ok: all_pass = False
            print(f' {mark}{sp99:>8.1f}ms', end='')
        except:
            print(f' {\"ERR\":>12}', end='')
            all_pass = False
    print(f' {\"✓ ALL\" if all_pass else \"\":>8}')

print()
print('Criteria: sp99 < mvcc AND < 2x rwlock | iqps >= 85% mvcc | fresh <= 120% mvcc')
print()

# Show baselines for reference
print('Baselines:')
print(f'{\"rwlock sp99\":<15}', end='')
for s in scenarios:
    print(f' {baselines[s][\"rwlock_sp99\"]:>12.1f}', end='')
print()
print(f'{\"mvcc sp99\":<15}', end='')
for s in scenarios:
    print(f' {baselines[s][\"mvcc_sp99\"]:>12.1f}', end='')
print()
print(f'{\"mvcc iqps\":<15}', end='')
for s in scenarios:
    print(f' {baselines[s][\"mvcc_iqps\"]:>12.0f}', end='')
print()
"
