#!/bin/bash
# ── M2 Optimization Loop ──
# Runs fiber preemption with different strategies across ALL scenarios.
# Checks pass/fail for each, logs results, repeats with adjusted params.
#
# Pass criteria: fiber should be competitive with the BEST of rwlock/mvcc on search p99,
# without sacrificing more than 30% on insert QPS vs mvcc.
#
# Strategies to sweep:
#   1. M2 classic (baseline, alpha sweep)
#   2. M2-pro (no convergence gate, no alpha, yield unless in lock)
#   3. PSI (yield between inserts)
#   4. PSI + M2-pro hybrid
#   5. Adaptive combinations

set -euo pipefail
cd "$(dirname "$0")/.."

BENCH_BIN="./bin/bench"
RESULT_BASE="./result/m2_optimize"
LOG="./result/m2_optimize/experiment_log.csv"

mkdir -p "$RESULT_BASE"

# All scenarios: ds|config_template|label|rwlock_sp99|mvcc_sp99|mvcc_iqps
# rwlock_sp99 and mvcc_sp99 are baseline targets from m2_complete_results.md
SCENARIOS=(
  # Losing scenarios (fiber currently worse than rwlock on search p99)
  "gist|gist_fiber_b20_w40000_r40000|gist_b20_bal|15.77|52.48|4326"
  "gist|gist_fiber_b20_w90000_r10000|gist_b20_wh|15.85|53.06|4297"
  "deep1M|deep1M_fiber_b20_w90000_r10000|deep1M_b20_wh|6.69|23.33|8712"
  # Winning scenarios (fiber should stay good)
  "deep1M|deep1M_fiber_b20_w10000_r90000|deep1M_b20_rh|6.34|22.82|12578"
  "deep1M|deep1M_fiber_b20_w40000_r40000|deep1M_b20_bal|6.94|22.51|9406"
  "gist|gist_fiber_b20_w10000_r90000|gist_b20_rh|21.12|54.38|4832"
)

# Strategies to test
STRATEGIES=(
  "baseline|"
  "m2pro|ANN_PREEMPT_M2_PRO=1"
  "psi|ANN_PREEMPT_INTER_INSERT_YIELD=1"
  "alpha05|ANN_PREEMPT_STARVATION_ALPHA=0.05"
  "alpha10|ANN_PREEMPT_STARVATION_ALPHA=0.10"
  "alpha30|ANN_PREEMPT_STARVATION_ALPHA=0.30"
  "psi_ckpt500|ANN_PREEMPT_INTER_INSERT_YIELD=1 ANN_PREEMPT_PSI_CHECKPOINT_US=500"
  "psi_ckpt2000|ANN_PREEMPT_INTER_INSERT_YIELD=1 ANN_PREEMPT_PSI_CHECKPOINT_US=2000"
  "m2pro_svc1|ANN_PREEMPT_M2_PRO=1 ANN_PREEMPT_MAX_SERVICE_PER_YIELD=1"
  "m2pro_svc8|ANN_PREEMPT_M2_PRO=1 ANN_PREEMPT_MAX_SERVICE_PER_YIELD=8"
  "m2pro_svc16|ANN_PREEMPT_M2_PRO=1 ANN_PREEMPT_MAX_SERVICE_PER_YIELD=16"
)

# Write CSV header
if [[ ! -f "$LOG" ]]; then
  echo "strategy,scenario,srch_p99,srch_mean,ins_qps,ins_p99,recall,fresh_mean,fresh_p99,rwlock_sp99,mvcc_sp99,mvcc_iqps,pass_search,pass_insert" > "$LOG"
fi

extract_metrics() {
  local csv="$1"
  python3 -c "
import csv, sys
with open('$csv') as f:
    rows = list(csv.DictReader(f))
if not rows:
    print('ERROR')
    sys.exit(1)
r = rows[-1]
print(f\"{r['search_p99_op_latency_ms (per-batch)']},{r['search_mean_op_latency_ms (per-batch)']},{r['insert_qps (per-point)']},{r['insert_p99_op_latency_ms (per-batch)']},{r['recall']},{r['freshness_total_pts_mean']},{r['freshness_total_pts_p99']}\")
" 2>/dev/null || echo "ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR"
}

total=0
passed=0
failed=0

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  M2 Optimization Loop: Full Scenario x Strategy Sweep    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Scenarios: ${#SCENARIOS[@]}, Strategies: ${#STRATEGIES[@]}"
echo "Total experiments: $(( ${#SCENARIOS[@]} * ${#STRATEGIES[@]} ))"
echo ""

for strategy_entry in "${STRATEGIES[@]}"; do
  IFS='|' read -r strategy_name env_vars <<< "$strategy_entry"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Strategy: $strategy_name"
  echo "  Env: $env_vars"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  for scenario_entry in "${SCENARIOS[@]}"; do
    IFS='|' read -r ds config_name label rwlock_sp99 mvcc_sp99 mvcc_iqps <<< "$scenario_entry"

    config_file="config/tbb_fiber_full/${config_name}.yaml"
    out_dir="$RESULT_BASE/${strategy_name}/${label}"
    result_csv="$out_dir/benchmark_results.csv"

    total=$((total + 1))

    # Skip if already done
    if [[ -f "$result_csv" ]] && [[ $(wc -c < "$result_csv") -gt 100 ]]; then
      echo "  [SKIP] $label (already done)"
      metrics=$(extract_metrics "$result_csv")
    else
      mkdir -p "$out_dir"
      # Create modified config with custom output dir
      sed "s|output_dir:.*|output_dir: $out_dir|" "$config_file" > "/tmp/m2opt_${strategy_name}_${label}.yaml"

      echo -n "  [RUN]  $label ... "

      if eval "$env_vars" LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} \
         "$BENCH_BIN" -config "/tmp/m2opt_${strategy_name}_${label}.yaml" > "$out_dir/stdout.log" 2>&1; then
        echo "done"
      else
        echo "FAILED"
        failed=$((failed + 1))
        echo "$strategy_name,$label,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,$rwlock_sp99,$mvcc_sp99,$mvcc_iqps,FAIL,FAIL" >> "$LOG"
        continue
      fi

      metrics=$(extract_metrics "$result_csv")
    fi

    if [[ "$metrics" == *"ERROR"* ]]; then
      echo "  [ERROR] Could not parse results for $label"
      echo "$strategy_name,$label,$metrics,$rwlock_sp99,$mvcc_sp99,$mvcc_iqps,FAIL,FAIL" >> "$LOG"
      failed=$((failed + 1))
      continue
    fi

    # Parse metrics
    IFS=',' read -r sp99 smean iqps ip99 recall fm f99 <<< "$metrics"

    # Check pass criteria:
    # search p99 should be <= 1.5x rwlock (competitive)
    # insert qps should be >= 0.7x mvcc (not too degraded)
    pass_search=$(python3 -c "print('PASS' if float('$sp99') <= 1.5 * float('$rwlock_sp99') else 'FAIL')")
    pass_insert=$(python3 -c "print('PASS' if float('$iqps') >= 0.7 * float('$mvcc_iqps') else 'FAIL')")

    status="✓"
    if [[ "$pass_search" == "FAIL" ]] || [[ "$pass_insert" == "FAIL" ]]; then
      status="✗"
      failed=$((failed + 1))
    else
      passed=$((passed + 1))
    fi

    echo "  [$status] $label: sp99=$sp99 (rwl=$rwlock_sp99) iqps=$iqps (mvcc=$mvcc_iqps) | search:$pass_search insert:$pass_insert"
    echo "$strategy_name,$label,$metrics,$rwlock_sp99,$mvcc_sp99,$mvcc_iqps,$pass_search,$pass_insert" >> "$LOG"
  done
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Summary                                                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "Total: $total | Passed: $passed | Failed: $failed"
echo ""
echo "Full results: $LOG"
echo ""
echo "── Analyze: ──"
echo "python3 scripts/analyze_m2_optimize.py"
