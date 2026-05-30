#!/bin/bash
# ── Yield Policy Sweep: Test adaptive alpha on M2's losing scenarios ──
#
# Hypothesis: M2 fails in write-heavy and gist because alpha=0.20 (static)
# is too aggressive. Sweeping alpha from 0.05 to 0.40 should reveal
# the optimal operating point for each workload.
#
# Losing scenarios to test:
#   1. gist b20 w40k/r40k (fiber srch_p99=51.20 vs rwlock 15.77)
#   2. gist b20 w90k/r10k (fiber srch_p99=50.37 vs rwlock 15.85)
#   3. deep1M b20 w90k/r10k (fiber srch_p99=12.22 vs rwlock 6.69)
#   4. sift b200 w90k/r10k (fiber srch_p99=90.97 vs rwlock 23.82)
#
# Also include winning scenarios as control:
#   5. deep1M b20 w10k/r90k (fiber wins big)
#   6. deep1M b200 w40k/r40k (fiber wins big)

set -euo pipefail
cd "$(dirname "$0")/.."

EXP_NAME="yield_policy_sweep"
BENCH_BIN="./bin/bench"
THREADS=16

# Alpha values to sweep
ALPHAS=(0.05 0.10 0.15 0.20 0.30 0.40)

# Scenarios: ds|data_path|incr_query|incr_gt|overall_query|overall_gt|M|ef_c|ef_s|batch|wrate|rrate|label
SCENARIOS=(
  "gist|../data/gist/gist_base.bin|../data/gist/gist_query_stream.bin|../data/gist/gist_stream_i100_offset_index.txt|../data/gist/gist_query.bin|../data/gist/gist_base.gt20|16|200|35|20|40000|40000|gist_b20_balanced"
  "gist|../data/gist/gist_base.bin|../data/gist/gist_query_stream.bin|../data/gist/gist_stream_i100_offset_index.txt|../data/gist/gist_query.bin|../data/gist/gist_base.gt20|16|200|35|20|90000|10000|gist_b20_writeheavy"
  "deep1M|../data/deep1M/deep1M_base.bin|../data/deep1M/deep1M_query_stream.bin|../data/deep1M/deep1M_stream_i100_offset_index.txt|../data/deep1M/deep1M_query.bin|../data/deep1M/deep1M_base.gt20|16|200|35|20|90000|10000|deep1M_b20_writeheavy"
  "sift|../data/sift/sift_base.bin|../data/sift/sift_query_stream.bin|../data/sift/sift_stream_i100_offset_index.txt|../data/sift/sift_query.bin|../data/sift/sift_base.gt20|16|200|35|200|90000|10000|sift_b200_writeheavy"
  "deep1M|../data/deep1M/deep1M_base.bin|../data/deep1M/deep1M_query_stream.bin|../data/deep1M/deep1M_stream_i100_offset_index.txt|../data/deep1M/deep1M_query.bin|../data/deep1M/deep1M_base.gt20|16|200|35|20|10000|90000|deep1M_b20_readheavy"
  "deep1M|../data/deep1M/deep1M_base.bin|../data/deep1M/deep1M_query_stream.bin|../data/deep1M/deep1M_stream_i100_offset_index.txt|../data/deep1M/deep1M_query.bin|../data/deep1M/deep1M_base.gt20|16|200|35|200|40000|40000|deep1M_b200_balanced"
)

generate_config() {
  local ds_name="$1" data_path="$2" incr_query="$3" incr_gt="$4"
  local overall_query="$5" overall_gt="$6"
  local m="$7" ef_c="$8" ef_s="$9" batch="${10}"
  local wrate="${11}" rrate="${12}" out_dir="${13}" alpha="${14}"

  cat <<YAML
data:
  dataset_name: ${ds_name}
  max_elements: -1
  begin_num: 50000
  data_type: float
  data_path: ${data_path}
  incr_query_path: ${incr_query}
  incr_gt_path: ${incr_gt}
  overall_query_path: ${overall_query}
  overall_gt_path: ${overall_gt}
index:
  index_type: annchor-preempt
  m: ${m}
  ef_construction: ${ef_c}
search:
  recall_at: 10
  ef_search: ${ef_s}
  enable_mvcc: true
  enable_preempt_m2: true
workload:
  batch_size: [${batch}]
  num_threads: [${THREADS}]
  queue_size: 0
  rate_groups(r/w):
    - [${wrate}, ${rrate}]
  with_external_rw_lock: false
  query_mode: chasing
compare:
  enabled: false
  simulate:
    enabled: false
result:
  output_dir: ${out_dir}
profile:
  memory_monitor_interval: 100
  enable_memory_profile: true
YAML
}

is_done() {
  local csv="$1/benchmark_results.csv"
  [[ -f "$csv" ]] && [[ $(wc -c < "$csv") -gt 100 ]]
}

total=0
skipped=0
failed=0

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Yield Policy Sweep: Alpha parameter tuning          ║"
echo "╚══════════════════════════════════════════════════════╝"

for scenario in "${SCENARIOS[@]}"; do
  IFS='|' read -r ds_name data_path incr_query incr_gt overall_query overall_gt m ef_c ef_s batch wrate rrate label <<< "$scenario"

  for alpha in "${ALPHAS[@]}"; do
    alpha_label=$(echo "$alpha" | tr '.' 'p')
    out_dir="./result/${EXP_NAME}/${label}/alpha_${alpha_label}"
    config_file="./config/${EXP_NAME}/${label}_alpha_${alpha_label}.yaml"

    total=$((total + 1))

    if is_done "$out_dir"; then
      echo "[SKIP] $out_dir (already done)"
      skipped=$((skipped + 1))
      continue
    fi

    mkdir -p "$(dirname "$config_file")"
    generate_config "$ds_name" "$data_path" "$incr_query" "$incr_gt" \
      "$overall_query" "$overall_gt" "$m" "$ef_c" "$ef_s" "$batch" \
      "$wrate" "$rrate" "$out_dir" "$alpha" > "$config_file"

    echo ""
    echo "================================================================"
    echo "[${label}] alpha=${alpha}"
    echo "================================================================"

    # Set alpha via environment variable
    if ANN_PREEMPT_STARVATION_ALPHA="$alpha" \
       LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} \
       "$BENCH_BIN" -config "$config_file" 2>&1 | \
       grep -E "Progress.*100%|Overall recall|Results written|Error|panic"; then
      echo "[DONE] $out_dir"
    else
      echo "[FAIL] $out_dir"
      failed=$((failed + 1))
    fi
  done
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Summary                                             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "Total: $total | Skipped: $skipped | Failed: $failed | Ran: $((total - skipped - failed))"
echo ""
echo "── Analyze with: ──"
echo "python3 scripts/analyze_yield_sweep.py"
