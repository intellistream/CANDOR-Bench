#!/bin/bash
# ── Graph Quality Drift Experiment ──
#
# Compares batch-built graph vs incremental-insert graph on identical data.
# Goal: quantify whether incremental insert produces worse graph quality
# (lower recall, more hops/distance computations per search) than batch build.
#
# This is the key structural advantage of segment systems:
# they periodically rebuild (seal) segments from scratch → better graph quality.
#
# Setup:
#   - Batch:  begin_num ≈ max_elements  (almost all points batch-built)
#   - Incr-X: begin_num = X% of max     (rest inserted incrementally)
#   - All configs end with same total points
#   - Single thread, no MVCC/fiber, no rate limit → pure graph quality comparison
#   - Search-only after all inserts complete (near-zero insert rate)
#
# Datasets: sift (100K, 128-dim), deep1M (1M, 96-dim)
# Metrics: overall recall@10, dist_computations, hops

set -euo pipefail
cd "$(dirname "$0")/.."

EXP_NAME="graph_quality_drift"
BENCH_BIN="./bin/bench"

# Dataset definitions: name|data_path|incr_query|incr_gt|overall_query|overall_gt|M|ef_c|ef_s|max_elements
# Use sift_base_200k (has matching GT) and deep1M
DATASETS=(
  "deep1M|../data/deep1M/deep1M_base.bin|../data/deep1M/deep1M_query_stream.bin|../data/deep1M/deep1M_stream_i100_offset_index.txt|../data/deep1M/deep1M_query.bin|../data/deep1M/deep1M_base.gt20|16|200|35|1000000"
)

THREADS=8

generate_config() {
  local ds_name="$1" data_path="$2" incr_query="$3" incr_gt="$4"
  local overall_query="$5" overall_gt="$6"
  local m="$7" ef_c="$8" ef_s="$9" max_elements="${10}"
  local begin_num="${11}" out_dir="${12}" batch_size="${13}"

  cat <<YAML
data:
  dataset_name: ${ds_name}
  max_elements: ${max_elements}
  begin_num: ${begin_num}
  data_type: float
  data_path: ${data_path}
  incr_query_path: ${incr_query}
  incr_gt_path: ${incr_gt}
  overall_query_path: ${overall_query}
  overall_gt_path: ${overall_gt}
index:
  index_type: hnsw
  m: ${m}
  ef_construction: ${ef_c}
search:
  recall_at: 10
  ef_search: ${ef_s}
  enable_mvcc: false
workload:
  batch_size: [${batch_size}]
  num_threads: [${THREADS}]
  queue_size: 0
  rate_groups(r/w):
    - [999999999, 999999999]
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
echo "║  Graph Quality Drift: Batch vs Incremental Build    ║"
echo "╚══════════════════════════════════════════════════════╝"

for ds_entry in "${DATASETS[@]}"; do
  IFS='|' read -r ds_name data_path incr_query incr_gt overall_query overall_gt m ef_c ef_s max_elem <<< "$ds_entry"

  # Define begin_num ratios to test
  # batch: 99% preloaded (need at least 1 batch to insert for benchmark to run)
  # incr_50: 50% preloaded
  # incr_10: 10% preloaded
  # incr_1: 1% preloaded (nearly full incremental)

  declare -A BEGIN_NUMS
  BEGIN_NUMS["batch_99pct"]=$(( max_elem * 99 / 100 ))
  BEGIN_NUMS["incr_50pct"]=$(( max_elem * 50 / 100 ))
  BEGIN_NUMS["incr_10pct"]=$(( max_elem * 10 / 100 ))
  BEGIN_NUMS["incr_1pct"]=$(( max_elem * 1 / 100 ))

  for label in batch_99pct incr_50pct incr_10pct incr_1pct; do
    begin_num=${BEGIN_NUMS[$label]}
    # Use batch_size=100 for sift, 1000 for deep1M (to keep experiment reasonable)
    if [[ "$ds_name" == "sift" ]]; then
      batch_size=100
    else
      batch_size=1000
    fi

    out_dir="./result/${EXP_NAME}/${ds_name}/${label}"
    config_file="./config/${EXP_NAME}/${ds_name}_${label}.yaml"

    total=$((total + 1))

    if is_done "$out_dir"; then
      echo "[SKIP] $out_dir (already done)"
      skipped=$((skipped + 1))
      continue
    fi

    mkdir -p "$(dirname "$config_file")"
    generate_config "$ds_name" "$data_path" "$incr_query" "$incr_gt" \
      "$overall_query" "$overall_gt" "$m" "$ef_c" "$ef_s" "$max_elem" \
      "$begin_num" "$out_dir" "$batch_size" > "$config_file"

    echo ""
    echo "================================================================"
    echo "[${ds_name}] ${label}: begin_num=${begin_num} / max=${max_elem}"
    echo "================================================================"

    if LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} "$BENCH_BIN" -config "$config_file" 2>&1 | \
       grep -E "Progress.*[0-9]+%|Overall recall|Results written|Error|panic|Build completed"; then
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
echo "python3 scripts/analyze_graph_quality_drift.py"
