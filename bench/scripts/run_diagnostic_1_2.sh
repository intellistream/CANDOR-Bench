#!/bin/bash
# ── Diagnostic Experiments #1 & #2 ──
#
# Diagnostic #1: Search interference tax
#   - "search-only" baseline: begin_num=99000, max_elements=100000, insert_rate=100
#   - Compare search p99 vs existing mixed workload results
#
# Diagnostic #2: Watermark HOL blocking
#   - MVCC mode, sweep batch sizes (20, 100, 200, 500)
#   - Balanced rate 40000/40000
#   - Measure freshness_contiguity vs freshness_total gap at different batch sizes
#   - Larger batch → more HOL blocking opportunity
#
# Datasets: sift, deep1M (skip gist for speed)
# Threads: 16

set -euo pipefail
cd "$(dirname "$0")/.."

EXP_NAME="diagnostic_1_2"
BENCH_BIN="./bin/bench"

DATASETS=(
  "sift|../data/sift/sift_base.bin|../data/sift/sift_query_stream.bin|../data/sift/sift_stream_i100_offset_index.txt|../data/sift/sift_query.bin|../data/sift/sift_base.gt20|16|200|35"
  "deep1M|../data/deep1M/deep1M_base.bin|../data/deep1M/deep1M_query_stream.bin|../data/deep1M/deep1M_stream_i100_offset_index.txt|../data/deep1M/deep1M_query.bin|../data/deep1M/deep1M_base.gt20|16|200|35"
)

THREADS=16

generate_config() {
  local ds_name="$1" data_path="$2" incr_query="$3" incr_gt="$4" overall_query="$5" overall_gt="$6"
  local m="$7" ef_c="$8" ef_s="$9" index_type="${10}" enable_mvcc="${11}" enable_preempt="${12}"
  local rw_lock="${13}" batch_size="${14}" write_rate="${15}" read_rate="${16}" out_dir="${17}"
  local begin_num="${18}" max_elements="${19}"

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
  index_type: ${index_type}
  m: ${m}
  ef_construction: ${ef_c}
search:
  recall_at: 10
  ef_search: ${ef_s}
  enable_mvcc: ${enable_mvcc}
  enable_preempt_m2: ${enable_preempt}
workload:
  batch_size: [${batch_size}]
  num_threads: [${THREADS}]
  queue_size: 0
  rate_groups(r/w):
    - [${write_rate}, ${read_rate}]
  with_external_rw_lock: ${rw_lock}
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

echo "╔══════════════════════════════════════════════╗"
echo "║  Diagnostic #1: Search Interference Tax      ║"
echo "║  (near search-only: 99k preloaded, 1k slow)  ║"
echo "╚══════════════════════════════════════════════╝"

# ── Diagnostic #1: Search-only baseline ──
# Modes: rwlock, mvcc, fiber — all with near-zero write pressure
# begin_num=99000, max_elements=100000, insert_rate=100, search_rate=100000
# batch_size=20 to match existing mixed experiments
D1_MODES=(
  "rwlock|annchor-preempt|false|false|true"
  "mvcc|annchor-preempt|true|false|false"
  "fiber|annchor-preempt|true|true|false"
)

for ds_entry in "${DATASETS[@]}"; do
  IFS='|' read -r ds_name data_path incr_query incr_gt overall_query overall_gt m ef_c ef_s <<< "$ds_entry"

  for mode_entry in "${D1_MODES[@]}"; do
    IFS='|' read -r mode_name index_type enable_mvcc enable_preempt rw_lock <<< "$mode_entry"

    out_dir="./result/${EXP_NAME}/${ds_name}/d1_searchonly_${mode_name}_b20"
    config_file="./config/${EXP_NAME}/${ds_name}_d1_searchonly_${mode_name}_b20.yaml"

    total=$((total + 1))

    if is_done "$out_dir"; then
      echo "[SKIP] $out_dir (already done)"
      skipped=$((skipped + 1))
      continue
    fi

    mkdir -p "$(dirname "$config_file")"
    generate_config "$ds_name" "$data_path" "$incr_query" "$incr_gt" "$overall_query" "$overall_gt" \
      "$m" "$ef_c" "$ef_s" "$index_type" "$enable_mvcc" "$enable_preempt" "$rw_lock" \
      20 100 100000 "$out_dir" 99000 100000 > "$config_file"

    echo ""
    echo "================================================================"
    echo "[D1] ${ds_name} / ${mode_name} / search-only baseline"
    echo "================================================================"

    if LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} "$BENCH_BIN" -config "$config_file" 2>&1 | \
       grep -E "Progress.*[0-9]+%|Overall recall|Results written|Error|panic"; then
      echo "[DONE] $out_dir"
    else
      echo "[FAIL] $out_dir"
      failed=$((failed + 1))
    fi
  done
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Diagnostic #2: Watermark HOL Blocking       ║"
echo "║  (MVCC, sweep batch size 20→500)             ║"
echo "╚══════════════════════════════════════════════╝"

# ── Diagnostic #2: Watermark HOL blocking ──
# MVCC mode only (watermark matters here, not in rwlock)
# Sweep batch sizes: 20, 100, 200, 500
# Balanced rate 40000/40000 (saturated)
# begin_num=50000 (same as fiber_exp)
# If HOL blocking is real, larger batch → disproportionately worse contiguity lag
D2_BATCH_SIZES=(20 100 200 500)

for ds_entry in "${DATASETS[@]}"; do
  IFS='|' read -r ds_name data_path incr_query incr_gt overall_query overall_gt m ef_c ef_s <<< "$ds_entry"

  for batch_size in "${D2_BATCH_SIZES[@]}"; do
    out_dir="./result/${EXP_NAME}/${ds_name}/d2_hol_mvcc_b${batch_size}_w40000_r40000"
    config_file="./config/${EXP_NAME}/${ds_name}_d2_hol_mvcc_b${batch_size}_w40000_r40000.yaml"

    total=$((total + 1))

    if is_done "$out_dir"; then
      echo "[SKIP] $out_dir (already done)"
      skipped=$((skipped + 1))
      continue
    fi

    mkdir -p "$(dirname "$config_file")"
    generate_config "$ds_name" "$data_path" "$incr_query" "$incr_gt" "$overall_query" "$overall_gt" \
      "$m" "$ef_c" "$ef_s" "annchor-preempt" "true" "false" "false" \
      "$batch_size" 40000 40000 "$out_dir" 50000 -1 > "$config_file"

    echo ""
    echo "================================================================"
    echo "[D2] ${ds_name} / MVCC / b${batch_size} / HOL sweep"
    echo "================================================================"

    if LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} "$BENCH_BIN" -config "$config_file" 2>&1 | \
       grep -E "Progress.*[0-9]+%|Overall recall|Results written|Error|panic"; then
      echo "[DONE] $out_dir"
    else
      echo "[FAIL] $out_dir"
      failed=$((failed + 1))
    fi
  done
done

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Summary                                     ║"
echo "╚══════════════════════════════════════════════╝"
echo "Total: $total | Skipped: $skipped | Failed: $failed | Ran: $((total - skipped - failed))"
echo ""
echo "── Analyze with: ──"
echo "python3 scripts/analyze_diagnostic_1_2.py"
