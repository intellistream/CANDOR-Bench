#!/bin/bash
# ── Per-Node Yield Experiment ──
# Tests the new per-node yield (no convergence gate) with L-based control.
# Compares against old budget-drain results on gist (960d) and deep1M (96d).
#
# Key change: yield at every node expansion, controlled purely by starvation_alpha.
# High-dim search is expensive → L spikes after one yield → self-adaptive.
#
# Usage:
#   bash scripts/run_pernode_yield_exp.sh          # run all
#   bash scripts/run_pernode_yield_exp.sh gist     # gist only
#   bash scripts/run_pernode_yield_exp.sh deep1M   # deep1M only

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Build ──
echo "Building bench binary..."
export CC=gcc
export CXX=g++
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$(pwd)/build/lib
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

go build -o bin/bench ./cmd/bench_dev/
echo "Build done."

BENCH_BIN="./bin/bench"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_ROOT="./result/pernode_yield_${TIMESTAMP}"
CONFIG_ROOT="./config/pernode_yield_${TIMESTAMP}"
mkdir -p "$RESULT_ROOT" "$CONFIG_ROOT"

FILTER="${1:-all}"
BEGIN_NUM=50000

# ── Alpha sweep values ──
# Current default is 0.20. Test tighter values for high-dim.
ALPHAS=(0.10 0.15 0.20 0.30 0.50)

# ── Budget values for drain (inter-insert) ──
# With shared L-ledger, drain is also L-controlled, so B is just upper bound.
BUDGETS=(4 8)

# ── Batch sizes ──
BATCH_SIZES=(20 200)

# ── Workloads: read_rate|write_rate|label ──
WORKLOADS=(
  "90000|10000|r90k_w10k"
  "40000|40000|r40k_w40k"
  "10000|90000|r10k_w90k"
)

# ── Datasets ──
declare -A DS_DATA DS_INCR_Q DS_INCR_GT DS_OVERALL_Q DS_OVERALL_GT
DS_DATA[gist]="../data/gist/gist_base.bin"
DS_INCR_Q[gist]="../data/gist/gist_query_stream.bin"
DS_INCR_GT[gist]="../data/gist/gist_base_i20_offset_index.txt"
DS_OVERALL_Q[gist]="../data/gist/gist_query.bin"
DS_OVERALL_GT[gist]="../data/gist/gist_base.gt20"

DS_DATA[deep1M]="../data/deep1M/deep1M_base.bin"
DS_INCR_Q[deep1M]="../data/deep1M/deep1M_query_stream.bin"
DS_INCR_GT[deep1M]="../data/deep1M/deep1M_stream_i100_offset_index.txt"
DS_OVERALL_Q[deep1M]="../data/deep1M/deep1M_query.bin"
DS_OVERALL_GT[deep1M]="../data/deep1M/deep1M_base.gt20"

DS_DATA[sift]="../data/sift/sift_base.bin"
DS_INCR_Q[sift]="../data/sift/sift_query_stream.bin"
DS_INCR_GT[sift]="../data/sift/sift_base_i20_offset_index.txt"
DS_OVERALL_Q[sift]="../data/sift/sift_query.bin"
DS_OVERALL_GT[sift]="../data/sift/sift_base.gt20"

DATASETS_LIST=(gist deep1M sift)

# ── Per-dataset index/search params for ~95% recall ──
declare -A DS_M DS_EFC DS_EFS
DS_M[sift]=16;    DS_EFC[sift]=200;    DS_EFS[sift]=50
DS_M[deep1M]=16;  DS_EFC[deep1M]=200;  DS_EFS[deep1M]=100
DS_M[gist]=24;    DS_EFC[gist]=400;    DS_EFS[gist]=200

generate_config() {
  local ds="$1" read_rate="$2" write_rate="$3" out_dir="$4" batch_size="$5"
  local m="${DS_M[$ds]}" efc="${DS_EFC[$ds]}" efs="${DS_EFS[$ds]}"
  cat <<YAML
data:
  dataset_name: ${ds}
  max_elements: -1
  begin_num: ${BEGIN_NUM}
  data_type: float
  data_path: ${DS_DATA[$ds]}
  incr_query_path: ${DS_INCR_Q[$ds]}
  incr_gt_path: ${DS_INCR_GT[$ds]}
  overall_query_path: ${DS_OVERALL_Q[$ds]}
  overall_gt_path: ${DS_OVERALL_GT[$ds]}
index:
  index_type: annchor-preempt
  m: ${m}
  ef_construction: ${efc}
search:
  recall_at: 10
  ef_search: ${efs}
  enable_mvcc: true
  enable_preempt_m2: true
workload:
  batch_size: [${batch_size}]
  num_threads: [16]
  queue_size: 0
  rate_groups(r/w):
    - [${read_rate}, ${write_rate}]
  with_external_rw_lock: false
  query_mode: chasing
compare:
  enabled: false
result:
  output_dir: ${out_dir}
profile:
  memory_monitor_interval: 100
  enable_memory_profile: false
YAML
}

run_one() {
  local config="$1" out_dir="$2" label="$3" logfile="$4"

  if [[ -f "$out_dir/benchmark_results.csv" ]]; then
    echo "[SKIP] $label — already done"
    return 0
  fi

  echo "[RUN]  $label"
  mkdir -p "$out_dir"

  set +e
  $BENCH_BIN -config "$config" > "$logfile" 2>&1
  local rc=$?
  set -e

  if [[ $rc -eq 0 ]] && [[ -f "$out_dir/benchmark_results.csv" ]]; then
    echo "[DONE] $label"
  else
    echo "[FAIL] $label (exit=$rc) — see $logfile"
  fi
}

# ── Main loop ──
total=0
for ds in "${DATASETS_LIST[@]}"; do
  [[ "$FILTER" != "all" && "$FILTER" != "$ds" ]] && continue

  for wl_entry in "${WORKLOADS[@]}"; do
    IFS='|' read -r read_rate write_rate wl_label <<< "$wl_entry"

    for batch_size in "${BATCH_SIZES[@]}"; do
      for alpha in "${ALPHAS[@]}"; do
        for budget in "${BUDGETS[@]}"; do
          tag="${ds}_${wl_label}_bs${batch_size}_a${alpha}_b${budget}"
          out_dir="${RESULT_ROOT}/${tag}"
          config="${CONFIG_ROOT}/${tag}.yaml"

          generate_config "$ds" "$read_rate" "$write_rate" "$out_dir" "$batch_size" > "$config"

          # Set env vars for this run
          export ANN_PREEMPT_STARVATION_ALPHA="$alpha"
          export ANN_PREEMPT_DRAIN_ALPHA="$alpha"
          export ANN_PREEMPT_SEARCH_BUDGET="$budget"

          logfile="${RESULT_ROOT}/${tag}.log"
          run_one "$config" "$out_dir" "$tag" "$logfile"
          total=$((total + 1))
        done
      done
    done
  done
done

echo ""
echo "========================================"
echo "  Per-node yield experiment complete"
echo "  Total runs: $total"
echo "  Results in: $RESULT_ROOT"
echo "========================================"

# ── Collect summary CSV ──
SUMMARY="${RESULT_ROOT}/summary.csv"
echo "dataset,workload,batch_size,alpha,budget,ins_qps,ins_p99,srch_qps,srch_p99_op,srch_p99_e2e,recall,fiber_yield_count,fiber_starvation_refuses,fiber_searches_serviced" > "$SUMMARY"

for csv in "$RESULT_ROOT"/*/benchmark_results.csv; do
  [[ -f "$csv" ]] || continue
  dir=$(dirname "$csv")
  tag=$(basename "$dir")

  # Parse tag: ds_wl_bsN_aX_bM
  ds=$(echo "$tag" | cut -d_ -f1)
  wl=$(echo "$tag" | sed 's/^[^_]*_//' | sed 's/_bs[0-9]*//' | sed 's/_a[0-9.]*//' | sed 's/_b[0-9]*$//')
  bs=$(echo "$tag" | grep -oP 'bs\K[0-9]+')
  alpha=$(echo "$tag" | grep -oP 'a\K[0-9.]+')
  budget=$(echo "$tag" | grep -oP 'b\K[0-9]+$')

  # Extract metrics from benchmark_results.csv (skip header)
  tail -n +2 "$csv" | while IFS=',' read -r line; do
    echo "${ds},${wl},${bs},${alpha},${budget},${line}"
  done
done >> "$SUMMARY"

echo "Summary written to: $SUMMARY"
