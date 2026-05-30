#!/bin/bash
# ── Sift TBB+Fiber experiment ──
# Compare: rwlock / mvcc / fiber on sift
# 3 workloads × 2 batch sizes = 6 per mode, 18 total
set -euo pipefail
cd "$(dirname "$0")/.."

BENCH_BIN="./bin/bench_dev"
EXP_DIR="./result/tbb_fiber_sift"
CONFIG_DIR="./config/tbb_fiber_sift"
mkdir -p "$CONFIG_DIR"

DS="sift"
DATA_PATH="../data/sift/sift_base.bin"
INCR_QUERY="../data/sift/sift_query_stream.bin"
INCR_GT="../data/sift/sift_stream_i100_offset_index.txt"
OVERALL_QUERY="../data/sift/sift_query.bin"
OVERALL_GT="../data/sift/sift_base.gt20"
M=16; EFC=200; EFS=35; THREADS=16; BEGIN=50000

# mode: name|enable_mvcc|enable_preempt|rw_lock
MODES=(
  "rwlock|false|false|true"
  "mvcc|true|false|false"
  "fiber|true|true|false"
)

RATES=(
  "40000|40000"
  "10000|90000"
  "90000|10000"
)

BATCHES=(20 200)

generate_config() {
  local mode="$1" enable_mvcc="$2" enable_preempt="$3" rw_lock="$4"
  local batch="$5" wrate="$6" rrate="$7" outdir="$8"
  cat <<YAML
data:
  dataset_name: $DS
  max_elements: -1
  begin_num: $BEGIN
  data_type: float
  data_path: $DATA_PATH
  incr_query_path: $INCR_QUERY
  incr_gt_path: $INCR_GT
  overall_query_path: $OVERALL_QUERY
  overall_gt_path: $OVERALL_GT
index:
  index_type: annchor-preempt
  m: $M
  ef_construction: $EFC
search:
  recall_at: 10
  ef_search: $EFS
  enable_mvcc: $enable_mvcc
  enable_preempt_m2: $enable_preempt
workload:
  batch_size: [$batch]
  num_threads: [$THREADS]
  queue_size: 0
  rate_groups(r/w):
    - [$wrate, $rrate]
  with_external_rw_lock: $rw_lock
  query_mode: chasing
compare:
  enabled: false
  simulate:
    enabled: false
result:
  output_dir: $outdir
profile:
  memory_monitor_interval: 100
  enable_memory_profile: true
YAML
}

is_done() {
  local csv="$1/benchmark_results.csv"
  [[ -f "$csv" ]] && [[ $(wc -c < "$csv") -gt 100 ]]
}

total=0; done=0; failed=0

for mode_entry in "${MODES[@]}"; do
  IFS='|' read -r mode enable_mvcc enable_preempt rw_lock <<< "$mode_entry"
  for batch in "${BATCHES[@]}"; do
    for rate_entry in "${RATES[@]}"; do
      IFS='|' read -r wrate rrate <<< "$rate_entry"
      outdir="$EXP_DIR/${mode}_b${batch}_w${wrate}_r${rrate}"
      config="$CONFIG_DIR/${mode}_b${batch}_w${wrate}_r${rrate}.yaml"
      total=$((total+1))

      if is_done "$outdir"; then
        echo "[SKIP] $outdir"
        done=$((done+1))
        continue
      fi

      generate_config "$mode" "$enable_mvcc" "$enable_preempt" "$rw_lock" \
        "$batch" "$wrate" "$rrate" "$outdir" > "$config"

      echo ""
      echo "========================================"
      echo "[RUN $((done+1))/$total] $mode b$batch w${wrate}-r${rrate}"
      echo "========================================"

      if LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} "$BENCH_BIN" -config "$config" 2>&1 | \
         grep -E "Progress.*100%|recall|Results written|Error|panic"; then
        echo "[DONE] $outdir"
        done=$((done+1))
      else
        echo "[FAIL] $outdir"
        failed=$((failed+1))
      fi
    done
  done
done

echo ""
echo "========================================"
echo "Summary: $done/$total done, $failed failed"
echo "========================================"
