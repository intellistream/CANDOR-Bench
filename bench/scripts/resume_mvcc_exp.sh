#!/bin/bash
# Resume mvcc experiment: skip configs that already have results
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/build/lib"

BENCH_BIN="./bin/bench"
CONFIG_DIR="./config/mvcc_exp"
RESULT_BASE="./result/mvcc_exp"

DATASETS=(
  "sift|../data/sift/sift_base.bin|../data/sift/sift_query_stream.bin|../data/sift/sift_stream_i100_offset_index.txt|../data/sift/sift_query.bin|../data/sift/sift_base.gt20|16|200|35"
  "gist|../data/gist/gist_base.bin|../data/gist/gist_query_stream.bin|../data/gist/gist_stream_i100_offset_index.txt|../data/gist/gist_query.bin|../data/gist/gist_base.gt20|24|500|80"
)
QUERY_MODES=("chasing" "round_robin")
LOCK_MODES=("rwlock" "nolock" "mvcc")

# Reuse generate_config from run_mvcc_exp.sh logic
generate_config() {
  local ds_name=$1 data_path=$2 incr_query=$3 incr_gt=$4
  local overall_query=$5 overall_gt=$6
  local m=$7 ef_c=$8 ef_s=$9
  local qmode=${10} lock=${11}

  local index_type="hnsw"
  local ext_rw="false"
  local mvcc_line=""
  if [ "$lock" = "rwlock" ]; then
    index_type="hnsw"
    ext_rw="true"
  elif [ "$lock" = "nolock" ]; then
    index_type="hnsw"
    ext_rw="false"
  elif [ "$lock" = "mvcc" ]; then
    index_type="annchor-m1"
    ext_rw="false"
    mvcc_line="  enable_mvcc: true"
  fi

  local fname="${CONFIG_DIR}/${ds_name}_${qmode}_${lock}.yaml"

  cat > "$fname" <<EOF
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
  index_type: ${index_type}
  m: ${m}
  ef_construction: ${ef_c}

search:
  recall_at: 10
  ef_search: ${ef_s}
${mvcc_line}

workload:
  batch_size: [20, 200]
  num_threads: [16, 64]
  queue_size: 0
  rate_groups(r/w):
    - [20000, 20000]
    - [4000, 36000]
    - [36000, 4000]
  with_external_rw_lock: ${ext_rw}
  query_mode: ${qmode}
  precompute_schedule: true

compare:
  enabled: false
  simulate:
    enabled: false

result:
  output_dir: ${RESULT_BASE}/${ds_name}/${qmode}_${lock}

profile:
  memory_monitor_interval: 100
  enable_memory_profile: true
EOF

  echo "$fname"
}

# Check if a config already has valid results (look for summary CSV)
is_done() {
  local result_dir="$1"
  # Consider done if there's at least one summary CSV with >1 line
  local csv_count
  csv_count=$(find "$result_dir" -name "*.csv" -size +100c 2>/dev/null | wc -l)
  [ "$csv_count" -gt 0 ]
}

total=12
count=0
skipped=0
for ds_entry in "${DATASETS[@]}"; do
  IFS='|' read -r ds_name data_path incr_query incr_gt overall_query overall_gt m ef_c ef_s <<< "$ds_entry"
  for qmode in "${QUERY_MODES[@]}"; do
    for lock in "${LOCK_MODES[@]}"; do
      count=$((count + 1))
      result_dir="${RESULT_BASE}/${ds_name}/${qmode}_${lock}"

      if is_done "$result_dir"; then
        echo "[$count/$total] SKIP (already done): ${ds_name} ${qmode} ${lock}"
        skipped=$((skipped + 1))
        continue
      fi

      cfg=$(generate_config "$ds_name" "$data_path" "$incr_query" "$incr_gt" \
                            "$overall_query" "$overall_gt" "$m" "$ef_c" "$ef_s" \
                            "$qmode" "$lock")
      echo "[$count/$total] Running: $cfg"
      mkdir -p "$result_dir"
      $BENCH_BIN -config "$cfg" 2>&1 | tee "${result_dir}/bench.log"
      echo "[$count/$total] Done: ${ds_name} ${qmode} ${lock}"
      echo ""
    done
  done
done

echo "=== Resume complete. Skipped ${skipped}, ran $((count - skipped)) configs ==="
