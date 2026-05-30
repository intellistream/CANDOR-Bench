#!/bin/bash
# Experiment: Compare external lock modes (rwlock / nolock / mvcc)
# across 4 datasets × 2 query modes × 3 lock modes
# Each config sweeps batch_size=[20,100,200] × threads=[16,32,64]
#
# Usage: cd bench && bash scripts/run_mvcc_exp.sh

set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/build/lib"

BENCH_BIN="./bin/bench"
CONFIG_DIR="./config/mvcc_exp"
RESULT_BASE="./result/mvcc_exp"

mkdir -p "$CONFIG_DIR" "$RESULT_BASE"

# ── Dataset configs ──────────────────────────────────────────────
# Format: name|data_path|incr_query|incr_gt|overall_query|overall_gt|m|ef_c|ef_s
DATASETS=(
  "sift|../data/sift/sift_base.bin|../data/sift/sift_query_stream.bin|../data/sift/sift_stream_i100_offset_index.txt|../data/sift/sift_query.bin|../data/sift/sift_base.gt20|16|200|35"
  "gist|../data/gist/gist_base.bin|../data/gist/gist_query_stream.bin|../data/gist/gist_stream_i100_offset_index.txt|../data/gist/gist_query.bin|../data/gist/gist_base.gt20|24|500|80"
)

QUERY_MODES=("chasing" "round_robin")
# Lock modes: rwlock = external rw lock; nolock = no external lock; mvcc = mvcc mode
LOCK_MODES=("rwlock" "nolock" "mvcc")

generate_config() {
  local ds_name=$1 data_path=$2 incr_query=$3 incr_gt=$4
  local overall_query=$5 overall_gt=$6
  local m=$7 ef_c=$8 ef_s=$9
  local qmode=${10} lock=${11}

  # Determine index_type and lock flags
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
    - [40000, 40000]
    - [10000, 90000]
    - [90000, 10000]
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

# ── Generate all configs & run ───────────────────────────────────
total=0
for ds_entry in "${DATASETS[@]}"; do
  IFS='|' read -r ds_name data_path incr_query incr_gt overall_query overall_gt m ef_c ef_s <<< "$ds_entry"
  for qmode in "${QUERY_MODES[@]}"; do
    for lock in "${LOCK_MODES[@]}"; do
      total=$((total + 1))
    done
  done
done

echo "=== MVCC Experiment: ${total} configs, each sweeps batch×threads×rate ==="
echo ""

count=0
for ds_entry in "${DATASETS[@]}"; do
  IFS='|' read -r ds_name data_path incr_query incr_gt overall_query overall_gt m ef_c ef_s <<< "$ds_entry"
  for qmode in "${QUERY_MODES[@]}"; do
    for lock in "${LOCK_MODES[@]}"; do
      count=$((count + 1))
      cfg=$(generate_config "$ds_name" "$data_path" "$incr_query" "$incr_gt" \
                            "$overall_query" "$overall_gt" "$m" "$ef_c" "$ef_s" \
                            "$qmode" "$lock")
      echo "[$count/$total] Running: $cfg"
      mkdir -p "${RESULT_BASE}/${ds_name}/${qmode}_${lock}"
      $BENCH_BIN -config "$cfg" 2>&1 | tee "${RESULT_BASE}/${ds_name}/${qmode}_${lock}/bench.log"
      echo "[$count/$total] Done: ${ds_name} ${qmode} ${lock}"
      echo ""
    done
  done
done

echo "=== All ${total} experiments complete. Results in ${RESULT_BASE}/ ==="
