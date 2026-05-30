#!/bin/bash
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/build/lib"

BENCH="./bin/bench"
RESULT="./result/fine_grain_tuning"
mkdir -p "$RESULT"

# Base config template
make_config() {
    local name=$1 budget=$2 cooldown=$3 threshold=$4
    local dir="./config/fine_grain_exp/tune_${name}.yaml"
    cat > "$dir" <<EOF
data:
  dataset_name: sift
  max_elements: -1
  begin_num: 50000
  data_type: float
  data_path: ../data/sift/sift_base.bin
  incr_query_path: ../data/sift/sift_query_stream.bin
  incr_gt_path: ../data/sift/sift_stream_i100_offset_index.txt
  overall_query_path: ../data/sift/sift_query.bin
  overall_gt_path: ../data/sift/sift_base.gt20

index:
  index_type: annchor-preempt
  m: 16
  ef_construction: 200

search:
  recall_at: 10
  ef_search: 35
  enable_mvcc: true
  enable_preempt_m2: true
  preempt_quantum_points: 1
  preempt_search_backlog_threshold: ${threshold}
  preempt_max_yields_per_batch: 100
  preempt_budget_pct: ${budget}
  preempt_budget_window_us: 10000

workload:
  batch_size: [10, 20]
  num_threads: [16]
  queue_size: 0
  rate_groups(r/w):
    - [20000, 20000]
  with_external_rw_lock: false
  query_mode: chasing

compare:
  enabled: false
  simulate:
    enabled: false

result:
  output_dir: ${RESULT}/${name}

profile:
  memory_monitor_interval: 100
  enable_memory_profile: true
EOF
    echo "$dir"
}

echo "=== Fine-grain tuning ==="

# A: cd=500us, budget=30%, threshold=2 (conservative)
cfg=$(make_config "A_cd500_b30_t2" 30.0 500 2)
echo "--- A: cd=500us budget=30% threshold=2 ---"
export ANN_FINE_GRAIN_YIELD_COOLDOWN_US=500
mkdir -p "$RESULT/A_cd500_b30_t2"
$BENCH -config "$cfg" 2>&1 | tee "$RESULT/A_cd500_b30_t2/bench.log"

# B: cd=500us, budget=50%, threshold=1 (moderate)
cfg=$(make_config "B_cd500_b50_t1" 50.0 500 1)
echo ""
echo "--- B: cd=500us budget=50% threshold=1 ---"
export ANN_FINE_GRAIN_YIELD_COOLDOWN_US=500
mkdir -p "$RESULT/B_cd500_b50_t1"
$BENCH -config "$cfg" 2>&1 | tee "$RESULT/B_cd500_b50_t1/bench.log"

# C: cd=1000us, budget=50%, threshold=1 (very conservative)
cfg=$(make_config "C_cd1000_b50_t1" 50.0 1000 1)
echo ""
echo "--- C: cd=1000us budget=50% threshold=1 ---"
export ANN_FINE_GRAIN_YIELD_COOLDOWN_US=1000
mkdir -p "$RESULT/C_cd1000_b50_t1"
$BENCH -config "$cfg" 2>&1 | tee "$RESULT/C_cd1000_b50_t1/bench.log"

# D: cd=200us, budget=20%, threshold=2 (tight budget)
cfg=$(make_config "D_cd200_b20_t2" 20.0 200 2)
echo ""
echo "--- D: cd=200us budget=20% threshold=2 ---"
export ANN_FINE_GRAIN_YIELD_COOLDOWN_US=200
mkdir -p "$RESULT/D_cd200_b20_t2"
$BENCH -config "$cfg" 2>&1 | tee "$RESULT/D_cd200_b20_t2/bench.log"

echo ""
echo "=== Tuning done ==="
