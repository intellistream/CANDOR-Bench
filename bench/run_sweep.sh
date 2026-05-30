#!/bin/bash
set -e
cd /home/junyao/code/ANN-CC-bench/bench

UNDO_LOG="algorithms/annchor/ANNchor/src/hnswlib/undo_log.h"
HNSWALG="algorithms/annchor/ANNchor/src/hnswlib/hnswalg.h"
RESULTS_FILE="result/sweep_results.csv"

mkdir -p result
echo "rcs_k,eul_max_eval,recall,search_qps,p95_lat,p99_lat,eul_dp_pruned,eul_dp_passed,prune_false_neg,prune_true_neg" > "$RESULTS_FILE"

run_config() {
    local K=$1
    local EVAL=$2
    local DIR="sweep_k${K}_eval${EVAL}"
    
    echo "=== Running K=$K, EVAL=$EVAL ==="
    
    # Edit parameters
    sed -i "s/static constexpr int RCS_K = [0-9]*/static constexpr int RCS_K = ${K}/" "$UNDO_LOG"
    sed -i "s/static constexpr size_t EUL_MAX_EVAL = [0-9]*/static constexpr size_t EUL_MAX_EVAL = ${EVAL}/" "$HNSWALG"
    
    # Verify
    echo "  RCS_K=$(grep 'static constexpr int RCS_K' $UNDO_LOG | grep -o '[0-9]*')"
    echo "  EUL_MAX_EVAL=$(grep 'static constexpr size_t EUL_MAX_EVAL' $HNSWALG | grep -o '[0-9]*')"
    
    # Rebuild
    cd build && make -j$(nproc) index 2>&1 | tail -2
    cd .. && go build -o bench_dp . 2>&1
    
    # Write config
    cat > "result/${DIR}_config.yaml" << EOF
data:
  dataset_name: sift
  max_elements: 1000000
  begin_num: 500000
  data_type: float
  data_path: ../data/sift/sift_base.bin
  incr_query_path: ../data/sift/sift_query_stream.bin
  overall_query_path: ../data/sift/sift_query.bin
  overall_gt_path: ../data/sift/sift_base.gt20
index:
  index_type: annchor
  m: 16
  ef_construction: 200
search:
  recall_at: 10
  ef_search: [64]
  enable_mvcc: [true]
workload:
  batch_size: [20]
  num_threads: 32
  queue_size: 0
  with_external_rw_lock: false
  query_mode: round_robin
  rate_groups(r/w):
    - {read: 999999999, write: 999999999}
result:
  output_dir: ./result/${DIR}
compare:
  enabled: false
  simulate:
    enabled: false
EOF
    
    # Clean and run
    rm -rf "result/${DIR}"
    LD_LIBRARY_PATH=build/lib ./bench_dp -config "result/${DIR}_config.yaml" 2>&1 | tail -10
    
    # Extract results using python
    local CSV="result/${DIR}/benchmark_results.csv"
    if [ -f "$CSV" ]; then
        python3 -c "
import csv, re
with open('$CSV') as f:
    reader = csv.DictReader(f)
    for row in reader:
        recall = row.get('recall', 'N/A')
        qps = row.get('search_qps (per-point)', 'N/A')
        p95 = row.get('search_p95_op_latency_ms (per-batch)', 'N/A')
        p99 = row.get('search_p99_op_latency_ms (per-batch)', 'N/A')
        stats = row.get('stats', '')
        def extract(s, key):
            m = re.search(key + r':(\d+)', s)
            return m.group(1) if m else 'N/A'
        pruned = extract(stats, 'eul_dp_pruned')
        passed = extract(stats, 'eul_dp_passed')
        fn = extract(stats, 'prune_false_neg')
        tn = extract(stats, 'prune_true_neg')
        print(f'$K,$EVAL,{recall},{qps},{p95},{p99},{pruned},{passed},{fn},{tn}')
" >> "$RESULTS_FILE"
        echo "  -> Done. Results appended."
    else
        echo "  -> ERROR: no results file"
        echo "$K,$EVAL,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS_FILE"
    fi
    echo ""
}

# Phase 1: RCS_K sweep with EUL_MAX_EVAL=8
echo "========== Phase 1: RCS_K sweep (EUL_MAX_EVAL=8) =========="
for K in 4 8 16 32; do
    run_config $K 8
done

echo "Phase 1 results:"
cat "$RESULTS_FILE"
echo ""

# Phase 2: EUL_MAX_EVAL sweep with RCS_K=16
echo "========== Phase 2: EUL_MAX_EVAL sweep (RCS_K=16) =========="
for EVAL in 4 16 32; do
    run_config 16 $EVAL
done

echo ""
echo "========== ALL RESULTS =========="
cat "$RESULTS_FILE"

# Restore defaults
sed -i "s/static constexpr int RCS_K = [0-9]*/static constexpr int RCS_K = 16/" "$UNDO_LOG"
sed -i "s/static constexpr size_t EUL_MAX_EVAL = [0-9]*/static constexpr size_t EUL_MAX_EVAL = 8/" "$HNSWALG"
echo "Restored defaults (K=16, EVAL=8)"
