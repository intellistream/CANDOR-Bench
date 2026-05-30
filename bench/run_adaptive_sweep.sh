#!/bin/bash
# Run adaptive CC sweep: 3 modes × 5 thread counts = 15 runs
set -e
cd /home/junyao/code/ANN-CC-bench/bench
export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH

RESULT_DIR=result/adaptive_cc_sweep
mkdir -p $RESULT_DIR

for threads in 1 4 16 32 64; do
    for mode in lock mvcc nolock; do
        case $mode in
            lock)   nl=true;  mvcc=false ;;
            mvcc)   nl=false; mvcc=true  ;;
            nolock) nl=false; mvcc=false ;;
        esac
        
        echo "=== ${mode} ${threads}T ==="
        cat > /tmp/cc_sweep_${mode}_${threads}.yaml <<EOF
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
  ef_search: 64
  use_node_lock: ${nl}
  enable_mvcc: ${mvcc}
  enable_s3: false
  enable_search_sharing: false
  enable_warm_start: false
workload:
  batch_size: 100
  num_threads: ${threads}
  queue_size: 0
  rate_groups(r/w):
    - [999999, 999999]
  with_external_rw_lock: false
  query_mode: round_robin
compare:
  enabled: false
result:
  output_dir: ${RESULT_DIR}/${mode}_${threads}t
EOF
        ./bench_runner -config /tmp/cc_sweep_${mode}_${threads}.yaml 2>&1 | grep -E "Progress.*100%|recall" || true
    done
done

echo "=== ALL DONE ==="
# Consolidate results
echo "mode,threads,insert_qps,search_qps,recall" > $RESULT_DIR/summary.csv
for threads in 1 4 16 32 64; do
    for mode in lock mvcc nolock; do
        csv="${RESULT_DIR}/${mode}_${threads}t/benchmark_results.csv"
        if [ -f "$csv" ]; then
            # Extract fields: insert_qps (col 13), search_qps (col 20), recall (col 28)
            data=$(tail -1 "$csv" | tr ',' '\n')
            ins_qps=$(echo "$data" | sed -n '13p')
            srch_qps=$(echo "$data" | sed -n '20p')
            recall=$(echo "$data" | sed -n '28p')
            echo "${mode},${threads},${ins_qps},${srch_qps},${recall}" >> $RESULT_DIR/summary.csv
        fi
    done
done
cat $RESULT_DIR/summary.csv
