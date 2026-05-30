#!/bin/bash
set -e
cd "$(dirname "$0")"

export LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH

echo "ratio,recall,search_qps,lr_total_nodes,lr_skipped_nodes,lr_recovered_edges,lr_skipped_edges"

for ratio in 1.0 0.9 0.8 0.7 0.5 0.3; do
    cat > /tmp/lr_config.yaml <<EOF
data:
  dataset_name: sift
  max_elements: 1000000
  begin_num: 50000
  data_type: float
  data_path: ../data/sift/sift_base.bin
  incr_query_path: ../data/sift/sift_query_stream.bin
  overall_query_path: ../data/sift/sift_query.bin
  overall_gt_path: ../data/sift/sift_base.gt20

index:
  index_type: annchor-dev
  m: 16
  ef_construction: 200

search:
  recall_at: 10
  ef_search: 35
  enable_mvcc: [true]
  lazy_recovery_ratio: [$ratio]

workload:
  batch_size: 20
  num_threads: [32]
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
  output_dir: ./result/lr_sweep

profile:
  enable_memory_profile: false
EOF

    output=$(./bin/bench -config /tmp/lr_config.yaml 2>&1)
    recall=$(echo "$output" | grep -oP 'recall@10 = \K[0-9.]+' | tail -1)
    stats=$(echo "$output" | grep "stats" | tail -1)
    search_qps=$(echo "$output" | grep -oP 'search_qps.*?(\d+\.\d+)' | grep -oP '\d+\.\d+' | tail -1)
    
    # Extract from CSV
    csv_line=$(tail -1 result/lr_sweep/benchmark_results.csv 2>/dev/null || echo "")
    if [ -n "$csv_line" ]; then
        recall_csv=$(echo "$csv_line" | awk -F',' '{print $(NF-3)}')
        sqps=$(echo "$csv_line" | awk -F',' '{print $18}')
        stats_str=$(echo "$csv_line" | awk -F'"' '{print $2}')
        lr_total=$(echo "$stats_str" | grep -oP 'lr_total_nodes:\K\d+')
        lr_skip=$(echo "$stats_str" | grep -oP 'lr_skipped_nodes:\K\d+')
        lr_rec=$(echo "$stats_str" | grep -oP 'lr_recovered_edges:\K\d+')
        lr_skip_e=$(echo "$stats_str" | grep -oP 'lr_skipped_edges:\K\d+')
        echo "$ratio,$recall_csv,$sqps,$lr_total,$lr_skip,$lr_rec,$lr_skip_e"
    fi
done
