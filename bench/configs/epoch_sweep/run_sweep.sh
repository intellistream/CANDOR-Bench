#!/bin/bash
set -e
cd /home/junyao/code/ANN-CC-bench/bench

RESULTS_DIR="/home/junyao/code/ANN-CC-bench/experiments/helping/build/results/epoch_sweep"
mkdir -p "$RESULTS_DIR"

BATCH_SIZES=(50 100 200 500 1000 2000 5000)
SHARING_VALUES=("false" "true")

for B in "${BATCH_SIZES[@]}"; do
  for S in "${SHARING_VALUES[@]}"; do
    LABEL="b${B}_s3_${S}"
    CONFIG="/tmp/epoch_sweep_${LABEL}.yaml"
    
    cat > "$CONFIG" << YAML
data:
  dataset_name: sift
  max_elements: 200000
  begin_num: 50000
  data_type: float
  data_path: ../data/sift/sift_base.bin
  incr_query_path: ../data/sift/sift_query_stream.bin
  overall_query_path: ../data/sift/sift_query.bin
  overall_gt_path: ../data/sift/sift_base.gt20

index:
  index_type: hnsw
  m: 16
  ef_construction: 200

search:
  recall_at: 10
  ef_search: 35
  use_node_lock: true
  enable_search_sharing: ${S}

workload:
  batch_size: ${B}
  num_threads: 8
  queue_size: 0
  insert_event_rate: 20000
  search_event_rate: 20000
  with_external_rw_lock: false
  query_mode: chasing

compare:
  enabled: false

result:
  output_dir: ${RESULTS_DIR}

profile:
  memory_monitor_interval: 100
  enable_memory_profile: true
YAML

    echo "=== Running $LABEL ==="
    CGO_LDFLAGS="-L$(pwd)/build/lib -lindex" \
    CGO_CXXFLAGS="-I$(pwd)/algorithms -std=c++17" \
    LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH \
    go run . -config "$CONFIG" 2>&1 | tee -a "$RESULTS_DIR/sweep.log"
    
    echo "=== Completed $LABEL ==="
  done
done

echo "All sweep runs completed!"
