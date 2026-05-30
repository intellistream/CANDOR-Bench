#!/bin/bash
# Run comprehensive saturate benchmark:
# 3 datasets × chasing + round_robin × MVCC on/off × 16/64 threads × 3 R/W ratios
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build/lib

# Dataset configs: name, M, ef_c, ef_s, query_mode(ch=chasing, rr=round_robin)
declare -A DS_M=( [sift]=16 [deep1M]=24 [msong]=8 )
declare -A DS_EFC=( [sift]=200 [deep1M]=200 [msong]=100 )
declare -A DS_EFS=( [sift]=35 [deep1M]=40 [msong]=40 )

configs=()

for dataset in sift deep1M msong; do
    for mode in chasing round_robin; do
        cfgdir="config/annchor/${dataset}"
        mkdir -p "$cfgdir"
        cfgfile="${cfgdir}/saturate_${mode}.yaml"

        cat > "$cfgfile" << YAML
data:
  dataset_name: ${dataset}
  max_elements: -1
  begin_num: 50000
  data_type: float
  data_path: ../data/${dataset}/${dataset}_base.bin
  incr_query_path: ../data/${dataset}/${dataset}_query_stream.bin
  incr_gt_path: ../data/${dataset}/${dataset}_stream_i20_offset_index.txt
  overall_query_path: ../data/${dataset}/${dataset}_query.bin
  overall_gt_path: ../data/${dataset}/${dataset}_base.gt20

index:
  index_type: annchor
  m: ${DS_M[$dataset]}
  ef_construction: ${DS_EFC[$dataset]}

search:
  recall_at: 10
  ef_search: ${DS_EFS[$dataset]}
  enable_mvcc: [true, false]

workload:
  batch_size: [20]
  num_threads: [16, 64]
  queue_size: 0
  rate_groups(r/w):
    - [20000, 20000]
    - [4000, 36000]
    - [36000, 4000]

  with_external_rw_lock: false
  query_mode: ${mode}

compare:
  enabled: false
  simulate:
    enabled: false

result:
  output_dir: ./result/saturate_comprehensive

profile:
  memory_monitor_interval: 100
  enable_memory_profile: true
YAML
        configs+=("$cfgfile")
        echo "Created: $cfgfile"
    done
done

echo ""
echo "Total configs: ${#configs[@]}"
echo "Each config sweeps: enable_mvcc[2] × threads[2] × rates[3] = 12 runs"
echo "Total runs: $((${#configs[@]} * 12))"
echo ""

go build -o bench .
if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

for config in "${configs[@]}"; do
    echo "========================================"
    echo "Running: $config"
    echo "========================================"
    ./bench -config "$config"
done

echo "ALL DONE"
