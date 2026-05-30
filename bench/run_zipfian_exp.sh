#!/bin/bash
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH

# Zipfian vs Uniform query distribution experiment
# SIFT-1M, 32 threads, MVCC+M2, per-query latency enabled

for qmode in "round_robin" "zipfian"; do
  for skew in 0.0 0.99 1.5; do
    # skip skew variations for round_robin
    if [ "$qmode" = "round_robin" ] && [ "$skew" != "0.0" ]; then
      continue
    fi
    # skip skew=0.0 for zipfian
    if [ "$qmode" = "zipfian" ] && [ "$skew" = "0.0" ]; then
      continue
    fi

    label="${qmode}"
    if [ "$qmode" = "zipfian" ]; then
      label="zipfian_s${skew}"
    fi
    outdir="./result/zipfian_exp/${label}"

    echo "=== Running $label ==="

    cat > /tmp/zipfian_exp.yaml <<EOF
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
  enable_mvcc: true
  lazy_recovery_ratio: 0.5
workload:
  batch_size: 20
  num_threads: 32
  queue_size: 0
  rate_groups(r/w):
    - [999999, 999999]
  with_external_rw_lock: false
  query_mode: $qmode
  zipfian_skew: $skew
  per_query_latency: true
compare:
  enabled: false
  simulate:
    enabled: false
result:
  output_dir: $outdir
profile:
  memory_monitor_interval: 100
  enable_memory_profile: false
EOF

    ./bin/bench -config /tmp/zipfian_exp.yaml 2>&1 | tee "${outdir}_run.log" || true
    echo "=== Done $label ==="
    echo ""
  done
done

echo "=== All done ==="
echo ""
echo "Results:"
for d in ./result/zipfian_exp/*/; do
  echo "--- $(basename $d) ---"
  if [ -f "$d/benchmark_results.csv" ]; then
    head -2 "$d/benchmark_results.csv"
  fi
  # grep conflict stats from log
  log="${d%/}_run.log"
  if [ -f "$log" ]; then
    grep -i "m2_search_expand\|conflict\|recover" "$log" | tail -5
  fi
  echo ""
done
