#!/bin/bash
# Measure lock contention vs cache line bouncing for write-write contention
# Usage: bash scripts/perf_ww_contention.sh
set -e

cd "$(dirname "$0")/.."

BENCH="./bin/bench"
OUTDIR="./result/exp_ww_contention_perf"
mkdir -p "$OUTDIR"

PERF_EVENTS="cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,context-switches,migrations,dTLB-load-misses"

for THREADS in 1 4 16 64; do
    echo "============================================"
    echo "Running insert-only with $THREADS threads"
    echo "============================================"

    # Create per-thread config
    cat > "$OUTDIR/config_t${THREADS}.yaml" <<EOF
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
  index_type: hnsw
  m: 16
  ef_construction: 200

search:
  recall_at: 10
  ef_search: 64
  use_node_lock: [true]
  enable_mvcc: [false]

workload:
  batch_size: 20
  num_threads: [$THREADS]
  queue_size: 0
  with_external_rw_lock: false
  query_mode: chasing
  rate_groups(r/w):
    - {read: 0, write: 999999999}

result:
  output_dir: $OUTDIR/t${THREADS}

repetitions: 1
EOF

    # Run with perf stat
    export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH
    perf stat -e "$PERF_EVENTS" -o "$OUTDIR/perf_t${THREADS}.txt" \
        "$BENCH" -config "$OUTDIR/config_t${THREADS}.yaml" 2>&1 | tail -5

    echo ""
    echo "--- perf stat for $THREADS threads ---"
    cat "$OUTDIR/perf_t${THREADS}.txt"
    echo ""
done

echo "============================================"
echo "Summary: compare perf counters across thread counts"
echo "============================================"
for THREADS in 1 4 16 64; do
    echo "--- $THREADS threads ---"
    grep -E "(cache-misses|LLC-load-misses|L1-dcache-load-misses|instructions|cycles|context-switches|migrations)" "$OUTDIR/perf_t${THREADS}.txt" 2>/dev/null || true
    echo ""
done
