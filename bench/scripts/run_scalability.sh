#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH
BENCH=./bin/bench_dev

echo "=== Scalability experiment: 4 modes × 6 threads × 2 reps ==="
echo "Start: $(date)"

for mode in rwlock hnsw mvcc preempt; do
    echo ""
    echo ">>> Running $mode ... $(date)"
    $BENCH -config config/exp_scale_${mode}.yaml 2>&1
    echo ">>> $mode done: $(date)"
done

echo ""
echo "=== All done: $(date) ==="
