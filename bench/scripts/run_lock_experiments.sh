#!/bin/bash
set -e

export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH
BENCH=./bin/bench
LOG_DIR=/tmp/lock_experiments
mkdir -p $LOG_DIR

echo "=== Lock overhead experiment suite ==="
echo "Start: $(date)"

# 1. GIST thread scaling (high-dim, lock overhead should be smaller)
echo ""
echo ">>> [1/4] GIST 960d thread scaling..."
$BENCH -config config/exp_lock_gist_threads.yaml 2>&1 | tee $LOG_DIR/gist_threads.log
echo ">>> [1/4] GIST done: $(date)"

# 2. Deep1M thread scaling (medium-dim)
echo ""
echo ">>> [2/4] Deep1M 256d thread scaling..."
$BENCH -config config/exp_lock_deep1m_threads.yaml 2>&1 | tee $LOG_DIR/deep1m_threads.log
echo ">>> [2/4] Deep1M done: $(date)"

# 3. Insert-only (write-write contention isolation)
echo ""
echo ">>> [3/4] Insert-only (write-write contention)..."
$BENCH -config config/exp_lock_insert_only.yaml 2>&1 | tee $LOG_DIR/insert_only.log
echo ">>> [3/4] Insert-only done: $(date)"

# 4. Small graph (higher collision rate)
echo ""
echo ">>> [4/4] Small graph (50k, higher contention)..."
$BENCH -config config/exp_lock_small_graph.yaml 2>&1 | tee $LOG_DIR/small_graph.log
echo ">>> [4/4] Small graph done: $(date)"

echo ""
echo "=== All experiments complete: $(date) ==="
echo ""
echo "Results:"
echo "  GIST:        result/exp_lock_gist_threads/benchmark_results.csv"
echo "  Deep1M:      result/exp_lock_deep1m_threads/benchmark_results.csv"
echo "  Insert-only: result/exp_lock_insert_only/benchmark_results.csv"
echo "  Small graph: result/exp_lock_small_graph/benchmark_results.csv"
echo "  (Previous)   result/exp_lock_overhead/benchmark_results.csv"
echo "  (Previous)   result/exp_lock_overhead_rw_ratio/benchmark_results.csv"
