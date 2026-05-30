#!/bin/bash
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/build/lib"

BENCH="./bin/bench"
CFG="./config/fine_grain_exp/sift_preempt.yaml"
RESULT_BASE="./result/fine_grain_sweep"
mkdir -p "$RESULT_BASE"

echo "=== Fine-grain yield cooldown sweep ==="

# Baseline: stable (no preempt)
echo "--- stable baseline ---"
mkdir -p "$RESULT_BASE/stable"
$BENCH -config "./config/fine_grain_exp/sift_stable.yaml" 2>&1 | tee "$RESULT_BASE/stable/bench.log"

# Sweep cooldown
for COOLDOWN in 0 50 200 500; do
    echo ""
    echo "--- preempt cooldown=${COOLDOWN}us ---"
    export ANN_FINE_GRAIN_YIELD_COOLDOWN_US=$COOLDOWN
    DIR="$RESULT_BASE/preempt_cd${COOLDOWN}"
    mkdir -p "$DIR"
    $BENCH -config "$CFG" 2>&1 | tee "$DIR/bench.log"
done

echo ""
echo "=== Sweep done. Comparing search latency ==="
echo "config,batch,search_op_avg_ms,search_op_p99_ms,insert_qps,yield_count"
for d in "$RESULT_BASE"/*/; do
    name=$(basename "$d")
    if [ -f "$d/benchmark_results.csv" ]; then
        tail -n+2 "$d/benchmark_results.csv" | while IFS=, read -r alg threads wb rb ds rest; do
            search_avg=$(echo "$rest" | cut -d, -f8)
            search_p99=$(echo "$rest" | cut -d, -f11)
            insert_qps=$(echo "$rest" | cut -d, -f1)
            echo "$name,b${wb},$search_avg,$search_p99,$insert_qps"
        done
    fi
done

echo "=== All done ==="
