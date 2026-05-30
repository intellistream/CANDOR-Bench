#!/bin/bash
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/build/lib"

BENCH="./bin/bench"
RESULT="./result/fine_grain_profiles"
mkdir -p "$RESULT"

echo "=== Fine-grain preempt profile comparison ==="

# 1. Stable baseline
echo "--- stable ---"
mkdir -p "$RESULT/stable"
$BENCH -config "./config/fine_grain_exp/sift_stable.yaml" 2>&1 | tee "$RESULT/stable/bench.log"
cp ./result/fine_grain_exp/sift_stable/benchmark_results.csv "$RESULT/stable/" 2>/dev/null || true

# 2. Balanced: budget=30%, cooldown=300us, threshold=2
echo ""
echo "--- balanced (budget=30%, cd=300us, threshold=2) ---"
export ANN_FINE_GRAIN_YIELD_COOLDOWN_US=300
mkdir -p "$RESULT/balanced"
$BENCH -config "./config/fine_grain_exp/sift_preempt_balanced.yaml" 2>&1 | tee "$RESULT/balanced/bench.log"
cp ./result/fine_grain_exp/sift_preempt_balanced/benchmark_results.csv "$RESULT/balanced/" 2>/dev/null || true

# 3. Aggressive: budget=50%, cooldown=100us, threshold=1
echo ""
echo "--- aggressive (budget=50%, cd=100us, threshold=1) ---"
export ANN_FINE_GRAIN_YIELD_COOLDOWN_US=100
mkdir -p "$RESULT/aggressive"
$BENCH -config "./config/fine_grain_exp/sift_preempt_aggressive.yaml" 2>&1 | tee "$RESULT/aggressive/bench.log"
cp ./result/fine_grain_exp/sift_preempt_aggressive/benchmark_results.csv "$RESULT/aggressive/" 2>/dev/null || true

echo ""
echo "=== Done. Results in $RESULT ==="
