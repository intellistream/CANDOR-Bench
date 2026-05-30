#!/bin/bash
set -euo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/build/lib"

BENCH="./bin/bench"
RESULT="./result/fine_grain_exp"
mkdir -p "$RESULT"

echo "=== Fine-grain yield experiment ==="
echo ""

for cfg in sift_stable sift_preempt; do
    echo "--- Running: $cfg ---"
    mkdir -p "$RESULT/$cfg"
    $BENCH -config "./config/fine_grain_exp/${cfg}.yaml" 2>&1 | tee "$RESULT/$cfg/bench.log"
    echo "--- Done: $cfg ---"
    echo ""
done

echo "=== All done ==="
