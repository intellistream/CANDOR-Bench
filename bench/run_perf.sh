#!/bin/bash
# Run benchmark with perf stat to measure cache misses
# Usage: ./run_perf.sh <config.yaml> <output_prefix>

set -e
CONFIG=$1
PREFIX=$2

cd /home/junyao/code/ANN-CC-bench/bench
export LD_LIBRARY_PATH=./build/lib:$LD_LIBRARY_PATH

echo "=== Running perf stat for $CONFIG ==="
perf stat -e cache-misses,cache-references,L1-dcache-load-misses,L1-dcache-loads,LLC-load-misses,LLC-loads,instructions,cycles \
    go run . -config "$CONFIG" 2>&1 | tee "${PREFIX}_perf.log"

echo "=== Done ==="
