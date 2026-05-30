#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/junyao/code/ANN-CC-Bench
BENCH_DIR="$ROOT/bench"
LOG_DIR="$ROOT/results"

mkdir -p "$LOG_DIR"
rm -rf "$ROOT/results/sift_occ_t16" "$ROOT/results/sift_occ_t32" "$ROOT/results/sift_occ_t64"

export LD_LIBRARY_PATH="$BENCH_DIR/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$BENCH_DIR"
./bench_dev -config ./config/sift_occ_sat_rates_t16.yaml | tee "$LOG_DIR/sift_occ_t16.log"
./bench_dev -config ./config/sift_occ_sat_rates_t32.yaml | tee "$LOG_DIR/sift_occ_t32.log"
./bench_dev -config ./config/sift_occ_sat_rates_t64.yaml | tee "$LOG_DIR/sift_occ_t64.log"
python3 ./scripts/summarize_sift_occ.py
