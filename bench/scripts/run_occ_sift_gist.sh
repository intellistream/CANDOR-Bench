#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/junyao/code/ANN-CC-Bench
BENCH_DIR="$ROOT/bench"
RESULT_DIR="$ROOT/results"

mkdir -p "$RESULT_DIR"
export LD_LIBRARY_PATH="$BENCH_DIR/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$BENCH_DIR"

cmake --build build --target index -j 8
/usr/local/go/bin/go build ./cmd/bench_dev

rm -rf \
  "$RESULT_DIR/sift_occ_t16" "$RESULT_DIR/sift_occ_t32" "$RESULT_DIR/sift_occ_t64" \
  "$RESULT_DIR/gist_occ_t16" "$RESULT_DIR/gist_occ_t32" "$RESULT_DIR/gist_occ_t64"

./bench_dev -config ./config/sift_occ_sat_rates_t16.yaml
./bench_dev -config ./config/sift_occ_sat_rates_t32.yaml
./bench_dev -config ./config/sift_occ_sat_rates_t64.yaml
python3 ./scripts/summarize_occ_table.py sift "$RESULT_DIR/sift_occ_table.csv"

./bench_dev -config ./config/gist_occ_sat_rates_t16.yaml
./bench_dev -config ./config/gist_occ_sat_rates_t32.yaml
./bench_dev -config ./config/gist_occ_sat_rates_t64.yaml
python3 ./scripts/summarize_occ_table.py gist "$RESULT_DIR/gist_occ_table.csv"
