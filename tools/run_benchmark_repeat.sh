#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ghr/candor-bench/SAGE-DB-Bench"
RESULTS_DIR="$ROOT_DIR/results/sift"
RUNBOOK="runbooks/general_experiment/general_experiment.yaml"
DATASET="sift"

run_and_move() {
  local algorithm="$1"
  local run_index="$2"
  local target_suffix="$3"
  local result_dir="$RESULTS_DIR/$algorithm/ef-120"
  local target_dir="$RESULTS_DIR/$algorithm/$target_suffix"

  python "$ROOT_DIR/run_benchmark.py" \
    --algorithm "$algorithm" \
    --dataset "$DATASET" \
    --runbook "$ROOT_DIR/$RUNBOOK" \
    --enable-cache-profiling

  if [[ ! -d "$result_dir" ]]; then
    echo "Expected result directory not found: $result_dir" >&2
    exit 1
  fi

  if [[ -e "$target_dir" ]]; then
    echo "Target directory already exists: $target_dir" >&2
    exit 1
  fi

  mv "$result_dir" "$target_dir"
}

for i in $(seq 15 20); do
  run_and_move "faiss_hnsw_incremental" "$i" "test${i}"
  #run_and_move "faiss_HNSW" "$i" "test${i}"
done
