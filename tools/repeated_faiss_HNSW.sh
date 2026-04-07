#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_BASE="${REPO_ROOT}/results/sift/faiss_HNSW"
SOURCE_DIR="${RESULT_BASE}/ef-120"

for i in $(seq 1 10); do
  echo "[Run ${i}/10] Starting benchmark..."

  (
    cd "${REPO_ROOT}"
    python run_benchmark.py \
      --algorithm faiss_HNSW \
      --dataset sift \
      --runbook runbooks/general_experiment/general_experiment.yaml \
      --enable-cache-profiling
  )

  TARGET_DIR="${RESULT_BASE}/test${i}"

  if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "Error: source directory not found: ${SOURCE_DIR}" >&2
    exit 1
  fi

  if [[ -e "${TARGET_DIR}" ]]; then
    echo "Error: target already exists: ${TARGET_DIR}" >&2
    echo "Please remove it or choose a different naming scheme." >&2
    exit 1
  fi

  mv "${SOURCE_DIR}" "${TARGET_DIR}"
  echo "[Run ${i}/10] Moved ${SOURCE_DIR} -> ${TARGET_DIR}"
done

echo "All 10 runs completed successfully."
