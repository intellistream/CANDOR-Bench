#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-sift-small}"
RUNBOOK="${2:-runbooks/general_experiment/general_experiment_sift_small.yaml}"
RESULTS_DIR="${3:-results_gammafresh_vs_freshdiskann}"
FIG_DIR="${4:-figures/paper-gammafresh-vs-freshdiskann}"

cd "$ROOT_DIR"

uv run python tools/bruteforce_streaming_gt.py --dataset "$DATASET" --runbook "$RUNBOOK"
uv run python run_benchmark.py --algorithm gammafresh --dataset "$DATASET" --runbook "$RUNBOOK" --output "$RESULTS_DIR"
uv run python run_benchmark.py --algorithm diskann --dataset "$DATASET" --runbook "$RUNBOOK" --output "$RESULTS_DIR"
uv run python tools/plot_paper_general.py \
  --dataset "$DATASET" \
  --runbook "$RUNBOOK" \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$FIG_DIR" \
  --algorithms gammafresh diskann
