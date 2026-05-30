#!/usr/bin/env bash
# Validate cause 1 independently: writer mutation burst.
#
# This wrapper removes search node-lock effects by running only `nolock`.
# If batch-E2E spikes still appear and grow with write pressure / batch size,
# then the cause is not node-lock convoy; it is writer/search interference from
# graph mutation and memory/cache pressure.
set -euo pipefail

cd "$(dirname "$0")/.."

export RESULT_BASE="${RESULT_BASE:-./result/m3_cause1_mutation_burst}"
export CONFIG_DIR="${CONFIG_DIR:-./config/m3_cause1_mutation_burst}"
export SYNC_MODES="${SYNC_MODES:-nolock}"
export BATCHES="${BATCHES:-20 200 400}"
export QUERY_MODES="${QUERY_MODES:-round_robin chasing}"

bash scripts/run_m3_fluctuation_rootcause.sh

python3 scripts/analyze_m3_cause12_validation.py "$RESULT_BASE" --cause cause1 \
  | tee "$RESULT_BASE/cause1_analysis.txt"
