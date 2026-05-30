#!/usr/bin/env bash
# Validate cause 2 independently: HNSW node-lock convoy.
#
# This wrapper runs paired `nodelock` and `nolock` cases under identical
# workload schedules. The effect size is the extra tail latency from enabling
# search-side shared locks while insert still mutates neighbor lists.
set -euo pipefail

cd "$(dirname "$0")/.."

export RESULT_BASE="${RESULT_BASE:-./result/m3_cause2_node_lock_convoy}"
export CONFIG_DIR="${CONFIG_DIR:-./config/m3_cause2_node_lock_convoy}"
export SYNC_MODES="${SYNC_MODES:-nodelock nolock}"
export BATCHES="${BATCHES:-20 200 400}"
export QUERY_MODES="${QUERY_MODES:-round_robin chasing}"

bash scripts/run_m3_fluctuation_rootcause.sh

python3 scripts/analyze_m3_cause12_validation.py "$RESULT_BASE" --cause cause2 \
  | tee "$RESULT_BASE/cause2_analysis.txt"
