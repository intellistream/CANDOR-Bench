#!/usr/bin/env bash
# Validate which insert-side graph-connection phase creates bursty search spikes.
#
# This run keeps the graph semantics unchanged. It adds fine-grained telemetry
# around mutuallyConnectNewElement and aligns top search batches with nearby
# insert batches.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_insert_burst_rootcause_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_insert_burst_rootcause_${STAMP}}"

mkdir -p "$ROOT" "$CONFIG_ROOT"

echo "[m3-insert-burst] root=$ROOT"
echo "[m3-insert-burst] config=$CONFIG_ROOT"

cmake --build ./build --target index -j "${BUILD_JOBS:-2}"
LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .

COMMON_ENV=(
  ANNCHOR_M3_BATCH_DIAG=1
  INSERT_RATIO="${INSERT_RATIO:-99}"
  DATASETS="${DATASETS:-sift}"
  BEGIN_NUM="${BEGIN_NUM:-50000}"
  MAX_ELEMENTS="${MAX_ELEMENTS:-300000}"
  THREADS="${THREADS:-64}"
  EF_SEARCH="${EF_SEARCH:-35}"
  EF_CONSTRUCTION="${EF_CONSTRUCTION:-200}"
  M="${M:-16}"
  BATCHES="${BATCHES:-200 400}"
  RATE_PAIRS="${RATE_PAIRS:-40000 40000;10000 90000}"
  QUERY_MODES="${QUERY_MODES:-round_robin chasing}"
  SYNC_MODES="${SYNC_MODES:-nolock nodelock rwlock}"
  REPETITIONS="${REPETITIONS:-1}"
  RUN_PERF=0
)

run_mode() {
  local mode="$1"
  local enable="$2"
  local out="$ROOT/recovery_${mode}"
  local cfg="$CONFIG_ROOT/recovery_${mode}"
  echo "[m3-insert-burst] run recovery_${mode}"
  env "${COMMON_ENV[@]}" \
    ENABLE_UNDO_RECOVERY="$enable" \
    RESULT_BASE="$out" \
    CONFIG_DIR="$cfg" \
    bash ./scripts/run_m3_fluctuation_rootcause.sh
  python3 ./scripts/analyze_m3_batch_diag_rootcause.py "$out" | tee "$out/m3_batch_diag_analyze.log"
  python3 ./scripts/analyze_m3_insert_burst_alignment.py "$out" 20 | tee "$out/m3_insert_burst_analyze_w20.log"
  python3 ./scripts/analyze_m3_insert_burst_alignment.py "$out" 50 | tee "$out/m3_insert_burst_analyze_w50.log"
}

if [[ "${RUN_RECOVERY_OFF:-1}" == "1" ]]; then
  run_mode off false
fi
run_mode on true

cat > "$ROOT/README.md" <<EOF
# M3 Insert Burst Root Cause

Started: ${STAMP}

Purpose: verify why writer/search contention is bursty by aligning top search
batches with fine-grained insert graph-connection phases.

Primary outputs:

\`\`\`text
$ROOT/recovery_off/m3_insert_burst_alignment.md
$ROOT/recovery_off/m3_insert_burst_alignment.csv
$ROOT/recovery_off/m3_insert_burst_examples.csv
$ROOT/recovery_on/m3_insert_burst_alignment.md
$ROOT/recovery_on/m3_insert_burst_alignment.csv
$ROOT/recovery_on/m3_insert_burst_examples.csv
\`\`\`

Key insert columns:

\`\`\`text
graph_new_node_link_ms
graph_old_update_loop_ms
graph_prune_heuristic_ms
graph_undo_record_ms
graph_existing_node_visits
graph_existing_node_appends
graph_existing_node_prunes
graph_pruned_edges
\`\`\`

Decision rule:

\`\`\`text
If top search batches align with graph_old_update_loop_ms,
graph_prune_heuristic_ms, existing_node_prunes, pruned_edges, or link_updates,
the burst phase is reciprocal old-node update / prune / link rewrite inside
mutuallyConnectNewElement.

If only graph_new_node_link_ms aligns, the burst is new-node link materialization.
If only lock_wait aligns, it is writer lock contention. If none align, the
resource burst is outside this graph-connect phase and needs TBB scheduler /
CPU-set instrumentation.
\`\`\`
EOF

echo "[m3-insert-burst] done root=$ROOT"
