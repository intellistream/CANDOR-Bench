#!/usr/bin/env bash
# Overnight root-cause run for the current ANNCHOR-MVCC latency spikes.
#
# This is observational only: no skip/shadow/dryrun graph mutation ablations.
# It aligns true search batch latency with per-batch ANNCHOR-MVCC work counters.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_current_latency_rootcause_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_current_latency_rootcause_${STAMP}}"

mkdir -p "$ROOT" "$CONFIG_ROOT"

echo "[m3-current-rootcause] root=$ROOT"
echo "[m3-current-rootcause] config=$CONFIG_ROOT"

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
  BATCHES="${BATCHES:-20 200 400}"
  RATE_PAIRS="${RATE_PAIRS:-40000 40000;10000 90000;90000 10000}"
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
  echo "[m3-current-rootcause] run recovery_${mode}"
  env "${COMMON_ENV[@]}" \
    ENABLE_UNDO_RECOVERY="$enable" \
    RESULT_BASE="$out" \
    CONFIG_DIR="$cfg" \
    bash ./scripts/run_m3_fluctuation_rootcause.sh
  python3 ./scripts/analyze_m3_batch_diag_rootcause.py "$out" | tee "$out/m3_batch_diag_analyze.log"
}

run_mode off false
run_mode on true

echo "[m3-current-rootcause] optional perf subset"
PERF_ROOT="$ROOT/perf_subset"
PERF_CFG="$CONFIG_ROOT/perf_subset"
env ANNCHOR_M3_BATCH_DIAG=1 \
  INSERT_RATIO="${INSERT_RATIO:-99}" \
  DATASETS="${DATASETS:-sift}" \
  BEGIN_NUM="${BEGIN_NUM:-50000}" \
  MAX_ELEMENTS="${PERF_MAX_ELEMENTS:-160000}" \
  THREADS="${THREADS:-64}" \
  EF_SEARCH="${EF_SEARCH:-35}" \
  EF_CONSTRUCTION="${EF_CONSTRUCTION:-200}" \
  M="${M:-16}" \
  BATCHES="${PERF_BATCHES:-200}" \
  RATE_PAIRS="${PERF_RATE_PAIRS:-40000 40000}" \
  QUERY_MODES="${PERF_QUERY_MODES:-round_robin chasing}" \
  SYNC_MODES="${PERF_SYNC_MODES:-nolock}" \
  REPETITIONS=1 \
  RUN_PERF="${RUN_PERF_SUBSET:-1}" \
  ENABLE_UNDO_RECOVERY=true \
  RESULT_BASE="$PERF_ROOT" \
  CONFIG_DIR="$PERF_CFG" \
  PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,cache-misses,LLC-load-misses,LLC-loads,LLC-stores,context-switches,cpu-migrations}" \
  bash ./scripts/run_m3_fluctuation_rootcause.sh || true
python3 ./scripts/analyze_m3_batch_diag_rootcause.py "$PERF_ROOT" | tee "$PERF_ROOT/m3_batch_diag_analyze.log" || true

cat > "$ROOT/README.md" <<EOF
# M3 Current Latency Root Cause Overnight

Started: ${STAMP}

This run does not change graph topology. It compares current ANNCHOR-MVCC search
latency against per-batch work counters.

Primary outputs:

\`\`\`text
$ROOT/recovery_off/m3_fluctuation_rootcause_summary.csv
$ROOT/recovery_off/m3_batch_diag_report.md
$ROOT/recovery_off/m3_batch_diag_summary.csv
$ROOT/recovery_on/m3_fluctuation_rootcause_summary.csv
$ROOT/recovery_on/m3_batch_diag_report.md
$ROOT/recovery_on/m3_batch_diag_summary.csv
$ROOT/perf_subset/*/perf_stat.txt
\`\`\`

Interpretation rule:

\`\`\`text
If p99/top batches are enriched in recovery_loop_ms / recovered_edges /
recovery_candidates, the immediate cause is snapshot restoration work tail.
If they are enriched in future_skip_hops, the immediate cause is live/future
edge traversal before visibility filtering.
If they are enriched in dist_comps / hops but not recovery counters, the cause
is traversal-work tail outside log recovery.
If neither explains top batches, inspect perf_subset for cache/LLC/context-switch
signals and then add narrower instrumentation.
\`\`\`
EOF

echo "[m3-current-rootcause] done root=$ROOT"
