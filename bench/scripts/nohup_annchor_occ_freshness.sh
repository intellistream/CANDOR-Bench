#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

STOP_OLD_PYTHON="${STOP_OLD_PYTHON:-1}"
if [[ "$STOP_OLD_PYTHON" == "1" || "$STOP_OLD_PYTHON" == "true" ]]; then
  pkill -TERM -f 'python.*(experiments/prior_validation|run_occ_cost|run_streaming|streaming_hierarchical|streaming_refined|streaming_step1)' || true
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/annchor_occ_freshness_full_$STAMP}"
mkdir -p "$OUT_ROOT"

GO_BIN="${GO_BIN:-/usr/local/go/bin/go}"
DATASETS="${DATASETS:-sift,msong,deep1M,gist}"
BATCH_SIZES="${BATCH_SIZES:-20,100,200,400}"
TOPKS="${TOPKS:-10}"
THREADS="${THREADS:-16,32,64}"
WARMUP="${WARMUP:-50000}"
MAX_ELEMENTS="${MAX_ELEMENTS:--1}"
EF_SEARCH="${EF_SEARCH:-35}"
RATE_GROUPS="${RATE_GROUPS:-90000:10000,40000:40000,10000:90000}"
QUERY_MODES="${QUERY_MODES:-round_robin,chasing}"
PRECOMPUTE_SCHEDULE="${PRECOMPUTE_SCHEDULE:-false}"
ENABLE_RECALL="${ENABLE_RECALL:-false}"

LOG="$OUT_ROOT/run.log"
PID_FILE="$OUT_ROOT/run.pid"

setsid env \
  GO_BIN="$GO_BIN" \
  OUT_ROOT="$OUT_ROOT" \
  DATASETS="$DATASETS" \
  BATCH_SIZES="$BATCH_SIZES" \
  TOPKS="$TOPKS" \
  THREADS="$THREADS" \
  WARMUP="$WARMUP" \
  MAX_ELEMENTS="$MAX_ELEMENTS" \
  EF_SEARCH="$EF_SEARCH" \
  RATE_GROUPS="$RATE_GROUPS" \
  QUERY_MODES="$QUERY_MODES" \
  PRECOMPUTE_SCHEDULE="$PRECOMPUTE_SCHEDULE" \
  ENABLE_RECALL="$ENABLE_RECALL" \
  bash "$ROOT/bench/scripts/run_annchor_occ_freshness_matrix.sh" \
  > "$LOG" 2>&1 < /dev/null &

PID="$!"
echo "$PID" > "$PID_FILE"

cat <<EOF
started annchor OCC freshness bench
pid: $PID
out: $OUT_ROOT
log: $LOG
csv: $OUT_ROOT/annchor_occ_freshness_summary.csv

watch:
  tail -f "$LOG"

stop:
  kill -TERM -"$PID"
EOF
