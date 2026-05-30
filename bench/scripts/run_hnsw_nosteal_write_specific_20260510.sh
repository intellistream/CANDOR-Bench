#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/hnsw_nosteal_write_specific_20260510}"
CFG_ROOT="${CFG_ROOT:-./config/hnsw_nosteal_write_specific_20260510}"
DURATION_MS="${DURATION_MS:-8000}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
COMPETING_WORKERS="${COMPETING_WORKERS:-20}"
BATCH_SIZE="${BATCH_SIZE:-20}"
QUERY_MODE="${QUERY_MODE:-peeking}"
DATASET_NAME="${DATASET_NAME:-sift200k}"
DATA_PATH="${DATA_PATH:-../data/sift/sift_base_200k.bin}"
QUERY_PATH="${QUERY_PATH:-../data/sift/sift_query_stream.bin}"
MAX_ELEMENTS="${MAX_ELEMENTS:-200000}"
BEGIN_NUM="${BEGIN_NUM:-180000}"
M="${M:-16}"
EF_SEARCH="${EF_SEARCH:-50}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-100}"
SEARCH_WORK_COUNTERS="${SEARCH_WORK_COUNTERS:-1}"
CASES="${CASES:-idle bg_search periodic_insert periodic_switch}"
MEASURED_CPU_LIST="${MEASURED_CPU_LIST:-0-7}"
COMPETING_CPU_LIST="${COMPETING_CPU_LIST:-8-27}"

mkdir -p "$OUT_ROOT" "$CFG_ROOT"

if [[ ! -x "./bin/bench" ]]; then
  echo "missing ./bin/bench" >&2
  exit 1
fi

case_mode() {
  case "$1" in
    idle) printf 'idle' ;;
    bg_search) printf 'search' ;;
    periodic_insert) printf 'search_plus_periodic_insert' ;;
    periodic_switch) printf 'periodic' ;;
    *) echo "unknown case $1" >&2; exit 1 ;;
  esac
}

case_bg_workers() {
  case "$1" in
    idle|periodic_insert) printf '0' ;;
    bg_search|periodic_switch) printf '%s' "$COMPETING_WORKERS" ;;
    *) echo "unknown case $1" >&2; exit 1 ;;
  esac
}

case_insert_workers() {
  case "$1" in
    idle|bg_search|periodic_switch) printf '0' ;;
    periodic_insert) printf '%s' "$COMPETING_WORKERS" ;;
    *) echo "unknown case $1" >&2; exit 1 ;;
  esac
}

write_config() {
  local cfg="$1"
  local out_dir="$2"
  local total_threads="$3"

  mkdir -p "$out_dir" "$(dirname "$cfg")"
  {
    printf 'data:\n'
    printf '  dataset_name: %s\n' "$DATASET_NAME"
    printf '  max_elements: %s\n' "$MAX_ELEMENTS"
    printf '  begin_num: %s\n' "$BEGIN_NUM"
    printf '  data_type: float\n'
    printf '  data_path: %s\n' "$DATA_PATH"
    printf '  incr_query_path: %s\n' "$QUERY_PATH"
    printf '  overall_query_path: %s\n' "$QUERY_PATH"
    printf 'index:\n'
    printf '  index_type: hnsw\n'
    printf '  m: %s\n' "$M"
    printf '  ef_construction: %s\n' "$EF_CONSTRUCTION"
    printf 'search:\n'
    printf '  recall_at: 10\n'
    printf '  ef_search: %s\n' "$EF_SEARCH"
    printf '  use_node_lock: true\n'
    printf '  enable_mvcc: false\n'
    printf '  enable_undo_recovery: false\n'
    printf 'workload:\n'
    printf '  batch_size: %s\n' "$BATCH_SIZE"
    printf '  num_threads: %s\n' "$total_threads"
    printf '  queue_size: 0\n'
    printf '  insert_event_rate: 0\n'
    printf '  search_event_rate: 0\n'
    printf '  insert_burst_schedule:\n'
    printf '    - {start_ms: 2500, duration_ms: 1000, insert_event_rate: 0}\n'
    printf '    - {start_ms: 6000, duration_ms: 1000, insert_event_rate: 0}\n'
    printf '  schedule_horizon_ms: %s\n' "$DURATION_MS"
    printf '  with_external_rw_lock: false\n'
    printf '  use_node_lock: true\n'
    printf '  query_mode: %s\n' "$QUERY_MODE"
    printf '  per_query_latency: true\n'
    printf '  precompute_schedule: false\n'
    printf 'result:\n'
    printf '  output_dir: %s\n' "$out_dir"
    printf 'repetitions: 1\n'
    printf 'profile:\n'
    printf '  memory_monitor_interval: 100\n'
    printf '  enable_memory_profile: false\n'
  } > "$cfg"
}

for case_name in $CASES; do
  mode="$(case_mode "$case_name")"
  bg_workers="$(case_bg_workers "$case_name")"
  insert_workers="$(case_insert_workers "$case_name")"
  total_threads=$((MEASURED_WORKERS + COMPETING_WORKERS))
  out="$OUT_ROOT/$QUERY_MODE/$case_name"
  cfg="$CFG_ROOT/${QUERY_MODE}_${case_name}.yaml"
  write_config "$cfg" "$out" "$total_threads"
  rm -rf "$out/raw_latency"
  mkdir -p "$out"
  echo "[hnsw-nosteal] query=$QUERY_MODE case=$case_name mode=$mode bg=$bg_workers insert=$insert_workers measured_cpu=$MEASURED_CPU_LIST competing_cpu=$COMPETING_CPU_LIST"
  env \
    M3_CLOSED_LOOP=1 \
    M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
    M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
    M3_CLOSED_LOOP_COMPETING_WORKERS="$bg_workers" \
    M3_CLOSED_LOOP_INSERT_WORKERS="$insert_workers" \
    M3_CLOSED_LOOP_COMPETING_MODE="$mode" \
    ANNCHOR_M3_BATCH_DIAG=1 \
    ANNCHOR_PHASE_OVERLAP_DIAG=0 \
    ANNCHOR_REWRITE_ACTIVE_DIAG=0 \
    ANNCHOR_SEARCH_WORK_COUNTERS="$SEARCH_WORK_COUNTERS" \
    HNSW_MEASURED_SEARCH_ARENA=1 \
    HNSW_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
    HNSW_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
    HNSW_SEARCH_CPU_LIST="$COMPETING_CPU_LIST" \
    HNSW_INSERT_CPU_LIST="$COMPETING_CPU_LIST" \
    ANNCHOR_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
    ANNCHOR_SEARCH_CPU_LIST="$COMPETING_CPU_LIST" \
    ANNCHOR_INSERT_CPU_LIST="$COMPETING_CPU_LIST" \
    OCC_RAW_LATENCY_DIR="$out/raw_latency" \
    RAW_LATENCY_SKIP_SEARCH_WIDE=0 \
    LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
    ./bin/bench -config "$cfg" > "$out/run.log" 2>&1
done

echo "[hnsw-nosteal] complete: $OUT_ROOT/$QUERY_MODE"
