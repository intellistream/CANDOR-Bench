#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/hnsw_external_rwlock_nodelock_p99_20260510}"
CFG_ROOT="${CFG_ROOT:-./config/hnsw_external_rwlock_nodelock_p99_20260510}"
REPS="${REPS:-3}"
DURATION_MS="${DURATION_MS:-8000}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
BACKGROUND_SEARCH_WORKERS="${BACKGROUND_SEARCH_WORKERS:-20}"
INSERT_WORKERS="${INSERT_WORKERS:-20}"
BATCH_SIZE="${BATCH_SIZE:-20}"
QUERY_MODE="${QUERY_MODE:-round_robin}"
MAX_ELEMENTS="${MAX_ELEMENTS:-200000}"
BEGIN_NUM="${BEGIN_NUM:-180000}"
EF_SEARCH="${EF_SEARCH:-50}"
SEARCH_WORK_COUNTERS="${SEARCH_WORK_COUNTERS:-1}"
MODES="${MODES:-nolock rwlock}"

mkdir -p "$OUT_ROOT" "$CFG_ROOT"

if [[ ! -x "./bin/bench" ]]; then
  echo "missing ./bin/bench" >&2
  exit 1
fi

write_config() {
  local cfg="$1"
  local out_dir="$2"
  local rwlock="$3"

  mkdir -p "$out_dir" "$(dirname "$cfg")"
  {
    printf 'data:\n'
    printf '  dataset_name: sift200k\n'
    printf '  max_elements: %s\n' "$MAX_ELEMENTS"
    printf '  begin_num: %s\n' "$BEGIN_NUM"
    printf '  data_type: float\n'
    printf '  data_path: ../data/sift/sift_base_200k.bin\n'
    printf '  incr_query_path: ../data/sift/sift_query_stream.bin\n'
    printf '  overall_query_path: ../data/sift/sift_query_stream.bin\n'
    printf 'index:\n'
    printf '  index_type: hnsw\n'
    printf '  m: 16\n'
    printf '  ef_construction: 100\n'
    printf 'search:\n'
    printf '  recall_at: 10\n'
    printf '  ef_search: %s\n' "$EF_SEARCH"
    printf '  use_node_lock: true\n'
    printf '  enable_mvcc: false\n'
    printf '  enable_undo_recovery: false\n'
    printf 'workload:\n'
    printf '  batch_size: %s\n' "$BATCH_SIZE"
    printf '  num_threads: %s\n' "$((MEASURED_WORKERS + BACKGROUND_SEARCH_WORKERS + INSERT_WORKERS))"
    printf '  queue_size: 0\n'
    printf '  insert_event_rate: 0\n'
    printf '  search_event_rate: 0\n'
    printf '  insert_burst_schedule:\n'
    printf '    - {start_ms: 2500, duration_ms: 1000, insert_event_rate: 0}\n'
    printf '    - {start_ms: 6000, duration_ms: 1000, insert_event_rate: 0}\n'
    printf '  schedule_horizon_ms: %s\n' "$DURATION_MS"
    printf '  with_external_rw_lock: %s\n' "$rwlock"
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

run_case() {
  local mode="$1"
  local rep="$2"
  local rwlock="false"
  if [[ "$mode" == "rwlock" ]]; then
    rwlock="true"
  fi

  local out="$OUT_ROOT/$QUERY_MODE/$mode/rep$rep"
  local cfg="$CFG_ROOT/${QUERY_MODE}_${mode}_rep${rep}.yaml"
  write_config "$cfg" "$out" "$rwlock"
  rm -rf "$out/raw_latency"
  mkdir -p "$out"
  echo "[hnsw-ext-rwlock-nodelock] query=$QUERY_MODE mode=$mode rep=$rep"
  env \
    M3_CLOSED_LOOP=1 \
    M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
    M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
    M3_CLOSED_LOOP_COMPETING_WORKERS="$BACKGROUND_SEARCH_WORKERS" \
    M3_CLOSED_LOOP_INSERT_WORKERS="$INSERT_WORKERS" \
    M3_CLOSED_LOOP_COMPETING_MODE="matched_rw" \
    ANNCHOR_M3_BATCH_DIAG=1 \
    ANNCHOR_PHASE_OVERLAP_DIAG=0 \
    ANNCHOR_REWRITE_ACTIVE_DIAG=0 \
    ANNCHOR_SEARCH_WORK_COUNTERS="$SEARCH_WORK_COUNTERS" \
    ANNCHOR_MEASURED_SEARCH_ARENA=1 \
    ANNCHOR_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
    ANNCHOR_SHARED_ARENA_THREADS="$BACKGROUND_SEARCH_WORKERS" \
    OCC_RAW_LATENCY_DIR="$out/raw_latency" \
    RAW_LATENCY_SKIP_SEARCH_WIDE=0 \
    LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
    ./bin/bench -config "$cfg" > "$out/run.log" 2>&1
}

for rep in $(seq 1 "$REPS"); do
  for mode in $MODES; do
    run_case "$mode" "$rep"
  done
done

echo "[hnsw-ext-rwlock-nodelock] complete: $OUT_ROOT/$QUERY_MODE"
