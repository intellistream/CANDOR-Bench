#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/m2_node_lock_cause_matrix_20260507}"
CFG_ROOT="${CFG_ROOT:-./config/m2_node_lock_cause_matrix_20260507}"
PROBE_SO="./tools/rwlock_probe.so"

DURATION_MS="${DURATION_MS:-12000}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
COMPETING_WORKERS="${COMPETING_WORKERS:-40}"
BATCH_SIZE="${BATCH_SIZE:-20}"
EF_SEARCH="${EF_SEARCH:-50}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-100}"
RWLOCK_PROBE_MIN_US="${RWLOCK_PROBE_MIN_US:-1000}"
SEARCH_WORK_COUNTERS="${SEARCH_WORK_COUNTERS:-0}"
RAW_LATENCY_SKIP_SEARCH_WIDE="${RAW_LATENCY_SKIP_SEARCH_WIDE:-1}"

mkdir -p "$OUT_ROOT" "$CFG_ROOT"

write_config() {
  local case_name="$1"
  local dataset_name="$2"
  local max_elements="$3"
  local begin_num="$4"
  local data_path="$5"
  local query_path="$6"
  local gt_path="$7"
  local node_lock="$8"
  local m="${9:-16}"
  local ef_construction="${10:-$EF_CONSTRUCTION}"
  local ef_search="${11:-$EF_SEARCH}"
  local out_dir="$OUT_ROOT/$case_name"
  local cfg="$CFG_ROOT/$case_name.yaml"

  mkdir -p "$out_dir"
  {
    printf 'data:\n'
    printf '  dataset_name: %s\n' "$dataset_name"
    printf '  max_elements: %s\n' "$max_elements"
    printf '  begin_num: %s\n' "$begin_num"
    printf '  data_type: float\n'
    printf '  data_path: %s\n' "$data_path"
    printf '  incr_query_path: %s\n' "$query_path"
    printf '  overall_query_path: %s\n' "$query_path"
    if [[ -n "$gt_path" ]]; then
      printf '  overall_gt_path: %s\n' "$gt_path"
    fi
    printf 'index:\n'
    printf '  index_type: annchor-m2\n'
    printf '  m: %s\n' "$m"
    printf '  ef_construction: %s\n' "$ef_construction"
    printf 'search:\n'
    printf '  recall_at: 10\n'
    printf '  ef_search: %s\n' "$ef_search"
    printf '  use_node_lock: %s\n' "$node_lock"
    printf '  enable_mvcc: true\n'
    printf '  enable_undo_recovery: true\n'
    printf 'workload:\n'
    printf '  batch_size: %s\n' "$BATCH_SIZE"
    printf '  num_threads: 48\n'
    printf '  queue_size: 0\n'
    printf '  insert_event_rate: 0\n'
    printf '  search_event_rate: 0\n'
    printf '  insert_burst_schedule:\n'
    printf '    - {start_ms: 2500, duration_ms: 1000, insert_event_rate: 0}\n'
    printf '    - {start_ms: 6000, duration_ms: 1000, insert_event_rate: 0}\n'
    printf '    - {start_ms: 9500, duration_ms: 1000, insert_event_rate: 0}\n'
    printf '  schedule_horizon_ms: %s\n' "$DURATION_MS"
    printf '  with_external_rw_lock: false\n'
    printf '  use_node_lock: %s\n' "$node_lock"
    printf '  query_mode: round_robin\n'
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

write_config "sift200k_node_lock_on" "sift200k" 200000 100000 "../data/sift/sift_base_200k.bin" "../data/sift/sift_query_1k.bin" "../data/sift/sift_base_200k.gt20" "true"
write_config "sift200k_node_lock_off" "sift200k" 200000 100000 "../data/sift/sift_base_200k.bin" "../data/sift/sift_query_1k.bin" "../data/sift/sift_base_200k.gt20" "false"
write_config "sift10m200k_node_lock_on" "sift10m200k" 200000 100000 "../data/sift10m/sift10m_base_200k.bin" "../data/sift10m/sift10m_query.bin" "" "true"
write_config "sift10m200k_node_lock_off" "sift10m200k" 200000 100000 "../data/sift10m/sift10m_base_200k.bin" "../data/sift10m/sift10m_query.bin" "" "false"
write_config "sift1m_node_lock_on" "sift1m" 1000000 800000 "../data/sift/sift_base.bin" "../data/sift/sift_query.bin" "../data/sift/sift_base.gt20" "true"
write_config "sift1m_node_lock_off" "sift1m" 1000000 800000 "../data/sift/sift_base.bin" "../data/sift/sift_query.bin" "../data/sift/sift_base.gt20" "false"
write_config "gist200k_node_lock_on" "gist200k" 200000 100000 "../data/gist/gist_base_200k.bin" "../data/gist/gist_query.bin" "../data/gist/gist_base_200k.gt20" "true" 16 200 80
write_config "gist200k_node_lock_off" "gist200k" 200000 100000 "../data/gist/gist_base_200k.bin" "../data/gist/gist_query.bin" "../data/gist/gist_base_200k.gt20" "false" 16 200 80
write_config "dbpedia50k_node_lock_on" "dbpedia50k" 50000 25000 "../data/dbpedia_openai/dbpedia_openai_50k.bin" "../data/dbpedia_openai/dbpedia_openai_query.bin" "" "true"
write_config "dbpedia50k_node_lock_off" "dbpedia50k" 50000 25000 "../data/dbpedia_openai/dbpedia_openai_50k.bin" "../data/dbpedia_openai/dbpedia_openai_query.bin" "" "false"

if [[ ! -x "./bin/bench" ]]; then
  echo "missing ./bin/bench" >&2
  exit 1
fi
if [[ ! -f "$PROBE_SO" ]]; then
  echo "missing $PROBE_SO" >&2
  exit 1
fi

if [[ "$#" -gt 0 ]]; then
  CASES=("$@")
else
  CASES=(
    sift200k_node_lock_on
    sift200k_node_lock_off
    sift10m200k_node_lock_on
    sift10m200k_node_lock_off
    sift1m_node_lock_on
    sift1m_node_lock_off
    gist200k_node_lock_on
    gist200k_node_lock_off
    dbpedia50k_node_lock_on
    dbpedia50k_node_lock_off
  )
fi

for case_name in "${CASES[@]}"; do
  cfg="$CFG_ROOT/$case_name.yaml"
  out="$OUT_ROOT/$case_name"
  rm -rf "$out/raw_latency" "$out/rwlock_probe.csv"
  mkdir -p "$out"
  echo "[m2-node-lock-cause] case=$case_name cfg=$cfg out=$out"
  env \
    M3_CLOSED_LOOP=1 \
    M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
    M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
    M3_CLOSED_LOOP_COMPETING_WORKERS="$COMPETING_WORKERS" \
    M3_CLOSED_LOOP_COMPETING_MODE=periodic \
    ANNCHOR_M3_BATCH_DIAG=0 \
    ANNCHOR_PHASE_OVERLAP_DIAG=0 \
    ANNCHOR_REWRITE_ACTIVE_DIAG=0 \
    ANNCHOR_SEARCH_WORK_COUNTERS="$SEARCH_WORK_COUNTERS" \
    ANNCHOR_MEASURED_SEARCH_ARENA=1 \
    ANNCHOR_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
    ANNCHOR_SHARED_ARENA_THREADS="$COMPETING_WORKERS" \
    OCC_RAW_LATENCY_DIR="$out/raw_latency" \
    RAW_LATENCY_SKIP_SEARCH_WIDE="$RAW_LATENCY_SKIP_SEARCH_WIDE" \
    RWLOCK_PROBE_LOG="$out/rwlock_probe.csv" \
    RWLOCK_PROBE_MIN_US="$RWLOCK_PROBE_MIN_US" \
    LD_PRELOAD="$PWD/$PROBE_SO" \
    LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
    ./bin/bench -config "$cfg" > "$out/run.log" 2>&1
done

echo "[m2-node-lock-cause] complete: $OUT_ROOT"
