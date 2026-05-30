#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/m2_tail_causality_matrix_20260508}"
CFG_ROOT="${CFG_ROOT:-./config/m2_tail_causality_matrix_20260508}"
MODE="${MODE:-clean}" # clean or probe
REPS="${REPS:-3}"
DURATION_MS="${DURATION_MS:-12000}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-20}"
RWLOCK_PROBE_MIN_US="${RWLOCK_PROBE_MIN_US:-1000}"
COMPETING_WORKER_SET="${COMPETING_WORKER_SET:-20 40}"
COMPETING_MODE="${COMPETING_MODE:-periodic}"
INSERT_WORKERS="${INSERT_WORKERS:-}"
PROBE_SO="./tools/rwlock_probe.so"

mkdir -p "$OUT_ROOT" "$CFG_ROOT"

if [[ ! -x "./bin/bench" ]]; then
  echo "missing ./bin/bench" >&2
  exit 1
fi
if [[ "$MODE" == "probe" && ! -f "$PROBE_SO" ]]; then
  echo "missing $PROBE_SO" >&2
  exit 1
fi
if [[ "$MODE" != "clean" && "$MODE" != "probe" ]]; then
  echo "MODE must be clean or probe" >&2
  exit 2
fi

write_config() {
  local cfg="$1"
  local out_dir="$2"
  local dataset_name="$3"
  local max_elements="$4"
  local begin_num="$5"
  local data_path="$6"
  local query_path="$7"
  local gt_path="$8"
  local node_lock="$9"
  local m="${10}"
  local ef_construction="${11}"
  local ef_search="${12}"
  local competing_workers="${13}"

  mkdir -p "$out_dir" "$(dirname "$cfg")"
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
    local insert_workers_for_threads=0
    if [[ "$COMPETING_MODE" == "search_plus_periodic_insert" || "$COMPETING_MODE" == "bg_search_periodic_insert" || "$COMPETING_MODE" == "matched_rw" ]]; then
      insert_workers_for_threads="${INSERT_WORKERS:-$competing_workers}"
    fi
    printf '  num_threads: %s\n' "$((MEASURED_WORKERS + competing_workers + insert_workers_for_threads))"
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

write_dataset_cases() {
  local node_lock="$1"
  cat <<EOF
sift200k $node_lock 200000 100000 ../data/sift/sift_base_200k.bin ../data/sift/sift_query_1k.bin ../data/sift/sift_base_200k.gt20 16 100 50
sift10m200k $node_lock 200000 100000 ../data/sift10m/sift10m_base_200k.bin ../data/sift10m/sift10m_query.bin - 16 100 50
sift1m $node_lock 1000000 800000 ../data/sift/sift_base.bin ../data/sift/sift_query.bin ../data/sift/sift_base.gt20 16 100 50
gist200k $node_lock 200000 100000 ../data/gist/gist_base_200k.bin ../data/gist/gist_query.bin ../data/gist/gist_base_200k.gt20 16 200 80
dbpedia50k $node_lock 50000 25000 ../data/dbpedia_openai/dbpedia_openai_50k.bin ../data/dbpedia_openai/dbpedia_openai_query.bin - 16 100 50
EOF
}

run_case() {
  local dataset_name="$1" node_lock="$2" max_elements="$3" begin_num="$4" data_path="$5"
  local query_path="$6" gt_path="$7" m="$8" ef_construction="$9" ef_search="${10}"
  local competing_workers="${11}" rep="${12}"

  local lock_label="node_lock_${node_lock}"
  local case_name="${dataset_name}_cw${competing_workers}_${lock_label}_rep${rep}"
  local out="$OUT_ROOT/$MODE/$case_name"
  local cfg="$CFG_ROOT/$MODE/$case_name.yaml"

  if [[ "$gt_path" == "-" ]]; then
    gt_path=""
  fi
  write_config "$cfg" "$out" "$dataset_name" "$max_elements" "$begin_num" "$data_path" "$query_path" \
    "$gt_path" "$node_lock" "$m" "$ef_construction" "$ef_search" "$competing_workers"

  rm -rf "$out/raw_latency" "$out/rwlock_probe.csv"
  mkdir -p "$out"
  {
    printf 'mode,%s\n' "$MODE"
    printf 'dataset,%s\n' "$dataset_name"
    printf 'node_lock,%s\n' "$node_lock"
    printf 'competing_workers,%s\n' "$competing_workers"
    printf 'competing_mode,%s\n' "$COMPETING_MODE"
    printf 'insert_workers,%s\n' "${INSERT_WORKERS:-$competing_workers}"
    printf 'measured_workers,%s\n' "$MEASURED_WORKERS"
    printf 'rep,%s\n' "$rep"
    printf 'duration_ms,%s\n' "$DURATION_MS"
    printf 'batch_size,%s\n' "$BATCH_SIZE"
    printf 'ef_search,%s\n' "$ef_search"
    printf 'ef_construction,%s\n' "$ef_construction"
  } > "$out/experiment_meta.csv"

  echo "[m2-tail-causality-matrix] mode=$MODE dataset=$dataset_name cw=$competing_workers node_lock=$node_lock rep=$rep"

  env_args=(
    M3_CLOSED_LOOP=1
    M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS"
    M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS"
    M3_CLOSED_LOOP_COMPETING_WORKERS="$competing_workers"
    M3_CLOSED_LOOP_COMPETING_MODE="$COMPETING_MODE"
    ANNCHOR_M3_BATCH_DIAG=0
    ANNCHOR_PHASE_OVERLAP_DIAG=0
    ANNCHOR_REWRITE_ACTIVE_DIAG=0
    ANNCHOR_SEARCH_WORK_COUNTERS=0
    ANNCHOR_MEASURED_SEARCH_ARENA=1
    ANNCHOR_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS"
    ANNCHOR_SHARED_ARENA_THREADS="$competing_workers"
    OCC_RAW_LATENCY_DIR="$out/raw_latency"
    RAW_LATENCY_SKIP_SEARCH_WIDE=1
    LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}"
  )
  if [[ -n "$INSERT_WORKERS" ]]; then
    env_args+=(M3_CLOSED_LOOP_INSERT_WORKERS="$INSERT_WORKERS")
  fi
  if [[ "$MODE" == "probe" ]]; then
    env_args+=(
      RWLOCK_PROBE_LOG="$out/rwlock_probe.csv"
      RWLOCK_PROBE_MIN_US="$RWLOCK_PROBE_MIN_US"
      LD_PRELOAD="$PWD/$PROBE_SO"
    )
  fi

  env "${env_args[@]}" ./bin/bench -config "$cfg" > "$out/run.log" 2>&1
}

case_filter="${CASE_FILTER:-}"
for competing_workers in $COMPETING_WORKER_SET; do
  for rep in $(seq 1 "$REPS"); do
    while read -r dataset_name node_lock max_elements begin_num data_path query_path gt_path m ef_construction ef_search; do
      if [[ -n "$case_filter" && "$dataset_name" != "$case_filter" ]]; then
        continue
      fi
      run_case "$dataset_name" "$node_lock" "$max_elements" "$begin_num" "$data_path" "$query_path" \
        "$gt_path" "$m" "$ef_construction" "$ef_search" "$competing_workers" "$rep"
    done < <(write_dataset_cases true)
    while read -r dataset_name node_lock max_elements begin_num data_path query_path gt_path m ef_construction ef_search; do
      if [[ -n "$case_filter" && "$dataset_name" != "$case_filter" ]]; then
        continue
      fi
      run_case "$dataset_name" "$node_lock" "$max_elements" "$begin_num" "$data_path" "$query_path" \
        "$gt_path" "$m" "$ef_construction" "$ef_search" "$competing_workers" "$rep"
    done < <(write_dataset_cases false)
  done
done

echo "[m2-tail-causality-matrix] complete: $OUT_ROOT/$MODE"
