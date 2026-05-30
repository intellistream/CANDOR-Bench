#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/m3_prune_only_tail_matrix_20260508}"
CFG_ROOT="${CFG_ROOT:-./config/m3_prune_only_tail_matrix_20260508}"
MODE="${MODE:-clean}" # clean or probe
REPS="${REPS:-3}"
DURATION_MS="${DURATION_MS:-12000}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
BACKGROUND_SEARCH_WORKERS="${BACKGROUND_SEARCH_WORKERS:-20}"
INSERT_WORKERS="${INSERT_WORKERS:-20}"
BATCH_SIZE="${BATCH_SIZE:-20}"
CASE_FILTER="${CASE_FILTER:-}"
ALGORITHM_FILTER="${ALGORITHM_FILTER:-}"
ALGORITHMS="${ALGORITHMS:-annchor-m3-off annchor-m3}"
DATASET_SET="${DATASET_SET:-core}" # core or all
SEARCH_WORK_COUNTERS="${SEARCH_WORK_COUNTERS:-0}"
RAW_LATENCY_SKIP_SEARCH_WIDE="${RAW_LATENCY_SKIP_SEARCH_WIDE:-1}"
QUERY_MODE="${QUERY_MODE:-round_robin}"
ZIPFIAN_SKEW="${ZIPFIAN_SKEW:-0.99}"
RWLOCK_PROBE_MIN_US="${RWLOCK_PROBE_MIN_US:-1000}"
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
  local algorithm="$3"
  local dataset_name="$4"
  local max_elements="$5"
  local begin_num="$6"
  local data_path="$7"
  local query_path="$8"
  local gt_path="$9"
  local m="${10}"
  local ef_construction="${11}"
  local ef_search="${12}"

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
    printf '  index_type: %s\n' "$algorithm"
    printf '  m: %s\n' "$m"
    printf '  ef_construction: %s\n' "$ef_construction"
    printf 'search:\n'
    printf '  recall_at: 10\n'
    printf '  ef_search: %s\n' "$ef_search"
    printf '  use_node_lock: true\n'
    printf '  enable_mvcc: true\n'
    printf '  enable_undo_recovery: true\n'
    printf 'workload:\n'
    printf '  batch_size: %s\n' "$BATCH_SIZE"
    printf '  num_threads: %s\n' "$((MEASURED_WORKERS + BACKGROUND_SEARCH_WORKERS + INSERT_WORKERS))"
    printf '  queue_size: 0\n'
    printf '  insert_event_rate: 0\n'
    printf '  search_event_rate: 0\n'
    printf '  insert_burst_schedule:\n'
    printf '    - {start_ms: 2500, duration_ms: 1000, insert_event_rate: 0}\n'
    printf '    - {start_ms: 6000, duration_ms: 1000, insert_event_rate: 0}\n'
    printf '    - {start_ms: 9500, duration_ms: 1000, insert_event_rate: 0}\n'
    printf '  schedule_horizon_ms: %s\n' "$DURATION_MS"
    printf '  with_external_rw_lock: false\n'
    printf '  use_node_lock: true\n'
    printf '  query_mode: %s\n' "$QUERY_MODE"
    printf '  zipfian_skew: %s\n' "$ZIPFIAN_SKEW"
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
  if [[ "$DATASET_SET" == "all" ]]; then
    cat <<EOF
sift200k ${SIFT200K_MAX_ELEMENTS:-200000} ${SIFT200K_BEGIN_NUM:-100000} ../data/sift/sift_base_200k.bin ${SIFT200K_QUERY_PATH:-../data/sift/sift_query_1k.bin} ${SIFT200K_GT_PATH:-../data/sift/sift_base_200k.gt20} 16 100 50
sift10m200k ${SIFT10M200K_MAX_ELEMENTS:-200000} ${SIFT10M200K_BEGIN_NUM:-100000} ../data/sift10m/sift10m_base_200k.bin ../data/sift10m/sift10m_query.bin - 16 100 50
sift1m ${SIFT1M_MAX_ELEMENTS:-1000000} ${SIFT1M_BEGIN_NUM:-800000} ../data/sift/sift_base.bin ../data/sift/sift_query.bin ../data/sift/sift_base.gt20 16 100 50
gist200k ${GIST200K_MAX_ELEMENTS:-200000} ${GIST200K_BEGIN_NUM:-100000} ../data/gist/gist_base_200k.bin ../data/gist/gist_query.bin ../data/gist/gist_base_200k.gt20 16 200 80
dbpedia50k ${DBPEDIA50K_MAX_ELEMENTS:-50000} ${DBPEDIA50K_BEGIN_NUM:-25000} ../data/dbpedia_openai/dbpedia_openai_50k.bin ../data/dbpedia_openai/dbpedia_openai_query.bin - 16 100 50
EOF
    return
  fi
  cat <<EOF
sift200k ${SIFT200K_MAX_ELEMENTS:-200000} ${SIFT200K_BEGIN_NUM:-100000} ../data/sift/sift_base_200k.bin ${SIFT200K_QUERY_PATH:-../data/sift/sift_query_1k.bin} ${SIFT200K_GT_PATH:-../data/sift/sift_base_200k.gt20} 16 100 50
sift10m200k ${SIFT10M200K_MAX_ELEMENTS:-200000} ${SIFT10M200K_BEGIN_NUM:-100000} ../data/sift10m/sift10m_base_200k.bin ../data/sift10m/sift10m_query.bin - 16 100 50
EOF
}

run_case() {
  local algorithm="$1" dataset_name="$2" max_elements="$3" begin_num="$4" data_path="$5"
  local query_path="$6" gt_path="$7" m="$8" ef_construction="$9" ef_search="${10}" rep="${11}"

  local case_name="${dataset_name}_${algorithm}_rep${rep}"
  local out="$OUT_ROOT/$MODE/$case_name"
  local cfg="$CFG_ROOT/$MODE/$case_name.yaml"

  if [[ "$gt_path" == "-" ]]; then
    gt_path=""
  fi
  write_config "$cfg" "$out" "$algorithm" "$dataset_name" "$max_elements" "$begin_num" \
    "$data_path" "$query_path" "$gt_path" "$m" "$ef_construction" "$ef_search"

  rm -rf "$out/raw_latency" "$out/rwlock_probe.csv"
  mkdir -p "$out"
  {
    printf 'mode,%s\n' "$MODE"
    printf 'algorithm,%s\n' "$algorithm"
    printf 'dataset,%s\n' "$dataset_name"
    printf 'background_search_workers,%s\n' "$BACKGROUND_SEARCH_WORKERS"
    printf 'insert_workers,%s\n' "$INSERT_WORKERS"
    printf 'measured_workers,%s\n' "$MEASURED_WORKERS"
    printf 'rep,%s\n' "$rep"
    printf 'duration_ms,%s\n' "$DURATION_MS"
    printf 'batch_size,%s\n' "$BATCH_SIZE"
    printf 'ef_search,%s\n' "$ef_search"
    printf 'ef_construction,%s\n' "$ef_construction"
  } > "$out/experiment_meta.csv"

  echo "[m3-prune-only-tail] mode=$MODE algorithm=$algorithm dataset=$dataset_name rep=$rep"

  env_args=(
    M3_CLOSED_LOOP=1
    M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS"
    M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS"
    M3_CLOSED_LOOP_COMPETING_WORKERS="$BACKGROUND_SEARCH_WORKERS"
    M3_CLOSED_LOOP_INSERT_WORKERS="$INSERT_WORKERS"
    M3_CLOSED_LOOP_COMPETING_MODE="matched_rw"
    ANNCHOR_M3_BATCH_DIAG="${M3_BATCH_DIAG:-0}"
    ANNCHOR_PHASE_OVERLAP_DIAG="${PHASE_OVERLAP_DIAG:-0}"
    ANNCHOR_REWRITE_ACTIVE_DIAG="${REWRITE_ACTIVE_DIAG:-0}"
    ANNCHOR_SEARCH_WORK_COUNTERS="$SEARCH_WORK_COUNTERS"
    ANNCHOR_HOT_REGION_REPAIR=0
    ANNCHOR_L0_ADJ_OVERLAY=0
    ANNCHOR_L0_EDGE_DELTA=0
    ANNCHOR_MEASURED_SEARCH_ARENA=1
    ANNCHOR_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS"
    ANNCHOR_SHARED_ARENA_THREADS="$BACKGROUND_SEARCH_WORKERS"
    OCC_RAW_LATENCY_DIR="$out/raw_latency"
    RAW_LATENCY_SKIP_SEARCH_WIDE="$RAW_LATENCY_SKIP_SEARCH_WIDE"
    LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}"
  )
  if [[ "$MODE" == "probe" ]]; then
    env_args+=(
      RWLOCK_PROBE_LOG="$out/rwlock_probe.csv"
      RWLOCK_PROBE_MIN_US="$RWLOCK_PROBE_MIN_US"
      LD_PRELOAD="$PWD/$PROBE_SO"
    )
  fi

  env "${env_args[@]}" ./bin/bench -config "$cfg" > "$out/run.log" 2>&1
}

for rep in $(seq 1 "$REPS"); do
  while read -r dataset_name max_elements begin_num data_path query_path gt_path m ef_construction ef_search; do
    if [[ -n "$CASE_FILTER" && "$dataset_name" != "$CASE_FILTER" ]]; then
      continue
    fi
    for algorithm in $ALGORITHMS; do
      if [[ -n "$ALGORITHM_FILTER" && "$ALGORITHM_FILTER" != "$algorithm" ]]; then
        continue
      fi
      run_case "$algorithm" "$dataset_name" "$max_elements" "$begin_num" "$data_path" \
        "$query_path" "$gt_path" "$m" "$ef_construction" "$ef_search" "$rep"
    done
  done < <(write_dataset_cases)
done

echo "[m3-prune-only-tail] complete: $OUT_ROOT/$MODE"
