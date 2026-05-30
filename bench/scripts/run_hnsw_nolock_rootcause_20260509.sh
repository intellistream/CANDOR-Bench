#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${OUT_ROOT:-./result/hnsw_nolock_rootcause_20260509}"
CFG_ROOT="${CFG_ROOT:-./config/hnsw_nolock_rootcause_20260509}"
MAX_ELEMENTS="${MAX_ELEMENTS:-150000}"
BEGIN_NUM="${BEGIN_NUM:-50000}"
THREADS="${THREADS:-32}"
BATCH_SIZE="${BATCH_SIZE:-200}"
INSERT_RATE="${INSERT_RATE:-40000}"
SEARCH_RATE="${SEARCH_RATE:-40000}"
QUERY_MODE="${QUERY_MODE:-chasing}"
REPS="${REPS:-1}"
MODES="${MODES:-readonly_noop cpu_dummy ann_search_dummy nolock_full rwlock_full}"
USE_NODE_LOCK="${USE_NODE_LOCK:-false}"

mkdir -p "$OUT_ROOT" "$CFG_ROOT"

if [[ ! -x "./bin/bench" ]]; then
  echo "missing ./bin/bench" >&2
  exit 1
fi

write_config() {
  local cfg="$1"
  local out="$2"
  local rwlock="$3"
  mkdir -p "$out" "$(dirname "$cfg")"
  {
    printf 'data:\n'
    printf '  dataset_name: sift\n'
    printf '  max_elements: %s\n' "$MAX_ELEMENTS"
    printf '  begin_num: %s\n' "$BEGIN_NUM"
    printf '  data_type: float\n'
    printf '  data_path: ../data/sift/sift_base.bin\n'
    printf '  incr_query_path: ../data/sift/sift_query_stream.bin\n'
    printf '  overall_query_path: ../data/sift/sift_query.bin\n'
    printf '  overall_gt_path: ../data/sift/sift_base.gt20\n'
    printf 'index:\n'
    printf '  index_type: hnsw\n'
    printf '  m: 16\n'
    printf '  ef_construction: 100\n'
    printf 'search:\n'
    printf '  recall_at: 10\n'
    printf '  ef_search: 50\n'
    printf '  use_node_lock: %s\n' "$USE_NODE_LOCK"
    printf '  enable_mvcc: false\n'
    printf '  enable_undo_recovery: false\n'
    printf 'workload:\n'
    printf '  batch_size: %s\n' "$BATCH_SIZE"
    printf '  num_threads: %s\n' "$THREADS"
    printf '  queue_size: 0\n'
    printf '  insert_event_rate: %s\n' "$INSERT_RATE"
    printf '  search_event_rate: %s\n' "$SEARCH_RATE"
    printf '  with_external_rw_lock: %s\n' "$rwlock"
    printf '  use_node_lock: %s\n' "$USE_NODE_LOCK"
    printf '  query_mode: %s\n' "$QUERY_MODE"
    printf '  per_query_latency: true\n'
    printf '  precompute_schedule: true\n'
    printf 'compare:\n'
    printf '  enabled: false\n'
    printf '  simulate:\n'
    printf '    enabled: false\n'
    printf 'result:\n'
    printf '  output_dir: %s\n' "$out"
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
  local dummy=""
  local loops=""
  case "$mode" in
    readonly_noop)
      dummy="noop"
      ;;
    cpu_dummy)
      dummy="cpu"
      loops="${DUMMY_CPU_LOOPS:-20000}"
      ;;
    ann_search_dummy)
      dummy="ann_search"
      loops="${DUMMY_ANN_SEARCH_LOOPS:-5}"
      ;;
    nolock_full)
      ;;
    rwlock_full)
      rwlock="true"
      ;;
    *)
      echo "unknown mode: $mode" >&2
      return 2
      ;;
  esac

  local out="$OUT_ROOT/$mode/rep$rep"
  local cfg="$CFG_ROOT/${mode}_rep${rep}.yaml"
  write_config "$cfg" "$out" "$rwlock"
  rm -rf "$out/raw_latency"
  mkdir -p "$out"
  echo "[hnsw-nolock-rootcause] mode=$mode rep=$rep"
  env \
    INSERT_DUMMY_MODE="$dummy" \
    INSERT_DUMMY_LOOPS="$loops" \
    ANNCHOR_M3_BATCH_DIAG=1 \
    ANNCHOR_SEARCH_WORK_COUNTERS=1 \
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

echo "[hnsw-nolock-rootcause] complete: $OUT_ROOT"
