#!/usr/bin/env bash
# Plain-HNSW c2c/address-attribution diagnostic for insert-vs-search root cause.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/hnsw_c2c_rootcause_20260513}"
CFG_ROOT="${CFG_ROOT:-./config/hnsw_c2c_rootcause_20260513}"

DATASET_NAME="${DATASET_NAME:-sift10m}"
DATA_PATH="${DATA_PATH:-../data/sift10m/sift10m_base.bin}"
QUERY_PATH="${QUERY_PATH:-../data/sift/sift_query_stream.bin}"
MAX_ELEMENTS="${MAX_ELEMENTS:-10000000}"
BEGIN_NUM="${BEGIN_NUM:-1000000}"
M_HNSW="${M_HNSW:-16}"
EF_SEARCH="${EF_SEARCH:-50}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-100}"
BATCH_SIZE="${BATCH_SIZE:-20}"
QUERY_MODE="${QUERY_MODE:-peeking}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
DURATION_MS="${DURATION_MS:-45000}"
SKIP_PRE_MS="${SKIP_PRE_MS:-10000}"
SAMPLE_MS="${SAMPLE_MS:-25000}"

MEASURED_CPU_LIST="${MEASURED_CPU_LIST:-0-7}"
COMPETING_CPU_LIST="${COMPETING_CPU_LIST:-8-23,24-47,56-79}"
MODES="${MODES:-bg_search insert_real}"

mkdir -p "$OUT_ROOT/cells" "$CFG_ROOT"

write_config() {
  local cfg="$1"
  local out_dir="$2"
  local total_threads="$3"
  mkdir -p "$out_dir" "$(dirname "$cfg")"
  {
    printf 'data: {dataset_name: %s, max_elements: %s, begin_num: %s, data_type: float, data_path: %s, incr_query_path: %s, overall_query_path: %s}\n' \
      "$DATASET_NAME" "$MAX_ELEMENTS" "$BEGIN_NUM" "$DATA_PATH" "$QUERY_PATH" "$QUERY_PATH"
    printf 'index: {index_type: hnsw, m: %s, ef_construction: %s}\n' "$M_HNSW" "$EF_CONSTRUCTION"
    printf 'search: {recall_at: 10, ef_search: %s, use_node_lock: true, enable_mvcc: false, enable_undo_recovery: false}\n' "$EF_SEARCH"
    printf 'workload:\n'
    printf '  batch_size: %s\n' "$BATCH_SIZE"
    printf '  num_threads: %s\n' "$total_threads"
    printf '  queue_size: 0\n  insert_event_rate: 0\n  search_event_rate: 0\n'
    printf '  schedule_horizon_ms: %s\n' "$DURATION_MS"
    printf '  with_external_rw_lock: false\n  use_node_lock: true\n'
    printf '  query_mode: %s\n  per_query_latency: true\n  precompute_schedule: false\n' "$QUERY_MODE"
    printf 'result: {output_dir: %s}\n' "$out_dir"
    printf 'repetitions: 1\n'
    printf 'profile: {memory_monitor_interval: 100, enable_memory_profile: false}\n'
  } > "$cfg"
}

wait_for_closed_loop() {
  local log="$1"
  for _ in $(seq 1 240); do
    grep -q "Closed-loop setup:" "$log" 2>/dev/null && return 0
    sleep 1
  done
  tail -80 "$log" >&2 || true
  return 1
}

run_cell() {
  local mode="$1"
  local closed="search"
  [[ "$mode" == "insert_real" ]] && closed="insert"
  local n=64
  local total_threads=$((MEASURED_WORKERS + n))
  local cell_dir="$OUT_ROOT/cells/${mode}_n64_c2c"
  local cfg="$CFG_ROOT/${mode}_n64_c2c.yaml"
  rm -rf "$cell_dir/raw_latency"
  mkdir -p "$cell_dir"
  write_config "$cfg" "$cell_dir" "$total_threads"
  {
    echo "mode=$mode"
    echo "closed_loop_mode=$closed"
    echo "n_competing=$n"
    echo "plain_hnsw=1"
    echo "fake_write=0"
    echo "measured_cpus=$MEASURED_CPU_LIST"
    echo "competing_cpus=$COMPETING_CPU_LIST"
    echo "duration_ms=$DURATION_MS"
    echo "skip_pre_ms=$SKIP_PRE_MS"
    echo "sample_ms=$SAMPLE_MS"
  } > "$cell_dir/cell.meta"

  echo "[hnsw-c2c] $mode -> $cell_dir"
  (
    env \
      M3_CLOSED_LOOP=1 \
      M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
      M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
      M3_CLOSED_LOOP_COMPETING_WORKERS="$n" \
      M3_CLOSED_LOOP_INSERT_WORKERS="$n" \
      M3_CLOSED_LOOP_COMPETING_MODE="$closed" \
      ANNCHOR_M3_BATCH_DIAG=1 \
      ANNCHOR_SEARCH_WORK_COUNTERS=1 \
      HNSW_MEASURED_SEARCH_ARENA=1 \
      HNSW_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
      HNSW_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      HNSW_INSERT_CPU_LIST="$COMPETING_CPU_LIST" \
      ANNCHOR_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      ANNCHOR_SEARCH_CPU_LIST="$COMPETING_CPU_LIST" \
      ANNCHOR_INSERT_CPU_LIST="$COMPETING_CPU_LIST" \
      OCC_RAW_LATENCY_DIR="$cell_dir/raw_latency" \
      LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
      ./bin/bench -config "$cfg"
  ) > "$cell_dir/run.log" 2>&1 &
  local bench_pid=$!
  wait_for_closed_loop "$cell_dir/run.log"
  echo "$bench_pid" > "$cell_dir/bench.pid"
  sleep "$(awk -v ms="$SKIP_PRE_MS" 'BEGIN{printf "%.6f", ms/1000.0}')"
  local all_cpus="${MEASURED_CPU_LIST},${COMPETING_CPU_LIST}"
  perf c2c record --all-user -p "$bench_pid" -C "$all_cpus" \
    -o "$cell_dir/perf_c2c.data" -- sleep "$(awk -v ms="$SAMPLE_MS" 'BEGIN{printf "%.6f", ms/1000.0}')" \
    > "$cell_dir/perf_c2c_record.stdout" 2>"$cell_dir/perf_c2c_record.stderr" || true
  wait "$bench_pid" 2>/dev/null || true
  perf c2c report --stdio --show-all -i "$cell_dir/perf_c2c.data" \
    > "$cell_dir/perf_c2c_report.txt" 2>"$cell_dir/perf_c2c_report.stderr" || true
  python3 scripts/analyze_hnsw_c2c_address_attribution_20260513.py --cell "$cell_dir" \
    > "$cell_dir/perf_c2c_address_attribution.txt" 2>&1 || true
}

{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "purpose=plain_hnsw_c2c_address_attribution"
  echo "dataset_name=$DATASET_NAME"
  echo "begin_num=$BEGIN_NUM"
  echo "max_elements=$MAX_ELEMENTS"
  echo "modes=$MODES"
} > "$OUT_ROOT/MANIFEST.txt"

for mode in $MODES; do
  run_cell "$mode"
done

date --iso-8601=seconds > "$OUT_ROOT/end_time.txt"
echo "[hnsw-c2c] all done -> $OUT_ROOT"
