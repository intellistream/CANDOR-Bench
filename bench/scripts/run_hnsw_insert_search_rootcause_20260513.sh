#!/usr/bin/env bash
# Plain-HNSW root-cause experiment for real insert hurting concurrent search P99.
#
# This script intentionally does not use fake writes and does not change insert
# semantics. It only adds measurement:
#   - search batch wall time, lock wait, edge count, per-edge cost;
#   - writer phase-overlap counters observed by search;
#   - perf stat for reader/writer LLC and uncore traffic;
#   - a periodic real-insert cell to distinguish dynamic interference from
#     persistent post-insert graph-structure effects.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/hnsw_insert_search_rootcause_20260513}"
CFG_ROOT="${CFG_ROOT:-./config/hnsw_insert_search_rootcause_20260513}"
BENCH_BIN="${BENCH_BIN:-./bin/bench}"
BENCH_LD_LIBRARY_PATH="${BENCH_LD_LIBRARY_PATH:-./build/lib:${LD_LIBRARY_PATH:-}}"

DATASET_NAME="${DATASET_NAME:-sift10m}"
DATA_PATH="${DATA_PATH:-../data/sift10m/sift10m_base.bin}"
QUERY_PATH="${QUERY_PATH:-../data/sift/sift_query_stream.bin}"
OVERALL_QUERY_PATH="${OVERALL_QUERY_PATH:-}"
OVERALL_GT_PATH="${OVERALL_GT_PATH:-}"
MAX_ELEMENTS="${MAX_ELEMENTS:-10000000}"
BEGIN_NUM="${BEGIN_NUM:-1000000}"
M_HNSW="${M_HNSW:-16}"
EF_SEARCH="${EF_SEARCH:-50}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-100}"
BATCH_SIZE="${BATCH_SIZE:-20}"
QUERY_MODE="${QUERY_MODE:-peeking}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
INSERT_EVENT_RATE="${INSERT_EVENT_RATE:-0}"
SEARCH_EVENT_RATE="${SEARCH_EVENT_RATE:-0}"

STEADY_DURATION_MS="${STEADY_DURATION_MS:-45000}"
STEADY_SKIP_PRE_MS="${STEADY_SKIP_PRE_MS:-10000}"
STEADY_SAMPLE_MS="${STEADY_SAMPLE_MS:-25000}"
PERIODIC_DURATION_MS="${PERIODIC_DURATION_MS:-60000}"
INSERT_WINDOW_START_MS="${INSERT_WINDOW_START_MS:-20000}"
INSERT_WINDOW_DURATION_MS="${INSERT_WINDOW_DURATION_MS:-20000}"
SEGMENT_GUARD_MS="${SEGMENT_GUARD_MS:-2000}"

MEASURED_CPU_LIST="${MEASURED_CPU_LIST:-0-7}"
COMPETING_CPU_POOL_START="${COMPETING_CPU_POOL_START:-8}"
COMPETING_CPU_POOL_END_S0="${COMPETING_CPU_POOL_END_S0:-23}"
COMPETING_CPU_POOL_START_S1="${COMPETING_CPU_POOL_START_S1:-24}"
COMPETING_CPU_POOL_END_S1="${COMPETING_CPU_POOL_END_S1:-47}"
COMPETING_HT_POOL_START="${COMPETING_HT_POOL_START:-56}"

N_VALUES="${N_VALUES:-16 64}"
REPS="${REPS:-1 2 3}"
MODES="${MODES:-baseline bg_search insert_real periodic_insert}"
ENABLE_PERF="${ENABLE_PERF:-1}"
ENABLE_UNCORE_PERF="${ENABLE_UNCORE_PERF:-1}"
ENABLE_INSERT_PHASE_TRACE="${ENABLE_INSERT_PHASE_TRACE:-1}"
M3_BATCH_DIAG="${M3_BATCH_DIAG:-0}"
PHASE_OVERLAP_DIAG="${PHASE_OVERLAP_DIAG:-0}"
TRACE_MODE="${TRACE_MODE:-insert_real}"
TRACE_N="${TRACE_N:-64}"
TRACE_REP="${TRACE_REP:-1}"
TRACE_FROM_DELTA="${TRACE_FROM_DELTA:-800000}"

PERF_EVENTS_CORE="${PERF_EVENTS_CORE:-cycles,instructions,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses}"
PERF_EVENTS_UNCORE="${PERF_EVENTS_UNCORE:-uncore_imc/cas_count_read/,uncore_imc/cas_count_write/}"

mkdir -p "$OUT_ROOT/cells" "$CFG_ROOT"

die() {
  echo "[hnsw-rootcause] ERROR: $*" >&2
  exit 1
}

record_platform_state() {
  local out="$1"
  {
    local gov="/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if [[ -r "$gov" ]]; then
      echo "cpu0_governor=$(cat "$gov")"
    else
      echo "cpu0_governor=unavailable"
    fi
    for pkg in /sys/devices/system/cpu/intel_uncore_frequency/package_*/; do
      [[ -d "$pkg" ]] || continue
      local name
      name="$(basename "$pkg")"
      local min="unavailable"
      local max="unavailable"
      [[ -r "$pkg/min_freq_khz" ]] && min="$(cat "$pkg/min_freq_khz")"
      [[ -r "$pkg/max_freq_khz" ]] && max="$(cat "$pkg/max_freq_khz")"
      echo "uncore_${name}_min_khz=$min"
      echo "uncore_${name}_max_khz=$max"
    done
    if [[ -r /var/run/anns_cstate_pin.pid ]] && kill -0 "$(cat /var/run/anns_cstate_pin.pid)" 2>/dev/null; then
      echo "cpu_dma_latency_pin_pid=$(cat /var/run/anns_cstate_pin.pid)"
    else
      echo "cpu_dma_latency_pin_pid=inactive_or_unavailable"
    fi
  } > "$out"
}

if [[ ! -x "$BENCH_BIN" ]]; then
  die "missing BENCH_BIN=$BENCH_BIN"
fi
if [[ "${ALLOW_SMALL_DATASET:-0}" != "1" ]]; then
  [[ "$MAX_ELEMENTS" -ge 1000000 ]] || die "MAX_ELEMENTS=$MAX_ELEMENTS below full-scale gate"
  [[ "$BEGIN_NUM" -ge 1000000 ]] || die "BEGIN_NUM=$BEGIN_NUM below full-scale gate"
  [[ $((MAX_ELEMENTS - BEGIN_NUM)) -ge 800000 ]] || die "insert headroom too small"
fi

competing_cpu_list_for_n() {
  local n="$1"
  if [[ "$n" -le 0 ]]; then echo ""; return; fi
  local s0=$((COMPETING_CPU_POOL_END_S0 - COMPETING_CPU_POOL_START + 1))
  if [[ "$n" -le "$s0" ]]; then
    echo "${COMPETING_CPU_POOL_START}-$((COMPETING_CPU_POOL_START + n - 1))"
    return
  fi
  local rem=$((n - s0))
  local s1_len=$((COMPETING_CPU_POOL_END_S1 - COMPETING_CPU_POOL_START_S1 + 1))
  if [[ "$rem" -le "$s1_len" ]]; then
    echo "${COMPETING_CPU_POOL_START}-${COMPETING_CPU_POOL_END_S0},${COMPETING_CPU_POOL_START_S1}-$((COMPETING_CPU_POOL_START_S1 + rem - 1))"
    return
  fi
  local rem2=$((rem - s1_len))
  echo "${COMPETING_CPU_POOL_START}-${COMPETING_CPU_POOL_END_S0},${COMPETING_CPU_POOL_START_S1}-${COMPETING_CPU_POOL_END_S1},${COMPETING_HT_POOL_START}-$((COMPETING_HT_POOL_START + rem2 - 1))"
}

write_config() {
  local cfg="$1"
  local out_dir="$2"
  local total_threads="$3"
  local duration_ms="$4"
  local with_burst="$5"
  local overall_query_path="${OVERALL_QUERY_PATH:-$QUERY_PATH}"
  mkdir -p "$out_dir" "$(dirname "$cfg")"
  {
    if [[ -n "$OVERALL_GT_PATH" ]]; then
      printf 'data: {dataset_name: %s, max_elements: %s, begin_num: %s, data_type: float, data_path: %s, incr_query_path: %s, overall_query_path: %s, overall_gt_path: %s}\n' \
        "$DATASET_NAME" "$MAX_ELEMENTS" "$BEGIN_NUM" "$DATA_PATH" "$QUERY_PATH" "$overall_query_path" "$OVERALL_GT_PATH"
    else
      printf 'data: {dataset_name: %s, max_elements: %s, begin_num: %s, data_type: float, data_path: %s, incr_query_path: %s, overall_query_path: %s}\n' \
        "$DATASET_NAME" "$MAX_ELEMENTS" "$BEGIN_NUM" "$DATA_PATH" "$QUERY_PATH" "$overall_query_path"
    fi
    printf 'index: {index_type: hnsw, m: %s, ef_construction: %s}\n' "$M_HNSW" "$EF_CONSTRUCTION"
    printf 'search: {recall_at: 10, ef_search: %s, use_node_lock: true, enable_mvcc: false, enable_undo_recovery: false}\n' "$EF_SEARCH"
    printf 'workload:\n'
    printf '  batch_size: %s\n' "$BATCH_SIZE"
    printf '  num_threads: %s\n' "$total_threads"
    printf '  queue_size: 0\n  insert_event_rate: %s\n  search_event_rate: %s\n' "$INSERT_EVENT_RATE" "$SEARCH_EVENT_RATE"
    printf '  schedule_horizon_ms: %s\n' "$duration_ms"
    if [[ "$with_burst" == "1" ]]; then
      printf '  insert_burst_schedule:\n'
      printf '    - {start_ms: %s, duration_ms: %s, insert_event_rate: 1}\n' \
        "$INSERT_WINDOW_START_MS" "$INSERT_WINDOW_DURATION_MS"
    fi
    printf '  with_external_rw_lock: false\n  use_node_lock: true\n'
    printf '  query_mode: %s\n  per_query_latency: true\n  precompute_schedule: false\n' "$QUERY_MODE"
    printf 'result: {output_dir: %s}\n' "$out_dir"
    printf 'repetitions: 1\n'
    printf 'profile: {memory_monitor_interval: 100, enable_memory_profile: false}\n'
  } > "$cfg"
}

wait_for_closed_loop() {
  local log="$1"
  local max_wait_s="${2:-240}"
  local waited=0
  while [[ "$waited" -lt "$max_wait_s" ]]; do
    if grep -q "Closed-loop setup:" "$log" 2>/dev/null; then return 0; fi
    if grep -q "panic:\\|fatal error:" "$log" 2>/dev/null; then
      tail -80 "$log" >&2 || true
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  tail -80 "$log" >&2 || true
  return 1
}

closed_mode_for_mode() {
  case "$1" in
    baseline) echo "idle" ;;
    bg_search) echo "search" ;;
    insert_real) echo "insert" ;;
    periodic_insert) echo "periodic" ;;
    *) die "unknown mode $1" ;;
  esac
}

run_perf_stat() {
  local cell_dir="$1"
  local competing_cpus="$2"
  local sample_s="$3"
  if [[ "$ENABLE_PERF" != "1" ]]; then return 0; fi
  perf stat -C "$MEASURED_CPU_LIST" -x, -e "$PERF_EVENTS_CORE" -o "$cell_dir/perf_reader.txt" -- sleep "$sample_s" 2>"$cell_dir/perf_reader.stderr" &
  local p_reader=$!
  local p_competing=""
  if [[ -n "$competing_cpus" ]]; then
    perf stat -C "$competing_cpus" -x, -e "$PERF_EVENTS_CORE" -o "$cell_dir/perf_competing.txt" -- sleep "$sample_s" 2>"$cell_dir/perf_competing.stderr" &
    p_competing=$!
  fi
  local p_uncore=""
  if [[ "$ENABLE_UNCORE_PERF" == "1" ]]; then
    perf stat -a -x, -e "$PERF_EVENTS_UNCORE" -o "$cell_dir/perf_uncore.txt" -- sleep "$sample_s" 2>"$cell_dir/perf_uncore.stderr" &
    p_uncore=$!
  fi
  wait "$p_reader" 2>/dev/null || true
  [[ -n "$p_competing" ]] && wait "$p_competing" 2>/dev/null || true
  [[ -n "$p_uncore" ]] && wait "$p_uncore" 2>/dev/null || true
}

run_cell() {
  local mode="$1"
  local n="$2"
  local rep="$3"
  if [[ "$mode" == "baseline" ]]; then n=0; fi
  local closed_mode
  closed_mode="$(closed_mode_for_mode "$mode")"
  local competing_cpus
  competing_cpus="$(competing_cpu_list_for_n "$n")"
  local total_threads=$((MEASURED_WORKERS + n))
  local duration_ms="$STEADY_DURATION_MS"
  local skip_pre_ms="$STEADY_SKIP_PRE_MS"
  local sample_ms="$STEADY_SAMPLE_MS"
  local with_burst=0
  local perf_skip_ms="$STEADY_SKIP_PRE_MS"
  local perf_sample_ms="$STEADY_SAMPLE_MS"
  if [[ "$mode" == "periodic_insert" ]]; then
    duration_ms="$PERIODIC_DURATION_MS"
    skip_pre_ms=0
    sample_ms="$PERIODIC_DURATION_MS"
    with_burst=1
    perf_skip_ms="$INSERT_WINDOW_START_MS"
    perf_sample_ms="$INSERT_WINDOW_DURATION_MS"
  fi

  local cell_dir="$OUT_ROOT/cells/${mode}_n${n}_rep${rep}"
  local cfg="$CFG_ROOT/${mode}_n${n}_rep${rep}.yaml"
  rm -rf "$cell_dir/raw_latency"
  mkdir -p "$cell_dir"
  write_config "$cfg" "$cell_dir" "$total_threads" "$duration_ms" "$with_burst"

  echo "[hnsw-rootcause] mode=$mode closed_mode=$closed_mode n=$n rep=$rep cpus=${competing_cpus:-none} -> $cell_dir"
  {
    echo "mode=$mode"
    echo "closed_loop_mode=$closed_mode"
    echo "n_competing=$n"
    echo "rep=$rep"
    echo "batch_size=$BATCH_SIZE"
    echo "duration_ms=$duration_ms"
    echo "skip_pre_ms=$skip_pre_ms"
    echo "sample_ms=$sample_ms"
    echo "segment_guard_ms=$SEGMENT_GUARD_MS"
    echo "insert_window_start_ms=$INSERT_WINDOW_START_MS"
    echo "insert_window_duration_ms=$INSERT_WINDOW_DURATION_MS"
    echo "dataset_name=$DATASET_NAME"
    echo "begin_num=$BEGIN_NUM"
    echo "max_elements=$MAX_ELEMENTS"
    echo "measured_workers=$MEASURED_WORKERS"
    echo "measured_cpus=$MEASURED_CPU_LIST"
    echo "competing_cpus=${competing_cpus:-}"
    echo "plain_hnsw=1"
    echo "fake_write=0"
    echo "m3_batch_diag=$M3_BATCH_DIAG"
    echo "phase_overlap_diag=$PHASE_OVERLAP_DIAG"
    echo "search_work_counters=1"
    echo "insert_event_rate=$INSERT_EVENT_RATE"
    echo "search_event_rate=$SEARCH_EVENT_RATE"
  } > "$cell_dir/cell.meta"
  record_platform_state "$cell_dir/platform_state.txt"

  local phase_trace=""
  if [[ "$ENABLE_INSERT_PHASE_TRACE" == "1" && "$mode" == "$TRACE_MODE" && "$n" == "$TRACE_N" && "$rep" == "$TRACE_REP" ]]; then
    phase_trace="$cell_dir/insert_phase_trace.csv"
    echo "insert_phase_trace=$phase_trace" >> "$cell_dir/cell.meta"
    echo "insert_phase_trace_from_id=$((BEGIN_NUM + TRACE_FROM_DELTA))" >> "$cell_dir/cell.meta"
  fi

  (
    env \
      M3_CLOSED_LOOP=1 \
      M3_CLOSED_LOOP_DURATION_MS="$duration_ms" \
      M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
      M3_CLOSED_LOOP_COMPETING_WORKERS="$n" \
      M3_CLOSED_LOOP_INSERT_WORKERS="$n" \
      M3_CLOSED_LOOP_COMPETING_MODE="$closed_mode" \
      ANNCHOR_M3_BATCH_DIAG="$M3_BATCH_DIAG" \
      ANNCHOR_PHASE_OVERLAP_DIAG="$PHASE_OVERLAP_DIAG" \
      ANNCHOR_SEARCH_WORK_COUNTERS=1 \
      HNSW_MEASURED_SEARCH_ARENA=1 \
      HNSW_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
      HNSW_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      HNSW_INSERT_CPU_LIST="$competing_cpus" \
      HNSW_INSERT_VECTOR_READ_TRACE_PATH="${HNSW_INSERT_VECTOR_READ_TRACE_PATH:-}" \
      HNSW_INSERT_VECTOR_READ_TRACE_TOPK="${HNSW_INSERT_VECTOR_READ_TRACE_TOPK:-}" \
      HNSW_INSERT_VECTOR_HOT_IDS_PATH="${HNSW_INSERT_VECTOR_HOT_IDS_PATH:-}" \
      ANNCHOR_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      ANNCHOR_SEARCH_CPU_LIST="$competing_cpus" \
      ANNCHOR_INSERT_CPU_LIST="$competing_cpus" \
      ANNCHOR_INSERT_PHASE_TRACE="$phase_trace" \
      ANNCHOR_INSERT_PHASE_TRACE_FROM_ID="$((BEGIN_NUM + TRACE_FROM_DELTA))" \
      OCC_RAW_LATENCY_DIR="$cell_dir/raw_latency" \
      LD_LIBRARY_PATH="$BENCH_LD_LIBRARY_PATH" \
      ${NUMA_LAUNCH_PREFIX:-} "$BENCH_BIN" -config "$cfg"
  ) > "$cell_dir/run.log" 2>&1 &
  local bench_pid=$!

  if ! wait_for_closed_loop "$cell_dir/run.log"; then
    kill "$bench_pid" 2>/dev/null || true
    echo "failed_before_closed_loop=1" >> "$cell_dir/cell.meta"
    return 1
  fi
  echo "$bench_pid" > "$cell_dir/bench.pid"
  sleep "$(awk -v ms="$perf_skip_ms" 'BEGIN{printf "%.6f", ms/1000.0}')"
  date +%s.%N > "$cell_dir/perf.start_unix.txt"
  run_perf_stat "$cell_dir" "$competing_cpus" "$(awk -v ms="$perf_sample_ms" 'BEGIN{printf "%.6f", ms/1000.0}')"
  date +%s.%N > "$cell_dir/perf.end_unix.txt"
  wait "$bench_pid" 2>/dev/null || true
  echo "completed=1" >> "$cell_dir/cell.meta"

  if [[ -n "$phase_trace" && -s "$phase_trace" ]]; then
    python3 scripts/bin_m3_insert_phase_trace.py "$phase_trace" \
      --search-wide "$cell_dir/raw_latency/search_wide.csv" \
      --bin-ms 5 \
      --out "$cell_dir/phase_trace_5ms_bins.csv" \
      > "$cell_dir/phase_trace_bin.log" 2>&1 || true
  fi
}

{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "purpose=plain_hnsw_insert_search_rootcause"
  echo "dataset_name=$DATASET_NAME"
  echo "data_path=$DATA_PATH"
  echo "query_path=$QUERY_PATH"
  echo "overall_query_path=${OVERALL_QUERY_PATH:-$QUERY_PATH}"
  echo "overall_gt_path=${OVERALL_GT_PATH:-}"
  echo "begin_num=$BEGIN_NUM"
  echo "max_elements=$MAX_ELEMENTS"
  echo "m=$M_HNSW ef_search=$EF_SEARCH ef_construction=$EF_CONSTRUCTION batch_size=$BATCH_SIZE"
  echo "insert_event_rate=$INSERT_EVENT_RATE"
  echo "search_event_rate=$SEARCH_EVENT_RATE"
  echo "modes=$MODES"
  echo "n_values=$N_VALUES"
  echo "reps=$REPS"
  echo "steady_duration_ms=$STEADY_DURATION_MS steady_sample=${STEADY_SKIP_PRE_MS}+${STEADY_SAMPLE_MS}"
  echo "periodic_duration_ms=$PERIODIC_DURATION_MS insert_window=${INSERT_WINDOW_START_MS}+${INSERT_WINDOW_DURATION_MS}"
  echo "fake_write=0"
  echo "plain_hnsw=1"
  echo "m3_batch_diag=$M3_BATCH_DIAG"
  echo "phase_overlap_diag=$PHASE_OVERLAP_DIAG"
  echo "search_work_counters=1"
  record_platform_state "$OUT_ROOT/platform_state.txt"
  uname -a
} > "$OUT_ROOT/MANIFEST.txt"

for mode in $MODES; do
  if [[ "$mode" == "baseline" ]]; then
    for rep in $REPS; do
      run_cell "$mode" 0 "$rep" || true
    done
    continue
  fi
  if [[ "$mode" == "periodic_insert" ]]; then
    for rep in $REPS; do
      run_cell "$mode" 64 "$rep" || true
    done
    continue
  fi
  for n in $N_VALUES; do
    for rep in $REPS; do
      run_cell "$mode" "$n" "$rep" || true
    done
  done
done

python3 scripts/analyze_hnsw_insert_search_rootcause_20260513.py --root "$OUT_ROOT" --out-dir "$OUT_ROOT" \
  > "$OUT_ROOT/analyze.log" 2>&1 || true

date --iso-8601=seconds > "$OUT_ROOT/end_time.txt"
echo "[hnsw-rootcause] all done -> $OUT_ROOT"
