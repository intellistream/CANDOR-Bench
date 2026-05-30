#!/usr/bin/env bash
# M3 writer lock-wait audit (2026-05-12)
# Question: how much of a writer's wall time is spent in lock wait?
#
# Two complementary measurements per cell:
# (a) bench internal counters in insert_wide.csv:
#     - graph_unique_lock_wait_ms : wall time blocked on unique_lock(link_list_locks_[neighbor])
#                                   when updating an existing hub neighbor.
#     - graph_link_critical_ms    : wall time inside the unique-lock critical section
#                                   (prune, rewrite, append).
#     - graph_insert_path_ms      : CPU time in HNSW upper+base search (multi-thread).
#     - graph_existing_neighbor_update_loop_ms : wall time for entire existing-neighbor loop
#                                                (lock_wait + critical_ms).
#     - op_ms                     : wall time for the batch insert call.
# (b) perf stat -C competing_cpus during the same window:
#     - cycles, instructions -> aggregate on-CPU time for writer pool.
#     - cycles / (N_writers * cpu_freq * window_s) -> on-CPU utilization fraction.
#     - 1 - utilization = approximate off-CPU (blocked) fraction.
# Plus, single cell N=20 with `perf record --pid <bench_pid>` to get a writer-thread
# on-CPU callstack profile (excluding avahi/idle squatters on idle CPUs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/m3_writer_lockwait_20260512}"
CFG_ROOT="${CFG_ROOT:-./config/m3_writer_lockwait_20260512}"

DATASET_NAME="${DATASET_NAME:-sift1m}"
DATA_PATH="${DATA_PATH:-../data/sift/sift_base.bin}"
QUERY_PATH="${QUERY_PATH:-../data/sift/sift_query_stream.bin}"
MAX_ELEMENTS="${MAX_ELEMENTS:-1000000}"
BEGIN_NUM="${BEGIN_NUM:-200000}"
M_HNSW="${M_HNSW:-16}"
EF_SEARCH="${EF_SEARCH:-50}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-100}"
BATCH_SIZE="${BATCH_SIZE:-20}"
QUERY_MODE="${QUERY_MODE:-peeking}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
DURATION_MS="${DURATION_MS:-35000}"
SKIP_PRE_MS="${SKIP_PRE_MS:-5000}"
SAMPLE_MS="${SAMPLE_MS:-25000}"

MEASURED_CPU_LIST="${MEASURED_CPU_LIST:-0-7}"
COMPETING_CPU_POOL_START="${COMPETING_CPU_POOL_START:-8}"
COMPETING_CPU_POOL_END_S0="${COMPETING_CPU_POOL_END_S0:-23}"
COMPETING_CPU_POOL_START_S1="${COMPETING_CPU_POOL_START_S1:-24}"
COMPETING_CPU_POOL_END_S1="${COMPETING_CPU_POOL_END_S1:-47}"

PERF_EVENTS_CORE="cycles,instructions,LLC-loads,LLC-load-misses"

mkdir -p "$OUT_ROOT" "$CFG_ROOT"

if [[ ! -x "./bin/bench" ]]; then
  echo "missing ./bin/bench" >&2; exit 1
fi

competing_cpu_list_for_n() {
  local n="$1"
  if [[ "$n" -le 0 ]]; then echo ""; return; fi
  local s0=$((COMPETING_CPU_POOL_END_S0 - COMPETING_CPU_POOL_START + 1))
  if [[ "$n" -le "$s0" ]]; then
    echo "${COMPETING_CPU_POOL_START}-$((COMPETING_CPU_POOL_START + n - 1))"
  else
    local rem=$((n - s0))
    local s1_end=$((COMPETING_CPU_POOL_START_S1 + rem - 1))
    if [[ "$s1_end" -gt "$COMPETING_CPU_POOL_END_S1" ]]; then
      local rem2=$((s1_end - COMPETING_CPU_POOL_END_S1))
      local ht_end=$((56 + rem2 - 1))
      echo "${COMPETING_CPU_POOL_START}-${COMPETING_CPU_POOL_END_S0},${COMPETING_CPU_POOL_START_S1}-${COMPETING_CPU_POOL_END_S1},56-${ht_end}"
    else
      echo "${COMPETING_CPU_POOL_START}-${COMPETING_CPU_POOL_END_S0},${COMPETING_CPU_POOL_START_S1}-${s1_end}"
    fi
  fi
}

write_config() {
  local cfg="$1"; local out_dir="$2"; local total_threads="$3"
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
    printf 'index:\n  index_type: hnsw\n  m: %s\n  ef_construction: %s\n' "$M_HNSW" "$EF_CONSTRUCTION"
    printf 'search:\n  recall_at: 10\n  ef_search: %s\n  use_node_lock: true\n  enable_mvcc: false\n  enable_undo_recovery: false\n' "$EF_SEARCH"
    printf 'workload:\n'
    printf '  batch_size: %s\n' "$BATCH_SIZE"
    printf '  num_threads: %s\n' "$total_threads"
    printf '  queue_size: 0\n  insert_event_rate: 0\n  search_event_rate: 0\n'
    printf '  schedule_horizon_ms: %s\n' "$DURATION_MS"
    printf '  with_external_rw_lock: false\n  use_node_lock: true\n'
    printf '  query_mode: %s\n  per_query_latency: true\n  precompute_schedule: false\n' "$QUERY_MODE"
    printf 'result:\n  output_dir: %s\nrepetitions: 1\nprofile:\n  memory_monitor_interval: 100\n  enable_memory_profile: false\n' "$out_dir"
  } > "$cfg"
}

wait_for_closed_loop() {
  local log="$1"; local max_wait_s="${2:-180}"
  local waited=0
  while [[ "$waited" -lt "$max_wait_s" ]]; do
    if grep -q "Closed-loop setup:" "$log" 2>/dev/null; then return 0; fi
    sleep 1; waited=$((waited+1))
  done
  return 1
}

run_cell() {
  local n="$1"; local cell_dir="$2"; local do_perf_record="${3:-0}"
  local total_threads=$((MEASURED_WORKERS + n))
  local cfg="$CFG_ROOT/$(basename "$cell_dir").yaml"
  write_config "$cfg" "$cell_dir" "$total_threads"
  local competing_cpus
  competing_cpus="$(competing_cpu_list_for_n "$n")"
  rm -rf "$cell_dir/raw_latency"
  mkdir -p "$cell_dir"
  echo "[lockwait] N=$n competing_cpus=$competing_cpus perf_record=$do_perf_record out=$cell_dir"
  echo "mode=insert n_writers=$n competing_cpus=$competing_cpus measured_cpus=$MEASURED_CPU_LIST duration_ms=$DURATION_MS skip_pre_ms=$SKIP_PRE_MS sample_ms=$SAMPLE_MS perf_record=$do_perf_record" > "$cell_dir/cell.meta"

  ( env \
      M3_CLOSED_LOOP=1 \
      M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
      M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
      M3_CLOSED_LOOP_COMPETING_WORKERS="$n" \
      M3_CLOSED_LOOP_INSERT_WORKERS="$n" \
      M3_CLOSED_LOOP_COMPETING_MODE=insert \
      ANNCHOR_M3_BATCH_DIAG=1 \
      ANNCHOR_SEARCH_WORK_COUNTERS=1 \
      HNSW_MEASURED_SEARCH_ARENA=1 \
      HNSW_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
      HNSW_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      HNSW_INSERT_CPU_LIST="$competing_cpus" \
      ANNCHOR_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      ANNCHOR_INSERT_CPU_LIST="$competing_cpus" \
      OCC_RAW_LATENCY_DIR="$cell_dir/raw_latency" \
      LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
      ./bin/bench -config "$cfg" ) > "$cell_dir/run.log" 2>&1 &
  local bench_pid=$!

  if ! wait_for_closed_loop "$cell_dir/run.log" 180; then
    echo "[lockwait] timeout $cell_dir"; kill $bench_pid 2>/dev/null || true; return 1
  fi
  echo "$bench_pid" > "$cell_dir/bench.pid"
  sleep "$(awk -v ms=$SKIP_PRE_MS 'BEGIN{print ms/1000}')"
  date +%s.%N > "$cell_dir/perf.start_unix.txt"
  local sample_s
  sample_s=$(awk -v ms=$SAMPLE_MS 'BEGIN{print ms/1000}')

  # Per-CPU perf stat: cycles + LLC events on competing CPUs
  perf stat -C "$competing_cpus" -x, -e "$PERF_EVENTS_CORE" -o "$cell_dir/perf_competing.txt" -- sleep "$sample_s" 2>/dev/null &
  local p_c=$!
  perf stat -C "$MEASURED_CPU_LIST" -x, -e "$PERF_EVENTS_CORE" -o "$cell_dir/perf_reader.txt" -- sleep "$sample_s" 2>/dev/null &
  local p_r=$!

  local p_rec=""
  if [[ "$do_perf_record" == "1" ]]; then
    # Double filter: pid (bench only, no avahi) + CPU (writer cores only).
    # This isolates the writer-thread on-CPU profile from reader threads.
    perf record -F 99 -g --call-graph=dwarf --pid "$bench_pid" -C "$competing_cpus" \
      -o "$cell_dir/perf.data" -- sleep "$sample_s" 2>"$cell_dir/perf_record.stderr" &
    p_rec=$!
  fi

  wait $p_c $p_r 2>/dev/null || true
  [[ -n "$p_rec" ]] && wait $p_rec 2>/dev/null || true
  date +%s.%N > "$cell_dir/perf.end_unix.txt"

  wait $bench_pid 2>/dev/null || true

  # Symbolicate perf.data immediately if recorded; tolerate pipe failures.
  if [[ "$do_perf_record" == "1" && -f "$cell_dir/perf.data" ]]; then
    set +eo pipefail
    perf report --stdio --no-children --sort comm,dso,symbol -i "$cell_dir/perf.data" 2>/dev/null \
      > "$cell_dir/perf_report_full.txt"
    perf report --stdio --no-children --sort dso,symbol -c bench -i "$cell_dir/perf.data" 2>/dev/null \
      > "$cell_dir/perf_report_bench.txt"
    set -eo pipefail
  fi
  echo "[lockwait] done $cell_dir"
}

P_root="$OUT_ROOT/cells"
mkdir -p "$P_root"

CELLS_N="${CELLS_N:-1 4 8 16 32 64}"
PERF_RECORD_AT_N="${PERF_RECORD_AT_N:-16}"

for n in $CELLS_N; do
  do_pr=0
  [[ "$n" == "$PERF_RECORD_AT_N" ]] && do_pr=1
  run_cell "$n" "$P_root/insert_n$n" "$do_pr"
done

date > "$OUT_ROOT/end_time.txt"
echo "[lockwait] all done"
