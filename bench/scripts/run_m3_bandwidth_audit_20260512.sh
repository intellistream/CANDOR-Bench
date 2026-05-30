#!/usr/bin/env bash
# M3 bandwidth audit (2026-05-12)
# Direct per-CPU-group and socket DRAM bandwidth measurement during reader+competing workload.
# Replaces the indirectly-estimated 568x per-thread bandwidth figure with measured cache-line traffic.
#
# Output layout:
#   $OUT_ROOT/
#     phase1_bandwidth/<mode>_n<N>/{run.log,benchmark_results.csv,
#                                   perf_reader.txt,perf_competing.txt,perf_uncore.txt,
#                                   uncore_freq.csv,cfg.yaml,timeline.txt}
#     phase2_profile/insert_n20/{perf.data,perf_report.txt,run.log,...}
#     phase3_hotcold/<mode>_rep<R>/{...same as phase1...}
#
# Designed for: Xeon Gold 6418H, 2 sockets, 24 physical cores per socket.
#   Reader  : CPU 0-7   (socket0 physical cores)
#   Compete : CPU 8-23  (socket0 physical) -> 24-47 (socket1 physical), no HT siblings of reader
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/m3_bandwidth_audit_20260512}"
CFG_ROOT="${CFG_ROOT:-./config/m3_bandwidth_audit_20260512}"

DATASET_NAME="${DATASET_NAME:-sift200k}"
DATA_PATH="${DATA_PATH:-../data/sift/sift_base_200k.bin}"
QUERY_PATH="${QUERY_PATH:-../data/sift/sift_query_stream.bin}"
MAX_ELEMENTS="${MAX_ELEMENTS:-200000}"
BEGIN_NUM="${BEGIN_NUM:-180000}"
M_HNSW="${M_HNSW:-16}"
EF_SEARCH="${EF_SEARCH:-50}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-100}"
BATCH_SIZE="${BATCH_SIZE:-20}"
QUERY_MODE="${QUERY_MODE:-peeking}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
DURATION_MS="${DURATION_MS:-35000}"

# Perf window: skip first SKIP_PRE_MS after closed-loop start, sample for SAMPLE_MS
SKIP_PRE_MS="${SKIP_PRE_MS:-5000}"
SAMPLE_MS="${SAMPLE_MS:-25000}"

# CPU pinning
MEASURED_CPU_LIST="${MEASURED_CPU_LIST:-0-7}"
# Competing CPU pool: physical cores only, both sockets, no HT siblings of reader.
COMPETING_CPU_POOL_START="${COMPETING_CPU_POOL_START:-8}"
COMPETING_CPU_POOL_END_S0="${COMPETING_CPU_POOL_END_S0:-23}"   # socket0 last physical
COMPETING_CPU_POOL_START_S1="${COMPETING_CPU_POOL_START_S1:-24}"
COMPETING_CPU_POOL_END_S1="${COMPETING_CPU_POOL_END_S1:-47}"  # socket1 last physical

PERF_EVENTS_CORE="cycles,instructions,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses"
PERF_EVENTS_UNCORE="uncore_imc/cas_count_read/,uncore_imc/cas_count_write/"

mkdir -p "$OUT_ROOT" "$CFG_ROOT"

if [[ ! -x "./bin/bench" ]]; then
  echo "missing ./bin/bench" >&2; exit 1
fi

# Determine competing CPU list for N competing workers, in physical-core order.
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
      # spill to HT siblings on socket0 56-71 (HT of 8-23)
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
    printf 'index:\n'
    printf '  index_type: hnsw\n'
    printf '  m: %s\n' "$M_HNSW"
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

# Sample uncore freq + CPU governor in background, with timestamps.
start_uncore_freq_sampler() {
  local out="$1"; local interval_s="${2:-0.5}"
  ( while true; do
      local ts
      ts=$(date +%s.%N)
      local f0 f1
      f0=$(cat /sys/devices/system/cpu/intel_uncore_frequency/package_00_die_00/current_freq_khz 2>/dev/null || echo NaN)
      f1=$(cat /sys/devices/system/cpu/intel_uncore_frequency/package_01_die_00/current_freq_khz 2>/dev/null || echo NaN)
      printf '%s,%s,%s\n' "$ts" "$f0" "$f1"
      sleep "$interval_s"
    done ) > "$out"
}

# Wait for "Closed-loop setup:" in log
wait_for_closed_loop() {
  local log="$1"; local max_wait_s="${2:-120}"
  local waited=0
  while [[ "$waited" -lt "$max_wait_s" ]]; do
    if grep -q "Closed-loop setup:" "$log" 2>/dev/null; then return 0; fi
    sleep 1; waited=$((waited+1))
  done
  return 1
}

run_one_cell() {
  local mode="$1"; local n_competing="$2"; local cell_dir="$3"; local extra_envs="${4:-}"
  local total_threads=$((MEASURED_WORKERS + n_competing))
  if [[ "$total_threads" -lt 1 ]]; then total_threads=1; fi
  local cfg="$CFG_ROOT/$(basename "$cell_dir").yaml"
  write_config "$cfg" "$cell_dir" "$total_threads"

  local competing_cpus
  competing_cpus="$(competing_cpu_list_for_n "$n_competing")"
  if [[ -z "$competing_cpus" ]]; then competing_cpus="$MEASURED_CPU_LIST"; fi  # unused but needs a value for envs

  rm -rf "$cell_dir/raw_latency"
  mkdir -p "$cell_dir"

  echo "[m3-bw] mode=$mode n=$n_competing competing_cpus=$competing_cpus out=$cell_dir"
  echo "mode=$mode n_competing=$n_competing measured_cpus=$MEASURED_CPU_LIST competing_cpus=$competing_cpus duration_ms=$DURATION_MS skip_pre_ms=$SKIP_PRE_MS sample_ms=$SAMPLE_MS" > "$cell_dir/cell.meta"

  # Start uncore freq sampler in background
  start_uncore_freq_sampler "$cell_dir/uncore_freq.csv" 0.5 &
  local sampler_pid=$!

  # Launch bench in background
  ( env \
      M3_CLOSED_LOOP=1 \
      M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
      M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
      M3_CLOSED_LOOP_COMPETING_WORKERS="$n_competing" \
      M3_CLOSED_LOOP_INSERT_WORKERS="$n_competing" \
      M3_CLOSED_LOOP_COMPETING_MODE="$mode" \
      ANNCHOR_M3_BATCH_DIAG=1 \
      ANNCHOR_SEARCH_WORK_COUNTERS=1 \
      HNSW_MEASURED_SEARCH_ARENA=1 \
      HNSW_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
      HNSW_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      HNSW_SEARCH_CPU_LIST="$competing_cpus" \
      HNSW_INSERT_CPU_LIST="$competing_cpus" \
      ANNCHOR_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      ANNCHOR_SEARCH_CPU_LIST="$competing_cpus" \
      ANNCHOR_INSERT_CPU_LIST="$competing_cpus" \
      OCC_RAW_LATENCY_DIR="$cell_dir/raw_latency" \
      RAW_LATENCY_SKIP_SEARCH_WIDE=0 \
      LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
      $extra_envs \
      ./bin/bench -config "$cfg" ) > "$cell_dir/run.log" 2>&1 &
  local bench_pid=$!

  # Wait for closed-loop to start
  if ! wait_for_closed_loop "$cell_dir/run.log" 180; then
    echo "[m3-bw] timeout waiting for closed-loop in $cell_dir" >&2
    kill $bench_pid 2>/dev/null || true
    kill $sampler_pid 2>/dev/null || true
    return 1
  fi
  date +%s.%N > "$cell_dir/perf.t0_unix.txt"
  python3 -c "import time;print(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))" > "$cell_dir/perf.t0_raw_ns.txt"

  # Skip pre-warm portion
  sleep "$(awk -v ms=$SKIP_PRE_MS 'BEGIN{print ms/1000}')"

  date +%s.%N > "$cell_dir/perf.start_unix.txt"
  python3 -c "import time;print(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))" > "$cell_dir/perf.start_raw_ns.txt"

  # Launch three perf instances in parallel for SAMPLE_MS
  local sample_s
  sample_s=$(awk -v ms=$SAMPLE_MS 'BEGIN{print ms/1000}')

  perf stat -C "$MEASURED_CPU_LIST" -x, -e "$PERF_EVENTS_CORE" -o "$cell_dir/perf_reader.txt" -- sleep "$sample_s" 2>/dev/null &
  local p_r=$!
  if [[ "$n_competing" -gt 0 ]]; then
    perf stat -C "$competing_cpus" -x, -e "$PERF_EVENTS_CORE" -o "$cell_dir/perf_competing.txt" -- sleep "$sample_s" 2>/dev/null &
    local p_c=$!
  else
    : > "$cell_dir/perf_competing.txt"
    local p_c=""
  fi
  perf stat -a -x, -e "$PERF_EVENTS_UNCORE" -o "$cell_dir/perf_uncore.txt" -- sleep "$sample_s" 2>/dev/null &
  local p_u=$!

  wait $p_r 2>/dev/null || true
  [[ -n "$p_c" ]] && wait $p_c 2>/dev/null || true
  wait $p_u 2>/dev/null || true

  date +%s.%N > "$cell_dir/perf.end_unix.txt"

  # Wait for bench to finish (closed-loop will end ~SAMPLE_MS+SKIP_PRE_MS later anyway)
  wait $bench_pid 2>/dev/null || true
  kill $sampler_pid 2>/dev/null || true
  wait $sampler_pid 2>/dev/null || true

  echo "[m3-bw] done $cell_dir"
}

############################################
# Phase 1: bandwidth N-sweep
############################################
phase1() {
  local p1_root="$OUT_ROOT/phase1_bandwidth"
  mkdir -p "$p1_root"

  # baseline (no competing)
  run_one_cell "idle" 0 "$p1_root/idle_n0"

  for n in 1 4 8 16 32 64; do
    run_one_cell "insert" "$n" "$p1_root/insert_n$n"
  done

  for n in 1 4 8 16 32 64; do
    run_one_cell "spin"   "$n" "$p1_root/spin_n$n"
  done

  for n in 1 4 8 16 32 64; do
    run_one_cell "search" "$n" "$p1_root/search_n$n"
  done
}

############################################
# Phase 2: CPU profile (lock-bound vs compute-bound)
############################################
phase2() {
  local p2_root="$OUT_ROOT/phase2_profile"
  mkdir -p "$p2_root"
  local cell="$p2_root/insert_n20"
  local mode="insert"; local n=20
  local total_threads=$((MEASURED_WORKERS + n))
  local cfg="$CFG_ROOT/p2_${mode}_n${n}.yaml"
  write_config "$cfg" "$cell" "$total_threads"
  local competing_cpus
  competing_cpus="$(competing_cpu_list_for_n "$n")"

  rm -rf "$cell/raw_latency"
  mkdir -p "$cell"
  echo "[m3-bw p2] insert n=$n competing_cpus=$competing_cpus"
  echo "mode=$mode n_competing=$n competing_cpus=$competing_cpus" > "$cell/cell.meta"

  start_uncore_freq_sampler "$cell/uncore_freq.csv" 0.5 &
  local sampler_pid=$!

  ( env \
      M3_CLOSED_LOOP=1 \
      M3_CLOSED_LOOP_DURATION_MS=45000 \
      M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
      M3_CLOSED_LOOP_COMPETING_WORKERS="$n" \
      M3_CLOSED_LOOP_INSERT_WORKERS="$n" \
      M3_CLOSED_LOOP_COMPETING_MODE="$mode" \
      ANNCHOR_M3_BATCH_DIAG=1 \
      ANNCHOR_SEARCH_WORK_COUNTERS=1 \
      HNSW_MEASURED_SEARCH_ARENA=1 \
      HNSW_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
      HNSW_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      HNSW_SEARCH_CPU_LIST="$competing_cpus" \
      HNSW_INSERT_CPU_LIST="$competing_cpus" \
      ANNCHOR_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      ANNCHOR_SEARCH_CPU_LIST="$competing_cpus" \
      ANNCHOR_INSERT_CPU_LIST="$competing_cpus" \
      OCC_RAW_LATENCY_DIR="$cell/raw_latency" \
      LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
      ./bin/bench -config "$cfg" ) > "$cell/run.log" 2>&1 &
  local bench_pid=$!
  if ! wait_for_closed_loop "$cell/run.log" 180; then
    echo "[m3-bw p2] timeout"; kill $bench_pid 2>/dev/null || true; kill $sampler_pid 2>/dev/null || true; return 1
  fi
  sleep 5
  date +%s.%N > "$cell/perf.start_unix.txt"
  # Profile call graph 25s
  perf record -F 99 -g -C "$competing_cpus" --call-graph=dwarf -o "$cell/perf.data" -- sleep 25 2>/dev/null
  date +%s.%N > "$cell/perf.end_unix.txt"
  perf report --stdio --no-children -i "$cell/perf.data" --sort comm,dso,symbol 2>/dev/null \
    | head -200 > "$cell/perf_report.txt" || true
  # Also do reader-side profile in same run? cleaner: separate cell. For now, just competing.
  wait $bench_pid 2>/dev/null || true
  kill $sampler_pid 2>/dev/null || true
  wait $sampler_pid 2>/dev/null || true
}

############################################
# Phase 3: hot vs cold writer (focused setting)
############################################
phase3() {
  local p3_root="$OUT_ROOT/phase3_hotcold"
  mkdir -p "$p3_root"
  local n=20
  local total_threads=$((MEASURED_WORKERS + n))
  local competing_cpus
  competing_cpus="$(competing_cpu_list_for_n "$n")"

  for variant in hot cold; do
    for rep in 1 2 3; do
      local cell="$p3_root/${variant}_rep${rep}"
      local cfg="$CFG_ROOT/p3_${variant}_rep${rep}.yaml"
      write_config "$cfg" "$cell" "$total_threads"
      rm -rf "$cell/raw_latency"
      mkdir -p "$cell"
      echo "[m3-bw p3] variant=$variant rep=$rep"
      echo "variant=$variant rep=$rep n_competing=$n competing_cpus=$competing_cpus" > "$cell/cell.meta"
      start_uncore_freq_sampler "$cell/uncore_freq.csv" 0.5 &
      local sampler_pid=$!
      ( env \
          M3_CLOSED_LOOP=1 \
          M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
          M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
          M3_CLOSED_LOOP_COMPETING_WORKERS="$n" \
          M3_CLOSED_LOOP_INSERT_WORKERS="$n" \
          M3_CLOSED_LOOP_COMPETING_MODE="search_plus_periodic_insert" \
          INSERT_DUMMY_MODE="graph_write_${variant}" \
          INSERT_DUMMY_LOOPS=4 \
          GRAPH_TOUCH_SAMPLE_QUERIES=512 \
          GRAPH_TOUCH_SEARCH_K=10 \
          GRAPH_TOUCH_MAX_NEIGHBORS=32 \
          GRAPH_TOUCH_LABEL_LIMIT=512 \
          GRAPH_TOUCH_LABELS_PER_TASK=512 \
          GRAPH_TOUCH_MAX_EDGES=4 \
          ANNCHOR_M3_BATCH_DIAG=1 \
          ANNCHOR_SEARCH_WORK_COUNTERS=1 \
          HNSW_MEASURED_SEARCH_ARENA=1 \
          HNSW_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
          HNSW_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
          HNSW_SEARCH_CPU_LIST="$competing_cpus" \
          HNSW_INSERT_CPU_LIST="$competing_cpus" \
          ANNCHOR_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
          ANNCHOR_SEARCH_CPU_LIST="$competing_cpus" \
          ANNCHOR_INSERT_CPU_LIST="$competing_cpus" \
          OCC_RAW_LATENCY_DIR="$cell/raw_latency" \
          LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
          ./bin/bench -config "$cfg" ) > "$cell/run.log" 2>&1 &
      local bench_pid=$!
      if ! wait_for_closed_loop "$cell/run.log" 180; then
        echo "[m3-bw p3] timeout $cell"; kill $bench_pid 2>/dev/null || true; kill $sampler_pid 2>/dev/null || true; continue
      fi
      sleep "$(awk -v ms=$SKIP_PRE_MS 'BEGIN{print ms/1000}')"
      date +%s.%N > "$cell/perf.start_unix.txt"
      local sample_s
      sample_s=$(awk -v ms=$SAMPLE_MS 'BEGIN{print ms/1000}')
      perf stat -C "$MEASURED_CPU_LIST" -x, -e "$PERF_EVENTS_CORE" -o "$cell/perf_reader.txt" -- sleep "$sample_s" 2>/dev/null &
      local pr=$!
      perf stat -C "$competing_cpus" -x, -e "$PERF_EVENTS_CORE" -o "$cell/perf_competing.txt" -- sleep "$sample_s" 2>/dev/null &
      local pc=$!
      perf stat -a -x, -e "$PERF_EVENTS_UNCORE" -o "$cell/perf_uncore.txt" -- sleep "$sample_s" 2>/dev/null &
      local pu=$!
      wait $pr $pc $pu 2>/dev/null || true
      date +%s.%N > "$cell/perf.end_unix.txt"
      wait $bench_pid 2>/dev/null || true
      kill $sampler_pid 2>/dev/null || true
      wait $sampler_pid 2>/dev/null || true
    done
  done
}

main() {
  PHASES="${PHASES:-phase1 phase2 phase3}"
  echo "[m3-bw] start phases: $PHASES OUT_ROOT=$OUT_ROOT"
  date > "$OUT_ROOT/start_time.txt"
  for ph in $PHASES; do
    case "$ph" in
      phase1) phase1 ;;
      phase2) phase2 ;;
      phase3) phase3 ;;
      *) echo "unknown phase $ph" >&2; exit 1 ;;
    esac
  done
  date > "$OUT_ROOT/end_time.txt"
  echo "[m3-bw] all done."
}

# Only run main when executed directly, not when sourced for function reuse.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main
fi
