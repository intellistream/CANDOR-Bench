#!/usr/bin/env bash
# M3 read-write tail causality audit.
#
# Goal:
#   Prove or falsify that sustained writes add a write-specific reader-tail
#   channel beyond generic CPU/thread/memory contention.
#
# Paper-facing controls:
#   baseline          measured readers only
#   spin              CPU-only competing workers, no graph access
#   bg_search         read-only graph traversal on competing CPUs
#   insert_real       writer-shaped workers with real HNSW insert/repair
#
# The important paper-facing contrasts are insert_real - spin and
# insert_real - bg_search at the same N and CPU placement. Fake-write modes
# such as insert_ann_search are diagnostics only and must not be used as a
# SIGMOD table/claim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/m3_rw_tail_causality_20260512}"
CFG_ROOT="${CFG_ROOT:-./config/m3_rw_tail_causality_20260512}"

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
DURATION_MS="${DURATION_MS:-35000}"
SKIP_PRE_MS="${SKIP_PRE_MS:-5000}"
SAMPLE_MS="${SAMPLE_MS:-25000}"

MEASURED_CPU_LIST="${MEASURED_CPU_LIST:-0-7}"
COMPETING_CPU_POOL_START="${COMPETING_CPU_POOL_START:-8}"
COMPETING_CPU_POOL_END_S0="${COMPETING_CPU_POOL_END_S0:-23}"
COMPETING_CPU_POOL_START_S1="${COMPETING_CPU_POOL_START_S1:-24}"
COMPETING_CPU_POOL_END_S1="${COMPETING_CPU_POOL_END_S1:-47}"
COMPETING_HT_POOL_START="${COMPETING_HT_POOL_START:-56}"

MODES="${MODES:-baseline spin bg_search insert_real}"
N_VALUES="${N_VALUES:-4 16 64}"
REPS="${REPS:-1 2 3 4 5}"
ENABLE_PERF="${ENABLE_PERF:-1}"
ENABLE_UNCORE_PERF="${ENABLE_UNCORE_PERF:-1}"
LOCK_VARIANT="${LOCK_VARIANT:-shared}"
M3_BATCH_DIAG="${M3_BATCH_DIAG:-0}"
SEARCH_WORK_COUNTERS="${SEARCH_WORK_COUNTERS:-1}"

PERF_EVENTS_CORE="${PERF_EVENTS_CORE:-cycles,instructions,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses}"
PERF_EVENTS_UNCORE="${PERF_EVENTS_UNCORE:-uncore_imc/cas_count_read/,uncore_imc/cas_count_write/}"

mkdir -p "$OUT_ROOT" "$CFG_ROOT"

die() {
  echo "[rw-causality] ERROR: $*" >&2
  exit 1
}

if [[ ! -x "./bin/bench" ]]; then
  die "missing ./bin/bench; build the benchmark first"
fi

for mode in $MODES; do
  case "$mode" in
    insert_ann_search|insert_cpu|insert_noop)
      if [[ "${ALLOW_FAKE_WRITE_DIAGNOSTIC:-0}" != "1" ]]; then
        die "$mode is a fake/debug insert mode and is not paper-facing; set ALLOW_FAKE_WRITE_DIAGNOSTIC=1 only for internal diagnostics"
      fi
      ;;
  esac
done

if [[ "${ALLOW_SMALL_DATASET:-0}" != "1" ]]; then
  if [[ "$MAX_ELEMENTS" -lt 1000000 ]]; then
    die "MAX_ELEMENTS=$MAX_ELEMENTS is below full-scale gate; set ALLOW_SMALL_DATASET=1 only for smoke tests"
  fi
  if [[ "$BEGIN_NUM" -lt 1000000 ]]; then
    die "BEGIN_NUM=$BEGIN_NUM is below full-scale gate; set ALLOW_SMALL_DATASET=1 only for smoke tests"
  fi
  headroom=$((MAX_ELEMENTS - BEGIN_NUM))
  if [[ "$headroom" -lt 800000 ]]; then
    die "insert headroom=$headroom is too small for M3 evidence"
  fi
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

write_manifest() {
  local out="$OUT_ROOT/MANIFEST.txt"
  {
    echo "created_at=$(date --iso-8601=seconds)"
    echo "dataset_name=$DATASET_NAME"
    echo "data_path=$DATA_PATH"
    echo "query_path=$QUERY_PATH"
    echo "begin_num=$BEGIN_NUM"
    echo "max_elements=$MAX_ELEMENTS"
    echo "insert_headroom=$((MAX_ELEMENTS - BEGIN_NUM))"
    echo "m_hnsw=$M_HNSW ef_search=$EF_SEARCH ef_construction=$EF_CONSTRUCTION batch_size=$BATCH_SIZE"
    echo "duration_ms=$DURATION_MS skip_pre_ms=$SKIP_PRE_MS sample_ms=$SAMPLE_MS"
    echo "measured_workers=$MEASURED_WORKERS measured_cpu_list=$MEASURED_CPU_LIST"
    echo "modes=$MODES"
    echo "n_values=$N_VALUES"
    echo "reps=$REPS"
    echo "lock_variant=$LOCK_VARIANT"
    echo "m3_batch_diag=$M3_BATCH_DIAG"
    echo "search_work_counters=$SEARCH_WORK_COUNTERS"
    echo "allow_fake_write_diagnostic=${ALLOW_FAKE_WRITE_DIAGNOSTIC:-0}"
    echo "uname=$(uname -a)"
    echo
    echo "[go binary]"
    case "$LOCK_VARIANT" in
      shared)
        LD_LIBRARY_PATH="$ROOT/build/lib:${LD_LIBRARY_PATH:-}" ./bin/bench -h 2>&1 | head -5 || true
        ;;
      unique)
        LD_LIBRARY_PATH="$ROOT/build/lib_unique_runtime:${ROOT}/build/lib:${LD_LIBRARY_PATH:-}" ./bin/bench -h 2>&1 | head -5 || true
        ;;
    esac
    echo
    echo "[cpu]"
    lscpu 2>/dev/null || true
    echo
    echo "[cpufreq_governors]"
    local gov_count=0
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
      [[ -r "$f" ]] || continue
      printf '%s=%s\n' "$f" "$(cat "$f")"
      gov_count=$((gov_count + 1))
      [[ "$gov_count" -ge 16 ]] && break
    done
    echo
    echo "[uncore_frequency]"
    for f in /sys/devices/system/cpu/intel_uncore_frequency/package_*/*freq_khz; do
      [[ -r "$f" ]] && printf '%s=%s\n' "$f" "$(cat "$f")"
    done
    echo
    echo "[cpuidle]"
    for f in /sys/devices/system/cpu/cpu0/cpuidle/state*/name /sys/devices/system/cpu/cpu0/cpuidle/state*/latency; do
      [[ -r "$f" ]] && printf '%s=%s\n' "$f" "$(cat "$f")"
    done
  } > "$out"
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

mode_to_closed_loop() {
  case "$1" in
    baseline) echo "idle" ;;
    spin) echo "spin" ;;
    bg_search) echo "search" ;;
    insert_real|insert_ann_search|insert_cpu|insert_noop) echo "insert" ;;
    *) die "unknown mode $1" ;;
  esac
}

mode_to_dummy() {
  case "$1" in
    insert_ann_search) echo "ann_search_query" ;;
    insert_cpu) echo "cpu" ;;
    insert_noop) echo "noop" ;;
    *) echo "" ;;
  esac
}

configure_lock_variant() {
  case "$LOCK_VARIANT" in
    shared)
      export LD_LIBRARY_PATH="$ROOT/build/lib:${LD_LIBRARY_PATH:-}"
      ;;
    unique)
      mkdir -p "$ROOT/build/lib_unique_runtime"
      cp -f "$ROOT/build/lib/libindex_unique_baseline.so" "$ROOT/build/lib_unique_runtime/libindex.so"
      export LD_LIBRARY_PATH="$ROOT/build/lib_unique_runtime:${LD_LIBRARY_PATH:-}"
      ;;
    *)
      die "LOCK_VARIANT must be shared or unique"
      ;;
  esac
}

run_perf_stat() {
  local cell_dir="$1"
  local competing_cpus="$2"
  local sample_s="$3"
  if [[ "$ENABLE_PERF" != "1" ]]; then
    return 0
  fi
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
  if [[ "$mode" == "baseline" ]]; then
    n=0
  fi
  local closed_mode
  closed_mode="$(mode_to_closed_loop "$mode")"
  local dummy_mode
  dummy_mode="$(mode_to_dummy "$mode")"
  local competing_cpus
  competing_cpus="$(competing_cpu_list_for_n "$n")"
  local total_threads=$((MEASURED_WORKERS + n))
  if [[ "$total_threads" -lt "$MEASURED_WORKERS" ]]; then
    total_threads="$MEASURED_WORKERS"
  fi

  local cell_dir="$OUT_ROOT/${mode}_n${n}_rep${rep}"
  local cfg="$CFG_ROOT/${mode}_n${n}_rep${rep}.yaml"
  rm -rf "$cell_dir/raw_latency"
  mkdir -p "$cell_dir"
  write_config "$cfg" "$cell_dir" "$total_threads"

  echo "[rw-causality] mode=$mode closed_mode=$closed_mode n=$n rep=$rep cpus=${competing_cpus:-none} -> $cell_dir"
  {
    echo "mode=$mode"
    echo "closed_loop_mode=$closed_mode"
    echo "dummy_mode=${dummy_mode:-real}"
    echo "n_competing=$n"
    echo "rep=$rep"
    echo "batch_size=$BATCH_SIZE"
    echo "duration_ms=$DURATION_MS"
    echo "skip_pre_ms=$SKIP_PRE_MS"
    echo "sample_ms=$SAMPLE_MS"
    echo "dataset_name=$DATASET_NAME"
    echo "begin_num=$BEGIN_NUM"
    echo "max_elements=$MAX_ELEMENTS"
    echo "insert_headroom=$((MAX_ELEMENTS - BEGIN_NUM))"
    echo "measured_workers=$MEASURED_WORKERS"
    echo "measured_cpus=$MEASURED_CPU_LIST"
    echo "competing_cpus=${competing_cpus:-}"
    echo "lock_variant=$LOCK_VARIANT"
  } > "$cell_dir/cell.meta"

  configure_lock_variant
  local sample_s
  sample_s="$(awk -v ms="$SAMPLE_MS" 'BEGIN{printf "%.6f", ms/1000.0}')"
  local skip_s
  skip_s="$(awk -v ms="$SKIP_PRE_MS" 'BEGIN{printf "%.6f", ms/1000.0}')"

  (
    env \
      M3_CLOSED_LOOP=1 \
      M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
      M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
      M3_CLOSED_LOOP_COMPETING_WORKERS="$n" \
      M3_CLOSED_LOOP_INSERT_WORKERS="$n" \
      M3_CLOSED_LOOP_COMPETING_MODE="$closed_mode" \
      INSERT_DUMMY_MODE="$dummy_mode" \
      INSERT_DUMMY_LOOPS="${INSERT_DUMMY_LOOPS:-1}" \
      ANNCHOR_M3_BATCH_DIAG="$M3_BATCH_DIAG" \
      ANNCHOR_SEARCH_WORK_COUNTERS="$SEARCH_WORK_COUNTERS" \
      HNSW_MEASURED_SEARCH_ARENA=1 \
      HNSW_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
      HNSW_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      ANNCHOR_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      ANNCHOR_SEARCH_CPU_LIST="$competing_cpus" \
      HNSW_INSERT_CPU_LIST="$competing_cpus" \
      ANNCHOR_INSERT_CPU_LIST="$competing_cpus" \
      OCC_RAW_LATENCY_DIR="$cell_dir/raw_latency" \
      ./bin/bench -config "$cfg"
  ) > "$cell_dir/run.log" 2>&1 &
  local bench_pid=$!

  if ! wait_for_closed_loop "$cell_dir/run.log"; then
    kill "$bench_pid" 2>/dev/null || true
    echo "failed_before_closed_loop=1" >> "$cell_dir/cell.meta"
    return 1
  fi
  echo "$bench_pid" > "$cell_dir/bench.pid"
  sleep "$skip_s"
  date +%s.%N > "$cell_dir/perf.start_unix.txt"
  run_perf_stat "$cell_dir" "$competing_cpus" "$sample_s"
  date +%s.%N > "$cell_dir/perf.end_unix.txt"
  wait "$bench_pid" 2>/dev/null || true
  echo "completed=1" >> "$cell_dir/cell.meta"
}

write_manifest

for mode in $MODES; do
  if [[ "$mode" == "baseline" ]]; then
    for rep in $REPS; do
      run_cell "$mode" 0 "$rep" || true
    done
    continue
  fi
  for n in $N_VALUES; do
    for rep in $REPS; do
      run_cell "$mode" "$n" "$rep" || true
    done
  done
done

date --iso-8601=seconds > "$OUT_ROOT/end_time.txt"
echo "[rw-causality] all done -> $OUT_ROOT"
