#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/m3_search_cpu_pinning_controls_20260508}"
CFG_ROOT="${CFG_ROOT:-./config/m3_search_cpu_pinning_controls_20260508}"
REPS="${REPS:-1}"
DURATION_MS="${DURATION_MS:-12000}"
DATASET="${DATASET:-sift200k}"
ALGORITHM="${ALGORITHM:-annchor-m3}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
BACKGROUND_SEARCH_WORKERS="${BACKGROUND_SEARCH_WORKERS:-20}"
INSERT_WORKERS="${INSERT_WORKERS:-20}"
BATCH_SIZE="${BATCH_SIZE:-20}"
CASES="${CASES:-searchonly noop cpu annread real}"

# Keep measured search isolated from background search and insert workers.
# On the current 2-socket / 48-core / SMT2 machine:
#   0-7      socket0 physical cores, measured search
#   8-23,56-59 socket0 background search, no measured-core SMT sharing
#   24-43    socket1 physical cores, insert
CPU_PINNING="${CPU_PINNING:-1}"
if [[ "$CPU_PINNING" == "1" ]]; then
  MEASURED_CPUS="${MEASURED_CPUS:-0-7}"
  BACKGROUND_CPUS="${BACKGROUND_CPUS:-8-23,56-59}"
  INSERT_CPUS="${INSERT_CPUS:-24-43}"
else
  MEASURED_CPUS=""
  BACKGROUND_CPUS=""
  INSERT_CPUS=""
fi

PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,cache-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses,l2_rqsts.all_rfo,l2_rqsts.rfo_miss,offcore_requests.l3_miss_demand_data_rd,offcore_requests_outstanding.l3_miss_demand_data_rd,context-switches,cpu-migrations}"
PERF_INTERVAL_MS="${PERF_INTERVAL_MS:-100}"
RUN_PERF="${RUN_PERF:-1}"
BUILD="${BUILD:-0}"

mkdir -p "$OUT_ROOT" "$CFG_ROOT"

if [[ "$BUILD" == "1" ]]; then
  cmake --build ./build --target index -j "${BUILD_JOBS:-8}"
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench ./main.go
fi

if [[ ! -x ./bin/bench ]]; then
  echo "missing ./bin/bench; rerun with BUILD=1 or build it first" >&2
  exit 1
fi

case_dummy_mode() {
  case "$1" in
    searchonly) printf 'noop\n' ;;
    noop) printf 'noop\n' ;;
    cpu) printf 'cpu\n' ;;
    annread) printf 'ann_search_query\n' ;;
    real) printf '\n' ;;
    *)
      echo "unknown case: $1" >&2
      return 2
      ;;
  esac
}

case_dummy_loops() {
  case "$1" in
    cpu) printf '%s\n' "${INSERT_CPU_LOOPS:-20000}" ;;
    annread) printf '%s\n' "${INSERT_ANNREAD_LOOPS:-1}" ;;
    *) printf '\n' ;;
  esac
}

case_insert_workers() {
  case "$1" in
    searchonly) printf '0\n' ;;
    *) printf '%s\n' "$INSERT_WORKERS" ;;
  esac
}

run_case() {
  local case_name="$1"
  local dummy_mode dummy_loops case_out case_cfg perf_out case_insert_workers
  dummy_mode="$(case_dummy_mode "$case_name")"
  dummy_loops="$(case_dummy_loops "$case_name")"
  case_insert_workers="$(case_insert_workers "$case_name")"
  case_out="$OUT_ROOT/$case_name"
  case_cfg="$CFG_ROOT/$case_name"
  perf_out="$case_out/measured_cpu_perf_interval.csv"
  mkdir -p "$case_out" "$case_cfg"

  cat > "$case_out/case_meta.csv" <<EOF
case,$case_name
dummy_mode,$dummy_mode
dummy_loops,$dummy_loops
measured_cpus,$MEASURED_CPUS
background_cpus,$BACKGROUND_CPUS
insert_cpus,$INSERT_CPUS
duration_ms,$DURATION_MS
dataset,$DATASET
algorithm,$ALGORITHM
measured_workers,$MEASURED_WORKERS
background_search_workers,$BACKGROUND_SEARCH_WORKERS
insert_workers,$INSERT_WORKERS
case_insert_workers,$case_insert_workers
batch_size,$BATCH_SIZE
perf_events,$PERF_EVENTS
perf_interval_ms,$PERF_INTERVAL_MS
cpu_pinning,$CPU_PINNING
EOF

  echo "[m3-search-cpu-pin] case=$case_name dummy=${dummy_mode:-real} out=$case_out"
  local cmd=(
    env
    OUT_ROOT="$case_out"
    CFG_ROOT="$case_cfg"
    MODE=clean
    REPS="$REPS"
    DURATION_MS="$DURATION_MS"
    MEASURED_WORKERS="$MEASURED_WORKERS"
    BACKGROUND_SEARCH_WORKERS="$BACKGROUND_SEARCH_WORKERS"
    INSERT_WORKERS="$case_insert_workers"
    BATCH_SIZE="$BATCH_SIZE"
    CASE_FILTER="$DATASET"
    ALGORITHM_FILTER="$ALGORITHM"
    DATASET_SET=core
    SEARCH_WORK_COUNTERS=1
    RAW_LATENCY_SKIP_SEARCH_WIDE=0
    ANNCHOR_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPUS"
    ANNCHOR_SEARCH_CPU_LIST="$BACKGROUND_CPUS"
    ANNCHOR_INSERT_CPU_LIST="$INSERT_CPUS"
    INSERT_DUMMY_MODE="$dummy_mode"
    INSERT_DUMMY_LOOPS="$dummy_loops"
    bash ./scripts/run_m3_prune_only_tail_matrix_20260508.sh
  )
  if [[ "$RUN_PERF" == "1" && -n "$MEASURED_CPUS" ]]; then
    date +%s%N > "$case_out/perf_start_unix_ns.txt"
    python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))' \
      > "$case_out/perf_start_raw_ns.txt" 2>/dev/null || rm -f "$case_out/perf_start_raw_ns.txt"
    perf stat -C "$MEASURED_CPUS" -I "$PERF_INTERVAL_MS" -x, \
      -e "$PERF_EVENTS" -o "$perf_out" -- "${cmd[@]}" \
      > "$case_out/runner.stdout.log" 2> "$case_out/runner.stderr.log"
  else
    date +%s%N > "$case_out/perf_start_unix_ns.txt"
    python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))' \
      > "$case_out/perf_start_raw_ns.txt" 2>/dev/null || rm -f "$case_out/perf_start_raw_ns.txt"
    "${cmd[@]}" > "$case_out/runner.stdout.log" 2> "$case_out/runner.stderr.log"
  fi
}

for case_name in $CASES; do
  run_case "$case_name"
done

python3 ./scripts/analyze_m3_search_cpu_pinning_controls_20260508.py \
  --root "$OUT_ROOT" \
  --out "$OUT_ROOT/summary.csv" \
  --readme "$OUT_ROOT/README.md"

echo "[m3-search-cpu-pin] complete root=$OUT_ROOT"
