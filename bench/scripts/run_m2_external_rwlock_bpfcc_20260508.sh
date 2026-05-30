#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/m2_external_rwlock_bpfcc_20260508}"
CFG_ROOT="${CFG_ROOT:-./config/m2_node_lock_cause_matrix_20260507}"
DURATION_MS="${DURATION_MS:-12000}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
COMPETING_WORKERS="${COMPETING_WORKERS:-40}"
MIN_US="${MIN_US:-1000}"
STACKS="${STACKS:-0}"
SEARCH_WORK_COUNTERS="${SEARCH_WORK_COUNTERS:-0}"
RAW_LATENCY_SKIP_SEARCH_WIDE="${RAW_LATENCY_SKIP_SEARCH_WIDE:-1}"
TRACE_TIMEOUT="${TRACE_TIMEOUT:-20}"

if [[ ! -x ./bin/bench ]]; then
  echo "missing ./bin/bench" >&2
  exit 1
fi
if [[ ! -x ./scripts/trace_rwlock_wait_bpfcc_20260508.sh ]]; then
  echo "missing ./scripts/trace_rwlock_wait_bpfcc_20260508.sh" >&2
  exit 1
fi

if [[ "$#" -gt 0 ]]; then
  CASES=("$@")
else
  CASES=(sift200k_node_lock_on sift200k_node_lock_off gist200k_node_lock_on gist200k_node_lock_off)
fi

mkdir -p "$OUT_ROOT"

for case_name in "${CASES[@]}"; do
  cfg="$CFG_ROOT/$case_name.yaml"
  out="$OUT_ROOT/$case_name"
  if [[ ! -f "$cfg" ]]; then
    echo "missing config: $cfg" >&2
    exit 1
  fi

  rm -rf "$out"
  mkdir -p "$out"
  python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))' > "$out/bench_launch_raw_ns.txt"

  echo "[m2-external-rwlock-bpfcc] case=$case_name out=$out min_us=$MIN_US stacks=$STACKS"
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
    LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
    ./bin/bench -config "$cfg" > "$out/run.log" 2>&1 &

  bench_pid=$!
  echo "$bench_pid" > "$out/bench.pid"
  sleep 0.2

  if [[ "$(id -u)" == "0" ]]; then
    timeout "$TRACE_TIMEOUT" env MIN_US="$MIN_US" STACKS="$STACKS" OUT="$out/funcslower_rdlock.txt" \
      ./scripts/trace_rwlock_wait_bpfcc_20260508.sh "$bench_pid" &
  else
    sudo timeout "$TRACE_TIMEOUT" env MIN_US="$MIN_US" STACKS="$STACKS" OUT="$out/funcslower_rdlock.txt" \
      ./scripts/trace_rwlock_wait_bpfcc_20260508.sh "$bench_pid" &
  fi
  tracer_pid=$!
  echo "$tracer_pid" > "$out/tracer.pid"

  bench_status=0
  wait "$bench_pid" || bench_status=$?
  if [[ "$bench_status" -ne 0 ]]; then
    echo "[m2-external-rwlock-bpfcc] bench failed: case=$case_name status=$bench_status" >&2
  fi

  tracer_status=0
  wait "$tracer_pid" || tracer_status=$?
  if [[ "$tracer_status" -ne 0 && "$tracer_status" -ne 124 ]]; then
    echo "[m2-external-rwlock-bpfcc] tracer failed: case=$case_name status=$tracer_status" >&2
  fi
  echo "$bench_status,$tracer_status" > "$out/status.csv"

  if [[ "$bench_status" -ne 0 ]]; then
    exit "$bench_status"
  fi
done

echo "[m2-external-rwlock-bpfcc] complete: $OUT_ROOT"
