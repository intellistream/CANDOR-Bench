#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/m2_node_lock_hardware_matrix_20260507}"
CFG_ROOT="${CFG_ROOT:-./config/m2_node_lock_hardware_matrix_20260507}"
PROBE_SO="./tools/rwlock_probe.so"
EVENTS="${EVENTS:-cycles,instructions,l2_rqsts.all_rfo,l2_rqsts.rfo_miss,LLC-load-misses,LLC-store-misses,longest_lat_cache.miss}"
INTERVAL_MS="${INTERVAL_MS:-50}"
DURATION_MS="${DURATION_MS:-12000}"

mkdir -p "$OUT_ROOT" "$CFG_ROOT"

write_config() {
  local case_name="$1" dataset_name="$2" max_elements="$3" begin_num="$4"
  local data_path="$5" query_path="$6" gt_path="$7" node_lock="$8"
  local m="${9:-16}" efc="${10:-100}" efs="${11:-50}"
  local out_dir="$OUT_ROOT/$case_name"
  local cfg="$CFG_ROOT/$case_name.yaml"
  mkdir -p "$out_dir"
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
    printf '  ef_construction: %s\n' "$efc"
    printf 'search:\n'
    printf '  recall_at: 10\n'
    printf '  ef_search: %s\n' "$efs"
    printf '  use_node_lock: %s\n' "$node_lock"
    printf '  enable_mvcc: true\n'
    printf '  enable_undo_recovery: true\n'
    printf 'workload:\n'
    printf '  batch_size: 20\n'
    printf '  num_threads: 48\n'
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

write_config "sift200k_node_lock_on" "sift200k" 200000 100000 "../data/sift/sift_base_200k.bin" "../data/sift/sift_query_1k.bin" "../data/sift/sift_base_200k.gt20" true 16 100 50
write_config "sift200k_node_lock_off" "sift200k" 200000 100000 "../data/sift/sift_base_200k.bin" "../data/sift/sift_query_1k.bin" "../data/sift/sift_base_200k.gt20" false 16 100 50
write_config "gist200k_node_lock_on" "gist200k" 200000 100000 "../data/gist/gist_base_200k.bin" "../data/gist/gist_query.bin" "../data/gist/gist_base_200k.gt20" true 16 200 80
write_config "gist200k_node_lock_off" "gist200k" 200000 100000 "../data/gist/gist_base_200k.bin" "../data/gist/gist_query.bin" "../data/gist/gist_base_200k.gt20" false 16 200 80

if [[ "$#" -gt 0 ]]; then
  CASES=("$@")
else
  CASES=(sift200k_node_lock_on sift200k_node_lock_off gist200k_node_lock_on gist200k_node_lock_off)
fi

for case_name in "${CASES[@]}"; do
  cfg="$CFG_ROOT/$case_name.yaml"
  out="$OUT_ROOT/$case_name"
  rm -rf "$out/raw_latency" "$out/rwlock_probe.csv" "$out/perf.csv"
  mkdir -p "$out"
  python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))' > "$out/perf_start_raw_ns.txt"
  echo "[m2-node-lock-hw] case=$case_name events=$EVENTS"
  perf stat -x, -I "$INTERVAL_MS" -e "$EVENTS" -o "$out/perf.csv" -- \
    env \
      M3_CLOSED_LOOP=1 \
      M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
      M3_CLOSED_LOOP_MEASURED_WORKERS=8 \
      M3_CLOSED_LOOP_COMPETING_WORKERS=40 \
      M3_CLOSED_LOOP_COMPETING_MODE=periodic \
      ANNCHOR_M3_BATCH_DIAG=0 \
      ANNCHOR_PHASE_OVERLAP_DIAG=0 \
      ANNCHOR_REWRITE_ACTIVE_DIAG=0 \
      ANNCHOR_SEARCH_WORK_COUNTERS=1 \
      ANNCHOR_MEASURED_SEARCH_ARENA=1 \
      ANNCHOR_MEASURED_SEARCH_ARENA_THREADS=8 \
      ANNCHOR_SHARED_ARENA_THREADS=40 \
      OCC_RAW_LATENCY_DIR="$out/raw_latency" \
      RAW_LATENCY_SKIP_SEARCH_WIDE=0 \
      RWLOCK_PROBE_LOG="$out/rwlock_probe.csv" \
      RWLOCK_PROBE_MIN_US=1000 \
      LD_PRELOAD="$PWD/$PROBE_SO" \
      LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
      ./bin/bench -config "$cfg" > "$out/run.log" 2>&1
done

echo "[m2-node-lock-hw] complete: $OUT_ROOT"
