#!/usr/bin/env bash
# VTune threading collection on bench at N=16 to professionally validate
# our C++ chrono-based lock-wait measurements.
set -euo pipefail
cd /home/junyao/code/ANN-CC-bench/bench

VTUNE=${VTUNE:-/opt/intel/oneapi/vtune/2025.6/bin64/vtune}
OUT_ROOT=${OUT_ROOT:-./result/m3_vtune_lockwait_20260512}
N_WRITERS=${N_WRITERS:-16}
DURATION_MS=${DURATION_MS:-35000}
SAMPLE_S=${SAMPLE_S:-20}
SKIP_PRE_S=${SKIP_PRE_S:-7}
MEASURED_CPU_LIST=${MEASURED_CPU_LIST:-0-7}
COMPETING_CPU_LIST=${COMPETING_CPU_LIST:-8-23}

rm -rf "$OUT_ROOT"; mkdir -p "$OUT_ROOT"
CFG="$OUT_ROOT/cfg.yaml"
cat > "$CFG" <<YAML
data:
  dataset_name: sift200k
  max_elements: 200000
  begin_num: 180000
  data_type: float
  data_path: ../data/sift/sift_base_200k.bin
  incr_query_path: ../data/sift/sift_query_stream.bin
  overall_query_path: ../data/sift/sift_query_stream.bin
index: {index_type: hnsw, m: 16, ef_construction: 100}
search:
  recall_at: 10
  ef_search: 50
  use_node_lock: true
  enable_mvcc: false
  enable_undo_recovery: false
workload:
  batch_size: 20
  num_threads: $((8 + N_WRITERS))
  queue_size: 0
  insert_event_rate: 0
  search_event_rate: 0
  schedule_horizon_ms: $DURATION_MS
  with_external_rw_lock: false
  use_node_lock: true
  query_mode: peeking
  per_query_latency: true
  precompute_schedule: false
result:
  output_dir: $OUT_ROOT
repetitions: 1
profile:
  memory_monitor_interval: 100
  enable_memory_profile: false
YAML

# Launch bench
( env \
    M3_CLOSED_LOOP=1 \
    M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
    M3_CLOSED_LOOP_MEASURED_WORKERS=8 \
    M3_CLOSED_LOOP_COMPETING_WORKERS="$N_WRITERS" \
    M3_CLOSED_LOOP_INSERT_WORKERS="$N_WRITERS" \
    M3_CLOSED_LOOP_COMPETING_MODE=insert \
    ANNCHOR_M3_BATCH_DIAG=1 \
    HNSW_MEASURED_SEARCH_ARENA=1 \
    HNSW_MEASURED_SEARCH_ARENA_THREADS=8 \
    HNSW_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
    HNSW_INSERT_CPU_LIST="$COMPETING_CPU_LIST" \
    ANNCHOR_INSERT_CPU_LIST="$COMPETING_CPU_LIST" \
    OCC_RAW_LATENCY_DIR="$OUT_ROOT/raw" \
    LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
    ./bin/bench -config "$CFG" ) > "$OUT_ROOT/run.log" 2>&1 &
BENCH_PID=$!
echo "$BENCH_PID" > "$OUT_ROOT/bench.pid"

# Wait for closed-loop
for i in $(seq 1 120); do
  grep -q "Closed-loop setup:" "$OUT_ROOT/run.log" 2>/dev/null && break
  sleep 1
done
sleep "$SKIP_PRE_S"

# VTune threading collection — sampling-and-waits mode for user-mode lock wait analysis
date +%s.%N > "$OUT_ROOT/vtune.start_unix.txt"
"$VTUNE" -collect threading \
    -r "$OUT_ROOT/vtune_result" \
    -duration "$SAMPLE_S" \
    -target-pid "$BENCH_PID" \
    > "$OUT_ROOT/vtune_collect.log" 2>&1
date +%s.%N > "$OUT_ROOT/vtune.end_unix.txt"

# Wait for bench to finish
wait $BENCH_PID 2>/dev/null || true

# Reports: hotspots (where time is spent), thread (per-thread breakdown),
# sync (lock/wait analysis with object stack)
"$VTUNE" -report summary -r "$OUT_ROOT/vtune_result" > "$OUT_ROOT/vtune_summary.txt" 2>&1 || true
"$VTUNE" -report hotspots -r "$OUT_ROOT/vtune_result" -group-by function -limit 40 > "$OUT_ROOT/vtune_hotspots.txt" 2>&1 || true
"$VTUNE" -report hotspots -r "$OUT_ROOT/vtune_result" -group-by sync-object -limit 30 > "$OUT_ROOT/vtune_syncobj.txt" 2>&1 || true
"$VTUNE" -report wait-time -r "$OUT_ROOT/vtune_result" -limit 30 > "$OUT_ROOT/vtune_waittime.txt" 2>&1 || true
"$VTUNE" -report top-down -r "$OUT_ROOT/vtune_result" -limit 40 > "$OUT_ROOT/vtune_topdown.txt" 2>&1 || true

echo "[vtune] done at $(date) — see $OUT_ROOT/*.txt"
