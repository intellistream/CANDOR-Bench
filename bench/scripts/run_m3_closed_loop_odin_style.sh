#!/usr/bin/env bash
# No-rate fixed-worker alternating-period run for an OdinANN-style timeline.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_closed_loop_odin_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_closed_loop_odin_${STAMP}}"
PAPER_ROOT="${PAPER_ROOT:-../paper-progress/m3/search-tail-rootcause-2026-05-01}"

BEGIN_NUM="${BEGIN_NUM:-500000}"
INSERT_POINTS="${INSERT_POINTS:-500000}"
DURATION_MS="${DURATION_MS:-12000}"
BATCH_SIZE="${BATCH_SIZE:-20}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
COMPETING_WORKERS="${COMPETING_WORKERS:-40}"
DATASET_NAME="${DATASET_NAME:-sift}"
DATA_PATH="${DATA_PATH:-../data/sift/sift_base.bin}"
QUERY_PATH="${QUERY_PATH:-../data/sift/sift_query.bin}"
GT_PATH="${GT_PATH:-../data/sift/sift_base.gt20}"
M="${M:-16}"
EF_SEARCH="${EF_SEARCH:-35}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-35}"
BUILD_JOBS="${BUILD_JOBS:-8}"
DISABLE_BUILD_STEP="${DISABLE_BUILD_STEP:-0}"
RUN_PERF="${RUN_PERF:-1}"
INTERVAL_MS="${INTERVAL_MS:-100}"
PERF_EVENTS="${PERF_EVENTS:-LLC-load-misses,dtlb_load_misses.walk_completed}"
PERF_STAT_ARGS="${PERF_STAT_ARGS:-}"
M3_BATCH_DIAG="${M3_BATCH_DIAG:-1}"
PHASE_OVERLAP_DIAG="${PHASE_OVERLAP_DIAG:-1}"
REWRITE_ACTIVE_DIAG="${REWRITE_ACTIVE_DIAG:-1}"
SEARCH_WORK_COUNTERS="${SEARCH_WORK_COUNTERS:-0}"
EXISTING_UPDATE_READ_SCAN_ONLY="${EXISTING_UPDATE_READ_SCAN_ONLY:-0}"
DISABLE_EXISTING_UNDO_RECORD="${DISABLE_EXISTING_UNDO_RECORD:-0}"
COMPETING_MODE="${COMPETING_MODE:-periodic}"
INSERT_DUMMY_MODE="${INSERT_DUMMY_MODE:-}"
INSERT_DUMMY_LOOPS="${INSERT_DUMMY_LOOPS:-}"
INSERT_POST_CPU_LOOPS="${INSERT_POST_CPU_LOOPS:-}"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$PAPER_ROOT"

if [[ "$DISABLE_BUILD_STEP" != "1" ]]; then
  cmake --build ./build --target index -j "$BUILD_JOBS"
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .
fi

MAX_ELEMENTS="${MAX_ELEMENTS:-$((BEGIN_NUM + INSERT_POINTS))}"

CFG="$CONFIG_ROOT/closed_loop_odin_efc${EF_CONSTRUCTION}.yaml"
OUT="$ROOT/closed_loop_odin_efc${EF_CONSTRUCTION}"
mkdir -p "$OUT"

cat > "$CFG" <<EOF
data:
  dataset_name: ${DATASET_NAME}
  max_elements: ${MAX_ELEMENTS}
  begin_num: ${BEGIN_NUM}
  data_type: float
  data_path: ${DATA_PATH}
  incr_query_path: ${QUERY_PATH}
  overall_query_path: ${QUERY_PATH}
  overall_gt_path: ${GT_PATH}
index:
  index_type: annchor-m1
  m: ${M}
  ef_construction: ${EF_CONSTRUCTION}
search:
  recall_at: 10
  ef_search: ${EF_SEARCH}
  use_node_lock: false
  enable_mvcc: true
  enable_undo_recovery: false
  enable_inflight_bruteforce: false
workload:
  batch_size: ${BATCH_SIZE}
  num_threads: $((MEASURED_WORKERS + COMPETING_WORKERS))
  queue_size: 0
  insert_event_rate: 0
  search_event_rate: 0
  insert_burst_schedule:
    - {start_ms: 2500, duration_ms: 1000, insert_event_rate: 0}
    - {start_ms: 6000, duration_ms: 1000, insert_event_rate: 0}
    - {start_ms: 9500, duration_ms: 1000, insert_event_rate: 0}
  schedule_horizon_ms: ${DURATION_MS}
  with_external_rw_lock: false
  query_mode: round_robin
  per_query_latency: true
  precompute_schedule: false
compare:
  enabled: false
  simulate:
    enabled: false
result:
  output_dir: ${OUT}
repetitions: 1
profile:
  memory_monitor_interval: 100
  enable_memory_profile: false
EOF

echo "[m3-closed-loop-odin] root=$ROOT"
echo "[m3-closed-loop-odin] config=$CFG"
echo "[m3-closed-loop-odin] dataset=${DATASET_NAME} m=${M} efs=${EF_SEARCH} efc=${EF_CONSTRUCTION} measured=${MEASURED_WORKERS} competing=${COMPETING_WORKERS} batch=${BATCH_SIZE} existing_update_read_scan_only=${EXISTING_UPDATE_READ_SCAN_ONLY} disable_existing_undo_record=${DISABLE_EXISTING_UNDO_RECORD}"
echo "[m3-closed-loop-odin] competing_mode=${COMPETING_MODE} insert_dummy_mode=${INSERT_DUMMY_MODE} insert_dummy_loops=${INSERT_DUMMY_LOOPS} insert_post_cpu_loops=${INSERT_POST_CPU_LOOPS}"
echo "[m3-closed-loop-odin] m3_batch_diag=${M3_BATCH_DIAG} phase_overlap_diag=${PHASE_OVERLAP_DIAG} rewrite_active_diag=${REWRITE_ACTIVE_DIAG} search_work_counters=${SEARCH_WORK_COUNTERS}"
echo "[m3-closed-loop-odin] run_perf=${RUN_PERF} interval_ms=${INTERVAL_MS} events=${PERF_EVENTS} perf_stat_args=${PERF_STAT_ARGS}"

date +%s%N > "$OUT/perf_start_unix_ns.txt"
python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))' \
  > "$OUT/perf_start_raw_ns.txt" 2>/dev/null || rm -f "$OUT/perf_start_raw_ns.txt"

ENV_CMD=(
  env
  M3_CLOSED_LOOP=1 \
  M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
  M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
  M3_CLOSED_LOOP_COMPETING_WORKERS="$COMPETING_WORKERS" \
  M3_CLOSED_LOOP_COMPETING_MODE="$COMPETING_MODE" \
  ANNCHOR_M3_BATCH_DIAG="$M3_BATCH_DIAG" \
  ANNCHOR_PHASE_OVERLAP_DIAG="$PHASE_OVERLAP_DIAG" \
  ANNCHOR_REWRITE_ACTIVE_DIAG="$REWRITE_ACTIVE_DIAG" \
  ANNCHOR_SEARCH_WORK_COUNTERS="$SEARCH_WORK_COUNTERS" \
  ANNCHOR_MEASURED_SEARCH_ARENA=1 \
  ANNCHOR_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
  ANNCHOR_SHARED_ARENA_THREADS="$COMPETING_WORKERS" \
  ANNCHOR_INSERT_PHASE_TRACE="$OUT/insert_phase_trace.csv" \
  ANNCHOR_INSERT_PHASE_TRACE_FROM_ID="$BEGIN_NUM" \
  INSERT_DUMMY_MODE="$INSERT_DUMMY_MODE" \
  INSERT_DUMMY_LOOPS="$INSERT_DUMMY_LOOPS" \
  INSERT_POST_CPU_LOOPS="$INSERT_POST_CPU_LOOPS" \
  OCC_RAW_LATENCY_DIR="$OUT/raw_latency" \
  LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}"
)
if [[ "$EXISTING_UPDATE_READ_SCAN_ONLY" == "1" ]]; then
  ENV_CMD+=(
    ANNCHOR_EXISTING_NEIGHBOR_UPDATE_READ_SCAN_ONLY=1
    ANNCHOR_EXISTING_NEIGHBOR_UPDATE_READ_SCAN_ONLY_FROM_ID="$BEGIN_NUM"
  )
fi
if [[ "$DISABLE_EXISTING_UNDO_RECORD" == "1" ]]; then
  ENV_CMD+=(
    ANNCHOR_DISABLE_EXISTING_NEIGHBOR_UNDO_RECORD=1
    ANNCHOR_DISABLE_EXISTING_NEIGHBOR_UNDO_RECORD_FROM_ID="$BEGIN_NUM"
  )
fi
BENCH_CMD=(./bin/bench -config "$CFG")

if [[ "$RUN_PERF" == "1" ]]; then
  read -r -a PERF_STAT_ARGS_ARR <<< "$PERF_STAT_ARGS"
  "${ENV_CMD[@]}" perf stat "${PERF_STAT_ARGS_ARR[@]}" -I "$INTERVAL_MS" -x, -e "$PERF_EVENTS" \
    -o "$OUT/perf_interval.csv" -- "${BENCH_CMD[@]}" 2>&1 | tee "$OUT/bench.stdout.log"
else
  "${ENV_CMD[@]}" "${BENCH_CMD[@]}" 2>&1 | tee "$OUT/bench.stdout.log"
fi

if [[ ! -s "$OUT/insert_phase_trace.csv" ]]; then
  cat > "$OUT/insert_phase_trace.csv" <<'EOF'
start_ns,end_ns,tid_hash,phase,point_id,level,existing_neighbor,existing_neighbor_degree_before,edges_loaded,prune_candidates,dist_comps,edges_written,edges_pruned
EOF
fi

python3 ./scripts/bin_m3_insert_phase_trace.py "$OUT/insert_phase_trace.csv" \
  --search-wide "$OUT/raw_latency/search_wide.csv" \
  --meta "$OUT/raw_latency/meta.csv" \
  --bin-ms 50 \
  --out "$OUT/periods.csv"

python3 ./scripts/plot_m3_period_groups.py "$OUT" \
  --perf-events "$PERF_EVENTS" \
  --out-csv "$OUT/period_groups.csv" \
  --out-fig "$OUT/period_groups.png" \
  > "$OUT/period_groups.log"

python3 ./scripts/plot_m3_closed_loop_odin_style.py "$OUT" \
  --config "$CFG" \
  --measured-workers "$MEASURED_WORKERS" \
  --competing-workers "$COMPETING_WORKERS" \
  --out "$ROOT/closed_loop_odin_tail.png" \
  --csv-out "$ROOT/closed_loop_odin_tail.csv" \
  --summary-out "$ROOT/closed_loop_odin_tail_summary.csv" \
  --boundaries-out "$ROOT/closed_loop_odin_tail_insert_boundaries.csv"

cp "$CFG" "$PAPER_ROOT/closed_loop_odin_config.yaml"
cp "$ROOT/closed_loop_odin_tail.png" "$PAPER_ROOT/closed_loop_odin_tail.png"
cp "$ROOT/closed_loop_odin_tail.csv" "$PAPER_ROOT/closed_loop_odin_tail.csv"
cp "$ROOT/closed_loop_odin_tail_summary.csv" "$PAPER_ROOT/closed_loop_odin_tail_summary.csv"
cp "$ROOT/closed_loop_odin_tail_insert_boundaries.csv" "$PAPER_ROOT/closed_loop_odin_tail_insert_boundaries.csv"

cat > "$ROOT/README.md" <<EOF
# M3 Closed-Loop OdinANN-Style Timeline

Measured search runs with ${MEASURED_WORKERS} fixed workers for the whole run.
The ${COMPETING_WORKERS} competing workers alternate between background search
and real insert according to the configured insert windows. There is no input
rate limiter; each worker immediately starts the next operation after finishing
the previous one.

Outputs are copied to:

\`\`\`text
$PAPER_ROOT/closed_loop_odin_tail.png
$PAPER_ROOT/closed_loop_odin_tail.csv
$PAPER_ROOT/closed_loop_odin_tail_summary.csv
$PAPER_ROOT/closed_loop_odin_tail_insert_boundaries.csv
$PAPER_ROOT/closed_loop_odin_config.yaml
\`\`\`
EOF

echo "[m3-closed-loop-odin] done root=$ROOT"
