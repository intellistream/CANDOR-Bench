#!/usr/bin/env bash
# Fixed-worker closed-loop M3 tail experiment.
#
# No input rate is set for the competing workload. Each competing worker starts
# the next operation immediately after finishing the previous one.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_closed_loop_worker_tail_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_closed_loop_worker_tail_${STAMP}}"
PAPER_ROOT="${PAPER_ROOT:-../paper-progress/m3/search-tail-rootcause-2026-05-01}"

BEGIN_NUM="${BEGIN_NUM:-500000}"
INSERT_POINTS="${INSERT_POINTS:-400000}"
DURATION_MS="${DURATION_MS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-20}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
WORKERS_LIST="${WORKERS_LIST:-2 8}"
EF_SEARCH="${EF_SEARCH:-35}"
BUILD_JOBS="${BUILD_JOBS:-8}"
SKIP_BUILD="${SKIP_BUILD:-0}"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$PAPER_ROOT"
MANIFEST="$ROOT/cases.csv"

echo "[m3-closed-loop] root=$ROOT"
echo "[m3-closed-loop] config=$CONFIG_ROOT"

if [[ "$SKIP_BUILD" != "1" ]]; then
  cmake --build ./build --target index -j "$BUILD_JOBS"
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .
fi

cat > "$MANIFEST" <<EOF
case,group,workers,ef_construction,mutation,duration_ms,run_dir,config
EOF

write_config() {
  local name="$1"
  local efc="$2"
  local max_elements="$3"
  local threads="$4"
  local out="$ROOT/$name"
  local cfg="$CONFIG_ROOT/${name}.yaml"
  cat > "$cfg" <<EOF
data:
  dataset_name: sift
  max_elements: ${max_elements}
  begin_num: ${BEGIN_NUM}
  data_type: float
  data_path: ../data/sift/sift_base.bin
  incr_query_path: ../data/sift/sift_query.bin
  overall_query_path: ../data/sift/sift_query.bin
  overall_gt_path: ../data/sift/sift_base.gt20
index:
  index_type: annchor-m1
  m: 16
  ef_construction: ${efc}
search:
  recall_at: 10
  ef_search: ${EF_SEARCH}
  use_node_lock: false
  enable_mvcc: true
  enable_undo_recovery: false
  enable_inflight_bruteforce: false
workload:
  batch_size: ${BATCH_SIZE}
  num_threads: ${threads}
  queue_size: 0
  insert_event_rate: 0
  search_event_rate: 0
  insert_burst_schedule: []
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
  output_dir: ${out}
repetitions: 1
profile:
  memory_monitor_interval: 100
  enable_memory_profile: false
EOF
  printf '%s\n' "$cfg"
}

run_case() {
  local name="$1"
  local group="$2"
  local workers="$3"
  local efc="$4"
  local mutation="$5"
  local max_elements="$6"
  local mode="$7"
  shift 7
  local threads=$((MEASURED_WORKERS + workers))
  local cfg
  cfg="$(write_config "$name" "$efc" "$max_elements" "$threads")"
  local out="$ROOT/$name"
  mkdir -p "$out"
  echo "${name},${group},${workers},${efc},${mutation},${DURATION_MS},${out},${cfg}" >> "$MANIFEST"
  echo "[m3-closed-loop] run $name mode=$mode workers=$workers efc=$efc"
  env \
    M3_CLOSED_LOOP=1 \
    M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
    M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
    M3_CLOSED_LOOP_COMPETING_WORKERS="$workers" \
    M3_CLOSED_LOOP_COMPETING_MODE="$mode" \
    ANNCHOR_M3_BATCH_DIAG=1 \
    ANNCHOR_PHASE_OVERLAP_DIAG=1 \
    ANNCHOR_MEASURED_SEARCH_ARENA=1 \
    ANNCHOR_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
    ANNCHOR_SHARED_ARENA_THREADS="$workers" \
    OCC_RAW_LATENCY_DIR="$out/raw_latency" \
    LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
    "$@" \
    ./bin/bench -config "$cfg" 2>&1 | tee "$out/bench.stdout.log"
}

run_case baseline baseline 0 35 none "$BEGIN_NUM" idle

for workers in $WORKERS_LIST; do
  run_case "bg_search_n${workers}" bg_search "$workers" 35 none "$BEGIN_NUM" search
done

for workers in $WORKERS_LIST; do
  max_elements=$((BEGIN_NUM + INSERT_POINTS))
  run_case "insert_n${workers}_efc200" insert "$workers" 200 full "$max_elements" insert
  run_case "insert_n${workers}_efc35" insert "$workers" 35 full "$max_elements" insert
  run_case "insert_n${workers}_efc35_skip_update" insert_diag "$workers" 35 skip_existing_update "$max_elements" insert \
    ANNCHOR_SKIP_EXISTING_NEIGHBOR_UPDATE=1 \
    ANNCHOR_SKIP_EXISTING_NEIGHBOR_UPDATE_FROM_ID="$BEGIN_NUM"
done

python3 ./scripts/summarize_m3_closed_loop_worker_tail.py \
  --manifest "$MANIFEST" \
  --queue-max-ms 0.1 \
  --out-summary "$ROOT/closed_loop_worker_summary.csv" \
  --out-pairs "$ROOT/closed_loop_worker_pairs.csv"

python3 ./scripts/plot_m3_closed_loop_worker_tail.py \
  --summary "$ROOT/closed_loop_worker_summary.csv" \
  --pairs "$ROOT/closed_loop_worker_pairs.csv" \
  --out "$ROOT/closed_loop_worker_tail.png"

cp "$MANIFEST" "$PAPER_ROOT/closed_loop_worker_cases.csv"
cp "$ROOT/closed_loop_worker_summary.csv" "$PAPER_ROOT/closed_loop_worker_summary.csv"
cp "$ROOT/closed_loop_worker_pairs.csv" "$PAPER_ROOT/closed_loop_worker_pairs.csv"
cp "$ROOT/closed_loop_worker_tail.png" "$PAPER_ROOT/closed_loop_worker_tail.png"

cat > "$ROOT/README.md" <<EOF
# M3 Closed-Loop Worker Tail

No input rate is set for competing work. The controlled variable is fixed
competing worker count.

Outputs are copied to:

\`\`\`text
$PAPER_ROOT/closed_loop_worker_cases.csv
$PAPER_ROOT/closed_loop_worker_summary.csv
$PAPER_ROOT/closed_loop_worker_pairs.csv
$PAPER_ROOT/closed_loop_worker_tail.png
\`\`\`
EOF

echo "[m3-closed-loop] done root=$ROOT"
