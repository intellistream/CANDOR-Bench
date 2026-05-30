#!/usr/bin/env bash
# Formal M3 search-tail experiment with occupancy-matched search-only controls.
#
# This is the replacement for equal-rate search-vs-insert comparison. It keeps
# measured search fixed, sweeps background-search rates, and matches controls by
# actual competing active-worker occupancy.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_occupancy_matched_tail_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_occupancy_matched_tail_${STAMP}}"
PAPER_ROOT="${PAPER_ROOT:-../paper-progress/m3/search-tail-rootcause-2026-05-01}"

BEGIN_NUM="${BEGIN_NUM:-500000}"
INSERT_POINTS="${INSERT_POINTS:-81000}"
MAX_ELEMENTS_INSERT=$((BEGIN_NUM + INSERT_POINTS))
THREADS="${THREADS:-48}"
BATCH_SIZE="${BATCH_SIZE:-20}"
MEASURED_RATE="${MEASURED_RATE:-4000}"
MEASURED_LANES_VALUE="${MEASURED_LANES_VALUE:-8}"
MEASURED_ARENA_THREADS="${MEASURED_ARENA_THREADS:-8}"
SHARED_ARENA_THREADS="${SHARED_ARENA_THREADS:-40}"
INSERT_RATE="${INSERT_RATE:-18000}"
EF_SEARCH="${EF_SEARCH:-35}"
EFC_BASE="${EFC_BASE:-200}"
EFC_MATCH="${EFC_MATCH:-35}"
SEARCH_RATES="${SEARCH_RATES:-5000 8000 10000 15000 20000 30000 40000 60000 80000 100000 120000 140000 160000}"
QUEUE_MAX_MS="${QUEUE_MAX_MS:-0.1}"
BUILD_JOBS="${BUILD_JOBS:-8}"
SKIP_BUILD="${SKIP_BUILD:-0}"
TRACE_INSERT_PHASES="${TRACE_INSERT_PHASES:-1}"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$PAPER_ROOT"
MANIFEST="$ROOT/cases.csv"

echo "[m3-occupancy-matched] root=$ROOT"
echo "[m3-occupancy-matched] config=$CONFIG_ROOT"
echo "[m3-occupancy-matched] paper=$PAPER_ROOT"

if [[ "$SKIP_BUILD" != "1" ]]; then
  cmake --build ./build --target index -j "$BUILD_JOBS"
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .
fi

cat > "$MANIFEST" <<EOF
case,kind,rate,ef_construction,mutation,run_dir,config
EOF

write_config() {
  local name="$1"
  local efc="$2"
  local max_elements="$3"
  local with_insert="$4"
  local out_dir="$ROOT/$name"
  local cfg="$CONFIG_ROOT/${name}.yaml"

  if [[ "$with_insert" == "1" ]]; then
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
  num_threads: ${THREADS}
  queue_size: 0
  insert_event_rate: ${INSERT_RATE}
  search_event_rate: ${MEASURED_RATE}
  insert_burst_schedule:
    - start_ms: 3000
      duration_ms: 1500
      insert_event_rate: ${INSERT_RATE}
    - start_ms: 8000
      duration_ms: 1500
      insert_event_rate: ${INSERT_RATE}
    - start_ms: 13000
      duration_ms: 1500
      insert_event_rate: ${INSERT_RATE}
  schedule_horizon_ms: 17000
  with_external_rw_lock: false
  query_mode: round_robin
  per_query_latency: true
  precompute_schedule: true
compare:
  enabled: false
  simulate:
    enabled: false
result:
  output_dir: ${out_dir}
repetitions: 1
profile:
  memory_monitor_interval: 100
  enable_memory_profile: false
EOF
  else
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
  num_threads: ${THREADS}
  queue_size: 0
  insert_event_rate: 0
  search_event_rate: ${MEASURED_RATE}
  insert_burst_schedule: []
  schedule_horizon_ms: 17000
  with_external_rw_lock: false
  query_mode: round_robin
  per_query_latency: true
  precompute_schedule: true
compare:
  enabled: false
  simulate:
    enabled: false
result:
  output_dir: ${out_dir}
repetitions: 1
profile:
  memory_monitor_interval: 100
  enable_memory_profile: false
EOF
  fi

  printf '%s\n' "$cfg"
}

run_case() {
  local name="$1"
  local cfg="$2"
  local bg_rate="$3"
  local trace_insert="$4"
  shift 4
  local out_dir="$ROOT/$name"
  mkdir -p "$out_dir"
  echo "[m3-occupancy-matched] run $name bg_rate=$bg_rate cfg=$cfg"
  local env_args=(
    ANNCHOR_M3_BATCH_DIAG=1
    ANNCHOR_PHASE_OVERLAP_DIAG=1
    ANNCHOR_MEASURED_SEARCH_ARENA=1
    ANNCHOR_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_ARENA_THREADS"
    ANNCHOR_SHARED_ARENA_THREADS="$SHARED_ARENA_THREADS"
    MEASURED_LANES="$MEASURED_LANES_VALUE"
    MEASURED_SEARCH_EVENT_RATE="$MEASURED_RATE"
    BACKGROUND_SEARCH_EVENT_RATE="$bg_rate"
    BACKGROUND_SEARCH_PAUSE_DURING_INSERT=1
    OCC_RAW_LATENCY_DIR="$out_dir/raw_latency"
    LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}"
  )
  if [[ "$trace_insert" == "1" && "$TRACE_INSERT_PHASES" == "1" ]]; then
    env_args+=(
      ANNCHOR_INSERT_PHASE_TRACE="$out_dir/insert_phase_trace.csv"
      ANNCHOR_INSERT_PHASE_TRACE_FROM_ID="$BEGIN_NUM"
    )
  fi
  env \
    "${env_args[@]}" \
    "$@" \
    ./bin/bench -config "$cfg" 2>&1 | tee "$out_dir/bench.stdout.log"
  if [[ "$trace_insert" == "1" && "$TRACE_INSERT_PHASES" == "1" && -s "$out_dir/insert_phase_trace.csv" ]]; then
    python3 ./scripts/bin_m3_insert_phase_trace.py "$out_dir/insert_phase_trace.csv" \
      --search-wide "$out_dir/raw_latency/search_wide.csv" \
      --meta "$out_dir/raw_latency/meta.csv" \
      --bin-ms 50 \
      --out "$out_dir/period_groups.csv"
  fi
}

append_case() {
  local name="$1"
  local kind="$2"
  local rate="$3"
  local efc="$4"
  local mutation="$5"
  local cfg="$6"
  echo "${name},${kind},${rate},${efc},${mutation},${ROOT}/${name},${cfg}" >> "$MANIFEST"
}

baseline_cfg="$(write_config baseline "$EFC_MATCH" "$BEGIN_NUM" 0)"
append_case baseline baseline 0 "$EFC_MATCH" none "$baseline_cfg"
run_case baseline "$baseline_cfg" 0 0

for rate in $SEARCH_RATES; do
  name="bg_search_${rate}"
  cfg="$(write_config "$name" "$EFC_MATCH" "$BEGIN_NUM" 0)"
  append_case "$name" bg_search "$rate" "$EFC_MATCH" none "$cfg"
  run_case "$name" "$cfg" "$rate" 0
done

insert_efc200_cfg="$(write_config insert_efc200 "$EFC_BASE" "$MAX_ELEMENTS_INSERT" 1)"
append_case insert_efc200 insert "$INSERT_RATE" "$EFC_BASE" full "$insert_efc200_cfg"
run_case insert_efc200 "$insert_efc200_cfg" 0 1

insert_efc35_cfg="$(write_config insert_efc35 "$EFC_MATCH" "$MAX_ELEMENTS_INSERT" 1)"
append_case insert_efc35 insert "$INSERT_RATE" "$EFC_MATCH" full "$insert_efc35_cfg"
run_case insert_efc35 "$insert_efc35_cfg" 0 1

insert_efc35_skip_cfg="$(write_config insert_efc35_skip_update "$EFC_MATCH" "$MAX_ELEMENTS_INSERT" 1)"
append_case insert_efc35_skip_update insert_diag "$INSERT_RATE" "$EFC_MATCH" skip_existing_update "$insert_efc35_skip_cfg"
run_case \
  insert_efc35_skip_update \
  "$insert_efc35_skip_cfg" \
  0 \
  1 \
  ANNCHOR_SKIP_EXISTING_NEIGHBOR_UPDATE=1 \
  ANNCHOR_SKIP_EXISTING_NEIGHBOR_UPDATE_FROM_ID="$BEGIN_NUM"

python3 ./scripts/summarize_m3_occupancy_matched_tail.py \
  --manifest "$MANIFEST" \
  --queue-max-ms "$QUEUE_MAX_MS" \
  --out-summary "$ROOT/occupancy_matched_summary.csv" \
  --out-matched "$ROOT/occupancy_matched_pairs.csv"

python3 ./scripts/plot_m3_occupancy_matched_tail.py \
  --summary "$ROOT/occupancy_matched_summary.csv" \
  --matched "$ROOT/occupancy_matched_pairs.csv" \
  --out "$ROOT/occupancy_matched_tail.png"

cp "$MANIFEST" "$PAPER_ROOT/occupancy_matched_cases.csv"
cp "$ROOT/occupancy_matched_summary.csv" "$PAPER_ROOT/occupancy_matched_summary.csv"
cp "$ROOT/occupancy_matched_pairs.csv" "$PAPER_ROOT/occupancy_matched_pairs.csv"
cp "$ROOT/occupancy_matched_tail.png" "$PAPER_ROOT/occupancy_matched_tail.png"

cat > "$ROOT/README.md" <<EOF
# M3 Occupancy-Matched Search Tail Experiment

Started: ${STAMP}

This run replaces same-rate search-vs-insert comparison with actual
active-worker occupancy matching.

Fixed measured workload:

\`\`\`text
measured search rate: ${MEASURED_RATE} q/s
measured lanes: ${MEASURED_LANES_VALUE}
measured C++ arena: ${MEASURED_ARENA_THREADS} threads
shared competing C++ arena: ${SHARED_ARENA_THREADS} threads
batch size: ${BATCH_SIZE}
low-queue filter: queue_ms <= ${QUEUE_MAX_MS}
\`\`\`

Cases:

\`\`\`text
baseline: measured search only
bg_search_*: background-search-only sweep; matched by background_search_workers
insert_efc200: real insert, ef_construction=${EFC_BASE}
insert_efc35: real insert, ef_construction=${EFC_MATCH}
insert_efc35_skip_update: diagnostic, normal initial build then skip existing-neighbor update
\`\`\`

Writer occupancy is computed from \`period_groups.csv\` by default, generated
from \`ANNCHOR_INSERT_PHASE_TRACE\`, because this is the C++ phase-occupancy
signal needed for fair matching. The trace is filtered with
\`ANNCHOR_INSERT_PHASE_TRACE_FROM_ID=${BEGIN_NUM}\`, so the initial build is not
recorded. Set \`TRACE_INSERT_PHASES=0\` only for a follow-up replay after the
matched background-search rate has already been chosen; then the summarizer
falls back to insert task intervals.

Primary outputs:

\`\`\`text
$ROOT/cases.csv
$ROOT/occupancy_matched_summary.csv
$ROOT/occupancy_matched_pairs.csv
$ROOT/occupancy_matched_tail.png
\`\`\`

The same tables are copied to:

\`\`\`text
$PAPER_ROOT/occupancy_matched_cases.csv
$PAPER_ROOT/occupancy_matched_summary.csv
$PAPER_ROOT/occupancy_matched_pairs.csv
$PAPER_ROOT/occupancy_matched_tail.png
\`\`\`
EOF

echo "[m3-occupancy-matched] done root=$ROOT"
