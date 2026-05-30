#!/usr/bin/env bash
# High-recall controls for separating write-request work from pure read concurrency.
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="${ROOT:-./result/20260504_m3_high_recall_write_read_controls}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/20260504_m3_high_recall_write_read_controls}"
DELIVER_ROOT="${DELIVER_ROOT:-../paper-progress/m3/search-tail-rootcause-2026-05-01/high_recall_write_read_controls_20260504}"
DATASETS="${DATASETS:-sift}"
CASES="${CASES:-background_search insert_ann_search real_insert}"
COMPETING_SET="${COMPETING_SET:-8 16 32 40}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-20}"
DURATION_MS="${DURATION_MS:-8000}"
BEGIN_NUM="${BEGIN_NUM:-900000}"
INSERT_POINTS="${INSERT_POINTS:-100000}"
BUILD_JOBS="${BUILD_JOBS:-8}"
RUN_PERF="${RUN_PERF:-0}"
INTERVAL_MS="${INTERVAL_MS:-100}"
PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,cache-misses,LLC-load-misses,dTLB-load-misses}"
SNAPSHOT_GT_PATH="${SNAPSHOT_GT_PATH:-}"
GRAPH_WRITE_LOOPS="${GRAPH_WRITE_LOOPS:-1}"
GRAPH_TOUCH_SAMPLE_QUERIES="${GRAPH_TOUCH_SAMPLE_QUERIES:-512}"
GRAPH_TOUCH_SEARCH_K="${GRAPH_TOUCH_SEARCH_K:-}"
GRAPH_TOUCH_MAX_NEIGHBORS="${GRAPH_TOUCH_MAX_NEIGHBORS:-32}"
GRAPH_TOUCH_LABEL_LIMIT="${GRAPH_TOUCH_LABEL_LIMIT:-4096}"
GRAPH_TOUCH_LABELS_PER_TASK="${GRAPH_TOUCH_LABELS_PER_TASK:-512}"
GRAPH_TOUCH_MAX_EDGES="${GRAPH_TOUCH_MAX_EDGES:-32}"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$DELIVER_ROOT/tables" "$DELIVER_ROOT/runs"

cmake --build ./build --target index -j "$BUILD_JOBS"
LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .

PARAMS="$DELIVER_ROOT/tables/run_params.csv"
COMMANDS="$DELIVER_ROOT/tables/run_commands.txt"
printf 'dataset,case_key,case_label,competing_workers,m,efc,efs,ann_search_loops,begin_num,insert_points,batch_size,run_root\n' > "$PARAMS"
: > "$COMMANDS"

dataset_params() {
  case "$1" in
    sift)
      printf '%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "../data/sift/sift_base.bin" \
        "../data/sift/sift_query.bin" \
        "../data/sift/sift_base.gt20" \
        "16" "200" "50" "4" "0.946"
      ;;
    deep1M)
      printf '%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "../data/deep1M/deep1M_base.bin" \
        "../data/deep1M/deep1M_query.bin" \
        "../data/deep1M/deep1M_base.gt20" \
        "16" "200" "100" "2" "0.948"
      ;;
    gist)
      printf '%s|%s|%s|%s|%s|%s|%s|%s\n' \
        "../data/gist/gist_base.bin" \
        "../data/gist/gist_query.bin" \
        "../data/gist/gist_base.gt20" \
        "24" "400" "200" "2" "0.957"
      ;;
    *)
      echo "unknown dataset: $1" >&2
      return 1
      ;;
  esac
}

case_settings() {
  case "$1" in
    background_search)
      printf '%s|%s|%s|%s\n' "Background Search" "background_search" "" ""
      ;;
    insert_ann_search)
      printf '%s|%s|%s|%s\n' "Insert Request ANN Search" "insert" "ann_search" "$2"
      ;;
    read_traversal_x1)
      printf '%s|%s|%s|%s\n' "Read Traversal x1" "insert" "ann_search" "1"
      ;;
    read_traversal_x2)
      printf '%s|%s|%s|%s\n' "Read Traversal x2" "insert" "ann_search" "2"
      ;;
    read_traversal_x4)
      printf '%s|%s|%s|%s\n' "Read Traversal x4" "insert" "ann_search" "4"
      ;;
    read_traversal_x8)
      printf '%s|%s|%s|%s\n' "Read Traversal x8" "insert" "ann_search" "8"
      ;;
    query_traversal_x1)
      printf '%s|%s|%s|%s\n' "Query Traversal x1" "insert" "ann_search_query" "1"
      ;;
    query_traversal_x2)
      printf '%s|%s|%s|%s\n' "Query Traversal x2" "insert" "ann_search_query" "2"
      ;;
    query_traversal_x4)
      printf '%s|%s|%s|%s\n' "Query Traversal x4" "insert" "ann_search_query" "4"
      ;;
    query_traversal_x8)
      printf '%s|%s|%s|%s\n' "Query Traversal x8" "insert" "ann_search_query" "8"
      ;;
    insert_cpu)
      printf '%s|%s|%s|%s\n' "Insert Request CPU Task" "insert" "cpu" "${INSERT_CPU_LOOPS:-20000}"
      ;;
    graph_write_hot)
      printf '%s|%s|%s|%s\n' "Hot Graph Writes" "insert" "graph_write_hot" "$GRAPH_WRITE_LOOPS"
      ;;
    graph_write_cold)
      printf '%s|%s|%s|%s\n' "Cold Graph Writes" "insert" "graph_write_cold" "$GRAPH_WRITE_LOOPS"
      ;;
    real_insert)
      printf '%s|%s|%s|%s\n' "Real Insert" "insert" "" ""
      ;;
    *)
      echo "unknown case: $1" >&2
      return 1
      ;;
  esac
}

for dataset in $DATASETS; do
  IFS='|' read -r data_path query_path gt_path m efc efs ann_loops recall_note < <(dataset_params "$dataset")
  if [[ -n "${M:-}" ]]; then
    m="$M"
  fi
  if [[ -n "${EF_CONSTRUCTION:-}" ]]; then
    efc="$EF_CONSTRUCTION"
  fi
  if [[ -n "${EF_SEARCH:-}" ]]; then
    efs="$EF_SEARCH"
  fi
  for competing_workers in $COMPETING_SET; do
    for case_key in $CASES; do
      IFS='|' read -r case_label competing_mode dummy_mode dummy_loops < <(case_settings "$case_key" "$ann_loops")
      case_gt_path="$gt_path"
      if [[ "$case_key" != "real_insert" && -n "$SNAPSHOT_GT_PATH" ]]; then
        case_gt_path="$SNAPSHOT_GT_PATH"
      fi
      run_name="${dataset}_cw${competing_workers}_${case_key}"
      case_root="$ROOT/$run_name"
      case_config="$CONFIG_ROOT/$run_name"
      paper_root="$DELIVER_ROOT/runs/$run_name"
      mkdir -p "$case_root" "$case_config" "$paper_root"

      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$dataset" "$case_key" "$case_label" "$competing_workers" "$m" "$efc" "$efs" \
        "$ann_loops" "$BEGIN_NUM" "$INSERT_POINTS" "$BATCH_SIZE" "$case_root" >> "$PARAMS"

      printf 'ROOT=%q CONFIG_ROOT=%q PAPER_ROOT=%q DATASET_NAME=%q DATA_PATH=%q QUERY_PATH=%q GT_PATH=%q M=%q EF_SEARCH=%q EF_CONSTRUCTION=%q BEGIN_NUM=%q INSERT_POINTS=%q BATCH_SIZE=%q MEASURED_WORKERS=%q COMPETING_WORKERS=%q COMPETING_MODE=%q DURATION_MS=%q RUN_PERF=%q INTERVAL_MS=%q PERF_EVENTS=%q INSERT_DUMMY_MODE=%q INSERT_DUMMY_LOOPS=%q GRAPH_TOUCH_SAMPLE_QUERIES=%q GRAPH_TOUCH_SEARCH_K=%q GRAPH_TOUCH_MAX_NEIGHBORS=%q GRAPH_TOUCH_LABEL_LIMIT=%q GRAPH_TOUCH_LABELS_PER_TASK=%q GRAPH_TOUCH_MAX_EDGES=%q DISABLE_BUILD_STEP=1 M3_BATCH_DIAG=0 PHASE_OVERLAP_DIAG=0 REWRITE_ACTIVE_DIAG=0 SEARCH_WORK_COUNTERS=1 bash scripts/run_m3_closed_loop_odin_style.sh\n' \
        "$case_root" "$case_config" "$paper_root" "$dataset" "$data_path" "$query_path" "$case_gt_path" \
        "$m" "$efs" "$efc" "$BEGIN_NUM" "$INSERT_POINTS" "$BATCH_SIZE" "$MEASURED_WORKERS" \
        "$competing_workers" "$competing_mode" "$DURATION_MS" "$RUN_PERF" "$INTERVAL_MS" "$PERF_EVENTS" \
        "$dummy_mode" "$dummy_loops" "$GRAPH_TOUCH_SAMPLE_QUERIES" "$GRAPH_TOUCH_SEARCH_K" \
        "$GRAPH_TOUCH_MAX_NEIGHBORS" "$GRAPH_TOUCH_LABEL_LIMIT" "$GRAPH_TOUCH_LABELS_PER_TASK" \
        "$GRAPH_TOUCH_MAX_EDGES" >> "$COMMANDS"

      env \
        ROOT="$case_root" \
        CONFIG_ROOT="$case_config" \
        PAPER_ROOT="$paper_root" \
        DATASET_NAME="$dataset" \
        DATA_PATH="$data_path" \
        QUERY_PATH="$query_path" \
        GT_PATH="$case_gt_path" \
        M="$m" \
        EF_SEARCH="$efs" \
        EF_CONSTRUCTION="$efc" \
        BEGIN_NUM="$BEGIN_NUM" \
        INSERT_POINTS="$INSERT_POINTS" \
        BATCH_SIZE="$BATCH_SIZE" \
        MEASURED_WORKERS="$MEASURED_WORKERS" \
        COMPETING_WORKERS="$competing_workers" \
        COMPETING_MODE="$competing_mode" \
        DURATION_MS="$DURATION_MS" \
        RUN_PERF="$RUN_PERF" \
        INTERVAL_MS="$INTERVAL_MS" \
        PERF_EVENTS="$PERF_EVENTS" \
        INSERT_DUMMY_MODE="$dummy_mode" \
        INSERT_DUMMY_LOOPS="$dummy_loops" \
        GRAPH_TOUCH_SAMPLE_QUERIES="$GRAPH_TOUCH_SAMPLE_QUERIES" \
        GRAPH_TOUCH_SEARCH_K="$GRAPH_TOUCH_SEARCH_K" \
        GRAPH_TOUCH_MAX_NEIGHBORS="$GRAPH_TOUCH_MAX_NEIGHBORS" \
        GRAPH_TOUCH_LABEL_LIMIT="$GRAPH_TOUCH_LABEL_LIMIT" \
        GRAPH_TOUCH_LABELS_PER_TASK="$GRAPH_TOUCH_LABELS_PER_TASK" \
        GRAPH_TOUCH_MAX_EDGES="$GRAPH_TOUCH_MAX_EDGES" \
        DISABLE_BUILD_STEP=1 \
        M3_BATCH_DIAG=0 \
        PHASE_OVERLAP_DIAG=0 \
        REWRITE_ACTIVE_DIAG=0 \
        SEARCH_WORK_COUNTERS=1 \
        bash scripts/run_m3_closed_loop_odin_style.sh
    done
  done
done

python3 ./scripts/analyze_m3_write_read_controls_20260504.py \
  --run-root "$ROOT" \
  --out-root "$DELIVER_ROOT" \
  --params "$PARAMS"

echo "[m3-write-read-controls] done root=$ROOT deliver=$DELIVER_ROOT"
