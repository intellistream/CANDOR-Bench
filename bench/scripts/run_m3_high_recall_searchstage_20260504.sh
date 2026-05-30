#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="${ROOT:-./result/20260504_m3_high_recall_searchstage}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/20260504_m3_high_recall_searchstage}"
DELIVER_ROOT="${DELIVER_ROOT:-../paper-progress/m3/search-tail-rootcause-2026-05-01/high_recall_searchstage_20260504}"
DATASETS="${DATASETS:-sift deep1M}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
COMPETING_WORKERS="${COMPETING_WORKERS:-40}"
BATCH_SIZE="${BATCH_SIZE:-20}"
DURATION_MS="${DURATION_MS:-12000}"
BEGIN_NUM="${BEGIN_NUM:-960000}"
INSERT_POINTS="${INSERT_POINTS:-40000}"
BUILD_JOBS="${BUILD_JOBS:-8}"
RUN_PERF="${RUN_PERF:-0}"
INTERVAL_MS="${INTERVAL_MS:-100}"
PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,cache-misses,LLC-load-misses,dTLB-load-misses,mem_load_retired.l3_miss,mem_load_l3_miss_retired.remote_hitm}"
PERF_STAT_ARGS="${PERF_STAT_ARGS:-}"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$DELIVER_ROOT/tables" "$DELIVER_ROOT/figures" "$DELIVER_ROOT/runs"

cmake --build ./build --target index -j "$BUILD_JOBS"
LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .

PARAMS="$DELIVER_ROOT/tables/run_params.csv"
COMMANDS="$DELIVER_ROOT/tables/run_commands.txt"
printf 'dataset,m,efc,efs,recall_note,begin_num,batch_size,measured_workers,competing_workers,run_root\n' > "$PARAMS"
: > "$COMMANDS"

dataset_params() {
  case "$1" in
    sift)
      printf '%s|%s|%s|%s|%s|%s|%s\n' \
        "../data/sift/sift_base.bin" \
        "../data/sift/sift_query.bin" \
        "../data/sift/sift_base.gt20" \
        "16" "200" "50" "0.946"
      ;;
    deep1M)
      printf '%s|%s|%s|%s|%s|%s|%s\n' \
        "../data/deep1M/deep1M_base.bin" \
        "../data/deep1M/deep1M_query.bin" \
        "../data/deep1M/deep1M_base.gt20" \
        "16" "200" "100" "0.948"
      ;;
    gist)
      printf '%s|%s|%s|%s|%s|%s|%s\n' \
        "../data/gist/gist_base.bin" \
        "../data/gist/gist_query.bin" \
        "../data/gist/gist_base.gt20" \
        "24" "400" "200" "0.957"
      ;;
    *)
      echo "unknown dataset: $1" >&2
      return 1
      ;;
  esac
}

for dataset in $DATASETS; do
  IFS='|' read -r data_path query_path gt_path m efc efs recall_note < <(dataset_params "$dataset")
  case_name="${dataset}_real_insert"
  case_root="$ROOT/$case_name"
  case_config="$CONFIG_ROOT/$case_name"
  paper_root="$DELIVER_ROOT/runs/$case_name"
  mkdir -p "$case_root" "$case_config" "$paper_root"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$dataset" "$m" "$efc" "$efs" "$recall_note" "$BEGIN_NUM" "$BATCH_SIZE" \
    "$MEASURED_WORKERS" "$COMPETING_WORKERS" "$case_root" >> "$PARAMS"

  printf 'ROOT=%q CONFIG_ROOT=%q PAPER_ROOT=%q DATASET_NAME=%q DATA_PATH=%q QUERY_PATH=%q GT_PATH=%q M=%q EF_SEARCH=%q EF_CONSTRUCTION=%q BEGIN_NUM=%q INSERT_POINTS=%q BATCH_SIZE=%q MEASURED_WORKERS=%q COMPETING_WORKERS=%q DURATION_MS=%q RUN_PERF=%q INTERVAL_MS=%q PERF_EVENTS=%q PERF_STAT_ARGS=%q DISABLE_BUILD_STEP=1 M3_BATCH_DIAG=0 PHASE_OVERLAP_DIAG=0 REWRITE_ACTIVE_DIAG=0 SEARCH_WORK_COUNTERS=1 bash scripts/run_m3_closed_loop_odin_style.sh\n' \
    "$case_root" "$case_config" "$paper_root" "$dataset" "$data_path" "$query_path" "$gt_path" \
    "$m" "$efs" "$efc" "$BEGIN_NUM" "$INSERT_POINTS" "$BATCH_SIZE" "$MEASURED_WORKERS" \
    "$COMPETING_WORKERS" "$DURATION_MS" "$RUN_PERF" "$INTERVAL_MS" "$PERF_EVENTS" "$PERF_STAT_ARGS" >> "$COMMANDS"

  env \
    ROOT="$case_root" \
    CONFIG_ROOT="$case_config" \
    PAPER_ROOT="$paper_root" \
    DATASET_NAME="$dataset" \
    DATA_PATH="$data_path" \
    QUERY_PATH="$query_path" \
    GT_PATH="$gt_path" \
    M="$m" \
    EF_SEARCH="$efs" \
    EF_CONSTRUCTION="$efc" \
    BEGIN_NUM="$BEGIN_NUM" \
    INSERT_POINTS="$INSERT_POINTS" \
    BATCH_SIZE="$BATCH_SIZE" \
    MEASURED_WORKERS="$MEASURED_WORKERS" \
    COMPETING_WORKERS="$COMPETING_WORKERS" \
    DURATION_MS="$DURATION_MS" \
    RUN_PERF="$RUN_PERF" \
    INTERVAL_MS="$INTERVAL_MS" \
    PERF_EVENTS="$PERF_EVENTS" \
    PERF_STAT_ARGS="$PERF_STAT_ARGS" \
    DISABLE_BUILD_STEP=1 \
    M3_BATCH_DIAG=0 \
    PHASE_OVERLAP_DIAG=0 \
    REWRITE_ACTIVE_DIAG=0 \
    SEARCH_WORK_COUNTERS=1 \
    bash scripts/run_m3_closed_loop_odin_style.sh
done

python3 ./scripts/analyze_m3_high_recall_searchstage_20260504.py \
  --run-root "$ROOT" \
  --out-root "$DELIVER_ROOT" \
  --params "$PARAMS"

echo "[m3-high-recall-searchstage] done root=$ROOT deliver=$DELIVER_ROOT"
