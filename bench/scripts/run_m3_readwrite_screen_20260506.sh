#!/usr/bin/env bash
# Broad screen for M3 read-write tail candidates.
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="${ROOT:-./result/20260506_m3_readwrite_screen}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/20260506_m3_readwrite_screen}"
DELIVER_ROOT="${DELIVER_ROOT:-../paper-progress/m3/search-tail-rootcause-2026-05-01/readwrite_screen_20260506}"
DATASETS="${DATASETS:-sift200k sift1m sift10m200k dbpedia50k}"
CONFIGS="${CONFIGS:-m16_efc100_efs50 m16_efc400_efs50 m16_efc800_efs50 m32_efc400_efs50}"
CASES="${CASES:-background_search real_insert}"
COMPETING_SET="${COMPETING_SET:-8 24 40}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
DURATION_MS="${DURATION_MS:-6000}"
BATCH_SIZE="${BATCH_SIZE:-20}"
BUILD_JOBS="${BUILD_JOBS:-8}"
RUN_PERF="${RUN_PERF:-0}"
INTERVAL_MS="${INTERVAL_MS:-100}"
PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,cache-misses,LLC-load-misses,dTLB-load-misses}"
INSERT_CPU_LOOPS="${INSERT_CPU_LOOPS:-20000}"
GRAPH_WRITE_LOOPS="${GRAPH_WRITE_LOOPS:-1}"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$DELIVER_ROOT/tables" "$DELIVER_ROOT/runs"

cmake --build ./build --target index -j "$BUILD_JOBS"
LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .

PARAMS="$DELIVER_ROOT/tables/run_params.csv"
COMMANDS="$DELIVER_ROOT/tables/run_commands.txt"
printf 'dataset,case_key,case_label,competing_workers,m,efc,efs,ann_search_loops,begin_num,insert_points,batch_size,run_root\n' > "$PARAMS"
: > "$COMMANDS"

dataset_spec() {
  case "$1" in
    sift200k)
      printf '%s|%s|%s|%s|%s|%s\n' \
        "../data/sift/sift_base_200k.bin" "../data/sift/sift_query.bin" "../data/sift/sift_base_200k.gt20" \
        "180000" "20000" "sift"
      ;;
    sift1m)
      printf '%s|%s|%s|%s|%s|%s\n' \
        "../data/sift/sift_base.bin" "../data/sift/sift_query.bin" "../data/sift/sift_base.gt20" \
        "900000" "100000" "sift"
      ;;
    sift10m200k)
      printf '%s|%s|%s|%s|%s|%s\n' \
        "../data/sift10m/sift10m_base_200k.bin" "../data/sift10m/sift10m_query.bin" "" \
        "180000" "20000" "sift10m"
      ;;
    sift10m2200k)
      printf '%s|%s|%s|%s|%s|%s\n' \
        "../data/sift10m/sift10m_base_2200k.bin" "../data/sift10m/sift10m_query.bin" "" \
        "2000000" "200000" "sift10m"
      ;;
    dbpedia50k)
      printf '%s|%s|%s|%s|%s|%s\n' \
        "../data/dbpedia_openai/dbpedia_openai_50k.bin" "../data/dbpedia_openai/dbpedia_openai_query.bin" "" \
        "45000" "5000" "dbpedia_openai"
      ;;
    dbpedia1m)
      printf '%s|%s|%s|%s|%s|%s\n' \
        "../data/dbpedia_openai/dbpedia_openai_base.bin" "../data/dbpedia_openai/dbpedia_openai_query.bin" "" \
        "900000" "100000" "dbpedia_openai"
      ;;
    *)
      echo "unknown dataset spec: $1" >&2
      return 1
      ;;
  esac
}

config_spec() {
  local cfg="$1"
  if [[ "$cfg" =~ ^m([0-9]+)_efc([0-9]+)_efs([0-9]+)$ ]]; then
    printf '%s|%s|%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
  else
    echo "unknown config spec: $cfg" >&2
    return 1
  fi
}

case_spec() {
  case "$1" in
    background_search)
      printf '%s|%s|%s|%s\n' "Pure search background" "background_search" "" "0"
      ;;
    real_insert)
      printf '%s|%s|%s|%s\n' "Real insert request" "insert" "" "0"
      ;;
    ann_graph_reads_x1)
      printf '%s|%s|%s|%s\n' "ANN graph reads x1" "insert" "ann_search" "1"
      ;;
    ann_graph_reads_x4)
      printf '%s|%s|%s|%s\n' "ANN graph reads x4" "insert" "ann_search" "4"
      ;;
    ann_graph_reads_x8)
      printf '%s|%s|%s|%s\n' "ANN graph reads x8" "insert" "ann_search" "8"
      ;;
    query_graph_reads_x4)
      printf '%s|%s|%s|%s\n' "Query graph reads x4" "insert" "ann_search_query" "4"
      ;;
    insert_period_cpu)
      printf '%s|%s|%s|%s\n' "CPU work inside insert period" "insert" "cpu" "$INSERT_CPU_LOOPS"
      ;;
    graph_writes_query_touched)
      printf '%s|%s|%s|%s\n' "Graph writes to query-touched labels" "insert" "graph_write_hot" "$GRAPH_WRITE_LOOPS"
      ;;
    graph_writes_other)
      printf '%s|%s|%s|%s\n' "Graph writes to other labels" "insert" "graph_write_cold" "$GRAPH_WRITE_LOOPS"
      ;;
    *)
      echo "unknown case: $1" >&2
      return 1
      ;;
  esac
}

for dataset_spec_name in $DATASETS; do
  IFS='|' read -r data_path query_path gt_path begin_num insert_points dataset_name < <(dataset_spec "$dataset_spec_name")
  if [[ ! -f "$data_path" ]]; then
    echo "[m3-readwrite-screen] skip missing data: $dataset_spec_name $data_path" >&2
    continue
  fi
  if [[ ! -f "$query_path" ]]; then
    echo "[m3-readwrite-screen] skip missing query: $dataset_spec_name $query_path" >&2
    continue
  fi
  if [[ -n "$gt_path" && ! -f "$gt_path" ]]; then
    echo "[m3-readwrite-screen] gt missing, recall disabled: $dataset_spec_name $gt_path" >&2
    gt_path=""
  fi
  for cfg_name in $CONFIGS; do
    IFS='|' read -r m efc efs < <(config_spec "$cfg_name")
    config_dataset="${dataset_spec_name}_${cfg_name}"
    for competing_workers in $COMPETING_SET; do
      for case_key in $CASES; do
        IFS='|' read -r case_label competing_mode dummy_mode dummy_loops < <(case_spec "$case_key")
        run_name="${config_dataset}_cw${competing_workers}_${case_key}"
        case_root="$ROOT/$run_name"
        case_config="$CONFIG_ROOT/$run_name"
        paper_root="$DELIVER_ROOT/runs/$run_name"
        max_elements=$((begin_num + insert_points))
        mkdir -p "$case_root" "$case_config" "$paper_root"

        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
          "$config_dataset" "$case_key" "$case_label" "$competing_workers" "$m" "$efc" "$efs" \
          "$dummy_loops" "$begin_num" "$insert_points" "$BATCH_SIZE" "$case_root" >> "$PARAMS"

        printf 'ROOT=%q CONFIG_ROOT=%q PAPER_ROOT=%q DATASET_NAME=%q DATA_PATH=%q QUERY_PATH=%q GT_PATH=%q M=%q EF_SEARCH=%q EF_CONSTRUCTION=%q BEGIN_NUM=%q INSERT_POINTS=%q MAX_ELEMENTS=%q BATCH_SIZE=%q MEASURED_WORKERS=%q COMPETING_WORKERS=%q COMPETING_MODE=%q DURATION_MS=%q RUN_PERF=%q INTERVAL_MS=%q PERF_EVENTS=%q INSERT_DUMMY_MODE=%q INSERT_DUMMY_LOOPS=%q DISABLE_BUILD_STEP=1 M3_BATCH_DIAG=0 PHASE_OVERLAP_DIAG=0 REWRITE_ACTIVE_DIAG=0 SEARCH_WORK_COUNTERS=1 bash scripts/run_m3_closed_loop_odin_style.sh\n' \
          "$case_root" "$case_config" "$paper_root" "$dataset_name" "$data_path" "$query_path" "$gt_path" \
          "$m" "$efs" "$efc" "$begin_num" "$insert_points" "$max_elements" "$BATCH_SIZE" "$MEASURED_WORKERS" \
          "$competing_workers" "$competing_mode" "$DURATION_MS" "$RUN_PERF" "$INTERVAL_MS" "$PERF_EVENTS" \
          "$dummy_mode" "$dummy_loops" >> "$COMMANDS"

        env \
          ROOT="$case_root" \
          CONFIG_ROOT="$case_config" \
          PAPER_ROOT="$paper_root" \
          DATASET_NAME="$dataset_name" \
          DATA_PATH="$data_path" \
          QUERY_PATH="$query_path" \
          GT_PATH="$gt_path" \
          M="$m" \
          EF_SEARCH="$efs" \
          EF_CONSTRUCTION="$efc" \
          BEGIN_NUM="$begin_num" \
          INSERT_POINTS="$insert_points" \
          MAX_ELEMENTS="$max_elements" \
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
          DISABLE_BUILD_STEP=1 \
          M3_BATCH_DIAG=0 \
          PHASE_OVERLAP_DIAG=0 \
          REWRITE_ACTIVE_DIAG=0 \
          SEARCH_WORK_COUNTERS=1 \
          bash scripts/run_m3_closed_loop_odin_style.sh
      done
    done
  done
done

python3 ./scripts/analyze_m3_write_read_controls_20260504.py \
  --run-root "$ROOT" \
  --out-root "$DELIVER_ROOT" \
  --params "$PARAMS"

echo "[m3-readwrite-screen] done root=$ROOT deliver=$DELIVER_ROOT"
