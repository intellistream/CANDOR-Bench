#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ROOT="${ROOT:-./result/20260504_m3_efc200_searchwork_b20_clean}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/20260504_m3_efc200_searchwork_b20_clean}"
DELIVER_ROOT="${DELIVER_ROOT:-../paper-progress/m3/search-tail-rootcause-2026-05-01/efc200_searchwork_20260504}"
MEASURED_WORKERS="${MEASURED_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-20}"
DURATION_MS="${DURATION_MS:-12000}"
EF_SEARCH="${EF_SEARCH:-35}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-200}"
BUILD_JOBS="${BUILD_JOBS:-8}"
INCLUDE_CW40="${INCLUDE_CW40:-0}"
ONLY_CW40="${ONLY_CW40:-0}"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$DELIVER_ROOT/tables" "$DELIVER_ROOT/figures" "$DELIVER_ROOT/runs"

cmake --build ./build --target index -j "$BUILD_JOBS"
LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .

COMMANDS="$DELIVER_ROOT/tables/run_commands.txt"
if [[ "$ONLY_CW40" == "1" ]]; then
  touch "$COMMANDS"
else
  : > "$COMMANDS"
fi

run_case() {
  local case_name="$1"
  local competing_workers="$2"
  shift 2
  local extra_env=("$@")
  local case_root="$ROOT/$case_name"
  local case_config="$CONFIG_ROOT/$case_name"
  local paper_root="$DELIVER_ROOT/runs/$case_name"
  mkdir -p "$case_root" "$case_config" "$paper_root"

  printf 'STAMP=20260504_m3_efc200_searchwork_%s ROOT=%q CONFIG_ROOT=%q PAPER_ROOT=%q MEASURED_WORKERS=%q COMPETING_WORKERS=%q EF_SEARCH=%q EF_CONSTRUCTION=%q BATCH_SIZE=%q DURATION_MS=%q RUN_PERF=0 DISABLE_BUILD_STEP=1 M3_BATCH_DIAG=0 PHASE_OVERLAP_DIAG=0 REWRITE_ACTIVE_DIAG=0 SEARCH_WORK_COUNTERS=1' \
    "$case_name" "$case_root" "$case_config" "$paper_root" "$MEASURED_WORKERS" "$competing_workers" "$EF_SEARCH" "$EF_CONSTRUCTION" "$BATCH_SIZE" "$DURATION_MS" >> "$COMMANDS"
  for kv in "${extra_env[@]}"; do
    printf ' %s' "$kv" >> "$COMMANDS"
  done
  printf ' bash scripts/run_m3_closed_loop_odin_style.sh\n' >> "$COMMANDS"

  env \
    STAMP="20260504_m3_efc200_searchwork_${case_name}" \
    ROOT="$case_root" \
    CONFIG_ROOT="$case_config" \
    PAPER_ROOT="$paper_root" \
    MEASURED_WORKERS="$MEASURED_WORKERS" \
    COMPETING_WORKERS="$competing_workers" \
    EF_SEARCH="$EF_SEARCH" \
    EF_CONSTRUCTION="$EF_CONSTRUCTION" \
    BATCH_SIZE="$BATCH_SIZE" \
    DURATION_MS="$DURATION_MS" \
    RUN_PERF=0 \
    DISABLE_BUILD_STEP=1 \
    M3_BATCH_DIAG=0 \
    PHASE_OVERLAP_DIAG=0 \
    REWRITE_ACTIVE_DIAG=0 \
    SEARCH_WORK_COUNTERS=1 \
    "${extra_env[@]}" \
    bash scripts/run_m3_closed_loop_odin_style.sh
}

if [[ "$ONLY_CW40" != "1" ]]; then
  run_case "cw32_neighbor_writes_on" 32
  run_case "cw32_neighbor_read_scan_no_writes" 32 EXISTING_UPDATE_READ_SCAN_ONLY=1
  run_case "cw32_undo_record_off" 32 DISABLE_EXISTING_UNDO_RECORD=1
fi

if [[ "$INCLUDE_CW40" == "1" || "$ONLY_CW40" == "1" ]]; then
  run_case "cw40_neighbor_writes_on" 40
  run_case "cw40_neighbor_read_scan_no_writes" 40 EXISTING_UPDATE_READ_SCAN_ONLY=1
  run_case "cw40_undo_record_off" 40 DISABLE_EXISTING_UNDO_RECORD=1
fi
