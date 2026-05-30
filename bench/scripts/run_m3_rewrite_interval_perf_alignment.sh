#!/usr/bin/env bash
# Short interval-perf run for aligning cache pressure with search latency.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_rewrite_interval_perf_alignment_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_rewrite_interval_perf_alignment_${STAMP}}"
HIGH_WRITER_CONFIG_CASE="${HIGH_WRITER_CONFIG_CASE:-normal_insert40k}"
BASE_CONFIG="${BASE_CONFIG:-./config/m3_rewrite_period_preflight_20260426_b200/${HIGH_WRITER_CONFIG_CASE}/sift_b200_chasing_w40000_r40000_nolock.yaml}"
INTERVAL_MS="${INTERVAL_MS:-100}"
PERF_EVENTS="${PERF_EVENTS:-L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses}"

mkdir -p "$ROOT" "$CONFIG_ROOT"

if [[ ! -x ./bin/bench ]]; then
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .
fi

run_case() {
  local case_name="$1"
  local skip_old="$2"
  local out="${ROOT}/${case_name}"
  local cfg="${CONFIG_ROOT}/${case_name}.yaml"
  mkdir -p "$out/raw_latency"

  sed "s#output_dir: .*#output_dir: ${out}#" "$BASE_CONFIG" > "$cfg"

  echo "[m3-interval-perf] run case=${case_name} skip_old=${skip_old}"
  date +%s%N > "${out}/perf_start_unix_ns.txt"
  python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))' \
    > "${out}/perf_start_raw_ns.txt" 2>/dev/null || rm -f "${out}/perf_start_raw_ns.txt"
  env \
    ANNCHOR_M3_BATCH_DIAG=1 \
    ANNCHOR_SKIP_EXISTING_NODE_REPAIR="${skip_old}" \
    ANNCHOR_SKIP_EXISTING_NEIGHBOR_UPDATE="${skip_old}" \
    ANNCHOR_REWRITE_RECENT_WINDOW="${ANNCHOR_REWRITE_RECENT_WINDOW:-100000}" \
    ANNCHOR_INSERT_PHASE_TRACE="${out}/insert_phase_trace.csv" \
    OCC_RAW_LATENCY_DIR="${out}/raw_latency" \
    LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
    perf stat -I "${INTERVAL_MS}" -x, -e "${PERF_EVENTS}" \
      -o "${out}/perf_interval.csv" -- \
      ./bin/bench -config "$cfg" > "${out}.log" 2>&1

  python3 ./scripts/bin_m3_insert_phase_trace.py "${out}/insert_phase_trace.csv" \
    --search-wide "${out}/raw_latency/search_wide.csv" \
    --meta "${out}/raw_latency/meta.csv" \
    --bin-ms "${PERIOD_BIN_MS:-5}" \
    --out "${out}/periods.csv"
  python3 ./scripts/plot_m3_period_groups.py "$out" \
    --perf-events "${PERF_EVENTS}" \
    --out-csv "${out}/period_groups.csv" \
    --out-fig "${out}/period_groups.png"
}

run_case baseline 0
run_case no_old_rewire 1

python3 ./scripts/analyze_m3_batch_diag_rootcause.py "$ROOT" >/dev/null

echo "[m3-interval-perf] done root=${ROOT}"
