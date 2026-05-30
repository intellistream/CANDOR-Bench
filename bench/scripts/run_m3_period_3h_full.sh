#!/usr/bin/env bash
# Full 3H-period experiment for the actual HNSW/ANNchor insert path:
# insert work, perf/cache pressure, and search latency on the same raw-clock
# period bins. Existing-neighbor-update ablation is excluded by default.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_period_3h_full_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_period_3h_full_${STAMP}}"
BASE_CONFIG_DIR="${BASE_CONFIG_DIR:-./config/m3_rewrite_period_preflight_20260426_b200}"
INTERVAL_MS="${INTERVAL_MS:-10}"
PERIOD_BIN_MS="${PERIOD_BIN_MS:-10}"
PERF_EVENTS="${PERF_EVENTS:-L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses}"
SEARCH_WORKERS="${SEARCH_WORKERS:-1}"
INCLUDE_ABLATION="${INCLUDE_ABLATION:-0}"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$ROOT/logs"

echo "[m3-3h-full] root=$ROOT"
echo "[m3-3h-full] base_config_dir=$BASE_CONFIG_DIR"
echo "[m3-3h-full] interval_ms=$INTERVAL_MS period_bin_ms=$PERIOD_BIN_MS events=$PERF_EVENTS"
echo "[m3-3h-full] include_ablation=$INCLUDE_ABLATION"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  cmake --build ./build --target index -j "${BUILD_JOBS:-4}"
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .
fi

run_one() {
  local base_cfg="$1"
  local rel="${base_cfg#${BASE_CONFIG_DIR}/}"
  local case_dir
  case_dir="$(dirname "$rel")"
  local label
  label="$(basename "$rel" .yaml)"
  local output_case_dir="$case_dir"
  local skip_existing=0
  local insert_workers=""

  if [[ "$case_dir" == no_old_rewire* ]]; then
    if [[ "$INCLUDE_ABLATION" != "1" ]]; then
      echo "[m3-3h-full] omit ablation config: ${case_dir}/${label}"
      return 0
    fi
    skip_existing=1
    output_case_dir="${case_dir/no_old_rewire/ablation_no_existing_neighbor_update}"
  elif [[ "$case_dir" == baseline* ]]; then
    output_case_dir="${case_dir/baseline/actual_hnsw_insert}"
  fi
  if [[ "$case_dir" =~ _iw([0-9]+)$ ]]; then
    insert_workers="${BASH_REMATCH[1]}"
  fi

  local out="${ROOT}/${output_case_dir}/${label}"
  local cfg="${CONFIG_ROOT}/${output_case_dir}/${label}.yaml"

  mkdir -p "$out/raw_latency" "$(dirname "$cfg")"
  sed "s#output_dir: .*#output_dir: ${out}#" "$base_cfg" > "$cfg"

  echo "[m3-3h-full] run case=${output_case_dir} label=${label} mode=actual_hnsw_insert skip_existing_neighbor_update=${skip_existing} insert_workers=${insert_workers:-config-default}"
  date +%s%N > "${out}/perf_start_unix_ns.txt"
  python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))' \
    > "${out}/perf_start_raw_ns.txt" 2>/dev/null || rm -f "${out}/perf_start_raw_ns.txt"

  env \
    ANNCHOR_M3_BATCH_DIAG=1 \
    ANNCHOR_SKIP_EXISTING_NODE_REPAIR="${skip_existing}" \
    ANNCHOR_SKIP_EXISTING_NEIGHBOR_UPDATE="${skip_existing}" \
    ANNCHOR_REWRITE_RECENT_WINDOW="${ANNCHOR_REWRITE_RECENT_WINDOW:-100000}" \
    ANNCHOR_INSERT_PHASE_TRACE="${out}/insert_phase_trace.csv" \
    OCC_RAW_LATENCY_DIR="${out}/raw_latency" \
    INSERT_WORKERS="${insert_workers}" \
    SEARCH_WORKERS="${SEARCH_WORKERS}" \
    LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}" \
    perf stat -I "${INTERVAL_MS}" -x, -e "${PERF_EVENTS}" \
      -o "${out}/perf_interval.csv" -- \
      ./bin/bench -config "$cfg" > "${out}/bench.log" 2>&1

  python3 ./scripts/bin_m3_insert_phase_trace.py "${out}/insert_phase_trace.csv" \
    --search-wide "${out}/raw_latency/search_wide.csv" \
    --meta "${out}/raw_latency/meta.csv" \
    --bin-ms "${PERIOD_BIN_MS}" \
    --out "${out}/periods.csv"

  python3 ./scripts/plot_m3_period_groups.py "$out" \
    --perf-events "${PERF_EVENTS}" \
    --out-csv "${out}/period_groups.csv" \
    --out-fig "${out}/period_groups.png" \
    | tee "${out}/period_groups.log"
}

mapfile -t CONFIGS < <(find "$BASE_CONFIG_DIR" -type f -name '*.yaml' | sort)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "[m3-3h-full] no configs found under $BASE_CONFIG_DIR" >&2
  exit 2
fi

for cfg in "${CONFIGS[@]}"; do
  run_one "$cfg"
done

export ROOT
python3 - <<'PY'
import csv
import pathlib
import os

root = pathlib.Path(os.environ["ROOT"])
rows = []
for path in sorted(root.glob("*/*/period_groups.csv")):
    case = path.parents[1].name
    label = path.parent.name
    periods = list(csv.DictReader(path.open()))
    if not periods:
        continue
    def f(row, key):
        try:
            return float(row.get(key, 0) or 0)
        except ValueError:
            return 0.0
    triple = [r for r in periods if f(r, "triple_high") > 0]
    rows.append({
        "case": case,
        "label": label,
        "periods": len(periods),
        "triple_high_periods": len(triple),
        "triple_high_rate": len(triple) / len(periods),
        "high_insert_periods": sum(1 for r in periods if f(r, "high_insert_work") > 0),
        "high_cache_periods": sum(1 for r in periods if f(r, "high_cache_miss") > 0),
        "high_search_periods": sum(1 for r in periods if f(r, "high_search_latency") > 0),
        "max_insert_update_active_workers": max(f(r, "existing_neighbor_update_active_workers") for r in periods),
        "max_prune_candidates": max(f(r, "existing_neighbor_prune_candidates") for r in periods),
        "max_search_e2e_ms": max(f(r, "search_finish_e2e_ms_max") for r in periods),
        "max_l2_miss_per_s": max(f(r, "perf_l2_rqsts_miss_per_s") for r in periods),
        "max_llc_miss_per_s": max(f(r, "perf_llc_load_misses_per_s") for r in periods),
    })

out = root / "period_3h_summary.csv"
if rows:
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(f"[m3-3h-full] wrote {out}")
PY

echo "[m3-3h-full] done root=${ROOT}"
