#!/usr/bin/env bash
# Run the M3 cache/phase control matrix on a shared producer schedule.
#
# Modes:
#   readonly_noop          producer/watermark path only, static graph
#   cpu_dummy              CPU-heavy dummy insert, no graph access
#   ann_search_dummy       insert-side ANN search only, no mutation
#   path_only_insert_control live insert path search + new-node links, existing-node repair removed
#   path_only_cpu_delay    repair-disabled insert plus calibrated CPU delay
#   full_insert            live HNSW insert
#   hot_filter             live HNSW insert with search-hot-region candidate filtering
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_cache_phase_controls_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_cache_phase_controls_${STAMP}}"
BASE_CONFIG_DIR="${BASE_CONFIG_DIR:-./config/m3_rewrite_period_preflight_20260426_b200}"
INTERVAL_MS="${INTERVAL_MS:-10}"
PERIOD_BIN_MS="${PERIOD_BIN_MS:-10}"
PERF_EVENTS="${PERF_EVENTS:-L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses,dtlb_load_misses.walk_completed}"
RUN_PERF="${RUN_PERF:-1}"
RUN_PHASE_TRACE="${RUN_PHASE_TRACE:-1}"
ANALYZE="${ANALYZE:-1}"
MODES="${MODES:-readonly_noop cpu_dummy ann_search_dummy path_only_insert_control full_insert hot_filter}"
LABEL_GLOB="${LABEL_GLOB:-sift_b200_*_w40000_r40000_nolock.yaml}"
MAX_CASES="${MAX_CASES:-0}"
REPEATS="${REPEATS:-1}"
DUMMY_CPU_LOOPS="${DUMMY_CPU_LOOPS:-20000}"
DUMMY_ANN_SEARCH_LOOPS="${DUMMY_ANN_SEARCH_LOOPS:-50}"
PATH_ONLY_POST_CPU_LOOPS="${PATH_ONLY_POST_CPU_LOOPS:-0}"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$ROOT/logs"

echo "[m3-cache-phase] root=$ROOT"
echo "[m3-cache-phase] base_config_dir=$BASE_CONFIG_DIR"
echo "[m3-cache-phase] modes=$MODES"
echo "[m3-cache-phase] label_glob=$LABEL_GLOB repeats=$REPEATS interval_ms=$INTERVAL_MS period_bin_ms=$PERIOD_BIN_MS events=$PERF_EVENTS run_perf=$RUN_PERF run_phase_trace=$RUN_PHASE_TRACE analyze=$ANALYZE"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  cmake --build ./build --target index -j "${BUILD_JOBS:-4}"
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .
fi

ensure_trace_header() {
  local trace="$1"
  if [[ ! -s "$trace" ]]; then
    cat > "$trace" <<'EOF'
start_ns,end_ns,tid_hash,phase,point_id,level,existing_neighbor,existing_neighbor_degree_before,edges_loaded,prune_candidates,dist_comps,edges_written,edges_pruned
EOF
  fi
}

patch_config() {
  local src="$1"
  local dst="$2"
  local out="$3"
  mkdir -p "$(dirname "$dst")"
  python3 - "$src" "$dst" "$out" <<'PY'
import sys
src, dst, out = sys.argv[1:4]
text = open(src, encoding="utf-8").read().splitlines()
seen_output = False
seen_precompute = False
with open(dst, "w", encoding="utf-8") as f:
    for line in text:
        stripped = line.strip()
        if stripped.startswith("output_dir:"):
            indent = line[:len(line) - len(line.lstrip())]
            f.write(f"{indent}output_dir: {out}\n")
            seen_output = True
        elif stripped.startswith("precompute_schedule:"):
            indent = line[:len(line) - len(line.lstrip())]
            f.write(f"{indent}precompute_schedule: true\n")
            seen_precompute = True
        else:
            f.write(line + "\n")
    if not seen_precompute:
        f.write("  precompute_schedule: true\n")
    if not seen_output:
        f.write(f"result:\n  output_dir: {out}\n")
PY
}

run_one() {
  local mode="$1"
  local base_cfg="$2"
  local rep="$3"
  local rel="${base_cfg#${BASE_CONFIG_DIR}/}"
  local source_case_dir
  source_case_dir="$(dirname "$rel")"
  local label
  label="$(basename "$rel" .yaml)"
  local concurrency_label="shared_worker_pool"
  local case_dir="${concurrency_label}"
  if [[ "$REPEATS" -gt 1 ]]; then
    case_dir="${case_dir}_rep$(printf '%02d' "$rep")"
  fi

  # Use the source config family only for workload parameters; modes control the mutation path.
  if [[ "$source_case_dir" == no_old_rewire* ]]; then
    return 0
  fi

  local out="${ROOT}/${mode}/${case_dir}/${label}"
  local cfg="${CONFIG_ROOT}/${mode}/${case_dir}/${label}.yaml"
  mkdir -p "$out/raw_latency"
  patch_config "$base_cfg" "$cfg" "$out"

  local dummy_mode=""
  local dummy_loops=""
  local post_cpu_loops=""
  local skip_existing=0
  local hot_repair=0
  local hot_filter=0
  case "$mode" in
    readonly_noop)
      dummy_mode="noop"
      ;;
    cpu_dummy)
      dummy_mode="cpu"
      dummy_loops="$DUMMY_CPU_LOOPS"
      ;;
    ann_search_dummy)
      dummy_mode="ann_search"
      dummy_loops="$DUMMY_ANN_SEARCH_LOOPS"
      ;;
    path_only_insert_control)
      skip_existing=1
      ;;
    path_only_cpu_delay)
      skip_existing=1
      post_cpu_loops="$PATH_ONLY_POST_CPU_LOOPS"
      ;;
    full_insert)
      ;;
    hot_filter)
      hot_repair=1
      hot_filter=1
      ;;
    *)
      echo "[m3-cache-phase] unknown mode: $mode" >&2
      return 2
      ;;
  esac

  echo "[m3-cache-phase] run mode=${mode} case=${case_dir} label=${label} rep=${rep} writer_concurrency=${concurrency_label}"
  date +%s%N > "${out}/perf_start_unix_ns.txt"
  python3 -c 'import time; print(time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))' \
    > "${out}/perf_start_raw_ns.txt" 2>/dev/null || rm -f "${out}/perf_start_raw_ns.txt"

  local trace_path="${out}/insert_phase_trace.csv"
  local trace_env=""
  if [[ "$RUN_PHASE_TRACE" == "1" ]]; then
    trace_env="$trace_path"
  fi

  local bench_cmd=(./bin/bench -config "$cfg")
  local env_cmd=(
    env
    ANNCHOR_M3_BATCH_DIAG=1 \
    ANNCHOR_INSERT_PHASE_TRACE="${trace_env}" \
    ANNCHOR_SKIP_EXISTING_NODE_REPAIR="${skip_existing}" \
    ANNCHOR_SKIP_EXISTING_NEIGHBOR_UPDATE="${skip_existing}" \
    ANNCHOR_HOT_REGION_REPAIR="${hot_repair}" \
    ANNCHOR_HOT_REGION_FILTER_CANDIDATES="${hot_filter}" \
    ANNCHOR_HOT_REGION_REORDER_SELECTED=0 \
    ANNCHOR_HOT_REGION_FILTER_MIN_COLD=0 \
    ANNCHOR_HOT_REGION_PRESERVE_NEAREST=0 \
    ANNCHOR_HOT_REGION_SHIFT=16 \
    ANNCHOR_HOT_REGION_TTL_EPOCHS=4096 \
    ANNCHOR_HOT_REGION_TABLE_BITS=20 \
    INSERT_DUMMY_MODE="${dummy_mode}" \
    INSERT_DUMMY_LOOPS="${dummy_loops}" \
    INSERT_POST_CPU_LOOPS="${post_cpu_loops}" \
    OCC_RAW_LATENCY_DIR="${out}/raw_latency" \
    LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}"
  )
  if [[ "$RUN_PERF" == "1" ]]; then
    "${env_cmd[@]}" perf stat -I "${INTERVAL_MS}" -x, -e "${PERF_EVENTS}" \
      -o "${out}/perf_interval.csv" -- "${bench_cmd[@]}" > "${out}/bench.log" 2>&1
  else
    "${env_cmd[@]}" "${bench_cmd[@]}" > "${out}/bench.log" 2>&1
  fi

  if [[ "$RUN_PHASE_TRACE" == "1" ]]; then
    ensure_trace_header "$trace_path"
    python3 ./scripts/bin_m3_insert_phase_trace.py "$trace_path" \
      --search-wide "${out}/raw_latency/search_wide.csv" \
      --meta "${out}/raw_latency/meta.csv" \
      --bin-ms "${PERIOD_BIN_MS}" \
      --out "${out}/periods.csv"

    python3 ./scripts/plot_m3_period_groups.py "$out" \
      --perf-events "${PERF_EVENTS}" \
      --out-csv "${out}/period_groups.csv" \
      --out-fig "${out}/period_groups.png" \
      | tee "${out}/period_groups.log"
  fi
}

mapfile -t CONFIGS < <(find "$BASE_CONFIG_DIR" -path "*/normal_*/*" -name "$LABEL_GLOB" | sort)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  mapfile -t CONFIGS < <(find "$BASE_CONFIG_DIR" -type f -name "$LABEL_GLOB" | sort)
fi
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "[m3-cache-phase] no configs matched under $BASE_CONFIG_DIR with $LABEL_GLOB" >&2
  exit 2
fi
if [[ "$MAX_CASES" -gt 0 && "${#CONFIGS[@]}" -gt "$MAX_CASES" ]]; then
  CONFIGS=("${CONFIGS[@]:0:${MAX_CASES}}")
fi

for cfg in "${CONFIGS[@]}"; do
  for rep in $(seq 1 "$REPEATS"); do
    for mode in $MODES; do
      run_one "$mode" "$cfg" "$rep"
    done
  done
done

if [[ "$ANALYZE" == "1" ]]; then
  python3 ./scripts/analyze_m3_cache_phase_controls.py "$ROOT"
fi

echo "[m3-cache-phase] done root=${ROOT}"
