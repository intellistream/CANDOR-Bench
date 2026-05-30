#!/usr/bin/env bash
# Root-cause matrix for search batch-E2E fluctuation under concurrent insert.
#
# The matrix intentionally keeps OCC disabled. It varies only synchronization
# mode, batch size, workload mix, and query mode so latency spikes can be
# attributed to writer/search interference rather than inflight merge work.
set -euo pipefail

cd "$(dirname "$0")/.."

BENCH_BIN="${BENCH_BIN:-./bin/bench}"
CONFIG_DIR="${CONFIG_DIR:-./config/m3_fluctuation_rootcause}"
RESULT_BASE="${RESULT_BASE:-./result/m3_fluctuation_rootcause}"
THREADS="${THREADS:-64}"
BEGIN_NUM="${BEGIN_NUM:-50000}"
MAX_ELEMENTS="${MAX_ELEMENTS:--1}"
EF_SEARCH="${EF_SEARCH:-35}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-200}"
M="${M:-16}"
REPETITIONS="${REPETITIONS:-1}"
RUN_PERF="${RUN_PERF:-0}"
PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,cache-misses,LLC-load-misses,context-switches,cpu-migrations}"
ENABLE_UNDO_RECOVERY="${ENABLE_UNDO_RECOVERY:-}"
ENABLE_MVCC="${ENABLE_MVCC:-true}"
PRECOMPUTE_SCHEDULE="${PRECOMPUTE_SCHEDULE:-false}"
INDEX_TYPE="${INDEX_TYPE:-annchor-m1}"

DATASETS=(${DATASETS:-sift})
BATCHES=(${BATCHES:-20 200 400})
if [[ -n "${RATE_PAIRS:-}" ]]; then
  IFS=';' read -r -a RATES <<< "$RATE_PAIRS"
else
  RATES=("40000 40000" "10000 90000" "90000 10000")
fi
QUERY_MODES=(${QUERY_MODES:-round_robin chasing})
SYNC_MODES=(${SYNC_MODES:-nodelock nolock rwlock})

mkdir -p "$CONFIG_DIR" "$RESULT_BASE" ./bin
export LD_LIBRARY_PATH="./build/lib:${LD_LIBRARY_PATH:-}"

needs_go_build=0
if [[ ! -x "$BENCH_BIN" ]]; then
  needs_go_build=1
elif find internal -type f -name '*.go' -newer "$BENCH_BIN" | grep -q .; then
  needs_go_build=1
fi

if [[ "$needs_go_build" == "1" ]]; then
  if command -v go >/dev/null 2>&1; then
    echo "[m3-rootcause] rebuilding $BENCH_BIN"
    LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o "$BENCH_BIN" .
  else
    echo "[m3-rootcause] $BENCH_BIN is missing or stale, and go is not available" >&2
    echo "  cd bench && LD_LIBRARY_PATH=./build/lib go build -o $BENCH_BIN ." >&2
    exit 1
  fi
fi

dataset_paths() {
  case "$1" in
    sift)
      echo "sift|../data/sift/sift_base.bin|../data/sift/sift_query_stream.bin|../data/sift/sift_query.bin|../data/sift/sift_base.gt20"
      ;;
    sift10m)
      echo "sift10m|../data/sift10m/sift10m_base.bin|../data/sift10m/sift10m_query_stream_200k.bin|../data/sift10m/sift10m_query.bin|"
      ;;
    dbpedia_openai)
      echo "dbpedia_openai|../data/dbpedia_openai/dbpedia_openai_base.bin|../data/dbpedia_openai/dbpedia_openai_query.bin|../data/dbpedia_openai/dbpedia_openai_query.bin|"
      ;;
    deep1M)
      echo "deep1M|../data/deep1M/deep1M_base.bin|../data/deep1M/deep1M_query_stream.bin|../data/deep1M/deep1M_query.bin|../data/deep1M/deep1M_base.gt20"
      ;;
    msong)
      echo "msong|../data/msong/msong_base.bin|../data/msong/msong_query_stream.bin|../data/msong/msong_query.bin|../data/msong/msong_base.gt20"
      ;;
    gist)
      echo "gist|../data/gist/gist_base.bin|../data/gist/gist_query_stream.bin|../data/gist/gist_query.bin|../data/gist/gist_base.gt20"
      ;;
    *)
      echo "[m3-rootcause] unknown dataset $1" >&2
      return 1
      ;;
  esac
}

generate_config() {
  local ds="$1" data="$2" incr_q="$3" overall_q="$4" overall_gt="$5"
  local batch="$6" wrate="$7" rrate="$8" qmode="$9" sync_mode="${10}" out_dir="${11}"
  local use_node_lock="true"
  local external_rw="false"

  case "$sync_mode" in
    nodelock)
      use_node_lock="true"
      external_rw="false"
      ;;
    nolock)
      use_node_lock="false"
      external_rw="false"
      ;;
    rwlock)
      # This serializes search and insert externally, so internal node locks are
      # disabled by EffectiveUseNodeLock.
      use_node_lock="false"
      external_rw="true"
      ;;
    *)
      echo "unknown sync mode: $sync_mode" >&2
      exit 1
      ;;
  esac

  local gt_line=""
  if [[ -n "$overall_gt" ]]; then
    gt_line="  overall_gt_path: ${overall_gt}"
  fi

  cat <<YAML
data:
  dataset_name: ${ds}
  max_elements: ${MAX_ELEMENTS}
  begin_num: ${BEGIN_NUM}
  data_type: float
  data_path: ${data}
  incr_query_path: ${incr_q}
  overall_query_path: ${overall_q}
${gt_line}
index:
  index_type: ${INDEX_TYPE}
  m: ${M}
  ef_construction: ${EF_CONSTRUCTION}
search:
  recall_at: 10
  ef_search: ${EF_SEARCH}
  use_node_lock: ${use_node_lock}
  enable_mvcc: ${ENABLE_MVCC}
$(if [[ -n "$ENABLE_UNDO_RECOVERY" ]]; then echo "  enable_undo_recovery: ${ENABLE_UNDO_RECOVERY}"; fi)
  enable_inflight_bruteforce: false
workload:
  batch_size: ${batch}
  num_threads: ${THREADS}
  queue_size: 0
  rate_groups(r/w):
    - [${wrate}, ${rrate}]
  with_external_rw_lock: ${external_rw}
  query_mode: ${qmode}
  per_query_latency: true
  precompute_schedule: ${PRECOMPUTE_SCHEDULE}
compare:
  enabled: false
  simulate:
    enabled: false
result:
  output_dir: ${out_dir}
repetitions: ${REPETITIONS}
profile:
  memory_monitor_interval: 100
  enable_memory_profile: false
YAML
}

run_one() {
  local label="$1" cfg="$2" out_dir="$3"
  mkdir -p "$out_dir/raw_latency"
  echo "[m3-rootcause] run $label"
  if [[ "$RUN_PERF" == "1" && -x "$(command -v perf || true)" ]]; then
    OCC_RAW_LATENCY_DIR="$out_dir/raw_latency" \
      perf stat -e "$PERF_EVENTS" \
      -o "$out_dir/perf_stat.txt" -- "$BENCH_BIN" -config "$cfg" > "${out_dir}.log" 2>&1
  else
    OCC_RAW_LATENCY_DIR="$out_dir/raw_latency" "$BENCH_BIN" -config "$cfg" > "${out_dir}.log" 2>&1
  fi
}

count=0
for ds_name in "${DATASETS[@]}"; do
  IFS='|' read -r ds data incr_q overall_q overall_gt <<< "$(dataset_paths "$ds_name")"
  if [[ ! -f "$data" || ! -f "$incr_q" || ! -f "$overall_q" ]]; then
    echo "[m3-rootcause] skip $ds: missing data/query files"
    continue
  fi
  for batch in "${BATCHES[@]}"; do
    for rate in "${RATES[@]}"; do
      read -r wrate rrate <<< "$rate"
      for qmode in "${QUERY_MODES[@]}"; do
        for sync_mode in "${SYNC_MODES[@]}"; do
          label="${ds}_b${batch}_${qmode}_w${wrate}_r${rrate}_${sync_mode}"
          out_dir="${RESULT_BASE}/${label}"
          cfg="${CONFIG_DIR}/${label}.yaml"
          generate_config "$ds" "$data" "$incr_q" "$overall_q" "$overall_gt" \
            "$batch" "$wrate" "$rrate" "$qmode" "$sync_mode" "$out_dir" > "$cfg"
          run_one "$label" "$cfg" "$out_dir"
          count=$((count + 1))
        done
      done
    done
  done
done

python3 ./scripts/analyze_m3_fluctuation_rootcause.py "$RESULT_BASE" | tee "$RESULT_BASE/summary.txt"
echo "[m3-rootcause] done runs=$count configs=$CONFIG_DIR results=$RESULT_BASE"
