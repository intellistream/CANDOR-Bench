#!/usr/bin/env bash
# Smoke-test deferred inflight OCC:
#   search first on the committed MVCC view, yield while the active insert batch
#   finishes, then batch-merge the query batch with that insert batch.
set -euo pipefail

cd "$(dirname "$0")/.."

BENCH_BIN="./bin/bench"
CONFIG_DIR="./config/deferred_occ_smoke"
RESULT_BASE="./result/deferred_occ_smoke"
THREADS="${THREADS:-16}"
BEGIN_NUM="${BEGIN_NUM:-50000}"
MAX_ELEMENTS="${MAX_ELEMENTS:-70000}"
EF_SEARCH="${EF_SEARCH:-35}"
EF_CONSTRUCTION="${EF_CONSTRUCTION:-200}"
M="${M:-16}"
DEFER_MAX_WAIT_US="${DEFER_MAX_WAIT_US:--1}"

DATASETS=(
  "sift|../data/sift/sift_base.bin|../data/sift/sift_query_stream.bin|../data/sift/sift_stream_i100_offset_index.txt|../data/sift/sift_query.bin|../data/sift/sift_base.gt20"
  "deep1M|../data/deep1M/deep1M_base.bin|../data/deep1M/deep1M_query_stream.bin|../data/deep1M/deep1M_stream_i100_offset_index.txt|../data/deep1M/deep1M_query.bin|../data/deep1M/deep1M_base.gt20"
  "msong|../data/msong/msong_base.bin|../data/msong/msong_query_stream.bin|../data/msong/msong_stream_i100_offset_index.txt|../data/msong/msong_query.bin|../data/msong/msong_base.gt20"
)
BATCHES=(200 400)
RATES=("40000 40000" "90000 10000")
MODES=("immediate" "deferred")

mkdir -p "$CONFIG_DIR" "$RESULT_BASE"

if ! command -v go >/dev/null 2>&1; then
  echo "[deferred-occ] go not found; cannot rebuild bench with deferred OCC changes" >&2
  exit 1
fi

echo "[deferred-occ] building bench binary"
LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o "$BENCH_BIN" .

generate_config() {
  local ds="$1" data="$2" incr_q="$3" incr_gt="$4" overall_q="$5" overall_gt="$6"
  local batch="$7" wrate="$8" rrate="$9" mode="${10}" out_dir="${11}"
  local defer_flag="false"
  if [[ "$mode" == "deferred" ]]; then
    defer_flag="true"
  fi

  cat <<YAML
data:
  dataset_name: ${ds}
  max_elements: ${MAX_ELEMENTS}
  begin_num: ${BEGIN_NUM}
  data_type: float
  data_path: ${data}
  incr_query_path: ${incr_q}
  incr_gt_path: ${incr_gt}
  overall_query_path: ${overall_q}
  overall_gt_path: ${overall_gt}
index:
  index_type: annchor-preempt
  m: ${M}
  ef_construction: ${EF_CONSTRUCTION}
search:
  recall_at: 10
  ef_search: ${EF_SEARCH}
  enable_mvcc: true
  enable_inflight_bruteforce: true
  defer_inflight_occ_after_insert: ${defer_flag}
  defer_inflight_occ_max_wait_us: ${DEFER_MAX_WAIT_US}
workload:
  batch_size: [${batch}]
  num_threads: [${THREADS}]
  queue_size: 0
  rate_groups(r/w):
    - [${wrate}, ${rrate}]
  with_external_rw_lock: false
  query_mode: chasing
compare:
  enabled: false
  simulate:
    enabled: false
result:
  output_dir: ${out_dir}
profile:
  memory_monitor_interval: 100
  enable_memory_profile: true
YAML
}

for row in "${DATASETS[@]}"; do
  IFS='|' read -r ds data incr_q incr_gt overall_q overall_gt <<< "$row"
  if [[ ! -f "$data" || ! -f "$incr_q" || ! -f "$overall_q" ]]; then
    echo "[deferred-occ] skip $ds: missing data/query files"
    continue
  fi
  for batch in "${BATCHES[@]}"; do
    for rate in "${RATES[@]}"; do
      read -r wrate rrate <<< "$rate"
      for mode in "${MODES[@]}"; do
        label="${ds}_b${batch}_w${wrate}_r${rrate}_${mode}"
        out_dir="${RESULT_BASE}/${label}"
        cfg="${CONFIG_DIR}/${label}.yaml"
        generate_config "$ds" "$data" "$incr_q" "$incr_gt" "$overall_q" "$overall_gt" \
          "$batch" "$wrate" "$rrate" "$mode" "$out_dir" > "$cfg"
        echo "[deferred-occ] run $label"
        LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} "$BENCH_BIN" -config "$cfg" \
          2>&1 | tee "${out_dir}.log" | grep -E "Progress.*100%|Overall recall|Results written|Error|panic" || true
      done
    done
  done
done

echo "[deferred-occ] results: $RESULT_BASE"
