#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT/bench"
STAMP="$(date +%Y%m%d_%H%M%S)"

OUT_ROOT="${OUT_ROOT:-$ROOT/results/annchor_occ_freshness_$STAMP}"
TMP_DIR="${TMP_DIR:-$OUT_ROOT/configs}"

DATASETS="${DATASETS:-sift,msong,deep1M,gist}"
BATCH_SIZES="${BATCH_SIZES:-20,100,200,400}"
TOPKS="${TOPKS:-10}"
THREADS="${THREADS:-16}"
WARMUP="${WARMUP:-50000}"
MAX_ELEMENTS="${MAX_ELEMENTS:--1}"
EF_SEARCH="${EF_SEARCH:-35}"
RATE_GROUPS="${RATE_GROUPS:-90000:10000,40000:40000,10000:90000}"
QUERY_MODES="${QUERY_MODES:-auto}"
ZIPFIAN_SKEW="${ZIPFIAN_SKEW:-0.99}"
PRECOMPUTE_SCHEDULE="${PRECOMPUTE_SCHEDULE:-false}"
ENABLE_RECALL="${ENABLE_RECALL:-false}"
GO_BIN="${GO_BIN:-go}"

if ! command -v "$GO_BIN" >/dev/null 2>&1; then
  if [[ -x /usr/local/go/bin/go ]]; then
    GO_BIN="/usr/local/go/bin/go"
  else
    echo "go binary not found; set GO_BIN=/path/to/go" >&2
    exit 2
  fi
fi

export LD_LIBRARY_PATH="$BENCH_DIR/build/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$TMP_DIR"

csv_list() {
  local raw="$1"
  local out="["
  local first=1
  IFS=',' read -ra vals <<< "$raw"
  for v in "${vals[@]}"; do
    v="${v// /}"
    [[ -z "$v" ]] && continue
    if [[ "$first" -eq 0 ]]; then
      out+=", "
    fi
    out+="$v"
    first=0
  done
  out+="]"
  printf '%s' "$out"
}

csv_string_list() {
  local raw="$1"
  local out="["
  local first=1
  IFS=',' read -ra vals <<< "$raw"
  for v in "${vals[@]}"; do
    v="${v// /}"
    [[ -z "$v" ]] && continue
    if [[ "$first" -eq 0 ]]; then
      out+=", "
    fi
    out+="\"$v\""
    first=0
  done
  out+="]"
  printf '%s' "$out"
}

write_rate_groups() {
  local raw="$1"
  IFS=',' read -ra groups <<< "$raw"
  for g in "${groups[@]}"; do
    g="${g// /}"
    [[ -z "$g" ]] && continue
    local read_rate="${g%%:*}"
    local write_rate="${g##*:}"
    echo "    - [$read_rate, $write_rate]"
  done
}

dataset_paths() {
  local ds="$1"
  case "$ds" in
    sift)
      DATA_PATH="../data/sift/sift_base.bin"
      INCR_QUERY_PATH="../data/sift/sift_query_stream.bin"
      OVERALL_QUERY_PATH="../data/sift/sift_query.bin"
      if [[ "$ENABLE_RECALL" == "true" || "$ENABLE_RECALL" == "1" ]]; then
        OVERALL_GT_PATH="../data/sift/sift_base.gt20"
      else
        OVERALL_GT_PATH=""
      fi
      DEFAULT_QUERY_MODES="round_robin,chasing,peeking,zipfian"
      ;;
    sift10m)
      DATA_PATH="../data/sift10m/sift10m_base.bin"
      INCR_QUERY_PATH="../data/sift10m/sift10m_query_stream_200k.bin"
      OVERALL_QUERY_PATH="../data/sift10m/sift10m_query.bin"
      OVERALL_GT_PATH=""
      DEFAULT_QUERY_MODES="round_robin,chasing,peeking,zipfian"
      ;;
    deep1M|deep)
      DATA_PATH="../data/deep1M/deep1M_base.bin"
      INCR_QUERY_PATH="../data/deep1M/deep1M_query_stream.bin"
      OVERALL_QUERY_PATH="../data/deep1M/deep1M_query.bin"
      if [[ "$ENABLE_RECALL" == "true" || "$ENABLE_RECALL" == "1" ]]; then
        OVERALL_GT_PATH="../data/deep1M/deep1M_base.gt20"
      else
        OVERALL_GT_PATH=""
      fi
      DEFAULT_QUERY_MODES="round_robin,chasing,peeking,zipfian"
      ;;
    gist)
      DATA_PATH="../data/gist/gist_base.bin"
      INCR_QUERY_PATH="../data/gist/gist_query_stream.bin"
      OVERALL_QUERY_PATH="../data/gist/gist_query.bin"
      if [[ "$ENABLE_RECALL" == "true" || "$ENABLE_RECALL" == "1" ]]; then
        OVERALL_GT_PATH="../data/gist/gist_base.gt20"
      else
        OVERALL_GT_PATH=""
      fi
      DEFAULT_QUERY_MODES="round_robin,chasing,peeking,zipfian"
      ;;
    msong)
      DATA_PATH="../data/msong/msong_base.bin"
      INCR_QUERY_PATH="../data/msong/msong_query_stream.bin"
      OVERALL_QUERY_PATH="../data/msong/msong_query.bin"
      if [[ "$ENABLE_RECALL" == "true" || "$ENABLE_RECALL" == "1" ]]; then
        OVERALL_GT_PATH="../data/msong/msong_base.gt20"
      else
        OVERALL_GT_PATH=""
      fi
      DEFAULT_QUERY_MODES="round_robin,chasing,peeking,zipfian"
      ;;
    dbpedia_openai)
      DATA_PATH="../data/dbpedia_openai/dbpedia_openai_base.bin"
      INCR_QUERY_PATH="../data/dbpedia_openai/dbpedia_openai_query.bin"
      OVERALL_QUERY_PATH="../data/dbpedia_openai/dbpedia_openai_query.bin"
      OVERALL_GT_PATH=""
      DEFAULT_QUERY_MODES="round_robin"
      ;;
    *)
      echo "unknown dataset: $ds" >&2
      exit 2
      ;;
  esac
}

run_one_dataset() {
  local ds="$1"
  dataset_paths "$ds"

  local modes="$QUERY_MODES"
  if [[ "$modes" == "auto" ]]; then
    modes="$DEFAULT_QUERY_MODES"
  fi

  local cfg="$TMP_DIR/${ds}_occ_freshness.yaml"
  local out_dir="$OUT_ROOT/$ds"
  mkdir -p "$out_dir"

  cat > "$cfg" <<EOF_CFG
data:
  dataset_name: $ds
  max_elements: $MAX_ELEMENTS
  begin_num: $WARMUP
  data_type: float
  data_path: $DATA_PATH
  incr_query_path: $INCR_QUERY_PATH
  overall_query_path: $OVERALL_QUERY_PATH
  overall_gt_path: $OVERALL_GT_PATH

index:
  index_type: annchor-m1
  m: 16
  ef_construction: 200

search:
  recall_at: $(csv_list "$TOPKS")
  ef_search: $EF_SEARCH
  enable_mvcc: true
  enable_inflight_bruteforce: true

workload:
  batch_size: $(csv_list "$BATCH_SIZES")
  num_threads: $(csv_list "$THREADS")
  queue_size: 0
  rate_groups(r/w):
$(write_rate_groups "$RATE_GROUPS")
  with_external_rw_lock: false
  query_mode: $(csv_string_list "$modes")
  zipfian_skew: $ZIPFIAN_SKEW
  per_query_latency: true
  precompute_schedule: $PRECOMPUTE_SCHEDULE

compare:
  enabled: false
  simulate:
    enabled: false

result:
  output_dir: $out_dir

profile:
  memory_monitor_interval: 100
  enable_memory_profile: false
EOF_CFG

  echo "[occ] running $ds with config $cfg"
  (cd "$BENCH_DIR" && "$GO_BIN" run ./cmd/bench_dev -config "$cfg")
}

IFS=',' read -ra datasets <<< "$DATASETS"
for ds in "${datasets[@]}"; do
  ds="${ds// /}"
  [[ -z "$ds" ]] && continue
  run_one_dataset "$ds"
done

combined="$OUT_ROOT/annchor_occ_freshness_summary.csv"
first=1
for f in "$OUT_ROOT"/*/benchmark_results.csv; do
  [[ -f "$f" ]] || continue
  if [[ "$first" -eq 1 ]]; then
    cat "$f" > "$combined"
    first=0
  else
    tail -n +2 "$f" >> "$combined"
  fi
done

echo "[occ] done"
echo "[occ] combined csv: $combined"
