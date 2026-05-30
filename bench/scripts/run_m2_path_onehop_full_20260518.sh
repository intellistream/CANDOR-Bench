#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/junyao/code/ANN-CC-bench"
BENCH="$ROOT/bench"
RUN_NAME="${M2_PATH_ONEHOP_RUN_NAME:-m2_path_onehop_full_20260518}"
OUT="$BENCH/result/$RUN_NAME"
CFG="$BENCH/config/$RUN_NAME"
LOG="$OUT/logs"
BIN="$BENCH/bin/bench_m2_path_onehop_full_20260518"
LD="$BENCH/build/lib:/home/junyao/local/tbb/lib:/home/junyao/local/lib"
FN_STRIDE="${M2_FN_STRIDE:-10}"

if [[ "${M2_KEEP_OLD_RESULTS:-0}" != "1" ]]; then
  rm -rf "$OUT"
fi
mkdir -p "$CFG" "$LOG" "$BENCH/bin"

cd "$BENCH"
cmake --build build --target index -j "${BUILD_JOBS:-8}"
LD_LIBRARY_PATH="$LD" go build -o "$BIN" .

write_cfg() {
  local dataset="$1"
  local method="$2"
  local max_elements="$3"
  local begin_num="$4"
  local data_path="$5"
  local query_stream="$6"
  local incr_gt="$7"
  local query_path="$8"
  local gt_path="$9"
  local m="${10}"
  local efc="${11}"
  local efs="${12}"
  local cfg_path="$CFG/${dataset}_${method}.yaml"
  local out_dir="./result/$RUN_NAME/$dataset/$method"

  {
    printf 'data:\n'
    printf '  dataset_name: %s\n' "$dataset"
    printf '  max_elements: %s\n' "$max_elements"
    printf '  begin_num: %s\n' "$begin_num"
    printf '  data_type: float\n'
    printf '  data_path: %s\n' "$data_path"
    printf '  incr_query_path: %s\n' "$query_stream"
    if [[ -n "$incr_gt" ]]; then
      printf '  incr_gt_path: %s\n' "$incr_gt"
    fi
    printf '  overall_query_path: %s\n' "$query_path"
    printf '  overall_gt_path: %s\n' "$gt_path"
    printf 'index:\n'
    printf '  index_type: annchor-m2\n'
    printf '  m: %s\n' "$m"
    printf '  ef_construction: %s\n' "$efc"
    printf 'search:\n'
    printf '  recall_at: 10\n'
    printf '  ef_search: %s\n' "$efs"
    printf '  enable_mvcc: true\n'
    if [[ "$method" == "path_onehop" ]]; then
      printf '  enable_inflight_join: true\n'
      printf '  inflight_join_path_one_hop: true\n'
      printf '  inflight_join_small_fresh_threshold: 200\n'
      printf '  inflight_join_query_result_prefix: 10\n'
      printf '  inflight_join_signature_workers: 8\n'
    else
      printf '  enable_inflight_bruteforce: true\n'
    fi
    printf 'workload:\n'
    printf '  batch_size: [400, 1000]\n'
    printf '  num_threads: 8\n'
    printf '  queue_size: 0\n'
    printf '  rate_groups(r/w):\n'
    printf '    - [999999999, 999999999]\n'
    printf '    - [40000, 40000]\n'
    printf '    - [90000, 10000]\n'
    printf '  with_external_rw_lock: false\n'
    printf '  query_mode: [round_robin, chasing, peeking]\n'
    printf '  per_query_latency: true\n'
    printf '  precompute_schedule: true\n'
    printf 'compare:\n'
    printf '  enabled: false\n'
    printf '  simulate:\n'
    printf '    enabled: false\n'
    printf 'result:\n'
    printf '  output_dir: %s\n' "$out_dir"
    printf 'repetitions: 1\n'
    printf 'profile:\n'
    printf '  memory_monitor_interval: 100\n'
    printf '  enable_memory_profile: false\n'
  } > "$cfg_path"
}

write_cfg sift brute 150000 50000 \
  "../data/sift/sift_base.bin" \
  "../data/sift/sift_query_stream.bin" \
  "../data/sift/sift_stream_i20_offset_index.txt" \
  "../data/sift/sift_query.bin" \
  "../data/sift/sift_base.gt20" \
  16 100 35
write_cfg sift path_onehop 150000 50000 \
  "../data/sift/sift_base.bin" \
  "../data/sift/sift_query_stream.bin" \
  "../data/sift/sift_stream_i20_offset_index.txt" \
  "../data/sift/sift_query.bin" \
  "../data/sift/sift_base.gt20" \
  16 100 35

write_cfg deep1M brute 150000 50000 \
  "../data/deep1M/deep1M_base.bin" \
  "../data/deep1M/deep1M_query_stream.bin" \
  "../data/deep1M/deep1M_stream_i20_offset_index.txt" \
  "../data/deep1M/deep1M_query.bin" \
  "../data/deep1M/deep1M_base.gt20" \
  24 200 40
write_cfg deep1M path_onehop 150000 50000 \
  "../data/deep1M/deep1M_base.bin" \
  "../data/deep1M/deep1M_query_stream.bin" \
  "../data/deep1M/deep1M_stream_i20_offset_index.txt" \
  "../data/deep1M/deep1M_query.bin" \
  "../data/deep1M/deep1M_base.gt20" \
  24 200 40

write_cfg gist brute 150000 50000 \
  "../data/gist/gist_base.bin" \
  "../data/gist/gist_query_stream.bin" \
  "../data/gist/gist_stream_i20_offset_index.txt" \
  "../data/gist/gist_query.bin" \
  "../data/gist/gist_base.gt20" \
  24 500 80
write_cfg gist path_onehop 150000 50000 \
  "../data/gist/gist_base.bin" \
  "../data/gist/gist_query_stream.bin" \
  "../data/gist/gist_stream_i20_offset_index.txt" \
  "../data/gist/gist_query.bin" \
  "../data/gist/gist_base.gt20" \
  24 500 80

printf 'config,started_at,exit_code\n' > "$OUT/run_status.csv"
for cfg in "$CFG"/*.yaml; do
  name="$(basename "$cfg" .yaml)"
  started="$(date --iso-8601=seconds)"
  echo "[$started] running $name"
  set +e
  ANNCHOR_M2_VALIDATE_FN=1 \
  ANNCHOR_M2_VALIDATE_FN_STRIDE="$FN_STRIDE" \
  LD_LIBRARY_PATH="$LD" "$BIN" -config "$cfg" > "$LOG/$name.log" 2>&1
  rc=$?
  set -e
  printf '%s,%s,%s\n' "$name" "$started" "$rc" >> "$OUT/run_status.csv"
  if [[ "$rc" -ne 0 ]]; then
    echo "run failed: $name rc=$rc; see $LOG/$name.log" >&2
    exit "$rc"
  fi
done

python3 "$BENCH/scripts/summarize_m2_path_onehop_full_20260518.py" "$OUT"
echo "all done: $OUT"
