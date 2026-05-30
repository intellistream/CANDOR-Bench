#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/junyao/code/ANN-CC-bench"
BENCH="$ROOT/bench"
OUT="${M2_PATH_CORRIDOR_OUT:-$ROOT/notes/results/m2_path_corridor_full_0508}"
OUT_REL="${M2_PATH_CORRIDOR_OUT_REL:-../notes/results/$(basename "$OUT")}"
CFG="$OUT/configs"
LOG="$OUT/logs"
BIN="${M2_PATH_CORRIDOR_BIN:-$BENCH/bin/anncc_bench_pathcorridor}"
LD="$ROOT/bench/build/lib:/home/junyao/local/tbb/lib:/home/junyao/local/lib"

mkdir -p "$CFG" "$LOG" "$BENCH/bin"

cd "$ROOT"
cmake --build bench/build --target index -j 8
cd "$BENCH"
LD_LIBRARY_PATH="$LD" go build -o "$BIN" .

write_cfg() {
  local name="$1"
  local dataset_name="$2"
  local max_elements="$3"
  local begin_num="$4"
  local data_path="$5"
  local incr_query_path="$6"
  local overall_query_path="$7"
  local overall_gt_path="$8"
  local batch="$9"
  local variant="${10}"
  local radius="${11}"
  local out_dir="${12}"
  local query_mode="${13}"
  local cfg_path="$CFG/$name.yaml"

  {
    printf 'data:\n'
    printf '  dataset_name: %s\n' "$dataset_name"
    printf '  max_elements: %s\n' "$max_elements"
    printf '  begin_num: %s\n' "$begin_num"
    printf '  data_type: float\n'
    printf '  data_path: %s\n' "$data_path"
    printf '  incr_query_path: %s\n' "$incr_query_path"
    printf '  overall_query_path: %s\n' "$overall_query_path"
    if [[ -n "$overall_gt_path" ]]; then
      printf '  overall_gt_path: %s\n' "$overall_gt_path"
    fi
    printf '\nindex:\n'
    printf '  index_type: annchor-m2\n'
    printf '  m: 16\n'
    printf '  ef_construction: 100\n'
    printf '\nsearch:\n'
    printf '  recall_at: 10\n'
    printf '  ef_search: 35\n'
    printf '  enable_mvcc: true\n'
    case "$variant" in
      brute)
        printf '  enable_inflight_bruteforce: true\n'
        ;;
      signature)
        printf '  enable_inflight_join: true\n'
        printf '  inflight_join_signature_bits: 128\n'
        printf '  inflight_join_signature_radius: %s\n' "$radius"
        printf '  inflight_join_signature_workers: 8\n'
        ;;
      pathsig)
        printf '  enable_inflight_join: true\n'
        printf '  inflight_join_signature_bits: 128\n'
        printf '  inflight_join_signature_radius: %s\n' "$radius"
        printf '  inflight_join_path_corridor: true\n'
        printf '  inflight_join_path_corridor_witnesses: 2\n'
        printf '  inflight_join_path_corridor_max_dim: 256\n'
        printf '  inflight_join_signature_workers: 8\n'
        ;;
      *)
        echo "unknown variant: $variant" >&2
        return 2
        ;;
    esac
    printf '\nworkload:\n'
    printf '  batch_size: %s\n' "$batch"
    printf '  num_threads: 8\n'
    printf '  queue_size: 0\n'
    printf '  rate_groups(r/w):\n'
    printf '    - [999999, 999999]\n'
    printf '  with_external_rw_lock: false\n'
    printf '  query_mode: %s\n' "$query_mode"
    printf '\ncompare:\n'
    printf '  enabled: false\n'
    printf '  simulate:\n'
    printf '    enabled: false\n'
    printf '\nresult:\n'
    printf '  output_dir: %s\n' "$out_dir"
    printf '\nrepetitions: 2\n'
    printf '\nprofile:\n'
    printf '  enable_memory_profile: false\n'
  } > "$cfg_path"
}

make_dataset_configs() {
  local ds="$1"
  local max_elements="$2"
  local begin_num="$3"
  local data_path="$4"
  local incr_query_path="$5"
  local overall_query_path="$6"
  local overall_gt_path="$7"
  local radius="$8"
  local query_mode="${9:-chasing}"

  for batch in 200 400 800; do
    for variant in brute signature pathsig; do
      local name="${ds}_b${batch}_${variant}"
      local out_dir="${OUT_REL}/${name}"
      write_cfg "$name" "$ds" "$max_elements" "$begin_num" "$data_path" \
        "$incr_query_path" "$overall_query_path" "$overall_gt_path" \
        "$batch" "$variant" "$radius" "$out_dir" "$query_mode"
    done
  done
}

make_dataset_configs \
  "sift1m" 1000000 960000 \
  "../data/sift/sift_base.bin" \
  "../data/sift/sift_query_stream.bin" \
  "../data/sift/sift_query.bin" \
  "../data/sift/sift_base.gt20" \
  33 \
  chasing

make_dataset_configs \
  "dbpedia50k" 50000 46000 \
  "../data/dbpedia_openai/dbpedia_openai_50k.bin" \
  "../data/dbpedia_openai/dbpedia_openai_query.bin" \
  "../data/dbpedia_openai/dbpedia_openai_query.bin" \
  "" \
  40 \
  round_robin

printf 'config,started_at,exit_code\n' > "$OUT/run_status.csv"
for cfg in "$CFG"/*.yaml; do
  name="$(basename "$cfg" .yaml)"
  started="$(date --iso-8601=seconds)"
  echo "[$started] running $name"
  set +e
  LD_LIBRARY_PATH="$LD" "$BIN" -config "$cfg" > "$LOG/$name.log" 2>&1
  rc=$?
  set -e
  printf '%s,%s,%s\n' "$name" "$started" "$rc" >> "$OUT/run_status.csv"
  if [[ "$rc" -ne 0 ]]; then
    echo "run failed: $name rc=$rc" >&2
    exit "$rc"
  fi
done

echo "all done: $OUT"
