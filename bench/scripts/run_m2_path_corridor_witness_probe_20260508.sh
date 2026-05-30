#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/junyao/code/ANN-CC-bench"
BENCH="$ROOT/bench"
OUT="$ROOT/notes/results/m2_path_corridor_witness_probe_0508"
CFG="$OUT/configs"
LOG="$OUT/logs"
BIN="$BENCH/bin/anncc_bench_pathcorridor_tight"
LD="$ROOT/bench/build/lib:/home/junyao/local/tbb/lib:/home/junyao/local/lib"

mkdir -p "$CFG" "$LOG" "$BENCH/bin"

cd "$BENCH"
LD_LIBRARY_PATH="$LD" go build -o "$BIN" .

write_cfg() {
  local name="$1"
  local dataset_name="$2"
  local max_elements="$3"
  local begin_num="$4"
  local data_path="$5"
  local query_stream="$6"
  local overall_query="$7"
  local overall_gt="$8"
  local variant="$9"
  local witnesses="${10}"
  local cfg_path="$CFG/$name.yaml"
  local out_dir="../notes/results/m2_path_corridor_witness_probe_0508/$name"

  {
    printf 'data:\n'
    printf '  dataset_name: %s\n' "$dataset_name"
    printf '  max_elements: %s\n' "$max_elements"
    printf '  begin_num: %s\n' "$begin_num"
    printf '  data_type: float\n'
    printf '  data_path: %s\n' "$data_path"
    printf '  incr_query_path: %s\n' "$query_stream"
    printf '  overall_query_path: %s\n' "$overall_query"
    printf '  overall_gt_path: %s\n' "$overall_gt"
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
        printf '  inflight_join_signature_radius: 33\n'
        printf '  inflight_join_signature_workers: 8\n'
        ;;
      pathsig*)
        printf '  enable_inflight_join: true\n'
        printf '  inflight_join_signature_bits: 128\n'
        printf '  inflight_join_signature_radius: 33\n'
        printf '  inflight_join_path_corridor: true\n'
        printf '  inflight_join_path_corridor_witnesses: %s\n' "$witnesses"
        printf '  inflight_join_path_corridor_max_dim: 256\n'
        printf '  inflight_join_signature_workers: 8\n'
        ;;
    esac
    printf '\nworkload:\n'
    printf '  batch_size: 400\n'
    printf '  num_threads: 8\n'
    printf '  queue_size: 0\n'
    printf '  rate_groups(r/w):\n'
    printf '    - [999999, 999999]\n'
    printf '  with_external_rw_lock: false\n'
    printf '  query_mode: chasing\n'
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

for ds in sift200k sift1m; do
  if [[ "$ds" == sift200k ]]; then
    max=200000
    begin=180000
    data="../data/sift/sift_base_200k.bin"
    gt="../data/sift/sift_base_200k.gt20"
  else
    max=1000000
    begin=960000
    data="../data/sift/sift_base.bin"
    gt="../data/sift/sift_base.gt20"
  fi
  qstream="../data/sift/sift_query_stream.bin"
  overall="../data/sift/sift_query.bin"
  write_cfg "${ds}_b400_brute" "$ds" "$max" "$begin" "$data" "$qstream" "$overall" "$gt" brute 0
  write_cfg "${ds}_b400_signature" "$ds" "$max" "$begin" "$data" "$qstream" "$overall" "$gt" signature 0
  write_cfg "${ds}_b400_pathsig1" "$ds" "$max" "$begin" "$data" "$qstream" "$overall" "$gt" pathsig1 1
  write_cfg "${ds}_b400_pathsig2" "$ds" "$max" "$begin" "$data" "$qstream" "$overall" "$gt" pathsig2 2
  write_cfg "${ds}_b400_pathsig4" "$ds" "$max" "$begin" "$data" "$qstream" "$overall" "$gt" pathsig4 4
done

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
