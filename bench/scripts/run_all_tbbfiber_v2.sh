#!/bin/bash
# ── TBB+Fiber experiment v2: correct 95% recall parameters ──
# sift: M=16 efc=200 efs=50 → recall ~0.946
# deep1M: M=16 efc=200 efs=100 → recall ~0.948
# gist: M=24 efc=400 efs=200 → recall ~0.957
set -euo pipefail
cd "$(dirname "$0")/.."

BENCH_BIN="./bin/bench_dev"
EXP_DIR="./result/tbb_fiber_v2"
CONFIG_DIR="./config/tbb_fiber_v2"
mkdir -p "$CONFIG_DIR"

THREADS=16; BEGIN=50000

DATASETS=(
  "sift|../data/sift/sift_base.bin|../data/sift/sift_query_stream.bin|../data/sift/sift_stream_i100_offset_index.txt|../data/sift/sift_query.bin|../data/sift/sift_base.gt20|16|200|50"
  "deep1M|../data/deep1M/deep1M_base.bin|../data/deep1M/deep1M_query_stream.bin|../data/deep1M/deep1M_stream_i100_offset_index.txt|../data/deep1M/deep1M_query.bin|../data/deep1M/deep1M_base.gt20|16|200|100"
  "gist|../data/gist/gist_base.bin|../data/gist/gist_query_stream.bin|../data/gist/gist_stream_i100_offset_index.txt|../data/gist/gist_query.bin|../data/gist/gist_base.gt20|24|400|200"
)

MODES=(
  "rwlock|false|false|true"
  "mvcc|true|false|false"
  "fiber|true|true|false"
)

RATES=(
  "40000|40000"
  "10000|90000"
  "90000|10000"
)

BATCHES=(20 200)

generate_config() {
  local ds="$1" dp="$2" iq="$3" ig="$4" oq="$5" og="$6"
  local m="$7" efc="$8" efs="$9" mode="${10}" emvcc="${11}" epreempt="${12}"
  local rwl="${13}" bs="${14}" wr="${15}" rr="${16}" out="${17}"
  cat <<YAML
data:
  dataset_name: $ds
  max_elements: -1
  begin_num: $BEGIN
  data_type: float
  data_path: $dp
  incr_query_path: $iq
  incr_gt_path: $ig
  overall_query_path: $oq
  overall_gt_path: $og
index:
  index_type: annchor-preempt
  m: $m
  ef_construction: $efc
search:
  recall_at: 10
  ef_search: $efs
  enable_mvcc: $emvcc
  enable_preempt_m2: $epreempt
workload:
  batch_size: [$bs]
  num_threads: [$THREADS]
  queue_size: 0
  rate_groups(r/w):
    - [$wr, $rr]
  with_external_rw_lock: $rwl
  query_mode: chasing
compare:
  enabled: false
  simulate:
    enabled: false
result:
  output_dir: $out
profile:
  memory_monitor_interval: 100
  enable_memory_profile: true
YAML
}

is_done() {
  [[ -f "$1/benchmark_results.csv" ]] && [[ $(wc -c < "$1/benchmark_results.csv") -gt 100 ]]
}

total=0; done=0; failed=0

for ds_entry in "${DATASETS[@]}"; do
  IFS='|' read -r ds dp iq ig oq og m efc efs <<< "$ds_entry"
  for mode_entry in "${MODES[@]}"; do
    IFS='|' read -r mode emvcc epreempt rwl <<< "$mode_entry"
    for bs in "${BATCHES[@]}"; do
      for rate_entry in "${RATES[@]}"; do
        IFS='|' read -r wr rr <<< "$rate_entry"
        out="$EXP_DIR/${ds}/${mode}_b${bs}_w${wr}_r${rr}"
        cfg="$CONFIG_DIR/${ds}_${mode}_b${bs}_w${wr}_r${rr}.yaml"
        total=$((total+1))

        if is_done "$out"; then
          echo "[SKIP] $out"
          done=$((done+1))
          continue
        fi

        mkdir -p "$(dirname "$cfg")"
        generate_config "$ds" "$dp" "$iq" "$ig" "$oq" "$og" \
          "$m" "$efc" "$efs" "$mode" "$emvcc" "$epreempt" "$rwl" \
          "$bs" "$wr" "$rr" "$out" > "$cfg"

        echo ""
        echo "========================================"
        echo "[RUN $((done+1))/$total] $ds / $mode / b$bs / w${wr}-r${rr}"
        echo "========================================"

        if LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} "$BENCH_BIN" -config "$cfg" 2>&1 | \
           grep -E "Progress.*100%|recall|Results written|Error|panic|Segmentation"; then
          echo "[DONE] $out"
          done=$((done+1))
        else
          echo "[FAIL] $out"
          failed=$((failed+1))
        fi
      done
    done
  done
done

echo ""
echo "========================================"
echo "Summary: $done/$total done, $failed failed"
echo "========================================"
