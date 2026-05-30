#!/bin/bash
# ── Write-heavy sweep: test fiber overhead across write ratios ──
# sift efs=50, b20 only (worst case for fiber overhead)
# Compare mvcc vs fiber at varying write ratios
set -euo pipefail
cd "$(dirname "$0")/.."

BENCH_BIN="./bin/bench_dev"
EXP_DIR="./result/writeheavy_sweep"
CONFIG_DIR="./config/writeheavy_sweep"
mkdir -p "$CONFIG_DIR"

THREADS=16; BEGIN=50000
DS="sift"
DP="../data/sift/sift_base.bin"
IQ="../data/sift/sift_query_stream.bin"
IG="../data/sift/sift_stream_i100_offset_index.txt"
OQ="../data/sift/sift_query.bin"
OG="../data/sift/sift_base.gt20"
M=16; EFC=200; EFS=50

MODES=(
  "mvcc|true|false|false"
  "fiber|true|true|false"
)

# Write-heavy sweep: from balanced to extreme write-heavy
RATES=(
  "50000|50000"
  "60000|40000"
  "70000|30000"
  "80000|20000"
  "90000|10000"
  "95000|5000"
  "99000|1000"
)

BATCHES=(20 200)

REPEATS=3

generate_config() {
  local mode="$1" emvcc="$2" epreempt="$3" rwl="$4" bs="$5" wr="$6" rr="$7" out="$8"
  cat <<YAML
data:
  dataset_name: $DS
  max_elements: -1
  begin_num: $BEGIN
  data_type: float
  data_path: $DP
  incr_query_path: $IQ
  incr_gt_path: $IG
  overall_query_path: $OQ
  overall_gt_path: $OG
index:
  index_type: annchor-preempt
  m: $M
  ef_construction: $EFC
search:
  recall_at: 10
  ef_search: $EFS
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

for rep in $(seq 1 $REPEATS); do
  for mode_entry in "${MODES[@]}"; do
    IFS='|' read -r mode emvcc epreempt rwl <<< "$mode_entry"
    for bs in "${BATCHES[@]}"; do
      for rate_entry in "${RATES[@]}"; do
        IFS='|' read -r wr rr <<< "$rate_entry"
        out="$EXP_DIR/rep${rep}/${mode}_b${bs}_w${wr}_r${rr}"
        cfg="$CONFIG_DIR/${mode}_b${bs}_w${wr}_r${rr}.yaml"
        total=$((total+1))

        if is_done "$out"; then
          echo "[SKIP] $out"
          done=$((done+1))
          continue
        fi

        mkdir -p "$(dirname "$cfg")"
        generate_config "$mode" "$emvcc" "$epreempt" "$rwl" \
          "$bs" "$wr" "$rr" "$out" > "$cfg"

        echo ""
        echo "========================================"
        echo "[RUN $((done+1))/$total] rep$rep / $mode / b$bs / w${wr}-r${rr}"
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
