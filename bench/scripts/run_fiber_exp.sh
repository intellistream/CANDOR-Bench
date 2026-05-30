#!/bin/bash
# ── M2 Fiber Preempt Experiment ──
# Compare: rwlock / mvcc (M2=off) / mvcc+fiber (M2=on)
# Datasets: sift, gist, deep1M
# Workloads: balanced, read-heavy, write-heavy
# Batch sizes: 20, 200
# Threads: 16

set -euo pipefail
cd "$(dirname "$0")/.."

EXP_NAME="fiber_exp"
BENCH_BIN="./bin/bench"

# ── Datasets: name|data_path|incr_query|incr_gt|overall_query|overall_gt|m|ef_c|ef_s ──
DATASETS=(
  "sift|../data/sift/sift_base.bin|../data/sift/sift_query_stream.bin|../data/sift/sift_stream_i100_offset_index.txt|../data/sift/sift_query.bin|../data/sift/sift_base.gt20|16|200|35"
  "gist|../data/gist/gist_base.bin|../data/gist/gist_query_stream.bin|../data/gist/gist_stream_i100_offset_index.txt|../data/gist/gist_query.bin|../data/gist/gist_base.gt20|16|200|35"
  "deep1M|../data/deep1M/deep1M_base.bin|../data/deep1M/deep1M_query_stream.bin|../data/deep1M/deep1M_stream_i100_offset_index.txt|../data/deep1M/deep1M_query.bin|../data/deep1M/deep1M_base.gt20|16|200|35"
)

QUERY_MODE="chasing"
THREADS=16
BEGIN_NUM=50000

generate_config() {
  local ds_name="$1" data_path="$2" incr_query="$3" incr_gt="$4" overall_query="$5" overall_gt="$6"
  local m="$7" ef_c="$8" ef_s="$9" index_type="${10}" enable_mvcc="${11}" enable_preempt="${12}"
  local rw_lock="${13}" batch_size="${14}" write_rate="${15}" read_rate="${16}" out_dir="${17}"

  cat <<YAML
data:
  dataset_name: ${ds_name}
  max_elements: -1
  begin_num: ${BEGIN_NUM}
  data_type: float
  data_path: ${data_path}
  incr_query_path: ${incr_query}
  incr_gt_path: ${incr_gt}
  overall_query_path: ${overall_query}
  overall_gt_path: ${overall_gt}
index:
  index_type: ${index_type}
  m: ${m}
  ef_construction: ${ef_c}
search:
  recall_at: 10
  ef_search: ${ef_s}
  enable_mvcc: ${enable_mvcc}
  enable_preempt_m2: ${enable_preempt}
workload:
  batch_size: [${batch_size}]
  num_threads: [${THREADS}]
  queue_size: 0
  rate_groups(r/w):
    - [${write_rate}, ${read_rate}]
  with_external_rw_lock: ${rw_lock}
  query_mode: ${QUERY_MODE}
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

is_done() {
  local csv="$1/benchmark_results.csv"
  [[ -f "$csv" ]] && [[ $(wc -c < "$csv") -gt 100 ]]
}

# ── Modes ──
# mode_name | index_type | enable_mvcc | enable_preempt | rw_lock
MODES=(
  "rwlock|annchor-preempt|false|false|true"
  "mvcc|annchor-preempt|true|false|false"
  "fiber|annchor-preempt|true|true|false"
)

# ── Workload rates (write_rate, read_rate) ──
RATE_GROUPS=(
  "40000|40000"    # balanced saturated
  "10000|90000"    # read-heavy
  "90000|10000"    # write-heavy
)

BATCH_SIZES=(20 200)

total=0
skipped=0
failed=0

for ds_entry in "${DATASETS[@]}"; do
  IFS='|' read -r ds_name data_path incr_query incr_gt overall_query overall_gt m ef_c ef_s <<< "$ds_entry"

  for mode_entry in "${MODES[@]}"; do
    IFS='|' read -r mode_name index_type enable_mvcc enable_preempt rw_lock <<< "$mode_entry"

    for batch_size in "${BATCH_SIZES[@]}"; do
      for rate_entry in "${RATE_GROUPS[@]}"; do
        IFS='|' read -r write_rate read_rate <<< "$rate_entry"

        out_dir="./result/${EXP_NAME}/${ds_name}/${mode_name}_b${batch_size}_w${write_rate}_r${read_rate}"
        config_file="./config/${EXP_NAME}/${ds_name}_${mode_name}_b${batch_size}_w${write_rate}_r${read_rate}.yaml"

        total=$((total + 1))

        if is_done "$out_dir"; then
          echo "[SKIP] $out_dir (already done)"
          skipped=$((skipped + 1))
          continue
        fi

        mkdir -p "$(dirname "$config_file")"
        generate_config "$ds_name" "$data_path" "$incr_query" "$incr_gt" "$overall_query" "$overall_gt" \
          "$m" "$ef_c" "$ef_s" "$index_type" "$enable_mvcc" "$enable_preempt" "$rw_lock" \
          "$batch_size" "$write_rate" "$read_rate" "$out_dir" > "$config_file"

        echo ""
        echo "================================================================"
        echo "[RUN] ${ds_name} / ${mode_name} / b${batch_size} / w${write_rate}-r${read_rate}"
        echo "================================================================"

        if LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} "$BENCH_BIN" -config "$config_file" 2>&1 | \
           grep -E "Progress.*[0-9]+%|Overall recall|Results written|Error|panic"; then
          echo "[DONE] $out_dir"
        else
          echo "[FAIL] $out_dir"
          failed=$((failed + 1))
        fi
      done
    done
  done
done

echo ""
echo "================================================================"
echo "Experiment complete: total=$total, skipped=$skipped, failed=$failed"
echo "================================================================"

# ── Summary ──
echo ""
echo "Generating summary..."
python3 << 'PYEOF'
import csv, os, glob

exp_dir = "./result/fiber_exp"
rows = []

for csv_path in sorted(glob.glob(f"{exp_dir}/**/benchmark_results.csv", recursive=True)):
    rel = os.path.relpath(csv_path, exp_dir)
    parts = rel.split("/")
    if len(parts) < 2:
        continue
    ds = parts[0]
    config = parts[1]  # e.g., fiber_b20_w40000_r40000

    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "ds": ds,
                "config": config,
                "iQPS": float(r["insert_qps (per-point)"]),
                "sQPS": float(r["search_qps (per-point)"]),
                "s_p99": float(r["search_p99_op_latency_ms (per-batch)"]),
                "s_mean": float(r["search_mean_op_latency_ms (per-batch)"]),
                "i_p99": float(r["insert_p99_op_latency_ms (per-batch)"]),
                "recall": float(r["recall"]),
                "fresh_mean": float(r["freshness_total_pts_mean"]),
                "fresh_p99": float(r["freshness_total_pts_p99"]),
            })

if not rows:
    print("No results found.")
else:
    hdr = f"{'ds':<8} {'config':<35} {'iQPS':>8} {'sQPS':>8} {'s_p99':>8} {'s_mean':>8} {'i_p99':>8} {'recall':>7} {'fresh_m':>8} {'fresh99':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['ds']:<8} {r['config']:<35} {r['iQPS']:>8.0f} {r['sQPS']:>8.0f} {r['s_p99']:>8.2f} {r['s_mean']:>8.2f} {r['i_p99']:>8.2f} {r['recall']:>7.3f} {r['fresh_mean']:>8.1f} {r['fresh_p99']:>8.0f}")
PYEOF
