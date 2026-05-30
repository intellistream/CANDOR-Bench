#!/bin/bash
# Experiment: Compare preemption modes
#   rwlock    = external rw lock baseline (hnsw)
#   mvcc      = MVCC only, no preempt (annchor-m1)
#   preempt   = MVCC + fine-grain cooperative preempt (annchor-preempt, Config C)
#
# Across 2 datasets × 2 query modes × 3 modes
# Each config sweeps batch_size=[20,200] × threads=[16,64] × 3 rate groups
#
# Usage: cd bench && bash scripts/run_preempt_exp.sh
#
# Config C (winner from tuning):
#   cooldown=1000us, budget=50%, threshold=1

set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/build/lib"
# Fine-grain yield cooldown (Config C)
export ANN_FINE_GRAIN_YIELD_COOLDOWN_US=1000

BENCH_BIN="./bin/bench"
CONFIG_DIR="./config/preempt_exp"
RESULT_BASE="./result/preempt_exp"

mkdir -p "$CONFIG_DIR" "$RESULT_BASE"

# ── Dataset configs ──────────────────────────────────────────────
# Format: name|data_path|incr_query|incr_gt|overall_query|overall_gt|m|ef_c|ef_s
DATASETS=(
  "sift|../data/sift/sift_base.bin|../data/sift/sift_query_stream.bin|../data/sift/sift_stream_i100_offset_index.txt|../data/sift/sift_query.bin|../data/sift/sift_base.gt20|16|200|35"
  "gist|../data/gist/gist_base.bin|../data/gist/gist_query_stream.bin|../data/gist/gist_stream_i100_offset_index.txt|../data/gist/gist_query.bin|../data/gist/gist_base.gt20|24|500|80"
)

QUERY_MODES=("chasing" "round_robin")

# Modes: only preempt — rwlock/mvcc results reused from mvcc_exp
# To run all three, uncomment: MODES=("rwlock" "mvcc" "preempt")
MODES=("preempt")

generate_config() {
  local ds_name=$1 data_path=$2 incr_query=$3 incr_gt=$4
  local overall_query=$5 overall_gt=$6
  local m=$7 ef_c=$8 ef_s=$9
  local qmode=${10} mode=${11}

  local index_type="hnsw"
  local ext_rw="false"
  local search_extra=""

  if [ "$mode" = "rwlock" ]; then
    index_type="hnsw"
    ext_rw="true"
  elif [ "$mode" = "mvcc" ]; then
    index_type="annchor-m1"
    ext_rw="false"
    search_extra="  enable_mvcc: true"
  elif [ "$mode" = "preempt" ]; then
    index_type="annchor-preempt"
    ext_rw="false"
    search_extra="  enable_mvcc: true
  enable_preempt_m2: true
  preempt_quantum_points: 1
  preempt_search_backlog_threshold: 1
  preempt_max_yields_per_batch: 100
  preempt_budget_pct: 5.0
  preempt_budget_window_us: 10000"
  fi

  local fname="${CONFIG_DIR}/${ds_name}_${qmode}_${mode}.yaml"

  cat > "$fname" <<EOF
data:
  dataset_name: ${ds_name}
  max_elements: -1
  begin_num: 50000
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
${search_extra}

workload:
  batch_size: [20, 200]
  num_threads: [16, 64]
  queue_size: 0
  rate_groups(r/w):
    - [40000, 40000]
    - [10000, 90000]
    - [90000, 10000]
  with_external_rw_lock: ${ext_rw}
  query_mode: ${qmode}

compare:
  enabled: false
  simulate:
    enabled: false

result:
  output_dir: ${RESULT_BASE}/${ds_name}/${qmode}_${mode}

profile:
  memory_monitor_interval: 100
  enable_memory_profile: true
EOF

  echo "$fname"
}

# ── Check if a config already has valid results ──────────────────
is_done() {
  local result_dir="$1"
  local csv_count
  csv_count=$(find "$result_dir" -name "*.csv" -size +100c 2>/dev/null | wc -l)
  [ "$csv_count" -gt 0 ]
}

# ── Count total configs ──────────────────────────────────────────
total=0
for ds_entry in "${DATASETS[@]}"; do
  IFS='|' read -r ds_name data_path incr_query incr_gt overall_query overall_gt m ef_c ef_s <<< "$ds_entry"
  for qmode in "${QUERY_MODES[@]}"; do
    for mode in "${MODES[@]}"; do
      total=$((total + 1))
    done
  done
done

echo "=== Preempt Experiment ==="
echo "Modes: rwlock (baseline), mvcc (MVCC only), preempt (MVCC + fine-grain yield)"
echo "Preempt Config C: cooldown=${ANN_FINE_GRAIN_YIELD_COOLDOWN_US}us, budget=50%, threshold=1"
echo "Datasets: $(echo "${DATASETS[@]}" | tr '|' ' ' | awk '{print $1}')"
echo "Total configs: ${total}, each sweeps batch=[20,200] × threads=[16,64] × 3 rates"
echo "Variants per config: 2×2×3 = 12, total runs: $((total * 12))"
echo ""

# ── Generate configs & run ───────────────────────────────────────
count=0
skipped=0
for ds_entry in "${DATASETS[@]}"; do
  IFS='|' read -r ds_name data_path incr_query incr_gt overall_query overall_gt m ef_c ef_s <<< "$ds_entry"
  for qmode in "${QUERY_MODES[@]}"; do
    for mode in "${MODES[@]}"; do
      count=$((count + 1))
      result_dir="${RESULT_BASE}/${ds_name}/${qmode}_${mode}"

      if is_done "$result_dir"; then
        echo "[$count/$total] SKIP (already done): ${ds_name} ${qmode} ${mode}"
        skipped=$((skipped + 1))
        continue
      fi

      cfg=$(generate_config "$ds_name" "$data_path" "$incr_query" "$incr_gt" \
                            "$overall_query" "$overall_gt" "$m" "$ef_c" "$ef_s" \
                            "$qmode" "$mode")
      echo "[$count/$total] Running: ${ds_name} ${qmode} ${mode}"
      echo "  Config: $cfg"
      mkdir -p "$result_dir"
      $BENCH_BIN -config "$cfg" 2>&1 | tee "$result_dir/bench.log"
      echo "[$count/$total] Done: ${ds_name} ${qmode} ${mode}"
      echo ""
    done
  done
done

echo "=== Experiment complete. Skipped ${skipped}, ran $((count - skipped)). ==="
echo "Results in ${RESULT_BASE}/"

# ── Summary table ────────────────────────────────────────────────
echo ""
echo "=== Summary (preempt_exp + mvcc_exp combined) ==="
python3 - "$RESULT_BASE" "./result/mvcc_exp" << 'PYEOF'
import csv, os, sys

def scan_results(base, source_label=""):
    rows = []
    if not os.path.isdir(base): return rows
    for ds in sorted(os.listdir(base)):
        ds_path = os.path.join(base, ds)
        if not os.path.isdir(ds_path): continue
        for qm_mode in sorted(os.listdir(ds_path)):
            csv_path = os.path.join(ds_path, qm_mode, "benchmark_results.csv")
            if not os.path.exists(csv_path): continue
            parts = qm_mode.rsplit("_", 1)
            qmode = parts[0] if len(parts) == 2 else qm_mode
            mode = parts[1] if len(parts) == 2 else "?"
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    wb = row["write_batch_size"]
                    th = row["threads"]
                    ir = float(row["insert_input_rate"])
                    sr = float(row["search_input_rate"])
                    rate = f"{ir:.0f}/{sr:.0f}"
                    sa = float(row["search_mean_op_latency_ms (per-batch)"])
                    sp = float(row["search_p99_op_latency_ms (per-batch)"])
                    iq = float(row["insert_qps (per-point)"])
                    fr = float(row["freshness_total_pts_mean"])
                    rc = float(row["recall"])
                    rows.append((ds, qmode, mode, wb, th, rate, sa, sp, iq, fr, rc))
    return rows

preempt_base = sys.argv[1]
mvcc_base = sys.argv[2]

all_rows = scan_results(preempt_base) + scan_results(mvcc_base)
# Deduplicate: keep preempt_exp version if both exist
seen = set()
unique = []
for r in all_rows:
    key = (r[0], r[1], r[2], r[3], r[4], r[5])
    if key not in seen:
        seen.add(key)
        unique.append(r)

unique.sort(key=lambda r: (r[0], r[1], r[5], r[3], r[4], r[2]))

print(f"{'dataset':<8} {'qmode':<12} {'mode':<10} {'batch':>5} {'thrd':>4} {'rate':>12} | {'srch_avg':>8} {'srch_p99':>8} {'ins_qps':>8} {'fresh':>6} {'recall':>6}")
print("-" * 110)
for ds, qm, mode, wb, th, rate, sa, sp, iq, fr, rc in unique:
    print(f"{ds:<8} {qm:<12} {mode:<10} {wb:>5} {th:>4} {rate:>12} | {sa:>8.2f} {sp:>8.2f} {iq:>8.1f} {fr:>6.1f} {rc:>6.3f}")
PYEOF
