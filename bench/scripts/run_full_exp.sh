#!/bin/bash
# ── ANNchor Full Experiment Suite ──
# Fixes from run_fiber_exp.sh:
#   1. rate_groups uses map format {write: N, read: M} to avoid r/w swap bug
#   2. enable_preempt_m2 explicitly set for all modes
#   3. Covers both workload-mix and thread-scalability experiments
#   4. Skips already-done configs (idempotent re-run)
#
# Usage:
#   bash scripts/run_full_exp.sh                    # run everything
#   bash scripts/run_full_exp.sh workload            # only workload-mix
#   bash scripts/run_full_exp.sh scale               # only scalability
#   bash scripts/run_full_exp.sh workload deep1M     # only deep1M workload-mix

set -euo pipefail
cd "$(dirname "$0")/.."

BENCH_BIN="./bin/bench"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# ── Datasets ──
# name|data_path|incr_query|incr_gt|overall_query|overall_gt|m|ef_c|ef_s
DATASETS=(
  "sift|../data/sift/sift_base.bin|../data/sift/sift_query_stream.bin|../data/sift/sift_stream_i100_offset_index.txt|../data/sift/sift_query.bin|../data/sift/sift_base.gt20|16|200|35"
  "gist|../data/gist/gist_base.bin|../data/gist/gist_query_stream.bin|../data/gist/gist_stream_i100_offset_index.txt|../data/gist/gist_query.bin|../data/gist/gist_base.gt20|16|200|35"
  "deep1M|../data/deep1M/deep1M_base.bin|../data/deep1M/deep1M_query_stream.bin|../data/deep1M/deep1M_stream_i100_offset_index.txt|../data/deep1M/deep1M_query.bin|../data/deep1M/deep1M_base.gt20|16|200|35"
)

QUERY_MODE="chasing"
BEGIN_NUM=50000

# ── Modes ──
# mode_name|index_type|enable_mvcc|enable_preempt_m2|with_external_rw_lock
MODES=(
  "rwlock|annchor-preempt|false|false|true"
  "mvcc|annchor-preempt|true|false|false"
  "fiber|annchor-preempt|true|true|false"
)

# ── Workload-mix rates: write_rate|read_rate ──
RATE_GROUPS=(
  "40000|40000"    # balanced
  "10000|90000"    # write-light / read-heavy
  "90000|10000"    # write-heavy / read-light
)

BATCH_SIZES=(20 200)
WORKLOAD_THREADS=16

# ── Scalability thread counts ──
SCALE_THREADS=(1 4 8 16 32 64)
SCALE_REPS=2

# ────────────────────────────────────────────────────────────────────────
# Config generator — uses map format to avoid r/w swap
# ────────────────────────────────────────────────────────────────────────
generate_workload_config() {
  local ds_name="$1" data_path="$2" incr_query="$3" incr_gt="$4"
  local overall_query="$5" overall_gt="$6"
  local m="$7" ef_c="$8" ef_s="$9"
  local index_type="${10}" enable_mvcc="${11}" enable_preempt="${12}"
  local rw_lock="${13}" batch_size="${14}"
  local write_rate="${15}" read_rate="${16}"
  local threads="${17}" out_dir="${18}" reps="${19:-1}"

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
  num_threads: [${threads}]
  queue_size: 0
  rate_groups(r/w):
    - {write: ${write_rate}, read: ${read_rate}}
  with_external_rw_lock: ${rw_lock}
  query_mode: ${QUERY_MODE}
compare:
  enabled: false
  simulate:
    enabled: false
result:
  output_dir: ${out_dir}
repetitions: ${reps}
profile:
  memory_monitor_interval: 100
  enable_memory_profile: true
YAML
}

generate_scale_config() {
  local ds_name="$1" data_path="$2" incr_query="$3" incr_gt="$4"
  local overall_query="$5" overall_gt="$6"
  local m="$7" ef_c="$8" ef_s="$9"
  local index_type="${10}" enable_mvcc="${11}" enable_preempt="${12}"
  local rw_lock="${13}" out_dir="${14}" reps="${15}"

  # Thread list as YAML array
  local thread_yaml=""
  for t in "${SCALE_THREADS[@]}"; do
    thread_yaml="${thread_yaml}, ${t}"
  done
  thread_yaml="[${thread_yaml:2}]"  # strip leading ", "

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
  batch_size: [20]
  num_threads: ${thread_yaml}
  queue_size: 0
  rate_groups(r/w):
    - {write: 9999999, read: 9999999}
  with_external_rw_lock: ${rw_lock}
  query_mode: ${QUERY_MODE}
compare:
  enabled: false
  simulate:
    enabled: false
result:
  output_dir: ${out_dir}
repetitions: ${reps}
profile:
  memory_monitor_interval: 100
  enable_memory_profile: true
YAML
}

# ────────────────────────────────────────────────────────────────────────
# is_done: check if output CSV has the expected number of data rows.
#   $1 = output directory
#   $2 = expected row count (default 1)
is_done() {
  local csv="$1/benchmark_results.csv"
  local expected="${2:-1}"
  [[ -f "$csv" ]] || return 1
  local actual
  actual=$(tail -n +2 "$csv" | wc -l)  # skip header
  [[ "$actual" -ge "$expected" ]]
}

run_bench() {
  local config_file="$1"
  local out_dir="$2"
  local label="$3"
  local tmplog
  tmplog=$(mktemp /tmp/bench_run_XXXXXX.log)

  # Run bench to completion first (no pipe — avoids orphan processes).
  set +e
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} "$BENCH_BIN" -config "$config_file" \
    > "$tmplog" 2>&1
  local bench_exit=$?
  set -e

  # Display filtered output
  grep -E "Progress.*[0-9]+%|Overall recall|Results written|Error|panic|SIGSEGV|SIGABRT" \
    "$tmplog" 2>/dev/null || true
  rm -f "$tmplog"

  if [[ "$bench_exit" -eq 0 ]] && [[ -f "$out_dir/benchmark_results.csv" ]]; then
    echo "[DONE] $out_dir"
    return 0
  else
    echo "[FAIL] $out_dir (exit=$bench_exit)"
    return 1
  fi
}

# ────────────────────────────────────────────────────────────────────────
# Experiment 1: Workload-mix (3 datasets × 3 modes × 3 rates × 2 batches)
# ────────────────────────────────────────────────────────────────────────
run_workload_exp() {
  local filter_ds="${1:-}"
  local exp_name="workload_exp"
  local total=0 skipped=0 done=0 failed=0

  echo "============================================================"
  echo "  Experiment: Workload Mix"
  echo "  Matrix: 3 datasets × 3 modes × 3 rates × 2 batches = 54"
  echo "  Threads: ${WORKLOAD_THREADS}"
  echo "============================================================"

  for ds_entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name data_path incr_query incr_gt overall_query overall_gt m ef_c ef_s <<< "$ds_entry"

    # Filter dataset if specified
    if [[ -n "$filter_ds" ]] && [[ "$ds_name" != "$filter_ds" ]]; then
      continue
    fi

    for mode_entry in "${MODES[@]}"; do
      IFS='|' read -r mode_name index_type enable_mvcc enable_preempt rw_lock <<< "$mode_entry"

      for batch_size in "${BATCH_SIZES[@]}"; do
        for rate_entry in "${RATE_GROUPS[@]}"; do
          IFS='|' read -r write_rate read_rate <<< "$rate_entry"

          out_dir="./result/${exp_name}/${ds_name}/${mode_name}_b${batch_size}_w${write_rate}_r${read_rate}"
          config_file="./config/${exp_name}/${ds_name}_${mode_name}_b${batch_size}_w${write_rate}_r${read_rate}.yaml"
          total=$((total + 1))

          if is_done "$out_dir"; then
            echo "[SKIP] $out_dir"
            skipped=$((skipped + 1))
            continue
          fi

          mkdir -p "$(dirname "$config_file")"
          generate_workload_config \
            "$ds_name" "$data_path" "$incr_query" "$incr_gt" "$overall_query" "$overall_gt" \
            "$m" "$ef_c" "$ef_s" "$index_type" "$enable_mvcc" "$enable_preempt" "$rw_lock" \
            "$batch_size" "$write_rate" "$read_rate" "$WORKLOAD_THREADS" "$out_dir" "1" \
            > "$config_file"

          echo ""
          echo "================================================================"
          echo "[RUN] ${ds_name} / ${mode_name} / b${batch_size} / w${write_rate}-r${read_rate}"
          echo "================================================================"

          if run_bench "$config_file" "$out_dir" "${ds_name}/${mode_name}"; then
            done=$((done + 1))
          else
            failed=$((failed + 1))
          fi
        done
      done
    done
  done

  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  Workload-mix complete: total=$total done=$done skipped=$skipped failed=$failed"
  echo "════════════════════════════════════════════════════════════"
}

# ────────────────────────────────────────────────────────────────────────
# Experiment 2: Thread Scalability (3 datasets × 3 modes × 6 thread counts)
# ────────────────────────────────────────────────────────────────────────
run_scale_exp() {
  local filter_ds="${1:-}"
  local exp_name="scale_exp"
  local total=0 skipped=0 done=0 failed=0

  echo "============================================================"
  echo "  Experiment: Thread Scalability"
  echo "  Matrix: 3 datasets × 3 modes × 6 threads = 54"
  echo "  Threads: ${SCALE_THREADS[*]}"
  echo "  Reps: ${SCALE_REPS}"
  echo "  Rate: saturated (9999999/9999999)"
  echo "============================================================"

  for ds_entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_name data_path incr_query incr_gt overall_query overall_gt m ef_c ef_s <<< "$ds_entry"

    if [[ -n "$filter_ds" ]] && [[ "$ds_name" != "$filter_ds" ]]; then
      continue
    fi

    for mode_entry in "${MODES[@]}"; do
      IFS='|' read -r mode_name index_type enable_mvcc enable_preempt rw_lock <<< "$mode_entry"

      out_dir="./result/${exp_name}/${ds_name}/${mode_name}"
      config_file="./config/${exp_name}/${ds_name}_${mode_name}.yaml"
      local expected_rows=$(( ${#SCALE_THREADS[@]} * SCALE_REPS ))
      total=$((total + 1))

      if is_done "$out_dir" "$expected_rows"; then
        echo "[SKIP] $out_dir"
        skipped=$((skipped + 1))
        continue
      fi

      mkdir -p "$(dirname "$config_file")"
      generate_scale_config \
        "$ds_name" "$data_path" "$incr_query" "$incr_gt" "$overall_query" "$overall_gt" \
        "$m" "$ef_c" "$ef_s" "$index_type" "$enable_mvcc" "$enable_preempt" "$rw_lock" \
        "$out_dir" "$SCALE_REPS" \
        > "$config_file"

      echo ""
      echo "================================================================"
      echo "[RUN] ${ds_name} / ${mode_name} / threads=[${SCALE_THREADS[*]}] / reps=${SCALE_REPS}"
      echo "================================================================"

      if run_bench "$config_file" "$out_dir" "${ds_name}/${mode_name}"; then
        done=$((done + 1))
      else
        failed=$((failed + 1))
      fi
    done
  done

  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  Scalability complete: total=$total done=$done skipped=$skipped failed=$failed"
  echo "════════════════════════════════════════════════════════════"
}

# ────────────────────────────────────────────────────────────────────────
# Summary generator
# ────────────────────────────────────────────────────────────────────────
generate_summary() {
  echo ""
  echo "Generating summary..."
  python3 << 'PYEOF'
import csv, os, glob, sys

for exp_name in ["workload_exp", "scale_exp"]:
    exp_dir = f"./result/{exp_name}"
    if not os.path.isdir(exp_dir):
        continue

    rows = []
    for csv_path in sorted(glob.glob(f"{exp_dir}/**/benchmark_results.csv", recursive=True)):
        rel = os.path.relpath(csv_path, exp_dir)
        parts = rel.split("/")
        ds = parts[0] if len(parts) >= 2 else "?"
        config = parts[1] if len(parts) >= 2 else parts[0]

        with open(csv_path) as f:
            for r in csv.DictReader(f):
                stats = r.get("stats", "")
                m2 = "?"
                fyc = "0"
                for kv in stats.split(", "):
                    if kv.startswith("preempt_m2_enabled:"):
                        m2 = kv.split(":")[1]
                    if kv.startswith("fiber_yield_count:"):
                        fyc = kv.split(":")[1]

                rows.append({
                    "ds": ds,
                    "config": config,
                    "t": r["threads"],
                    "iQPS": float(r["insert_qps (per-point)"]),
                    "sQPS": float(r["search_qps (per-point)"]),
                    "s_p99": float(r.get("search_p99_op_latency_ms (per-batch)", 0)),
                    "recall": float(r["recall"]),
                    "fresh_m": float(r.get("freshness_total_pts_mean", 0)),
                    "m2": m2,
                    "fyc": fyc,
                })

    if not rows:
        continue

    print(f"\n{'='*100}")
    print(f"  {exp_name}: {len(rows)} rows")
    print(f"{'='*100}")
    hdr = f"{'ds':<8} {'config':<35} {'t':>3} {'iQPS':>8} {'sQPS':>8} {'s_p99':>8} {'recall':>7} {'fresh_m':>8} {'m2':>3} {'fyc':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['ds']:<8} {r['config']:<35} {r['t']:>3} {r['iQPS']:>8.0f} {r['sQPS']:>8.0f} {r['s_p99']:>8.2f} {r['recall']:>7.3f} {r['fresh_m']:>8.1f} {r['m2']:>3} {r['fyc']:>10}")

    # Sanity check: flag fiber/preempt rows with m2=0
    bad = [r for r in rows if ("fiber" in r["config"] or "preempt" in r["config"]) and r["m2"] == "0"]
    if bad:
        print(f"\n⚠  WARNING: {len(bad)} fiber/preempt rows have preempt_m2_enabled=0!")
        for r in bad:
            print(f"   {r['ds']}/{r['config']} t={r['t']}")
PYEOF
}

# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
EXP_TYPE="${1:-all}"
FILTER_DS="${2:-}"

case "$EXP_TYPE" in
  workload)
    run_workload_exp "$FILTER_DS" 2>&1 | tee "$LOG_DIR/workload_exp.log"
    ;;
  scale)
    run_scale_exp "$FILTER_DS" 2>&1 | tee "$LOG_DIR/scale_exp.log"
    ;;
  all)
    {
      run_workload_exp "$FILTER_DS"
      run_scale_exp "$FILTER_DS"
      generate_summary
    } 2>&1 | tee "$LOG_DIR/full_exp.log"
    ;;
  summary)
    generate_summary
    ;;
  *)
    echo "Usage: $0 [workload|scale|all|summary] [dataset_filter]"
    echo "  e.g.: $0 workload deep1M"
    exit 1
    ;;
esac
