#!/usr/bin/env bash
# Full-scale method harness for the M3 root-cause follow-up.
#
# This is not a toy/noop/partial-insert run. It keeps real HNSW insertion and
# changes only the foreground construction-search budget (`ef_construction`).
# The intended method direction is foreground-cheap attach/search with a later
# background refinement contract; this harness measures whether the foreground
# budget attacks the observed search-P99 memory-pressure cause before a larger
# refinement implementation is worth building.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="${STAMP:-20260513}"
OUT_BASE="${OUT_BASE:-./result/hnsw_foreground_ef_method_${STAMP}}"
CFG_BASE="${CFG_BASE:-./config/hnsw_foreground_ef_method_${STAMP}}"
EF_VALUES="${EF_VALUES:-50 100}"
METHOD_REPS="${METHOD_REPS:-1}"
# Closed-loop insert is rate-controlled for method screens so `ef_construction`
# is not confounded with different saturated write throughput. Keep this below
# the measured efc=100 saturated capacity on SIFT10M/N=64 unless re-calibrated.
INSERT_EVENT_RATE="${INSERT_EVENT_RATE:-20000}"
# Optional quality guard. Leave empty unless the ground truth matches the
# visible prefix/window being tested; the local SIFT10M gt is for a static full
# index and is not valid for the 1M-begin live-insert root-cause cells.
OVERALL_QUERY_PATH="${OVERALL_QUERY_PATH:-}"
OVERALL_GT_PATH="${OVERALL_GT_PATH:-}"

mkdir -p "$OUT_BASE" "$CFG_BASE"

rep_values() {
  if [[ "$METHOD_REPS" =~ ^[0-9]+$ ]]; then
    seq 1 "$METHOD_REPS"
  else
    printf '%s\n' "$METHOD_REPS"
  fi
}

{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "purpose=full_scale_hnsw_foreground_ef_method_probe"
  echo "ef_values=$EF_VALUES"
  echo "dataset=sift10m"
  echo "begin_num=1000000"
  echo "max_elements=10000000"
  echo "real_insert=1"
  echo "fake_write=0"
  echo "insert_event_rate=$INSERT_EVENT_RATE"
  echo "modes=bg_search insert_real"
  echo "n_values=64"
  echo "reps=$METHOD_REPS"
  echo "overall_query_path=$OVERALL_QUERY_PATH"
  echo "overall_gt_path=$OVERALL_GT_PATH"
} > "$OUT_BASE/MANIFEST.txt"

for efc in $EF_VALUES; do
  out_root="$OUT_BASE/efc${efc}"
  cfg_root="$CFG_BASE/efc${efc}"
  echo "[hnsw-foreground-ef] ef_construction=${efc} -> ${out_root}"
  OUT_ROOT="$out_root" \
    CFG_ROOT="$cfg_root" \
    EF_CONSTRUCTION="$efc" \
    OVERALL_QUERY_PATH="$OVERALL_QUERY_PATH" \
    OVERALL_GT_PATH="$OVERALL_GT_PATH" \
    INSERT_EVENT_RATE="$INSERT_EVENT_RATE" \
    M3_BATCH_DIAG=0 \
    PHASE_OVERLAP_DIAG=0 \
    ANNCHOR_SEARCH_WORK_COUNTERS=1 \
    MODES="bg_search insert_real" \
    N_VALUES="64" \
    REPS="$(rep_values)" \
    ENABLE_INSERT_PHASE_TRACE=0 \
    bash scripts/run_hnsw_insert_search_rootcause_20260513.sh

  python3 scripts/analyze_hnsw_insert_search_rootcause_20260513.py \
    --root "$out_root" \
    --out-dir "$out_root" \
    > "$out_root/analyze_recheck.log" 2>&1
done

python3 - <<'PY' "$OUT_BASE"
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for agg in sorted(root.glob("efc*/hnsw_rootcause_aggregate.csv")):
    efc = agg.parent.name.replace("efc", "")
    with agg.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("n") != "64" or row.get("segment") != "steady":
                continue
            rows.append({
                "ef_construction": efc,
                "mode": row["mode"],
                "search_p99_ms": row["raw_wall_ms_p99_median"],
                "lock_max_p99_ms": row["lock_wait_max_ms_p99_median"],
                "residual_p99_ms": row["residual_after_max_lock_ms_p99_median"],
                "l0_edges_p99": row["l0_edges_sum_p99_median"],
                "per_l0_edge_p99_us": row["raw_us_per_l0_edge_p99_median"],
                "reader_llc_load_misses": row.get("reader_llc_load_misses_median", ""),
                "reader_llc_miss_ratio": row.get("reader_llc_miss_ratio_median", ""),
                "uncore_reads": row.get("uncore_reads_median", ""),
                "uncore_writes": row.get("uncore_writes_median", ""),
            })

out = root / "foreground_ef_summary.csv"
with out.open("w", newline="") as f:
    if rows:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    else:
        f.write("")
print(f"[hnsw-foreground-ef] summary -> {out}")
PY
