#!/usr/bin/env bash
# Full-source static recall guard for the M3 foreground-ef mitigation screen.
#
# This is not a latency run and does not use sampled data. It builds a full
# SIFT10M HNSW index for each ef_construction value and measures Recall@10 on
# the standard SIFT10M 1k-query ground truth. Use it as a quality guard for
# rate-controlled latency screens, not as latency evidence.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAMP="${STAMP:-20260513}"
OUT_BASE="${OUT_BASE:-./result/hnsw_ef_quality_guard_${STAMP}}"
CFG_BASE="${CFG_BASE:-./config/hnsw_ef_quality_guard_${STAMP}}"
EF_VALUES="${EF_VALUES:-50 100}"
DATASET_NAME="${DATASET_NAME:-sift10m}"
DATA_PATH="${DATA_PATH:-../data/sift10m/sift10m_base.bin}"
QUERY_PATH="${QUERY_PATH:-../data/sift10m/sift10m_query.bin}"
GT_PATH="${GT_PATH:-../data/sift10m/sift10m_base.gt20}"
M_HNSW="${M_HNSW:-16}"
EF_SEARCH="${EF_SEARCH:-50}"
THREADS="${THREADS:-64}"
BATCH_SIZE="${BATCH_SIZE:-20}"
DRY_RUN="${DRY_RUN:-0}"
BENCH_BIN="${BENCH_BIN:-./bin/bench}"
BENCH_LD_LIBRARY_PATH="${BENCH_LD_LIBRARY_PATH:-./build/lib:${LD_LIBRARY_PATH:-}}"

mkdir -p "$OUT_BASE" "$CFG_BASE"
LOCK_DIR="${OUT_BASE}.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[hnsw-ef-quality] another quality guard appears active for $OUT_BASE; lock=$LOCK_DIR"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "missing benchmark binary: $BENCH_BIN" >&2
  exit 1
fi
for p in "$DATA_PATH" "$QUERY_PATH" "$GT_PATH"; do
  [[ -f "$p" ]] || { echo "missing required file: $p" >&2; exit 1; }
done

{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "purpose=full_source_hnsw_ef_quality_guard"
  echo "dataset_name=$DATASET_NAME"
  echo "data_path=$DATA_PATH"
  echo "query_path=$QUERY_PATH"
  echo "gt_path=$GT_PATH"
  echo "m=$M_HNSW"
  echo "ef_search=$EF_SEARCH"
  echo "ef_values=$EF_VALUES"
  echo "bench_bin=$BENCH_BIN"
  echo "bench_ld_library_path=$BENCH_LD_LIBRARY_PATH"
  echo "dry_run=$DRY_RUN"
  echo "full_source=1"
  echo "fake_write=0"
  uname -a
} > "$OUT_BASE/MANIFEST.txt"

for efc in $EF_VALUES; do
  cfg="$CFG_BASE/efc${efc}.yaml"
  out="$OUT_BASE/efc${efc}"
  mkdir -p "$out"
  {
    printf 'data: {dataset_name: %s, max_elements: -1, begin_num: -1, data_type: float, data_path: %s, incr_query_path: %s, overall_query_path: %s, overall_gt_path: %s}\n' \
      "$DATASET_NAME" "$DATA_PATH" "$QUERY_PATH" "$QUERY_PATH" "$GT_PATH"
    printf 'index: {index_type: hnsw, m: %s, ef_construction: %s}\n' "$M_HNSW" "$efc"
    printf 'search: {recall_at: 10, ef_search: %s, use_node_lock: true, enable_mvcc: false, enable_undo_recovery: false}\n' "$EF_SEARCH"
    printf 'workload: {batch_size: %s, num_threads: %s, queue_size: 0, insert_event_rate: 0, search_event_rate: 0, with_external_rw_lock: false, use_node_lock: true, query_mode: round_robin, per_query_latency: false, precompute_schedule: false}\n' "$BATCH_SIZE" "$THREADS"
    printf 'compare: {enabled: false}\n'
    printf 'result: {output_dir: %s}\n' "$out"
    printf 'repetitions: 1\n'
    printf 'profile: {enable_memory_profile: false}\n'
  } > "$cfg"

  echo "[hnsw-ef-quality] ef_construction=$efc -> $out"
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  (
    env LD_LIBRARY_PATH="$BENCH_LD_LIBRARY_PATH" "$BENCH_BIN" -config "$cfg"
  ) > "$out/run.log" 2>&1
done

python3 - <<'PY' "$OUT_BASE"
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for result in sorted(root.glob("efc*/benchmark_results.csv")):
    efc = result.parent.name.replace("efc", "")
    with result.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "ef_construction": efc,
                "recall": row.get("recall", ""),
                "search_p99_ms": row.get("search_p99_op_latency_ms (per-batch)", ""),
                "insert_qps": row.get("insert_qps (per-point)", ""),
                "search_qps": row.get("search_qps (per-point)", ""),
                "stats": row.get("stats", ""),
            })

out = root / "quality_guard_summary.csv"
with out.open("w", newline="") as f:
    if rows:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    else:
        f.write("")
print(f"[hnsw-ef-quality] summary -> {out}")
PY
