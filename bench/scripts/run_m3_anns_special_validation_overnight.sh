#!/usr/bin/env bash
# Validate whether nolock search e2e spikes are specific to live HNSW mutation.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_anns_special_validation_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_anns_special_validation_${STAMP}}"

mkdir -p "$ROOT" "$CONFIG_ROOT"

echo "[m3-anns-special] root=$ROOT"
echo "[m3-anns-special] config=$CONFIG_ROOT"

cmake --build ./build --target index -j "${BUILD_JOBS:-2}"
LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .

COMMON_ENV=(
  ANNCHOR_M3_BATCH_DIAG=1
  DATASETS="${DATASETS:-sift}"
  BATCHES="${BATCHES:-400}"
  RATE_PAIRS="${RATE_PAIRS:-40000 40000;10000 90000}"
  QUERY_MODES="${QUERY_MODES:-round_robin chasing}"
  SYNC_MODES=nolock
  THREADS="${THREADS:-48}"
  SEARCH_WORKERS="${SEARCH_WORKERS:-1}"
  PRECOMPUTE_SCHEDULE=false
  REPETITIONS=1
  RUN_PERF=0
  ENABLE_UNDO_RECOVERY=false
)

run_case() {
  local name="$1"
  local begin_num="$2"
  local max_elements="$3"
  local insert_workers="$4"
  shift 4
  local out="$ROOT/$name"
  local cfg="$CONFIG_ROOT/$name"

  echo "[m3-anns-special] run $name begin=$begin_num max=$max_elements iw=$insert_workers extra=$*"
  env "${COMMON_ENV[@]}" \
    BEGIN_NUM="$begin_num" \
    MAX_ELEMENTS="$max_elements" \
    INSERT_WORKERS="$insert_workers" \
    RESULT_BASE="$out" \
    CONFIG_DIR="$cfg" \
    "$@" \
    bash ./scripts/run_m3_fluctuation_rootcause.sh

  python3 ./scripts/analyze_m3_batch_diag_rootcause.py "$out" \
    | tee "$out/m3_batch_diag_analyze.log"
}

# Live HNSW admission-depth scaling. This is the target behavior.
for workers in ${LIVE_INSERT_WORKERS:-4 16 32}; do
  label="live_hnsw_workers_${workers}"
  if [[ "$workers" -ge 32 ]]; then
    label="live_hnsw_high_concurrency"
  fi
  run_case "$label" 300000 360000 "$workers"
done

# Generic CPU-heavy insert control. Same Go dispatcher/admission path, no graph mutation.
run_case "dummy_cpu_high_concurrency" 300000 550000 47 \
  INSERT_DUMMY_MODE=cpu INSERT_DUMMY_LOOPS="${DUMMY_CPU_LOOPS:-20000}"

# HNSW/TBB read-only control. Enters ANN search/TBB path from insert workers,
# but does not mutate the graph.
run_case "dummy_ann_search_high_concurrency" 300000 360000 47 \
  INSERT_DUMMY_MODE=ann_search INSERT_DUMMY_LOOPS="${DUMMY_ANN_SEARCH_LOOPS:-50}"

export ROOT
python3 - <<'PY'
import csv
import pathlib
import statistics

root = pathlib.Path(__import__("os").environ["ROOT"])
rows = []
for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    summary = case_dir / "m3_fluctuation_rootcause_summary.csv"
    if not summary.exists():
        continue
    for r in csv.DictReader(open(summary)):
        rows.append({
            "case": case_dir.name,
            "label": r["label"],
            "mean_e2e": float(r["mean_batch_e2e_ms"]),
            "p99_e2e": float(r["p99_batch_e2e_ms"]),
            "max_e2e": float(r["max_batch_e2e_ms"]),
            "worst100_e2e": float(r["worst100_batch_e2e_ms"]),
            "worst100_op": float(r["worst100_batch_op_ms"]),
            "worst100_queue": float(r["worst100_queue_ms_est"]),
            "active_max": float(r["max_active_front_pts"]),
            "missed_max": float(r["max_missed_e2e_pts"]),
        })

out = root / "anns_special_combined_summary.csv"
with open(out, "w", newline="") as f:
    fieldnames = list(rows[0].keys()) if rows else []
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if rows:
        w.writeheader()
        w.writerows(rows)

insert_rows = []
for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    for insert_wide in sorted(case_dir.glob("*/raw_latency/insert_wide.csv")):
        vals = [float(r["op_ms"]) for r in csv.DictReader(open(insert_wide))]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        insert_rows.append({
            "case": case_dir.name,
            "label": insert_wide.parents[1].name,
            "insert_op_mean": statistics.mean(vals),
            "insert_op_p95": vals_sorted[int(0.95 * (len(vals_sorted)-1))],
            "insert_op_p99": vals_sorted[int(0.99 * (len(vals_sorted)-1))],
        })

insert_out = root / "anns_special_insert_cost_summary.csv"
with open(insert_out, "w", newline="") as f:
    fieldnames = list(insert_rows[0].keys()) if insert_rows else []
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if insert_rows:
        w.writeheader()
        w.writerows(insert_rows)

print(f"[m3-anns-special] wrote {out}")
print(f"[m3-anns-special] wrote {insert_out}")
PY

cat > "$ROOT/README.md" <<EOF
# M3 ANNS-Special Spike Validation

Started: ${STAMP}

Purpose: validate whether nolock search batch e2e spikes require live HNSW
mutation, or whether generic CPU/TBB-heavy background work is sufficient.

Primary outputs:

\`\`\`text
$ROOT/anns_special_combined_summary.csv
$ROOT/anns_special_insert_cost_summary.csv
$ROOT/live_hnsw_workers_4
$ROOT/live_hnsw_workers_16
$ROOT/live_hnsw_high_concurrency
$ROOT/dummy_cpu_high_concurrency
$ROOT/dummy_ann_search_high_concurrency
\`\`\`

Decision rule:

\`\`\`text
ANNS-special confirmed if live_hnsw_high_concurrency has high search p99_e2e / worst100_e2e,
while dummy_cpu_high_concurrency and dummy_ann_search_high_concurrency have comparable background cost
but much lower search p99_e2e. Admission-depth root cause is supported if
live_hnsw p99_e2e increases from workers_4 -> workers_16 -> high_concurrency.
\`\`\`
EOF

echo "[m3-anns-special] done root=$ROOT"
