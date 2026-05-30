#!/usr/bin/env bash
# Mechanism-only no-lock HNSW/ANNS spike validation.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_nolock_anns_mechanism_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_nolock_anns_mechanism_${STAMP}}"

mkdir -p "$ROOT" "$CONFIG_ROOT"

echo "[m3-nolock-anns-mechanism] root=$ROOT"
echo "[m3-nolock-anns-mechanism] config=$CONFIG_ROOT"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  cmake --build ./build --target index -j "${BUILD_JOBS:-2}"
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .
fi

COMMON_ENV=(
  ANNCHOR_M3_BATCH_DIAG=1
  DATASETS="${DATASETS:-sift}"
  BATCHES="${BATCHES:-400}"
  RATE_PAIRS="${RATE_PAIRS:-10000 90000}"
  QUERY_MODES="${QUERY_MODES:-chasing round_robin}"
  SYNC_MODES=nolock
  THREADS="${THREADS:-48}"
  SEARCH_WORKERS="${SEARCH_WORKERS:-1}"
  INSERT_WORKERS="${INSERT_WORKERS:-47}"
  BEGIN_NUM="${BEGIN_NUM:-300000}"
  MAX_ELEMENTS="${MAX_ELEMENTS:-360000}"
  PRECOMPUTE_SCHEDULE=false
  REPETITIONS="${REPETITIONS:-1}"
  RUN_PERF=0
  ENABLE_UNDO_RECOVERY=false
)

run_case() {
  local name="$1"
  shift
  local out="$ROOT/$name"
  local cfg="$CONFIG_ROOT/$name"

  echo "[m3-nolock-anns-mechanism] run $name extra=$*"
  env "${COMMON_ENV[@]}" \
    RESULT_BASE="$out" \
    CONFIG_DIR="$cfg" \
    "$@" \
    bash ./scripts/run_m3_fluctuation_rootcause.sh

  python3 ./scripts/analyze_m3_batch_diag_rootcause.py "$out" \
    | tee "$out/m3_batch_diag_analyze.log"
}

# Pure HNSW, no MVCC/undo. If this spikes, the phenomenon is not ANNCHOR-MVCC-specific.
run_case "pure_hnsw_nolock_high_concurrency" \
  INDEX_TYPE=hnsw ENABLE_MVCC=false

# Same ANNchor-M1 implementation with MVCC disabled. This isolates wrapper/codepath effects.
run_case "annchor_m1_disabled_nolock_high_concurrency" \
  INDEX_TYPE=annchor-m1 ENABLE_MVCC=false ANNCHOR_SKIP_EXISTING_NODE_REPAIR=0

# Target production-like no-lock path, but undo recovery/OCC merge stays disabled.
run_case "annchor_m1_enabled_nolock_high_concurrency" \
  INDEX_TYPE=annchor-m1 ENABLE_MVCC=true ANNCHOR_SKIP_EXISTING_NODE_REPAIR=0

# Mechanism ablation: still inserts into HNSW, but does not rewrite existing node
# adjacency lists. This is not a correctness-preserving implementation; it is
# only a causal test for online graph rewiring.
run_case "annchor_old_repair_removed_nolock_high_concurrency" \
  INDEX_TYPE=annchor-m1 ENABLE_MVCC=true ANNCHOR_SKIP_EXISTING_NODE_REPAIR=1

export ROOT
python3 - <<'PY'
import csv
import os
import pathlib
import statistics

root = pathlib.Path(os.environ["ROOT"])

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

out = root / "nolock_anns_mechanism_combined_summary.csv"
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
        with open(insert_wide) as f:
            r0 = next(csv.DictReader(f), None)
        extra = {}
        if r0:
            for key in [
                "graph_old_update_loop_ms",
                "graph_prune_heuristic_ms",
                "graph_undo_record_ms",
                "graph_existing_node_visits",
                "graph_existing_node_appends",
                "graph_existing_node_prunes",
                "graph_pruned_edges",
            ]:
                extra[f"{key}_first"] = float(r0.get(key, 0.0) or 0.0)
        # Full-column means for insertion mechanism counters.
        columns = {}
        with open(insert_wide) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in [
                    "graph_old_update_loop_ms",
                    "graph_prune_heuristic_ms",
                    "graph_undo_record_ms",
                    "graph_existing_node_visits",
                    "graph_existing_node_appends",
                    "graph_existing_node_prunes",
                    "graph_pruned_edges",
                ]:
                    columns.setdefault(key, []).append(float(row.get(key, 0.0) or 0.0))
        insert_rows.append({
            "case": case_dir.name,
            "label": insert_wide.parents[1].name,
            "insert_op_mean": statistics.mean(vals),
            "insert_op_p95": vals_sorted[int(0.95 * (len(vals_sorted)-1))],
            "insert_op_p99": vals_sorted[int(0.99 * (len(vals_sorted)-1))],
            **{f"{k}_mean": statistics.mean(v) for k, v in columns.items() if v},
            **extra,
        })

insert_out = root / "nolock_anns_mechanism_insert_summary.csv"
with open(insert_out, "w", newline="") as f:
    fieldnames = list(insert_rows[0].keys()) if insert_rows else []
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if insert_rows:
        w.writeheader()
        w.writerows(insert_rows)

print(f"[m3-nolock-anns-mechanism] wrote {out}")
print(f"[m3-nolock-anns-mechanism] wrote {insert_out}")
PY

cat > "$ROOT/README.md" <<EOF
# M3 No-Lock ANNS Mechanism Validation

Purpose: decide whether no-lock search batch e2e spikes are just generic
resource competition, or whether they require HNSW/ANNS online graph mutation.

Cases:

\`\`\`text
pure_hnsw_nolock_high_concurrency                 HNSW no-lock, no MVCC
annchor_m1_disabled_nolock_high_concurrency     ANNchor-M1 codepath with MVCC off
annchor_m1_enabled_nolock_high_concurrency      target no-lock path, MVCC on, undo recovery off
annchor_old_repair_removed_nolock_high_concurrency mechanism ablation: existing-node repair removed
\`\`\`

Primary outputs:

\`\`\`text
$ROOT/nolock_anns_mechanism_combined_summary.csv
$ROOT/nolock_anns_mechanism_insert_summary.csv
\`\`\`

Decision rule:

\`\`\`text
If pure_hnsw and annchor_m1_enabled both spike, but no_old_rewire collapses
the spike and graph_old_update_loop/prune counters collapse, the deeper root
cause is HNSW/ANNS online graph rewiring under no-lock concurrent search/insert.
If no_old_rewire still spikes, the cause is broader than existing-node rewiring.
If only annchor_m1_enabled spikes, the cause is MVCC/visibility instrumentation,
not generic ANNS.
\`\`\`
EOF

echo "[m3-nolock-anns-mechanism] done root=$ROOT"
