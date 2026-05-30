#!/usr/bin/env bash
# Focused preflight for no-lock HNSW rewrite-period spike evidence.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_rewrite_period_preflight_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_rewrite_period_preflight_${STAMP}}"
LOG_DIR="${ROOT}/logs"

DATASET="${DATASET:-sift}"
BATCHES_MATRIX="${BATCHES_MATRIX:-200}"
RATE_PAIRS_MATRIX="${RATE_PAIRS_MATRIX:-10000 90000;40000 40000}"
QUERY_MODES_MATRIX="${QUERY_MODES_MATRIX:-chasing round_robin}"
INSERT_WORKERS_MATRIX="${INSERT_WORKERS_MATRIX:-8 47}"
CASES="${CASES:-baseline no_old_rewire}"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$LOG_DIR"

echo "[m3-period-preflight] root=$ROOT"
echo "[m3-period-preflight] dataset=$DATASET batches=$BATCHES_MATRIX rates=$RATE_PAIRS_MATRIX modes=$QUERY_MODES_MATRIX workers=$INSERT_WORKERS_MATRIX cases=$CASES"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  cmake --build ./build --target index -j "${BUILD_JOBS:-2}"
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .
fi

run_case() {
  local case_name="$1"
  local insert_workers="$2"
  local skip_old="$3"
  local out="${ROOT}/${case_name}_workers${insert_workers}"
  local cfg="${CONFIG_ROOT}/${case_name}_workers${insert_workers}"

  mkdir -p "$out" "$cfg"
  echo "[m3-period-preflight] run case=${case_name} insert_workers=${insert_workers} skip_old=${skip_old}"
  env \
    ANNCHOR_M3_BATCH_DIAG=1 \
    ANNCHOR_SKIP_EXISTING_NODE_REPAIR="${skip_old}" \
    ANNCHOR_REWRITE_RECENT_WINDOW="${ANNCHOR_REWRITE_RECENT_WINDOW:-100000}" \
    DATASETS="${DATASET}" \
    BATCHES="${BATCHES_MATRIX}" \
    RATE_PAIRS="${RATE_PAIRS_MATRIX}" \
    QUERY_MODES="${QUERY_MODES_MATRIX}" \
    SYNC_MODES=nolock \
    THREADS="${THREADS:-48}" \
    SEARCH_WORKERS="${SEARCH_WORKERS:-1}" \
    INSERT_WORKERS="${insert_workers}" \
    BEGIN_NUM="${BEGIN_NUM:-300000}" \
    MAX_ELEMENTS="${MAX_ELEMENTS:-340000}" \
    PRECOMPUTE_SCHEDULE=false \
    REPETITIONS="${REPETITIONS:-1}" \
    RUN_PERF="${RUN_PERF:-0}" \
    PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,cache-misses,LLC-load-misses,context-switches,cpu-migrations}" \
    ENABLE_UNDO_RECOVERY=false \
    INDEX_TYPE=annchor-m1 \
    ENABLE_MVCC=true \
    RESULT_BASE="$out" \
    CONFIG_DIR="$cfg" \
    bash ./scripts/run_m3_fluctuation_rootcause.sh \
    2>&1 | tee "${LOG_DIR}/${case_name}_workers${insert_workers}.log"

  python3 ./scripts/analyze_m3_batch_diag_rootcause.py "$out" \
    | tee "${LOG_DIR}/${case_name}_workers${insert_workers}_batch_diag.log"
}

for iw in $INSERT_WORKERS_MATRIX; do
  for case_name in $CASES; do
    case "$case_name" in
      baseline)
        run_case "$case_name" "$iw" 0
        ;;
      no_old_rewire)
        run_case "$case_name" "$iw" 1
        ;;
      *)
        echo "[m3-period-preflight] unknown case: $case_name" >&2
        exit 1
        ;;
    esac
  done
done

export ROOT
python3 - <<'PY'
import csv
import os
import pathlib

root = pathlib.Path(os.environ["ROOT"])
latency = {}
for path in sorted(root.glob("*_workers*/m3_fluctuation_rootcause_summary.csv")):
    case = path.parent.name
    for row in csv.DictReader(open(path)):
        latency[(case, row["label"])] = row

rows = []
for path in sorted(root.glob("*_workers*/m3_batch_diag_summary.csv")):
    case = path.parent.name
    for row in csv.DictReader(open(path)):
        lat = latency.get((case, row["label"]), {})
        rows.append({
            "case": case,
            "label": row["label"],
            "query_mode": row.get("query_mode", ""),
            "write_rate": row.get("write_rate", ""),
            "read_rate": row.get("read_rate", ""),
            "batch": row.get("batch", ""),
            "num_batches": row.get("num_batches", ""),
            "p99_e2e_ms": lat.get("p99_batch_e2e_ms", row.get("p99_e2e_ms", "")),
            "max_e2e_ms": lat.get("max_batch_e2e_ms", ""),
            "p99_op_ms": row.get("p99_op_ms", ""),
            "period_query_hit_prob_mean": row.get("rewrite_period_query_hit_prob_mean", ""),
            "period_query_hit_prob_top": row.get("rewrite_period_query_hit_prob_top_mean", ""),
            "period_query_hit_prob_corr_op": row.get("rewrite_period_query_hit_prob_corr_op", ""),
            "period_expansion_hits_per_hop_mean": row.get("rewrite_period_expansion_hits_per_hop_mean", ""),
            "period_expansion_hits_per_hop_top": row.get("rewrite_period_expansion_hits_per_hop_top_mean", ""),
            "period_expansion_hits_per_hop_corr_op": row.get("rewrite_period_expansion_hits_per_hop_corr_op", ""),
            "period_active_writers_per_exp_mean": row.get("rewrite_period_active_writers_per_expansion_mean", ""),
            "period_active_writers_per_exp_top": row.get("rewrite_period_active_writers_per_expansion_top_mean", ""),
            "period_active_writer_max_mean": row.get("diag_rewrite_period_active_max_mean", ""),
            "period_active_writer_max_top": row.get("diag_rewrite_period_active_max_top_mean", ""),
            "active_node_query_hit_prob_mean": row.get("rewrite_active_query_hit_prob_mean", ""),
            "recent_node_query_hit_prob_mean": row.get("rewrite_recent_query_hit_prob_mean", ""),
            "dist_comps_top_enrich": row.get("diag_dist_comps_top_enrich", ""),
            "hops_top_enrich": row.get("diag_hops_top_enrich", ""),
        })

out = root / "combined_period_preflight_summary.csv"
if rows:
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(f"[m3-period-preflight] wrote {out}")
PY

echo "[m3-period-preflight] done root=${ROOT}"
