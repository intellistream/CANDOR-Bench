#!/usr/bin/env bash
# Validate whether rewrite-hit probability explains no-lock search batch spikes.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_rewrite_hit_probability_sweep_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_rewrite_hit_probability_sweep_${STAMP}}"
LOG_DIR="${ROOT}/logs"

WINDOWS="${WINDOWS:-0 100 1000 10000 100000}"
QUERY_MODES_MATRIX="${QUERY_MODES_MATRIX:-chasing round_robin}"
RATE_PAIRS_MATRIX="${RATE_PAIRS_MATRIX:-10000 90000}"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$LOG_DIR"

echo "[m3-hit-prob-sweep] root=$ROOT"
echo "[m3-hit-prob-sweep] windows=$WINDOWS"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  cmake --build ./build --target index -j "${BUILD_JOBS:-2}"
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .
fi

for window in $WINDOWS; do
  out="${ROOT}/window_${window}"
  cfg="${CONFIG_ROOT}/window_${window}"
  mkdir -p "$out" "$cfg"
  echo "[m3-hit-prob-sweep] run recent_window=${window}"
  env \
    ANNCHOR_M3_BATCH_DIAG=1 \
    ANNCHOR_SKIP_EXISTING_NODE_REPAIR=0 \
    ANNCHOR_REWRITE_RECENT_WINDOW="${window}" \
    DATASETS=sift \
    BATCHES=400 \
    RATE_PAIRS="${RATE_PAIRS_MATRIX}" \
    QUERY_MODES="${QUERY_MODES_MATRIX}" \
    SYNC_MODES=nolock \
    THREADS="${THREADS:-48}" \
    SEARCH_WORKERS="${SEARCH_WORKERS:-1}" \
    INSERT_WORKERS="${INSERT_WORKERS:-47}" \
    BEGIN_NUM="${BEGIN_NUM:-300000}" \
    MAX_ELEMENTS="${MAX_ELEMENTS:-360000}" \
    PRECOMPUTE_SCHEDULE=false \
    REPETITIONS="${REPETITIONS:-1}" \
    RUN_PERF=0 \
    ENABLE_UNDO_RECOVERY=false \
    INDEX_TYPE=annchor-m1 \
    ENABLE_MVCC=true \
    RESULT_BASE="$out" \
    CONFIG_DIR="$cfg" \
    bash ./scripts/run_m3_fluctuation_rootcause.sh \
    2>&1 | tee "${LOG_DIR}/window_${window}.log"

  python3 ./scripts/analyze_m3_batch_diag_rootcause.py "$out" \
    | tee "${LOG_DIR}/window_${window}_batch_diag.log"
done

export ROOT
python3 - <<'PY'
import csv
import os
import pathlib

root = pathlib.Path(os.environ["ROOT"])
rows = []
for summary in sorted(root.glob("window_*/m3_batch_diag_summary.csv")):
    window = summary.parent.name.removeprefix("window_")
    for r in csv.DictReader(open(summary)):
        rows.append({
            "recent_window": window,
            "label": r["label"],
            "query_mode": r.get("query_mode", ""),
            "p99_e2e_ms": r.get("p99_e2e_ms", ""),
            "p99_op_ms": r.get("p99_op_ms", ""),
            "active_query_hit_prob_mean": r.get("rewrite_active_query_hit_prob_mean", ""),
            "active_query_hit_prob_top": r.get("rewrite_active_query_hit_prob_top_mean", ""),
            "active_query_hit_prob_corr_op": r.get("rewrite_active_query_hit_prob_corr_op", ""),
            "recent_query_hit_prob_mean": r.get("rewrite_recent_query_hit_prob_mean", ""),
            "recent_query_hit_prob_top": r.get("rewrite_recent_query_hit_prob_top_mean", ""),
            "recent_query_hit_prob_corr_op": r.get("rewrite_recent_query_hit_prob_corr_op", ""),
            "recent_expansion_hits_per_hop_mean": r.get("rewrite_recent_expansion_hits_per_hop_mean", ""),
            "recent_expansion_hits_per_hop_top": r.get("rewrite_recent_expansion_hits_per_hop_top_mean", ""),
            "recent_expansion_hits_per_hop_corr_op": r.get("rewrite_recent_expansion_hits_per_hop_corr_op", ""),
        })

out = root / "combined_hit_probability_summary.csv"
if rows:
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(f"[m3-hit-prob-sweep] wrote {out}")
PY

echo "[m3-hit-prob-sweep] done root=${ROOT}"
