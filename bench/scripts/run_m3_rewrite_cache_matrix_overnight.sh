#!/usr/bin/env bash
# Broader no-lock rewrite/prune cache-chain validation.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_rewrite_cache_matrix_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_rewrite_cache_matrix_${STAMP}}"
LOG_DIR="${ROOT}/logs"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$LOG_DIR"

echo "[m3-rewrite-cache-matrix] root=$ROOT"
echo "[m3-rewrite-cache-matrix] config=$CONFIG_ROOT"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  cmake --build ./build --target index -j "${BUILD_JOBS:-2}"
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .
fi

DATASET_CASES_DEFAULT=(
  "sift:300000:360000"
  "sift10m:300000:360000"
  "dbpedia_openai:50000:60000"
)

if [[ -n "${DATASET_CASES:-}" ]]; then
  IFS=';' read -r -a DATASET_CASES_ARR <<< "$DATASET_CASES"
else
  DATASET_CASES_ARR=("${DATASET_CASES_DEFAULT[@]}")
fi

RATE_PAIRS_MATRIX="${RATE_PAIRS_MATRIX:-10000 90000;40000 40000;90000 10000}"
QUERY_MODES_MATRIX="${QUERY_MODES_MATRIX:-chasing round_robin}"
CASES="${CASES:-baseline no_old_rewire}"

run_one_case() {
  local case_name="$1"
  local dataset="$2"
  local begin_num="$3"
  local max_elements="$4"
  local skip_old="$5"

  local out="${ROOT}/${case_name}/${dataset}"
  local cfg="${CONFIG_ROOT}/${case_name}/${dataset}"
  mkdir -p "$out" "$cfg"

  echo "[m3-rewrite-cache-matrix] run case=${case_name} dataset=${dataset} begin=${begin_num} max=${max_elements} skip=${skip_old}"
  env \
    ANNCHOR_M3_BATCH_DIAG=1 \
    ANNCHOR_SKIP_EXISTING_NODE_REPAIR="${skip_old}" \
    ANNCHOR_REWRITE_RECENT_WINDOW="${ANNCHOR_REWRITE_RECENT_WINDOW:-100000}" \
    DATASETS="${dataset}" \
    BATCHES="${BATCHES:-400}" \
    RATE_PAIRS="${RATE_PAIRS_MATRIX}" \
    QUERY_MODES="${QUERY_MODES_MATRIX}" \
    SYNC_MODES=nolock \
    THREADS="${THREADS:-48}" \
    SEARCH_WORKERS="${SEARCH_WORKERS:-1}" \
    INSERT_WORKERS="${INSERT_WORKERS:-47}" \
    BEGIN_NUM="${begin_num}" \
    MAX_ELEMENTS="${max_elements}" \
    PRECOMPUTE_SCHEDULE=false \
    REPETITIONS="${REPETITIONS:-1}" \
    RUN_PERF="${RUN_PERF:-1}" \
    PERF_EVENTS="${PERF_EVENTS:-cycles,instructions,cache-misses,LLC-load-misses,context-switches,cpu-migrations}" \
    ENABLE_UNDO_RECOVERY=false \
    INDEX_TYPE=annchor-m1 \
    ENABLE_MVCC=true \
    RESULT_BASE="$out" \
    CONFIG_DIR="$cfg" \
    bash ./scripts/run_m3_fluctuation_rootcause.sh \
    2>&1 | tee "${LOG_DIR}/${case_name}_${dataset}.log"

  python3 ./scripts/analyze_m3_batch_diag_rootcause.py "$out" \
    | tee "${LOG_DIR}/${case_name}_${dataset}_batch_diag.log"
}

for dataset_case in "${DATASET_CASES_ARR[@]}"; do
  IFS=':' read -r dataset begin_num max_elements <<< "$dataset_case"
  for case_name in ${CASES}; do
    case "$case_name" in
      baseline)
        run_one_case "$case_name" "$dataset" "$begin_num" "$max_elements" 0
        ;;
      no_old_rewire)
        run_one_case "$case_name" "$dataset" "$begin_num" "$max_elements" 1
        ;;
      *)
        echo "[m3-rewrite-cache-matrix] unknown case: $case_name" >&2
        exit 1
        ;;
    esac
  done
done

export ROOT
python3 - <<'PY'
import csv
import pathlib
import re
import os

root = pathlib.Path(os.environ["ROOT"])
rows = []
perf_rows = []

for summary in sorted(root.glob("*/*/m3_fluctuation_rootcause_summary.csv")):
    case = summary.parents[1].name
    dataset = summary.parents[0].name
    for r in csv.DictReader(open(summary)):
        rows.append({"case": case, "dataset": dataset, **r})

for perf in sorted(root.glob("*/*/*/perf_stat.txt")):
    case = perf.parents[2].name
    dataset = perf.parents[1].name
    label = perf.parents[0].name
    metrics = {"case": case, "dataset": dataset, "label": label}
    for line in open(perf, errors="ignore"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r"([0-9,]+)\s+([A-Za-z0-9_.-]+)", stripped)
        if m:
            metrics[m.group(2)] = int(m.group(1).replace(",", ""))
        elif "seconds time elapsed" in stripped:
            try:
                metrics["elapsed_sec"] = float(stripped.split()[0])
            except Exception:
                pass
    perf_rows.append(metrics)

def write_csv(path, data):
    if not data:
        return
    fields = []
    seen = set()
    for row in data:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(data)

write_csv(root / "combined_latency_summary.csv", rows)
write_csv(root / "combined_perf_summary.csv", perf_rows)

joined = []
perf_by = {(r["case"], r["dataset"], r["label"]): r for r in perf_rows}
for row in rows:
    key = (row["case"], row["dataset"], row["label"])
    merged = dict(row)
    if key in perf_by:
        for k, v in perf_by[key].items():
            if k not in {"case", "dataset", "label"}:
                merged[f"perf_{k}"] = v
    joined.append(merged)
write_csv(root / "combined_latency_perf_summary.csv", joined)

print(f"[m3-rewrite-cache-matrix] wrote {root / 'combined_latency_summary.csv'}")
print(f"[m3-rewrite-cache-matrix] wrote {root / 'combined_perf_summary.csv'}")
print(f"[m3-rewrite-cache-matrix] wrote {root / 'combined_latency_perf_summary.csv'}")
PY

cat > "$ROOT/README.md" <<EOF
# M3 Rewrite Cache Matrix

Purpose: validate across datasets, rates, and query modes that no-lock e2e
spikes come from HNSW/ANNchor existing-node rewrite/prune causing cache/LLC
pressure, not direct active-node hit or query path length.

Cases:

\`\`\`text
baseline          ANNchor-M1 no-lock insert with existing-node rewrite/prune
no_old_rewire     Same path but existing-node rewrite disabled
\`\`\`

Primary outputs:

\`\`\`text
$ROOT/combined_latency_summary.csv
$ROOT/combined_perf_summary.csv
$ROOT/combined_latency_perf_summary.csv
\`\`\`

Dataset windows:

\`\`\`text
${DATASET_CASES_ARR[*]}
\`\`\`

Rates:

\`\`\`text
$RATE_PAIRS_MATRIX
\`\`\`

Query modes:

\`\`\`text
$QUERY_MODES_MATRIX
\`\`\`
EOF

echo "[m3-rewrite-cache-matrix] done root=$ROOT"
