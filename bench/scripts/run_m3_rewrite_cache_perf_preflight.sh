#!/usr/bin/env bash
# Perf-counter preflight for rewrite-period cache pressure.
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-./result/m3_rewrite_cache_perf_preflight_${STAMP}}"
CONFIG_ROOT="${CONFIG_ROOT:-./config/m3_rewrite_cache_perf_preflight_${STAMP}}"
LOG_DIR="${ROOT}/logs"

mkdir -p "$ROOT" "$CONFIG_ROOT" "$LOG_DIR"

echo "[m3-cache-perf] root=$ROOT"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  cmake --build ./build --target index -j "${BUILD_JOBS:-2}"
  LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .
fi

EVENT_GROUPS=(
  "l1|cycles,instructions,L1-dcache-loads,L1-dcache-load-misses"
  "l2_llc|l2_rqsts.references,l2_rqsts.miss,LLC-loads,LLC-load-misses"
  "cache_tlb|cache-references,cache-misses,dtlb_load_misses.walk_completed,context-switches,cpu-migrations"
)

for group in "${EVENT_GROUPS[@]}"; do
  IFS='|' read -r group_name events <<< "$group"
  echo "[m3-cache-perf] run group=${group_name} events=${events}"
  env \
    ROOT="${ROOT}/${group_name}" \
    CONFIG_ROOT="${CONFIG_ROOT}/${group_name}" \
    SKIP_BUILD=1 \
    RUN_PERF=1 \
    PERF_EVENTS="${events}" \
    DATASET="${DATASET:-sift}" \
    BATCHES_MATRIX="${BATCHES_MATRIX:-200}" \
    RATE_PAIRS_MATRIX="${RATE_PAIRS_MATRIX:-40000 40000}" \
    QUERY_MODES_MATRIX="${QUERY_MODES_MATRIX:-chasing round_robin}" \
    INSERT_WORKERS_MATRIX="${INSERT_WORKERS_MATRIX:-47}" \
    CASES="${CASES:-baseline no_old_rewire}" \
    THREADS="${THREADS:-48}" \
    SEARCH_WORKERS="${SEARCH_WORKERS:-1}" \
    BEGIN_NUM="${BEGIN_NUM:-300000}" \
    MAX_ELEMENTS="${MAX_ELEMENTS:-340000}" \
    bash ./scripts/run_m3_rewrite_period_preflight.sh \
    2>&1 | tee "${LOG_DIR}/${group_name}.log"
done

export ROOT
python3 - <<'PY'
import csv
import os
import pathlib
import re

root = pathlib.Path(os.environ["ROOT"])

def parse_perf(path: pathlib.Path) -> dict[str, str]:
    out = {}
    for line in open(path, errors="ignore"):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("Performance"):
            continue
        if "seconds time elapsed" in s:
            out["seconds_time_elapsed"] = s.split()[0]
            continue
        m = re.match(r"([0-9,]+)\s+([A-Za-z0-9_.:-]+)", s)
        if m:
            out[m.group(2)] = m.group(1).replace(",", "")
    return out

lat = {}
diag = {}
for summary in root.glob("*/*_workers*/m3_fluctuation_rootcause_summary.csv"):
    group = summary.parents[1].name
    case = summary.parent.name
    for row in csv.DictReader(open(summary)):
        lat[(group, case, row["label"])] = row

for summary in root.glob("*/*_workers*/m3_batch_diag_summary.csv"):
    group = summary.parents[1].name
    case = summary.parent.name
    for row in csv.DictReader(open(summary)):
        diag[(group, case, row["label"])] = row

rows = []
for perf in sorted(root.glob("*/*_workers*/*/perf_stat.txt")):
    group = perf.parents[2].name
    case = perf.parents[1].name
    label = perf.parent.name
    row = {
        "event_group": group,
        "case": case,
        "label": label,
        **parse_perf(perf),
    }
    l = lat.get((group, case, label), {})
    d = diag.get((group, case, label), {})
    row.update({
        "query_mode": l.get("query_mode", ""),
        "write_rate": l.get("write_rate", ""),
        "read_rate": l.get("read_rate", ""),
        "p99_e2e_ms": l.get("p99_batch_e2e_ms", ""),
        "mean_e2e_ms": l.get("mean_batch_e2e_ms", ""),
        "period_query_hit_prob_mean": d.get("rewrite_period_query_hit_prob_mean", ""),
        "active_node_query_hit_prob_mean": d.get("rewrite_active_query_hit_prob_mean", ""),
    })
    rows.append(row)

fields = []
seen = set()
for row in rows:
    for k in row:
        if k not in seen:
            seen.add(k)
            fields.append(k)

out = root / "combined_cache_perf_summary.csv"
if rows:
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
print(f"[m3-cache-perf] wrote {out}")
PY

echo "[m3-cache-perf] done root=${ROOT}"
