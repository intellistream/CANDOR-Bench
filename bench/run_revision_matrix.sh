#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="/home/junyao/code/ANN-CC-bench/bench"
BIN="$BENCH_DIR/bench_new2"
CFG_DIR="$BENCH_DIR/config/revision_matrix"
LOG_DIR="$BENCH_DIR/result/revision_matrix_logs"

mkdir -p "$LOG_DIR"
cd "$BENCH_DIR"
export LD_LIBRARY_PATH="$BENCH_DIR/build/lib:${LD_LIBRARY_PATH:-}"

DATASET_FILTER="${DATASET_FILTER:-}"
QUERY_FILTER="${QUERY_FILTER:-}"
WORKLOAD_FILTER="${WORKLOAD_FILTER:-}"
MODE_FILTER="${MODE_FILTER:-}"
LIMIT="${LIMIT:-0}"

count=0
shopt -s nullglob
for cfg in "$CFG_DIR"/*.yaml; do
  name=$(basename "$cfg" .yaml)
  [[ -n "$DATASET_FILTER" && "$name" != ${DATASET_FILTER}_* ]] && continue
  [[ -n "$QUERY_FILTER" && "$name" != *_${QUERY_FILTER}_* ]] && continue
  [[ -n "$WORKLOAD_FILTER" && "$name" != *_${WORKLOAD_FILTER}_* ]] && continue
  [[ -n "$MODE_FILTER" && "$name" != *_${MODE_FILTER} ]] && continue

  log="$LOG_DIR/${name}.log"
  echo "[BEGIN] $(date '+%F %T') $name"
  "$BIN" -config "$cfg" > "$log" 2>&1
  echo "[END]   $(date '+%F %T') $name"

  out_dir=$(python3 - <<'PY' "$cfg"
import sys, yaml
with open(sys.argv[1]) as f:
    cfg=yaml.safe_load(f)
print(cfg['result']['output_dir'])
PY
)

  python3 - <<'PY' "$BENCH_DIR/$out_dir/benchmark_results.csv" "$name"
import sys, pandas as pd
path, name = sys.argv[1], sys.argv[2]
df = pd.read_csv(path)
r = df.iloc[-1]
def get(*names):
    for n in names:
        if n in r.index:
            return r[n]
    return 'NA'
print('[SUMMARY] {name} insert_qps={iq:.1f} search_qps={sq:.1f} search_p99_ms={p99:.3f} recall={rec:.3f} freshness_p99={fp99}'.format(
    name=name,
    iq=float(get('insert_qps (per-point)','insert_qps')),
    sq=float(get('search_qps (per-point)','search_qps')),
    p99=float(get('search_p99_op_latency_ms (per-batch)','search_latency_p99_ms','op_latency_p99_ms')),
    rec=float(get('recall')),
    fp99=get('freshness_total_pts_p99')
))
PY

  count=$((count+1))
  if [[ "$LIMIT" != "0" && "$count" -ge "$LIMIT" ]]; then
    echo "[STOP] reached LIMIT=$LIMIT"
    break
  fi
done

echo "[DONE_ALL] $(date '+%F %T') ran=$count"
