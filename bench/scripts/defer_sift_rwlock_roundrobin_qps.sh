#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WAIT_PID="${1:?wait pid required}"
OUT_DIR="${2:?out dir required}"

mkdir -p "$OUT_DIR"

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

cd "$ROOT/bench"
env \
  INSERT_RATIO=1 \
  LD_LIBRARY_PATH="$ROOT/bench/build/lib:${LD_LIBRARY_PATH:-}" \
  /usr/local/go/bin/go run ./cmd/bench_dev -config ./config/sift_rwlock_roundrobin_qps_matrix.yaml \
  > "$OUT_DIR/run.log" 2>&1
