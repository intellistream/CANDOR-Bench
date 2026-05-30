#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$BENCH_DIR/.." && pwd)"

OUT_ROOT="${1:-/tmp/m3_evidence_$(date +%Y%m%d_%H%M%S)}"
REPS="${2:-1}"

mkdir -p "$OUT_ROOT"

# Rebuild native and go bench to avoid stale lib/binary mismatch.
cmake --build "$BENCH_DIR/build" -j --target index
(
  cd "$BENCH_DIR"
  mkdir -p /tmp/go-build-cache /tmp/go-mod-cache
  GOCACHE=/tmp/go-build-cache GOMODCACHE=/tmp/go-mod-cache /usr/local/go/bin/go build -o "$OUT_ROOT/bench" .
)

make_cfg() {
  local src_cfg="$1"
  local out_cfg="$2"
  local out_dir="$3"
  sed -E \
    -e "s|^  output_dir:.*$|  output_dir: ${out_dir}|" \
    -e "s|^repetitions:.*$|repetitions: ${REPS}|" \
    "$src_cfg" > "$out_cfg"
}

run_one() {
  local cfg_src="$1"
  local tag="$2"
  local cfg_tmp="$OUT_ROOT/${tag}.yaml"
  local out_dir="$OUT_ROOT/${tag}"
  make_cfg "$cfg_src" "$cfg_tmp" "$out_dir"
  echo "=== RUN ${tag} ==="
  (
    cd "$BENCH_DIR"
    export LD_LIBRARY_PATH="$BENCH_DIR/build/lib"
    "$OUT_ROOT/bench" -config "$cfg_tmp"
  )
}

# Chasing matrix: deep1M / gist / msong x (m3, m3-off, m3-ablation)
run_one "$BENCH_DIR/config/m3_chasing_deep1M.yaml"      "m3_deep1M_chasing"
run_one "$BENCH_DIR/config/m3off_chasing_deep1M.yaml"   "m3off_deep1M_chasing"
run_one "$BENCH_DIR/config/m3abl_chasing_deep1M.yaml"   "m3abl_deep1M_chasing"

run_one "$BENCH_DIR/config/m3_chasing_gist.yaml"        "m3_gist_chasing"
run_one "$BENCH_DIR/config/m3off_chasing_gist.yaml"     "m3off_gist_chasing"
run_one "$BENCH_DIR/config/m3abl_chasing_gist.yaml"     "m3abl_gist_chasing"

run_one "$BENCH_DIR/config/m3_chasing_msong.yaml"       "m3_msong_chasing"
run_one "$BENCH_DIR/config/m3off_chasing_msong.yaml"    "m3off_msong_chasing"
run_one "$BENCH_DIR/config/m3abl_chasing_msong.yaml"    "m3abl_msong_chasing"

python "$SCRIPT_DIR/m3_evidence_summary.py" "$OUT_ROOT"

echo "Evidence root: $OUT_ROOT"
