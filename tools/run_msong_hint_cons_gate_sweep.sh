#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT_DIR/bench/algorithms/streamseed_hybrid/config.yaml"
BACKUP="$(mktemp /tmp/streamseed_hybrid_config.XXXXXX.yaml)"

cp "$CONFIG" "$BACKUP"
restore_config() {
  cp "$BACKUP" "$CONFIG"
  rm -f "$BACKUP"
}
trap restore_config EXIT

set_hint_cons_gate() {
  local algorithm="$1"
  local gate="$2"

  python - "$CONFIG" "$algorithm" "$gate" <<'PY'
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
algorithm = sys.argv[2]
gate = sys.argv[3]

text = config_path.read_text()
lines = text.splitlines(keepends=True)

try:
    msong_start = next(i for i, line in enumerate(lines) if line == "msong:\n")
except StopIteration:
    raise SystemExit("Could not find top-level msong section")

msong_end = len(lines)
for i in range(msong_start + 1, len(lines)):
    if lines[i] and not lines[i].startswith((" ", "\n")):
        msong_end = i
        break

heading = f"  {algorithm}:\n"
try:
    alg_start = next(i for i in range(msong_start + 1, msong_end) if lines[i] == heading)
except StopIteration:
    raise SystemExit(f"Could not find msong.{algorithm} section")

alg_end = msong_end
for i in range(alg_start + 1, msong_end):
    if re.match(r"^  [^ ].*:\s*$", lines[i]):
        alg_end = i
        break

section = "".join(lines[alg_start:alg_end])
updated, count = re.subn(
    r'("hint_cons_gate":\s*)-?\d+(?:\.\d+)?',
    rf"\g<1>{gate}",
    section,
    count=1,
)
if count != 1:
    raise SystemExit(f"Could not update hint_cons_gate for msong.{algorithm}")

lines[alg_start:alg_end] = updated.splitlines(keepends=True)
config_path.write_text("".join(lines))
PY
}

run_one() {
  local algorithm="$1"
  local omp_threads="$2"
  local output_dir="$3"
  local gate="$4"
  local trial="$5"

  local result_base="$ROOT_DIR/results/msong/$algorithm"
  local source="$result_base/$output_dir"
  local target="$result_base/test${trial}-cons${gate}"

  if [[ -e "$target" ]]; then
    echo "ERROR: target result directory already exists: $target" >&2
    echo "Move or remove it before rerunning this sweep." >&2
    exit 1
  fi

  if [[ -e "$source" ]]; then
    local stale="${source}.pre_sweep_$(date +%Y%m%d_%H%M%S)_$$"
    echo "Preserving existing source result directory:"
    echo "  $source -> $stale"
    mv "$source" "$stale"
  fi

  echo
  echo "=== $algorithm | test=$trial | OMP_NUM_THREADS=$omp_threads | hint_cons_gate=$gate ==="
  set_hint_cons_gate "$algorithm" "$gate"

  (
    cd "$ROOT_DIR"
    OMP_NUM_THREADS="$omp_threads" python run_benchmark.py \
      --algorithm "$algorithm" \
      --dataset msong \
      --runbook runbooks/general_experiment/general_experiment.yaml
  )

  if [[ ! -d "$source" ]]; then
    echo "ERROR: expected result directory was not created: $source" >&2
    exit 1
  fi

  mv "$source" "$target"
  echo "Saved result directory: $target"
}

GATES=(-1.0 0.3 0.5 0.7 0.8 0.9 1.0)
TRIALS=(1 2 3)

for gate in "${GATES[@]}"; do
  for trial in "${TRIALS[@]}"; do
    run_one streamseed_hybrid 4 ef-120 "$gate" "$trial"
  done
done

for gate in "${GATES[@]}"; do
  for trial in "${TRIALS[@]}"; do
    run_one streamseed_hybrid_symphonyqg 4 ef-120 "$gate" "$trial"
  done
done

for gate in "${GATES[@]}"; do
  for trial in "${TRIALS[@]}"; do
    run_one streamseed_hybrid_freshdiskann 16 L-120_R-48 "$gate" "$trial"
  done
done
