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

set_hint_slot_params() {
  local algorithm="$1"
  local slots="$2"
  local capacity="$3"

  python - "$CONFIG" "$algorithm" "$slots" "$capacity" <<'PY'
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
algorithm = sys.argv[2]
slots = sys.argv[3]
capacity = sys.argv[4]

text = config_path.read_text()
lines = text.splitlines(keepends=True)

try:
    sift_start = next(i for i, line in enumerate(lines) if line == "sift:\n")
except StopIteration:
    raise SystemExit("Could not find top-level sift section")

sift_end = len(lines)
for i in range(sift_start + 1, len(lines)):
    if lines[i] and not lines[i].startswith((" ", "\n")):
        sift_end = i
        break

heading = f"  {algorithm}:\n"
try:
    alg_start = next(i for i in range(sift_start + 1, sift_end) if lines[i] == heading)
except StopIteration:
    raise SystemExit(f"Could not find sift.{algorithm} section")

alg_end = sift_end
for i in range(alg_start + 1, sift_end):
    if re.match(r"^  [^ ].*:\s*$", lines[i]):
        alg_end = i
        break

section = "".join(lines[alg_start:alg_end])
section, slots_count = re.subn(
    r'("hint_table_slots":\s*)\d+',
    rf"\g<1>{slots}",
    section,
    count=1,
)
section, capacity_count = re.subn(
    r'("hint_slot_capacity":\s*)\d+',
    rf"\g<1>{capacity}",
    section,
    count=1,
)
if slots_count != 1:
    raise SystemExit(f"Could not update hint_table_slots for sift.{algorithm}")
if capacity_count != 1:
    raise SystemExit(f"Could not update hint_slot_capacity for sift.{algorithm}")

lines[alg_start:alg_end] = section.splitlines(keepends=True)
config_path.write_text("".join(lines))
PY
}

run_one() {
  local algorithm="$1"
  local omp_threads="$2"
  local output_dir="$3"
  local slots="$4"
  local capacity="$5"
  local trial="$6"

  local result_base="$ROOT_DIR/results/sift/$algorithm"
  local source="$result_base/$output_dir"
  local target="$result_base/test${trial}-slot${slots}-capacity${capacity}"

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
  echo "=== $algorithm | test=$trial | OMP_NUM_THREADS=$omp_threads | hint_table_slots=$slots | hint_slot_capacity=$capacity ==="
  set_hint_slot_params "$algorithm" "$slots" "$capacity"

  (
    cd "$ROOT_DIR"
    OMP_NUM_THREADS="$omp_threads" python run_benchmark.py \
      --algorithm "$algorithm" \
      --dataset sift \
      --runbook runbooks/general_experiment/general_experiment.yaml
  )

  if [[ ! -d "$source" ]]; then
    echo "ERROR: expected result directory was not created: $source" >&2
    exit 1
  fi

  mv "$source" "$target"
  echo "Saved result directory: $target"
}

SLOTS=(6000)
CAPACITIES=(20 10)
TRIALS=(1 2 3)

for slots in "${SLOTS[@]}"; do
  for capacity in "${CAPACITIES[@]}"; do
    for trial in "${TRIALS[@]}"; do
      run_one streamseed_hybrid 4 ef-120 "$slots" "$capacity" "$trial"
    done
  done
done

for slots in "${SLOTS[@]}"; do
  for capacity in "${CAPACITIES[@]}"; do
    for trial in "${TRIALS[@]}"; do
      run_one streamseed_hybrid_symphonyqg 4 ef-120 "$slots" "$capacity" "$trial"
    done
  done
done

for slots in "${SLOTS[@]}"; do
  for capacity in "${CAPACITIES[@]}"; do
    for trial in "${TRIALS[@]}"; do
      run_one streamseed_hybrid_freshdiskann 16 L-120_R-48 "$slots" "$capacity" "$trial"
    done
  done
done
