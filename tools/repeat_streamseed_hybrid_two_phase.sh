#!/usr/bin/env bash
set -euo pipefail

# Two-phase benchmark workflow for streamseed_hybrid:
# 1) Run 5 repeats for sift/msong/glove using current config and archive as test1..test5.
# 2) Archive phase1 results per dataset into streamseed_hybrid_phase1_archive_<timestamp>.
# 3) For sift/msong/glove, set streamseed_hybrid_off to phase1 results.
# 4) Update config: set streamseed_mode to streamseed_core for streamseed_hybrid in
#    sift/msong/glove.
# 5) Run another 5 repeats for sift/msong/glove and archive as test1..test5 under
#    fresh results/*/streamseed_hybrid directories.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/sage-db-bench"

RUNBOOK="runbooks/general_experiment/general_experiment.yaml"
ALGO="streamseed_hybrid"
THREADS=4
REPEATS=5
DATASETS=("sift" "msong" "glove")
CONFIG_FILE="${REPO_ROOT}/bench/algorithms/streamseed_hybrid/config.yaml"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

activate_venv() {
  local had_u=0
  case $- in
    *u*) had_u=1 ;;
  esac
  set +u
  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
  elif [[ -f "${VENV_DIR}/local/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR}/local/bin/activate"
  else
    die "Virtualenv not found under ${VENV_DIR}. Please run deploy.sh first."
  fi
  if [[ ${had_u} -eq 1 ]]; then
    set -u
  fi
  echo "Using python: $(command -v python)"
}

ensure_pycandy() {
  if python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('PyCANDYAlgo') else 1)
PY
  then
    echo "PyCANDYAlgo is available."
    return
  fi

  echo "PyCANDYAlgo not found. Building/installing via build_PyCANDY.sh..."
  bash "${REPO_ROOT}/build_PyCANDY.sh"

  if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('PyCANDYAlgo') else 1)
PY
  then
    die "PyCANDYAlgo is still unavailable after build_PyCANDY.sh"
  fi

  echo "PyCANDYAlgo installed successfully."
}

backup_dir_if_exists() {
  local src="$1"
  local tag="$2"
  if [[ -d "${src}" ]]; then
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local dst="${src}_${tag}_${ts}"
    mv "${src}" "${dst}"
    echo "Moved ${src} -> ${dst}"
  fi
}

run_cycle() {
  local phase="$1"
  echo ""
  echo "==================== ${phase}: benchmark cycle start ===================="

  for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "===== Dataset: ${dataset} (${phase}) ====="

    local result_base="results/${dataset}/${ALGO}"
    local src_dir="${result_base}/ef-120"

    mkdir -p "${result_base}"

    for i in $(seq 1 "${REPEATS}"); do
      local dst_dir="${result_base}/test${i}"

      echo ""
      echo "[${dataset}][${phase}] Run ${i}/${REPEATS}"

      if [[ -e "${dst_dir}" ]]; then
        die "Target directory already exists: ${dst_dir}. Please remove/rename it and rerun."
      fi

      OMP_NUM_THREADS="${THREADS}" python run_benchmark.py \
        --algorithm "${ALGO}" \
        --dataset "${dataset}" \
        --runbook "${RUNBOOK}" \
        --enable-cache-profiling

      [[ -d "${src_dir}" ]] || die "Expected result directory not found: ${src_dir}"
      mv "${src_dir}" "${dst_dir}"
      echo "[${dataset}][${phase}] Archived result: ${dst_dir}"
    done
  done

  echo "==================== ${phase}: benchmark cycle done ====================="
}

switch_all_dataset_modes_to_core() {
  [[ -f "${CONFIG_FILE}" ]] || die "Config file not found: ${CONFIG_FILE}"

  CONFIG_PATH="${CONFIG_FILE}" python - <<'PY'
from pathlib import Path
import os
import re

cfg_path = Path(os.environ["CONFIG_PATH"])
lines = cfg_path.read_text(encoding="utf-8").splitlines(keepends=True)

targets = {"sift", "msong", "glove"}
updated = {k: False for k in targets}
current_dataset = None
in_streamseed_hybrid = False

dataset_re = re.compile(r"^(sift|msong|glove):\s*$")

for i, line in enumerate(lines):
    if line.startswith(" "):
        pass
    else:
        current_dataset = None
        in_streamseed_hybrid = False
        m = dataset_re.match(line.strip("\n"))
        if m:
            current_dataset = m.group(1)
        continue

    if current_dataset not in targets:
        continue

    if re.match(r"^  streamseed_hybrid:\s*$", line):
        in_streamseed_hybrid = True
        continue

    if re.match(r"^  [A-Za-z0-9_\-]+:\s*$", line) and not re.match(r"^  streamseed_hybrid:\s*$", line):
        in_streamseed_hybrid = False
        continue

    if in_streamseed_hybrid and (not updated[current_dataset]) and '"streamseed_mode": "off"' in line:
        lines[i] = line.replace('"streamseed_mode": "off"', '"streamseed_mode": "streamseed_core"', 1)
        updated[current_dataset] = True

cfg_path.write_text("".join(lines), encoding="utf-8")

for ds in ("sift", "msong", "glove"):
    state = "updated" if updated[ds] else "already-core-or-not-found"
    print(f"[{ds}] {state}")
PY

  echo "Applied streamseed_mode switch to streamseed_hybrid sections in ${CONFIG_FILE}"
}

cd "${REPO_ROOT}"
activate_venv
ensure_pycandy

echo "Repo root: ${REPO_ROOT}"
echo "Algorithm: ${ALGO}"
echo "Runbook: ${RUNBOOK}"
echo "OMP_NUM_THREADS: ${THREADS}"
echo "Repeats per dataset: ${REPEATS}"

# Prepare clean directories for phase 1.
for dataset in "${DATASETS[@]}"; do
  backup_dir_if_exists "results/${dataset}/${ALGO}" "prephase1_backup"
done

# Phase 1 with current config.
run_cycle "phase1"

# Archive phase1 results and generate *_off for all datasets.
phase1_ts=$(date +%Y%m%d_%H%M%S)
for dataset in "${DATASETS[@]}"; do
  src="results/${dataset}/${ALGO}"
  archive="results/${dataset}/${ALGO}_phase1_archive_${phase1_ts}"
  off_dir="results/${dataset}/${ALGO}_off"

  [[ -d "${src}" ]] || die "Missing ${src} after phase1"
  mv "${src}" "${archive}"
  echo "Archived phase1: ${src} -> ${archive}"

  backup_dir_if_exists "${off_dir}" "pre_rename_backup"
  cp -a "${archive}" "${off_dir}"
  echo "Created off baseline: ${off_dir}"
done

# Update config for all dataset mode switches.
switch_all_dataset_modes_to_core

# Phase 2 after mode switch.
run_cycle "phase2"

echo ""
echo "Two-phase workflow completed successfully."
