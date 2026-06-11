#!/usr/bin/env bash
set -euo pipefail

# Run streamseed_hybrid_freshdiskann benchmark 10 times per dataset and archive
# each run result directory as test{1..10}.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/sage-db-bench"

RUNBOOK="runbooks/general_experiment/general_experiment.yaml"
CONFIG="bench/algorithms/streamseed_hybrid/config.yaml"
ALGO="streamseed_hybrid_freshdiskann"
THREADS=16
REPEATS=10
TEST_START=1
DATASETS=("sift" "msong" "glove")

cd "${REPO_ROOT}"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

activate_venv() {
  # Some activate scripts reference unset vars (e.g. LD_LIBRARY_PATH).
  # Temporarily disable nounset while sourcing.
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

expected_result_dir() {
  local dataset="$1"
  python - "${CONFIG}" "${dataset}" <<'PY'
import json
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
dataset = sys.argv[2]
text = config_path.read_text()


def section(src, start_pattern, end_pattern=None):
    match = re.search(start_pattern, src, flags=re.MULTILINE)
    if not match:
        raise SystemExit(f"missing section: {start_pattern}")
    start = match.start()
    if end_pattern is None:
        return src[start:]
    end_match = re.search(end_pattern, src[match.end() :], flags=re.MULTILINE)
    end = len(src) if end_match is None else match.end() + end_match.start()
    return src[start:end]


dataset_block = section(text, rf"^{re.escape(dataset)}:\s*$", r"^\S")
algo_block = section(dataset_block, r"^  streamseed_hybrid_freshdiskann:\s*$", r"^  \S")
args_match = re.search(r"args:\s*\|\n((?:          .+\n)+)", algo_block)
if not args_match:
    raise SystemExit("missing args block")
args_text = "\n".join(line[10:] for line in args_match.group(1).splitlines())
args = json.loads(args_text)[0]
print(f"L-{int(args['L'])}_R-{int(args['R'])}")
PY
}

archive_result_dir() {
  local result_base="$1"
  local before_file="$2"
  local expected_name="$3"
  local dst_dir="$4"

  [[ -e "${dst_dir}" ]] && die "Target directory already exists: ${dst_dir}. Please remove/rename it and rerun."

  local src_dir=""
  while IFS= read -r candidate; do
    if ! grep -Fxq "${candidate}" "${before_file}"; then
      src_dir="${candidate}"
      break
    fi
  done < <(find "${result_base}" -mindepth 1 -maxdepth 1 -type d ! -name 'test*' -printf '%p\n' | sort)

  if [[ -z "${src_dir}" && -d "${result_base}/${expected_name}" ]]; then
    src_dir="${result_base}/${expected_name}"
  fi

  [[ -n "${src_dir}" ]] || die "Could not find new result directory under ${result_base}"
  [[ -d "${src_dir}" ]] || die "Expected result directory not found: ${src_dir}"

  mv "${src_dir}" "${dst_dir}"
  echo "Archived result: ${dst_dir}"
}

activate_venv
ensure_pycandy

echo "Repo root: ${REPO_ROOT}"
echo "Algorithm: ${ALGO}"
echo "Runbook: ${RUNBOOK}"
echo "OMP_NUM_THREADS: ${THREADS}"
echo "Repeats per dataset: ${REPEATS}"
echo "Archive names: test${TEST_START}..test$((TEST_START + REPEATS - 1))"

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "===== Dataset: ${dataset} ====="

  result_base="results/${dataset}/${ALGO}"
  expected_name="$(expected_result_dir "${dataset}")"
  mkdir -p "${result_base}"

  for i in $(seq 1 "${REPEATS}"); do
    test_id=$((TEST_START + i - 1))
    dst_dir="${result_base}/test${test_id}"
    before_file="$(mktemp)"

    echo ""
    echo "[${dataset}] Run ${i}/${REPEATS} -> test${test_id}"
    echo "[${dataset}] Expected raw result directory: ${result_base}/${expected_name}"

    find "${result_base}" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' | sort > "${before_file}"

    OMP_NUM_THREADS="${THREADS}" python run_benchmark.py \
      --algorithm "${ALGO}" \
      --dataset "${dataset}" \
      --runbook "${RUNBOOK}" \
      --enable-cache-profiling

    archive_result_dir "${result_base}" "${before_file}" "${expected_name}" "${dst_dir}"
    rm -f "${before_file}"
  done
done

echo ""
echo "All runs completed successfully."
