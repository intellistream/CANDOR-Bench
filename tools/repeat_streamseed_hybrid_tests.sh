#!/usr/bin/env bash
set -euo pipefail

# Run streamseed_hybrid benchmark 10 times per dataset and archive each run
# result directory from ef-120 to test{1..10}.

#python tools/plot_glove_streamseed_hybrid_mean_qps.py   --base-dir results/sift/streamseed_hybrid   --output tools/sift_streamseed_hybrid_mean_qps.png   --trim-k 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/sage-db-bench"

RUNBOOK="runbooks/general_experiment/general_experiment.yaml"
ALGO="streamseed_hybrid"
THREADS=4
REPEATS=5
REPEATS=10
TEST_START=1
DATASETS=("sift" "msong" "glove")

cd "${REPO_ROOT}"

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

activate_venv
ensure_pycandy

echo "Repo root: ${REPO_ROOT}"
echo "Algorithm: ${ALGO}"
echo "Runbook: ${RUNBOOK}"
echo "OMP_NUM_THREADS: ${THREADS}"
echo "Repeats per dataset: ${REPEATS}"
echo "Archive names: test${TEST_START}..test$((TEST_START + REPEATS - 1))"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "===== Dataset: ${dataset} ====="

  result_base="results/${dataset}/${ALGO}"
  src_dir="${result_base}/ef-120"

  mkdir -p "${result_base}"

  for i in $(seq 1 "${REPEATS}"); do
    test_id=$((TEST_START + i - 1))
    dst_dir="${result_base}/test${test_id}"

    echo ""
    echo "[${dataset}] Run ${i}/${REPEATS} -> test${test_id}"

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
    echo "[${dataset}] Archived result: ${dst_dir}"
  done

done

echo ""
echo "All runs completed successfully."
