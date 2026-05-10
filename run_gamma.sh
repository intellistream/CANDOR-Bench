#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/scripts/candor_native_env.sh"

OUTPUT_DIR="results/gamma_three_algos_sift_1m_no_recall"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_gamma_${RUN_ID}.log"
STATUS_FILE="$LOG_DIR/run_gamma_${RUN_ID}.status"

echo "run_gamma log: $LOG_FILE"
echo "run_gamma status: $STATUS_FILE"

set +e
{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "cwd=$PWD"
    echo "python=$(uv run python -c 'import sys; print(sys.executable)')"
    echo "command=uv run sage-bench --experiment gamma_sweep --dataset sift --config runbooks/gamma/gamma_three_algos_sift_1m_no_recall.yaml --output $OUTPUT_DIR --plot"

    uv run sage-bench \
    --experiment gamma_sweep \
    --dataset sift \
    --config runbooks/gamma/gamma_three_algos_sift_1m_no_recall.yaml \
        --output "$OUTPUT_DIR" \
    --plot

    status=$?
    echo "finished_at=$(date --iso-8601=seconds)"
    echo "exit_code=$status"
    echo "$status" > "$STATUS_FILE"
    exit "$status"
} > "$LOG_FILE" 2>&1
status=$?
set -e

if [ "$status" -eq 0 ]; then
    echo "run_gamma completed successfully"
else
    echo "run_gamma failed with exit code $status"
    echo "last log lines:"
    tail -80 "$LOG_FILE"
fi

exit "$status"
