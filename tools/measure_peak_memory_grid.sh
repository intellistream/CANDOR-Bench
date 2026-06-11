#!/usr/bin/env bash
set -u

# Measure peak RSS for a small benchmark grid without requiring /usr/bin/time.
# It samples /proc/<pid>/status while each benchmark process is running.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/memory_peak}"
RUNBOOK="${RUNBOOK:-runbooks/general_experiment/general_experiment.yaml}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1}"

mkdir -p "$OUT_DIR"

SUMMARY="$OUT_DIR/peak_memory_summary.csv"
echo "dataset,algorithm,omp_threads,runbook,exit_code,elapsed_sec,peak_rss_kb,peak_rss_gb,trace_file,log_file" > "$SUMMARY"

DATASETS=(sift msong glove)

run_one() {
    local dataset="$1"
    local algorithm="$2"
    local omp_threads="$3"

    local tag="${dataset}_${algorithm}_omp${omp_threads}"
    local trace_file="$OUT_DIR/${tag}_mem_trace.csv"
    local log_file="$OUT_DIR/${tag}.log"

    echo "============================================================"
    echo "Running: dataset=$dataset algorithm=$algorithm OMP_NUM_THREADS=$omp_threads"
    echo "Trace: $trace_file"
    echo "Log:   $log_file"

    cd "$ROOT_DIR" || exit 1

    local start_sec
    start_sec=$(date +%s)

    env OMP_NUM_THREADS="$omp_threads" python run_benchmark.py \
        --algorithm "$algorithm" \
        --dataset "$dataset" \
        --runbook "$RUNBOOK" \
        --enable-cache-profiling \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "time_sec,elapsed_sec,vmrss_kb,vmhwm_kb" > "$trace_file"

    while kill -0 "$pid" 2>/dev/null; do
        local now elapsed rss hwm
        now=$(date +%s)
        elapsed=$((now - start_sec))
        rss=$(awk '/VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)
        hwm=$(awk '/VmHWM:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)
        rss=${rss:-0}
        hwm=${hwm:-0}
        echo "$now,$elapsed,$rss,$hwm" >> "$trace_file"
        sleep "$SAMPLE_INTERVAL"
    done

    wait "$pid"
    local exit_code=$?

    local end_sec elapsed_sec peak_kb peak_gb
    end_sec=$(date +%s)
    elapsed_sec=$((end_sec - start_sec))
    peak_kb=$(awk -F, 'NR>1 && $4>max {max=$4} END {print max+0}' "$trace_file")
    peak_gb=$(awk -v kb="$peak_kb" 'BEGIN {printf "%.4f", kb/1024/1024}')

    echo "$dataset,$algorithm,$omp_threads,$RUNBOOK,$exit_code,$elapsed_sec,$peak_kb,$peak_gb,$trace_file,$log_file" >> "$SUMMARY"

    if [[ "$exit_code" -eq 0 ]]; then
        echo "Done: peak_rss=${peak_gb} GB"
    else
        echo "Failed with exit_code=$exit_code: peak_rss=${peak_gb} GB"
        echo "See log: $log_file"
    fi
}

for dataset in "${DATASETS[@]}"; do
    run_one "$dataset" "streamseed_hybrid" "4"
    run_one "$dataset" "streamseed_hybrid_symphonyqg" "16"
    run_one "$dataset" "streamseed_hybrid_freshdiskann" "16"
done

echo "============================================================"
echo "Peak memory summary saved to: $SUMMARY"
cat "$SUMMARY"
