#!/bin/bash

# Monitor progress of the SIGMOD benchmark suite

BENCH_DIR="/home/junyao/code/ANN-CC-bench/bench"
LOG_FILE="$BENCH_DIR/sigmod_full_benchmark.log"
RESULT_DIR="$BENCH_DIR/result/sigmod_full"

cd "$BENCH_DIR"

echo "SIGMOD Benchmark Progress Monitor"
echo "=================================="
echo "Time: $(date)"
echo ""

# Check if benchmark is still running
if pgrep -f "run_sigmod_full.sh" > /dev/null; then
    echo "✓ Benchmark suite is RUNNING"
else
    echo "✗ Benchmark suite is NOT running"
fi

echo ""

# Show last 10 lines of log
echo "Recent Progress:"
echo "---------------"
tail -10 "$LOG_FILE" 2>/dev/null || echo "No log file found"

echo ""

# Count completed experiments
echo "Results Summary:"
echo "---------------"
if [ -d "$RESULT_DIR" ]; then
    csv_count=$(find "$RESULT_DIR" -name "*.csv" | wc -l)
    echo "CSV result files: $csv_count"
    
    # List result directories
    echo "Result directories:"
    find "$RESULT_DIR" -type d -not -path "$RESULT_DIR" | sort
else
    echo "No results directory found yet"
fi

echo ""
echo "For continuous monitoring: watch -n 30 ./monitor_progress.sh"