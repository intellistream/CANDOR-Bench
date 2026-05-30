#!/bin/bash

# Monitor benchmark progress and post results to Slack
SLACK_CHANNEL="C0AFHJBD58R"
THREAD_ID="1771212891.447279"
BENCH_DIR="/home/junyao/code/ANN-CC-bench/bench"

cd $BENCH_DIR

# Function to post to Slack
post_to_slack() {
    local message="$1"
    echo "$(date): $message"
    # We'll implement Slack posting when results are ready
}

# Function to check if benchmark is complete
check_completion() {
    local log_file="$1"
    local experiment_name="$2"
    
    if [ -f "$log_file" ] && grep -q "Complete ===" "$log_file"; then
        return 0
    else
        return 1
    fi
}

# Function to gather results
gather_results() {
    local result_dir="$1"
    local experiment_name="$2"
    
    if [ -d "$result_dir" ]; then
        echo "=== $experiment_name Results ==="
        
        # Find CSV result files
        find "$result_dir" -name "*.csv" -exec echo "Found: {}" \;
        
        # Count total experiments run
        csv_count=$(find "$result_dir" -name "*.csv" | wc -l)
        echo "Total result files: $csv_count"
        
        # Show memory usage summary if available
        if [ -f "$result_dir/benchmark_results.csv" ]; then
            echo "Memory usage summary:"
            cut -d',' -f1,2,19,20 "$result_dir/benchmark_results.csv" | head -10
        fi
    fi
}

# Monitor loop
while true; do
    echo "$(date): Checking benchmark status..."
    
    # Check E1 (Lock vs MVCC vs MVCC+S3)
    if check_completion "e1_benchmark.log" "E1"; then
        echo "E1 experiments completed!"
        gather_results "result/sigmod_e1" "E1" > e1_results_summary.txt
        # post_to_slack "E1 (Lock vs MVCC vs MVCC+S3) experiments completed! Results ready."
        echo "E1 complete marker created"
        touch e1_complete
    fi
    
    # Check E2 (Scalability)
    if check_completion "e2_benchmark.log" "E2"; then
        echo "E2 experiments completed!"
        gather_results "result/sigmod_e2" "E2" > e2_results_summary.txt
        # post_to_slack "E2 (Scalability) experiments completed! Results ready."
        echo "E2 complete marker created"
        touch e2_complete
    fi
    
    # Show current progress
    echo "=== Current Progress ==="
    echo "E1 Progress:"
    tail -2 e1_benchmark.log 2>/dev/null || echo "E1 log not found"
    echo ""
    echo "E2 Progress:"
    tail -2 e2_benchmark.log 2>/dev/null || echo "E2 log not found"
    echo ""
    
    # Check if both are complete
    if [ -f "e1_complete" ] && [ -f "e2_complete" ]; then
        echo "All experiments completed!"
        # post_to_slack "All SIGMOD benchmarks completed! E1 and E2 results ready for analysis."
        break
    fi
    
    # Wait 5 minutes before next check
    sleep 300
done

echo "Monitoring complete. Results summarized in e1_results_summary.txt and e2_results_summary.txt"