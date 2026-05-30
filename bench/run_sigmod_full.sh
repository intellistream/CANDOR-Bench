#!/bin/bash

set -e  # Exit on error

# SIGMOD Full Benchmark Suite
# This script runs comprehensive benchmarks across all datasets for the SIGMOD paper

BENCH_DIR="/home/junyao/code/ANN-CC-bench/bench"
BINARY="$BENCH_DIR/bin/bench"
RESULT_BASE="$BENCH_DIR/result/sigmod_full"

cd "$BENCH_DIR"

# Set up environment for shared libraries
export CC=gcc
export CXX=g++
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build/lib
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

echo "Starting SIGMOD Full Benchmark Suite at $(date)"
echo "Results will be saved to: $RESULT_BASE"

# Ensure result directory exists
mkdir -p "$RESULT_BASE"

# Function to run a benchmark
run_benchmark() {
    local config_file="$1"
    local description="$2"
    
    echo "===================="
    echo "Running: $description"
    echo "Config: $config_file"
    echo "Time: $(date)"
    echo "===================="
    
    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config file not found: $config_file"
        return 1
    fi
    
    timeout 1800 "$BINARY" -config "$config_file" || {
        echo "ERROR: Benchmark failed or timed out (30 min limit)"
        return 1
    }
    
    echo "Completed: $description at $(date)"
    echo ""
}

# Task 1: Run GIST, Deep1M, MSong (SIFT already done)
# For each dataset: HNSW and ANNchor × balanced/read-heavy/write-heavy × 8,16,32 threads

echo "=== TASK 1: GIST, Deep1M, MSong Experiments ==="

datasets=("gist" "deep1M" "msong")
algorithms=("hnsw" "annchor") 
workloads=("ch" "pk" "rr")  # chasing (balanced), peeking (read-heavy), round_robin (write-heavy)

for dataset in "${datasets[@]}"; do
    for algo in "${algorithms[@]}"; do
        for workload in "${workloads[@]}"; do
            config_file="config/${algo}/${dataset}/${workload}_b20.yaml"
            description="${dataset} ${algo} ${workload} workload"
            
            # We need to modify the config to have the right thread counts and output directory
            # Let's create a temporary modified config
            temp_config="temp_${dataset}_${algo}_${workload}.yaml"
            
            # Copy and modify the config
            cp "$config_file" "$temp_config"
            
            # Modify num_threads to [8, 16, 32]
            sed -i 's/num_threads: \[.*\]/num_threads: [8, 16, 32]/' "$temp_config"
            
            # Update result output directory
            sed -i "s|output_dir: ./result/.*|output_dir: ./result/sigmod_full|" "$temp_config"
            
            run_benchmark "$temp_config" "$description"
            
            # Clean up temp config
            rm -f "$temp_config"
        done
    done
done

echo "=== TASK 1 COMPLETE ==="
echo "Proceeding to Task 2: Scalability Sweep"
echo ""

# Task 2: Scalability sweep (E2) - SIFT 1M only
# Thread counts: 1, 2, 4, 8, 16, 32, 64
# Both HNSW and ANNchor, Balanced workload only

echo "=== TASK 2: SIFT Scalability Sweep ==="

# Create scalability configs for HNSW and ANNchor
for algo in hnsw annchor; do
    temp_config="temp_sift_scalability_${algo}.yaml"
    
    # Use the existing scalability config as a base
    cp "config/sigmod_e2_scalability.yaml" "$temp_config"
    
    # Modify for specific algorithm
    if [ "$algo" == "hnsw" ]; then
        sed -i 's/index_type: \[.*\]/index_type: hnsw/' "$temp_config"
        sed -i 's/use_node_lock: \[.*\]/with_node_lock: [true, false]/' "$temp_config"
        sed -i '/enable_mvcc:/d' "$temp_config"
        sed -i '/enable_s3:/d' "$temp_config"
    else
        sed -i 's/index_type: \[.*\]/index_type: annchor/' "$temp_config"
        sed -i '/use_node_lock:/d' "$temp_config"
        sed -i '/enable_mvcc:/d' "$temp_config"
        sed -i '/enable_s3:/d' "$temp_config"
        sed -i '/with_node_lock:/d' "$temp_config"
    fi
    
    # Update result directory
    sed -i "s|output_dir: ./result/sigmod_e2|output_dir: ./result/sigmod_full|" "$temp_config"
    
    run_benchmark "$temp_config" "SIFT Scalability ${algo}"
    
    rm -f "$temp_config"
done

echo "=== TASK 2 COMPLETE ==="
echo "Proceeding to Task 3: SIFT 10M"
echo ""

# Task 3: SIFT 10M (E6) - 16 threads, balanced workload
echo "=== TASK 3: SIFT 10M Experiments ==="

# Check if SIFT 10M data exists
SIFT10M_PATH="../data/sift10m/sift10m_base.bin"
if [ ! -f "$SIFT10M_PATH" ]; then
    echo "ERROR: SIFT 10M data not found at $SIFT10M_PATH"
    echo "Skipping Task 3"
else
    for algo in hnsw annchor; do
        temp_config="temp_sift10m_${algo}.yaml"
        
        # Create config based on SIFT 1M config
        cp "config/${algo}/sift/ch_b20.yaml" "$temp_config"
        
        # Update for 10M dataset
        sed -i 's|data_path: ../data/sift/sift_base.bin|data_path: ../data/sift10m/sift10m_base.bin|' "$temp_config"
        sed -i 's|incr_query_path: ../data/sift/sift_query_stream.bin|incr_query_path: ../data/sift10m/sift10m_query_stream.bin|' "$temp_config"
        sed -i 's|incr_gt_path: ../data/sift/sift_stream_i20_offset_index.txt|incr_gt_path: ../data/sift10m/sift10m_stream_i20_offset_index.txt|' "$temp_config"
        sed -i 's|overall_query_path: ../data/sift/sift_query.bin|overall_query_path: ../data/sift10m/sift10m_query.bin|' "$temp_config"
        sed -i 's|overall_gt_path: ../data/sift/sift_base.gt20|overall_gt_path: ../data/sift10m/sift10m_base.gt20|' "$temp_config"
        sed -i 's/dataset_name: sift/dataset_name: sift10m/' "$temp_config"
        
        # Set 16 threads only
        sed -i 's/num_threads: \[.*\]/num_threads: [16]/' "$temp_config"
        
        # Update result directory
        sed -i "s|output_dir: ./result/.*|output_dir: ./result/sigmod_full|" "$temp_config"
        
        run_benchmark "$temp_config" "SIFT 10M ${algo} 16 threads"
        
        rm -f "$temp_config"
    done
fi

echo "=== TASK 3 COMPLETE ==="
echo "Proceeding to Task 4: Memory Analysis"
echo ""

# Task 4: Memory overhead analysis will be done after all benchmarks complete
echo "=== TASK 4: Memory Analysis ==="
echo "Analyzing peak memory usage from all results..."

# Find all CSV result files and extract memory data
find "$RESULT_BASE" -name "*.csv" -type f > temp_result_files.txt

if [ -s temp_result_files.txt ]; then
    echo "Found result files:"
    cat temp_result_files.txt
    
    echo ""
    echo "Extracting memory overhead data..."
    
    # Create memory analysis script
    python3 << 'EOF'
import pandas as pd
import glob
import os

result_dir = "/home/junyao/code/ANN-CC-bench/bench/result/sigmod_full"
memory_data = []

# Find all CSV files in the result directory
csv_files = glob.glob(os.path.join(result_dir, "**/*.csv"), recursive=True)

print(f"Found {len(csv_files)} CSV files")

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        if 'peak_memory_mb' in df.columns:
            # Extract relevant columns for memory analysis
            for _, row in df.iterrows():
                memory_data.append({
                    'file': os.path.basename(csv_file),
                    'dataset': row.get('dataset_name', 'unknown'),
                    'algorithm': row.get('index_type', 'unknown'), 
                    'threads': row.get('num_threads', 'unknown'),
                    'workload': row.get('query_mode', 'unknown'),
                    'peak_memory_mb': row.get('peak_memory_mb', 0),
                    'avg_qps': row.get('avg_qps', 0),
                    'recall': row.get('recall', 0)
                })
        else:
            print(f"No peak_memory_mb column in {csv_file}")
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

if memory_data:
    memory_df = pd.DataFrame(memory_data)
    memory_summary_file = os.path.join(result_dir, "memory_analysis_summary.csv")
    memory_df.to_csv(memory_summary_file, index=False)
    print(f"Memory analysis saved to: {memory_summary_file}")
    
    # Print summary
    print("\nMemory Usage Summary by Algorithm:")
    algo_summary = memory_df.groupby(['algorithm', 'dataset'])['peak_memory_mb'].agg(['mean', 'max', 'min']).round(2)
    print(algo_summary)
else:
    print("No memory data found in result files")
EOF

else
    echo "No result files found yet - analysis will be performed after benchmarks complete"
fi

rm -f temp_result_files.txt

echo "=== ALL TASKS COMPLETE ==="
echo "Full benchmark suite completed at $(date)"
echo "Results saved in: $RESULT_BASE"
echo ""
echo "Summary:"
echo "- Task 1: GIST, Deep1M, MSong with HNSW/ANNchor across workloads and thread counts"
echo "- Task 2: SIFT scalability sweep (1-64 threads)"  
echo "- Task 3: SIFT 10M experiments"
echo "- Task 4: Memory overhead analysis"