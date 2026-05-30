#!/bin/bash
set -e

cd /home/junyao/code/ANN-CC-bench/bench
export LD_LIBRARY_PATH=./build/lib

LOG=/tmp/run_all_rerun_full.log
echo "=== Full Rerun Started at $(date) ===" | tee $LOG

# All configs in order: small datasets first
CONFIGS=(
  rerun_sift_satbal.yaml
  rerun_sift.yaml
  rerun_sift_rwlock.yaml
  rerun_msong_satbal.yaml
  rerun_msong.yaml
  rerun_msong_rwlock.yaml
  rerun_deep1M_satbal.yaml
  rerun_deep1M.yaml
  rerun_deep1M_rwlock.yaml
  rerun_gist_satbal.yaml
  rerun_gist.yaml
  rerun_gist_rwlock.yaml
  rerun_sift10m_satbal.yaml
  rerun_sift10m.yaml
  rerun_sift10m_rwlock.yaml
  rerun_deep10M_satbal.yaml
  rerun_deep10M.yaml
  rerun_deep10M_rwlock.yaml
)

THREADS=(16 64)

TOTAL=$((${#CONFIGS[@]} * ${#THREADS[@]}))
COUNT=0

for threads in "${THREADS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    COUNT=$((COUNT + 1))
    
    # Update num_threads in config
    CONFIG_PATH="config/$config"
    sed -i "s/num_threads: [0-9]*/num_threads: $threads/" "$CONFIG_PATH"
    
    echo "" | tee -a $LOG
    echo "========================================" | tee -a $LOG
    echo "[$COUNT/$TOTAL] Running: $config (threads=$threads)" | tee -a $LOG
    echo "Started at: $(date)" | tee -a $LOG
    echo "========================================" | tee -a $LOG
    
    ./bench -config "$CONFIG_PATH" 2>&1 | tee -a $LOG
    
    echo "[$COUNT/$TOTAL] Finished: $config (threads=$threads) at $(date)" | tee -a $LOG
  done
done

echo "" | tee -a $LOG
echo "=== ALL DONE at $(date) ===" | tee -a $LOG
