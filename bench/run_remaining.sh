#!/bin/bash
set -e

cd /home/junyao/code/ANN-CC-bench/bench
export LD_LIBRARY_PATH=./build/lib

LOG=/tmp/run_all_rerun_full.log

# Skip list: 10m rwlock configs
SKIP="rerun_sift10m_rwlock.yaml rerun_deep10M_rwlock.yaml"

# Remaining configs after gist_rwlock (which is currently running in the old script)
CONFIGS=(
  rerun_sift10m_satbal.yaml
  rerun_sift10m.yaml
  rerun_sift10m_rwlock.yaml
  rerun_deep10M_satbal.yaml
  rerun_deep10M.yaml
  rerun_deep10M_rwlock.yaml
)

THREADS=(16 64)

# Also need to redo all configs with 64 threads
ALL_CONFIGS=(
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

# Count completed
DONE=$(grep -c 'Finished:' $LOG 2>/dev/null || echo 0)
# Total = 18 configs * 2 threads - 2 skipped configs * 2 threads = 32
TOTAL=32
COUNT=$DONE

echo "" | tee -a $LOG
echo "=== Resuming (skip 10m rwlock) at $(date) ===" | tee -a $LOG

# Phase 1: remaining 16-thread configs (sift10m, deep10M minus rwlock)
for config in "${CONFIGS[@]}"; do
  if echo "$SKIP" | grep -q "$config"; then
    echo "SKIPPING: $config (10m rwlock)" | tee -a $LOG
    continue
  fi
  
  CONFIG_PATH="config/$config"
  sed -i "s/num_threads: [0-9]*/num_threads: 16/" "$CONFIG_PATH"
  COUNT=$((COUNT + 1))
  
  echo "" | tee -a $LOG
  echo "========================================" | tee -a $LOG
  echo "[$COUNT/$TOTAL] Running: $config (threads=16)" | tee -a $LOG
  echo "Started at: $(date)" | tee -a $LOG
  echo "========================================" | tee -a $LOG
  
  ./bench -config "$CONFIG_PATH" 2>&1 | tee -a $LOG
  
  echo "[$COUNT/$TOTAL] Finished: $config (threads=16) at $(date)" | tee -a $LOG
done

# Phase 2: all configs with 64 threads (skip 10m rwlock)
for config in "${ALL_CONFIGS[@]}"; do
  if echo "$SKIP" | grep -q "$config"; then
    echo "SKIPPING: $config (10m rwlock)" | tee -a $LOG
    continue
  fi
  
  CONFIG_PATH="config/$config"
  sed -i "s/num_threads: [0-9]*/num_threads: 64/" "$CONFIG_PATH"
  COUNT=$((COUNT + 1))
  
  echo "" | tee -a $LOG
  echo "========================================" | tee -a $LOG
  echo "[$COUNT/$TOTAL] Running: $config (threads=64)" | tee -a $LOG
  echo "Started at: $(date)" | tee -a $LOG
  echo "========================================" | tee -a $LOG
  
  ./bench -config "$CONFIG_PATH" 2>&1 | tee -a $LOG
  
  echo "[$COUNT/$TOTAL] Finished: $config (threads=64) at $(date)" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "=== ALL DONE at $(date) ===" | tee -a $LOG
