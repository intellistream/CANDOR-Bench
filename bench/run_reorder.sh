#!/bin/bash
set -e

cd /home/junyao/code/ANN-CC-bench/bench
export LD_LIBRARY_PATH=./build/lib

LOG=/tmp/run_all_rerun_full.log
COUNT=$(grep -c 'Finished:' $LOG 2>/dev/null || echo 0)

echo "" | tee -a $LOG
echo "=== Reordered run started at $(date) ===" | tee -a $LOG

# Phase 1: 64-thread non-10M configs
NON10M_CONFIGS=(
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
)

# Phase 2: 16-thread 10M configs (skip rwlock)
TEN_M_CONFIGS_16=(
  rerun_sift10m_satbal.yaml
  rerun_sift10m.yaml
  rerun_deep10M_satbal.yaml
  rerun_deep10M.yaml
)

# Phase 3: 64-thread 10M configs (skip rwlock)
TEN_M_CONFIGS_64=(
  rerun_sift10m_satbal.yaml
  rerun_sift10m.yaml
  rerun_deep10M_satbal.yaml
  rerun_deep10M.yaml
)

# Total: 12 (64t non-10M) + 4 (16t 10M) + 4 (64t 10M) = 20
TOTAL=$((COUNT + 12 + 4 + 4))

run_config() {
  local config=$1
  local threads=$2
  local CONFIG_PATH="config/$config"
  
  sed -i "s/num_threads: [0-9]*/num_threads: $threads/" "$CONFIG_PATH"
  COUNT=$((COUNT + 1))
  
  echo "" | tee -a $LOG
  echo "========================================" | tee -a $LOG
  echo "[$COUNT/$TOTAL] Running: $config (threads=$threads)" | tee -a $LOG
  echo "Started at: $(date)" | tee -a $LOG
  echo "========================================" | tee -a $LOG
  
  ./bench -config "$CONFIG_PATH" 2>&1 | tee -a $LOG
  
  echo "[$COUNT/$TOTAL] Finished: $config (threads=$threads) at $(date)" | tee -a $LOG
}

echo "=== Phase 1: 64-thread non-10M ===" | tee -a $LOG
for config in "${NON10M_CONFIGS[@]}"; do
  run_config "$config" 64
done

echo "=== Phase 2: 16-thread 10M (no rwlock) ===" | tee -a $LOG
for config in "${TEN_M_CONFIGS_16[@]}"; do
  run_config "$config" 16
done

echo "=== Phase 3: 64-thread 10M (no rwlock) ===" | tee -a $LOG
for config in "${TEN_M_CONFIGS_64[@]}"; do
  run_config "$config" 64
done

echo "" | tee -a $LOG
echo "=== ALL DONE at $(date) ===" | tee -a $LOG
