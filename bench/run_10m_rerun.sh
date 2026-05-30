#!/bin/bash
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH

CONFIGS=(
  config/rerun_sift10m_satbal.yaml
  config/rerun_sift10m.yaml
  config/rerun_deep10M_satbal.yaml
  config/rerun_deep10M.yaml
)

THREADS=(16 64)

for cfg in "${CONFIGS[@]}"; do
  for t in "${THREADS[@]}"; do
    echo "===== $(date): Running $cfg with threads=$t ====="
    # Temporarily modify num_threads in config
    sed -i "s/num_threads: [0-9]*/num_threads: $t/" "$cfg"
    ./build/bench -config "$cfg" 2>&1 | tail -20
    echo "===== $(date): Finished $cfg threads=$t ====="
  done
done

echo "ALL 10M RERUNS COMPLETE at $(date)"
