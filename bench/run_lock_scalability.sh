#!/bin/bash
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$(pwd)/build/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

echo "Starting Lock-only thread scalability at $(date)"
go run main.go --config config/thread_scale_1M_lock_only.yaml 2>&1 | tee result/thread_scalability_1M_lock/run.log
echo "Completed at $(date)"
