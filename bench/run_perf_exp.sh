#!/bin/bash
set -e
cd /home/junyao/code/ANN-CC-bench/bench
export LD_LIBRARY_PATH=./build/lib

echo "=== Exp A: MVCC saturated (baseline) ==="
perf stat -e cache-misses,cache-references,L1-dcache-load-misses,LLC-load-misses,context-switches \
  ./bench -config config/exp_mvcc_sat.yaml 2>&1 | tee /tmp/exp_perf_mvcc.log

echo "=== Exp B: MVCC throttle 3200 ==="  
perf stat -e cache-misses,cache-references,L1-dcache-load-misses,LLC-load-misses,context-switches \
  ./bench -config config/exp_mvcc_throttle.yaml 2>&1 | tee /tmp/exp_perf_throttle.log

echo "=== Exp C: RWLock saturated ==="
perf stat -e cache-misses,cache-references,L1-dcache-load-misses,LLC-load-misses,context-switches \
  ./bench -config config/exp_rwlock_sat.yaml 2>&1 | tee /tmp/exp_perf_rwlock.log

echo "=== ALL PERF DONE ==="
