#!/bin/bash
set -e
cd /home/junyao/code/ANN-CC-bench/bench
export LD_LIBRARY_PATH=./build/lib

echo "=== Exp A-ch: MVCC sat chasing ==="
perf stat -e cache-misses,cache-references,L1-dcache-load-misses,LLC-load-misses,context-switches \
  ./bench -config config/exp_mvcc_sat_ch.yaml 2>&1 | tee /tmp/exp_perf_mvcc_ch.log

echo "=== Exp B-ch: MVCC throttle chasing ==="
perf stat -e cache-misses,cache-references,L1-dcache-load-misses,LLC-load-misses,context-switches \
  ./bench -config config/exp_mvcc_throttle_ch.yaml 2>&1 | tee /tmp/exp_perf_throttle_ch.log

echo "=== Exp C-ch: RWLock sat chasing ==="
perf stat -e cache-misses,cache-references,L1-dcache-load-misses,LLC-load-misses,context-switches \
  ./bench -config config/exp_rwlock_sat_ch.yaml 2>&1 | tee /tmp/exp_perf_rwlock_ch.log

echo "=== ALL PERF CHASING DONE ==="
