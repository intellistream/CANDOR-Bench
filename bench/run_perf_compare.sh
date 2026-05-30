#!/bin/bash
set -e
cd /home/junyao/code/ANN-CC-bench/bench

EVENTS="cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses"

echo "=== [1/2] MVCC/NoLock (concurrent r/w, no external lock) ==="
echo "Start: $(date)"
perf stat -e $EVENTS -o /tmp/perf_gist_nolock.txt ./bench -config config/perf_gist_mvcc.yaml 2>&1 | tail -5
echo "End: $(date)"
echo

echo "=== [2/2] RWLock (serialized r/w) ==="
echo "Start: $(date)"
perf stat -e $EVENTS -o /tmp/perf_gist_rwlock.txt ./bench -config config/perf_gist_rwlock.yaml 2>&1 | tail -5
echo "End: $(date)"
echo

echo "=== RESULTS ==="
echo "--- NoLock ---"
cat /tmp/perf_gist_nolock.txt
echo
echo "--- RWLock ---"
cat /tmp/perf_gist_rwlock.txt
