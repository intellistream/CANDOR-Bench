#!/usr/bin/env bash
# Read-write tail-latency: does writer impact reader P99/P999?
# Cells: (lock_variant ∈ {unique, shared}) × (N_writer ∈ {0,1,4,16,64}) × 3 reps
# Same SIFT-1M setup as FULL audit, 25s perf window.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-./result/m3_rw_tail_20260512}"
CFG_ROOT="${CFG_ROOT:-./config/m3_rw_tail_20260512}"

DATASET_NAME="sift10m"
DATA_PATH="../data/sift10m/sift10m_base.bin"
QUERY_PATH="../data/sift/sift_query_stream.bin"
MAX_ELEMENTS=10000000
BEGIN_NUM=200000
M_HNSW=16
EF_SEARCH=50
EF_CONSTRUCTION=100
BATCH_SIZE=20
QUERY_MODE=peeking
MEASURED_WORKERS=8
DURATION_MS=35000
SKIP_PRE_MS=5000
SAMPLE_MS=25000

MEASURED_CPU_LIST=0-7

mkdir -p "$OUT_ROOT" "$CFG_ROOT"

competing_cpu_list_for_n() {
  local n="$1"
  if [[ "$n" -le 0 ]]; then echo ""; return; fi
  local s0=16
  if [[ "$n" -le "$s0" ]]; then echo "8-$((7+n))"; return; fi
  local rem=$((n - s0))
  local s1_end=$((24 + rem - 1))
  if [[ "$s1_end" -gt 47 ]]; then
    local rem2=$((s1_end - 47))
    echo "8-23,24-47,56-$((55+rem2))"
  else
    echo "8-23,24-$((23+rem))"
  fi
}

write_config() {
  local cfg="$1"; local out_dir="$2"; local total_threads="$3"
  mkdir -p "$out_dir" "$(dirname "$cfg")"
  cat > "$cfg" <<YAML
data: {dataset_name: $DATASET_NAME, max_elements: $MAX_ELEMENTS, begin_num: $BEGIN_NUM, data_type: float, data_path: $DATA_PATH, incr_query_path: $QUERY_PATH, overall_query_path: $QUERY_PATH}
index: {index_type: hnsw, m: $M_HNSW, ef_construction: $EF_CONSTRUCTION}
search:
  recall_at: 10
  ef_search: $EF_SEARCH
  use_node_lock: true
  enable_mvcc: false
  enable_undo_recovery: false
workload:
  batch_size: $BATCH_SIZE
  num_threads: $total_threads
  queue_size: 0
  insert_event_rate: 0
  search_event_rate: 0
  schedule_horizon_ms: $DURATION_MS
  with_external_rw_lock: false
  use_node_lock: true
  query_mode: $QUERY_MODE
  per_query_latency: true
  precompute_schedule: false
result: {output_dir: $out_dir}
repetitions: 1
profile: {memory_monitor_interval: 100, enable_memory_profile: false}
YAML
}

wait_for_closed_loop() {
  local log="$1"; local max_wait_s=180
  local waited=0
  while [[ "$waited" -lt "$max_wait_s" ]]; do
    if grep -q "Closed-loop setup:" "$log" 2>/dev/null; then return 0; fi
    sleep 1; waited=$((waited+1))
  done
  return 1
}

run_cell() {
  local variant="$1"  # unique | shared
  local n="$2"
  local rep="$3"
  local cell_dir="$OUT_ROOT/${variant}_n${n}_rep${rep}"
  local total_threads=$((MEASURED_WORKERS + n))
  [[ "$total_threads" -lt 1 ]] && total_threads=1
  local cfg="$CFG_ROOT/$(basename "$cell_dir").yaml"
  write_config "$cfg" "$cell_dir" "$total_threads"

  local mode=insert
  [[ "$n" -eq 0 ]] && mode=idle

  local competing_cpus
  competing_cpus="$(competing_cpu_list_for_n "$n")"
  [[ -z "$competing_cpus" ]] && competing_cpus=8-9  # placeholder, unused at n=0

  rm -rf "$cell_dir/raw_latency"
  mkdir -p "$cell_dir"
  echo "[rw-tail] variant=$variant n=$n rep=$rep cpus=$competing_cpus → $cell_dir"
  echo "variant=$variant n_writers=$n rep=$rep measured_cpus=$MEASURED_CPU_LIST competing_cpus=$competing_cpus duration_ms=$DURATION_MS sample_ms=$SAMPLE_MS" > "$cell_dir/cell.meta"

  # Pick the right libindex.so for this variant
  if [[ "$variant" == "unique" ]]; then
    ln -sf libindex_unique_baseline.so "$ROOT/build/lib/libindex.so.run"
    export LD_LIBRARY_PATH="$ROOT/build/lib_unique_runtime:$LD_LIBRARY_PATH"
    mkdir -p "$ROOT/build/lib_unique_runtime"
    cp -f "$ROOT/build/lib/libindex_unique_baseline.so" "$ROOT/build/lib_unique_runtime/libindex.so"
  else
    export LD_LIBRARY_PATH="$ROOT/build/lib:$LD_LIBRARY_PATH"
  fi

  ( env \
      M3_CLOSED_LOOP=1 \
      M3_CLOSED_LOOP_DURATION_MS="$DURATION_MS" \
      M3_CLOSED_LOOP_MEASURED_WORKERS="$MEASURED_WORKERS" \
      M3_CLOSED_LOOP_COMPETING_WORKERS="$n" \
      M3_CLOSED_LOOP_INSERT_WORKERS="$n" \
      M3_CLOSED_LOOP_COMPETING_MODE="$mode" \
      ANNCHOR_M3_BATCH_DIAG=1 \
      HNSW_MEASURED_SEARCH_ARENA=1 \
      HNSW_MEASURED_SEARCH_ARENA_THREADS="$MEASURED_WORKERS" \
      HNSW_MEASURED_SEARCH_CPU_LIST="$MEASURED_CPU_LIST" \
      HNSW_INSERT_CPU_LIST="$competing_cpus" \
      ANNCHOR_INSERT_CPU_LIST="$competing_cpus" \
      OCC_RAW_LATENCY_DIR="$cell_dir/raw_latency" \
      ./bin/bench -config "$cfg" ) > "$cell_dir/run.log" 2>&1 &
  local pid=$!

  wait_for_closed_loop "$cell_dir/run.log" || { echo "[rw-tail] timeout"; kill $pid 2>/dev/null || true; return 1; }
  echo "$pid" > "$cell_dir/bench.pid"
  wait $pid 2>/dev/null || true
  echo "[rw-tail] done $cell_dir"
}

VARIANTS="${VARIANTS:-unique shared}"
CELLS_N="${CELLS_N:-0 1 4 16 64}"
REPS="${REPS:-1 2 3}"

for variant in $VARIANTS; do
  for n in $CELLS_N; do
    for r in $REPS; do
      run_cell "$variant" "$n" "$r" || true
    done
  done
done
echo "[rw-tail] all done $(date)"
