#!/bin/bash
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=build/lib:$LD_LIBRARY_PATH

# Sweep: 3 datasets × 3 thread counts × 3 configs (nolock, mvcc, mvcc+m2)
# Unlimited rate to see real throughput and tail latency

for ds in sift deep1M gist; do
  if [ "$ds" = "sift" ]; then
    base="../data/sift/sift_base.bin"; qs="../data/sift/sift_query_stream.bin"
    q="../data/sift/sift_query.bin"; gt="../data/sift/sift_base.gt20"
    threads="8 32 64"
  elif [ "$ds" = "deep1M" ]; then
    base="../data/deep1M/deep1M_base.bin"; qs="../data/deep1M/deep1M_query_stream.bin"
    q="../data/deep1M/deep1M_query.bin"; gt="../data/deep1M/deep1M_base.gt20"
    threads="8 32 64"
  else
    base="../data/gist/gist_base.bin"; qs="../data/gist/gist_query_stream.bin"
    q="../data/gist/gist_query.bin"; gt="../data/gist/gist_base.gt20"
    threads="8 16"
  fi

  for t in $threads; do
    for mode in "nolock:false:1.0" "mvcc:true:1.0" "mvcc_m2:true:0.5"; do
      IFS=: read -r mode_name mvcc ratio <<< "$mode"
      
      outdir="./result/m2_full/${ds}_t${t}_${mode_name}"
      
      cat > /tmp/m2_full.yaml <<EOF
data:
  dataset_name: $ds
  max_elements: 1000000
  begin_num: 50000
  data_type: float
  data_path: $base
  incr_query_path: $qs
  overall_query_path: $q
  overall_gt_path: $gt
index:
  index_type: annchor-dev
  m: 16
  ef_construction: 200
search:
  recall_at: 10
  ef_search: 35
  enable_mvcc: $mvcc
  lazy_recovery_ratio: $ratio
workload:
  batch_size: 20
  num_threads: $t
  queue_size: 0
  rate_groups(r/w):
    - [999999, 999999]
  with_external_rw_lock: false
  query_mode: chasing
compare:
  enabled: false
  simulate:
    enabled: false
result:
  output_dir: $outdir
profile:
  enable_memory_profile: false
EOF
      
      echo ">>> $ds | t=$t | $mode_name"
      ./bin/bench -config /tmp/m2_full.yaml 2>&1 | grep -E "recall@|Progress.*100%" || true
    done
  done
done

echo "=== ALL DONE ==="

# Collect results
python3 << 'PYEOF'
import csv, re, os, glob

print(f"{'dataset':>8} {'threads':>4} {'mode':>10} {'recall':>7} {'QPS':>8} {'mean':>7} {'p95':>7} {'p99':>7}")
print("-" * 72)

for csvf in sorted(glob.glob("result/m2_full/*/benchmark_results.csv")):
    dirname = os.path.basename(os.path.dirname(csvf))
    for ds in ["deep1M", "sift", "gist"]:
        if dirname.startswith(ds + "_"):
            rest = dirname[len(ds)+1:]
            break
    # parse t{N}_{mode}
    m = re.match(r't(\d+)_(.+)', rest)
    if not m: continue
    threads, mode = m.group(1), m.group(2)
    
    with open(csvf) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sqps = float(row['search_qps (per-point)'])
            if sqps == 0: continue
            recall = row['recall']
            s_mean = row['search_mean_op_latency_ms (per-batch)']
            s_p95 = row['search_p95_op_latency_ms (per-batch)']
            s_p99 = row['search_p99_op_latency_ms (per-batch)']
            print(f"{ds:>8} {threads:>4} {mode:>10} {recall:>7} {sqps:>8.0f} {s_mean:>7} {s_p95:>7} {s_p99:>7}")
PYEOF
