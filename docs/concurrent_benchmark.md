# Running the concurrent benchmark

The concurrency track measures an index under simultaneous inserts and
searches. Insert and search producers feed two worker pools through
rate limiters; searches run against the committed prefix of the insert
stream; latency is recorded per batch on both sides.

All commands below run from the repository root.

Build the native engine once (needs pybind11: `pip install pybind11`):

    cmake -B src/index/concurrent/build -S src/index/concurrent \
          -DCMAKE_BUILD_TYPE=Release \
          -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
    cmake --build src/index/concurrent/build -j --target concurrency_native

The shipped configs expect aligned-bin datasets under `./data/<name>/`;
`data/download.py` fetches the common ones and `data/fvecs_to_bin.py`
converts fvecs. Any config of your own can point at absolute paths
instead. Then:

    PYTHONPATH=src python3 -m concurrency.runner \
        --config configs/round_robin_baseline_sift.yaml

## The config, by section

    data:
      data_path: sift_base.bin       # aligned-bin, see data/fvecs_to_bin.py
      begin_num: 50000               # built before the clock starts
      max_elements: 200000           # insert stream stops here
      query_path: sift_query.bin
      overall_gt_path: sift.gt       # optional, full-visibility recall
      incr_gt_path: sift_incr_offset_index.txt   # optional, see
                                     # docs/incremental_groundtruth.md
    index:
      index_type: hnsw               # hnsw | vamana | parlayhnsw
                                     # | parlayvamana | segmented
      m: 16
      ef_construction: 200
    search:
      recall_at: 10
      ef_search: 35
    workload:
      batch_size: 20                 # multiple of 10
      num_threads: 8                 # split half insert / half search
      queue_size: 0                  # 0 = unbuffered handoff
      insert_event_rate: 20000       # points/sec; <= 0 means unthrottled
      search_event_rate: 20000
      query_mode: round_robin        # | chasing | peeking | zipfian
      warmup_points: 0               # exclude first N inserted points
                                     # from all measurements
    result:
      output_dir: results/run1

List-valued fields sweep (cross product), and
`rate_groups(r/w): [[r1,w1],[r2,w2]]` expands into one variant per
pair. Each variant appends one row to
`<output_dir>/benchmark_results.csv`.

parlayhnsw and parlayvamana ignore `ef_search`; give them
`search.beam_width` and `search.visit_limit` (and `index.level_m` for
parlayhnsw) or search quality collapses. `segmented` takes
`index.sealed_type` and `index.seal_threshold`.

## Reading the output

The progress line is the live picture:

    Progress: 14450/25000 (57%), Insert QPS: 14446, Search QPS: 22444,
    offset_lag: 0, search_snapshot_lag(avg/max): 52/450 pts

`search_snapshot_lag` is how many inserted-but-not-yet-committed points
existed when searches ran — the freshness gap that incremental recall
prices in. The CSV row carries per-point QPS for both sides, op and
end-to-end latency percentiles per batch, recall columns (empty cell =
not configured, not zero), and memory if `profile.enable_memory_profile`
is set.

## Rate limiting has a burst — use warmup_points

The limiter is a token bucket: rate `event_rate / batch_size` batches
per second, burst `batch_size` batches, starting full. Until the burst
drains, batches go out unthrottled — and the bucket refills while it
drains, so the unshaped phase is longer than the burst itself:

    free_points ≈ burst_points * device_rate / (device_rate - target_rate)

where burst_points = batch_size² and device_rate is what the index
sustains unthrottled. On a short run that phase can be the whole
workload, in which case the configured rates shaped nothing and the
run measured peak device throughput with extra steps.

The remedy is `workload.warmup_points`: samples (QPS, latency, lag)
start counting only after that many points are inserted, so the burst
and cold caches sit outside the measured window. Size it with the
formula above, then confirm the measured insert QPS lands on the
configured rate — in our validation, warmup sized that way brought a
+10.9% overshoot down to +0.6%. Expect the search QPS to read a few
percent under its target: the search stream is cut off when inserts
finish, and the truncated final interval undercounts.

With `warmup_points: 0` (the default) everything is measured, which is
the historical behavior and fine for limit tests. Either way, check
measured-vs-configured QPS before trusting a run as rate-controlled.

## Other modes

Same config file, different switch:

- `throughput_sweep.enabled` with `start_rate`/`end_rate`/`steps`:
  log-spaced search rates, fresh index per step, writes
  `throughput_latency_curve.csv`.
- `compare.enabled`: run A records its search schedule without the
  external RW lock, run B replays the identical schedule under the
  lock from the same snapshot; `compare_stats.csv` holds the ratios.
  hnsw only (needs snapshots).
- `compare.enabled` plus `compare.simulate.enabled`: the sequential
  lag-timeline experiment with optional partial-insert trials; results
  land as `.res` files plus a recall-diff CSV.
