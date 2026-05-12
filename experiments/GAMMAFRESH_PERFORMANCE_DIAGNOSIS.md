# GammaFresh Performance Diagnosis

## Summary

GammaFresh is expected to outperform plain Faiss HNSW and IVFPQ on dynamic workloads by keeping short-lived or low-value vectors in a cheap buffer and protecting the graph from unnecessary updates. The current implementation does not realize that model. In the measured path, GammaFresh is slower because several hot operations are implemented as full scans or near-full scans:

1. every online insert recomputes gamma statistics by scanning all live vector states;
2. every online insert scans all partitions to assign the vector;
3. every query scans all partition centroids, then scans buffered vectors in selected partitions;
4. the FaissHNSW backend makes graph queries extremely expensive once tombstones exist by setting `fetch_k = total`;
5. the current benchmark issues measurement inserts one vector at a time, maximizing the overhead of the above paths.

The largest code-level causes are therefore implementation overheads, not an inherent failure of the GammaFresh idea.

## Evidence From Existing Results

Sample result file:

`results/gamma_three_algos_sift_1m_no_recall/gamma_sweep_20260511_002957/data/benchmark_runs.csv`

This sample run used SIFT, 100k measurement operations, 1% prefill, `compute_recall=false`. It is not the full 1M-operation run, but it shows the same performance shape.

| gamma | gammafresh total | faiss_HNSW total | faiss_IVFPQ total | gammafresh throughput | faiss_HNSW throughput | faiss_IVFPQ throughput |
|:--|--:|--:|--:|--:|--:|--:|
| 0.01 | 34.42s | 6.52s | 2.23s | 2,905 ops/s | 15,348 ops/s | 44,867 ops/s |
| 1.0 | 89.54s | 3.91s | 2.10s | 1,117 ops/s | 25,578 ops/s | 47,532 ops/s |
| 100.0 | 40.04s | 1.30s | 1.61s | 2,498 ops/s | 77,007 ops/s | 62,108 ops/s |

Latency breakdown from:

`results/gamma_three_algos_sift_1m_no_recall/gamma_sweep_20260511_002957/data/benchmark_latencies.csv`

At `gamma=1`, GammaFresh is slow in both insert and query:

| algorithm | insert avg | query avg | delete avg |
|:--|--:|--:|--:|
| gammafresh | 1.194 ms | 1.177 ms | 0.0148 ms |
| faiss_HNSW | 0.109 ms | 0.0222 ms | 0.0035 ms |
| faiss_IVFPQ | 0.041 ms | 0.0201 ms | 0.0029 ms |

So the main issue is not delete. The slowdown is concentrated in GammaFresh insert and query.

## Workload Shape

The gamma workload computes write probability as:

```python
write_prob = 1.0 / (1.0 + gamma)
```

With `delete_ratio=0.5`, the approximate operation mix is:

| gamma | insert | delete | query |
|:--|--:|--:|--:|
| 0.01 | 49.5% | 49.5% | 1.0% |
| 1.0 | 25.0% | 25.0% | 50.0% |
| 100.0 | 0.5% | 0.5% | 99.0% |

`gamma=1` is the worst middle case for the current implementation: it has enough inserts/deletes to create buffer and tombstone pressure, and enough queries to pay the query-side penalties repeatedly.

## Root Cause 1: Insert Recomputes Gamma By Scanning All Live Vectors

Insert path:

`algorithms_impl/GammaFresh/src/index/gamma_fresh/execution/mutation.cpp`

```cpp
ctx_.vector_life.partition_gamma(...);
ctx_.vector_life.global_gamma(...);
```

The implementation is in:

`algorithms_impl/GammaFresh/src/index/gamma_fresh/control/vector_life_table.cpp`

```cpp
for (const auto& [_, state] : states_) {
    ...
}
```

Both `partition_gamma()` and `global_gamma()` iterate over `states_`, which is the live vector table. That means a single insert is not just local work. It is approximately:

```text
O(number_of_live_vectors)
```

On SIFT1M after prefill and online operations, this can mean tens or hundreds of thousands of entries per insert. This directly explains why GammaFresh insert latency is far above Faiss HNSW and IVFPQ.

Expected design:

```text
insert should update or read O(1) / O(log partitions) aggregate statistics
```

Current behavior:

```text
insert scans live vector state to estimate gamma
```

## Root Cause 2: Insert Scans All Partitions For Assignment

Assignment path:

`algorithms_impl/GammaFresh/src/index/gamma_fresh/partition/spatial_partition_strategy.cpp`

```cpp
for (const auto& partition : partitions) {
    ...
    distance = query_score(point, *partition, engine);
}
```

Every insert computes distance from the new vector to every active partition centroid. With current `split_factor=4.0`, target partition size is roughly:

```text
target_partition_size = split_factor * sqrt(N)
```

At 1M vectors, that is about 4000 vectors per target partition and roughly hundreds of partitions. This is not catastrophic alone, but combined with the full `VectorLifeTable` scan it makes insert expensive.

Expected design:

```text
route to candidate partitions using a routing index or bounded candidate set
```

Current behavior:

```text
linear scan over partitions for every insert
```

## Root Cause 3: Query Scans All Partitions, Then Scans Buffer Vectors

Query path:

`algorithms_impl/GammaFresh/src/index/gamma_fresh/execution/query.cpp`

```cpp
auto combined_results = query_graph(...);
const auto candidates = collect_candidates(...);
append_buffer_hits(...);
```

Candidate collection:

`algorithms_impl/GammaFresh/src/index/gamma_fresh/partition/spatial_partition_strategy.cpp`

```cpp
for (const auto& partition : partitions) {
    const double score = query_score(query, *partition, engine);
    entries.push_back({score, partition});
}
```

Buffer scan:

`algorithms_impl/GammaFresh/src/index/gamma_fresh/telemetry/result_merger.cpp`

```cpp
partition->scan([&](VectorId id) {
    ...
    engine->EuclideanDistanceSequential(record.data_, vec_ptr->data_);
    combined_results.push_back({dist, id, true});
});
```

So every query does:

```text
graph search
+ O(number_of_partitions) centroid scoring
+ exact scan of buffered vectors in selected partitions
+ result merge/dedup
```

When `maintain_interval=0` or maintenance is sparse, buffered vectors remain in partitions longer. That makes query latency much higher than Faiss HNSW and IVFPQ, which perform only their own index search.

## Root Cause 4: FaissHNSW Tombstones Can Turn Query Into Fetch-All

Backend path:

`algorithms_impl/GammaFresh/src/index/faiss_hnsw.cpp`

```cpp
const auto fetch_k =
    tombstones_.empty() ? std::min<std::size_t>(static_cast<std::size_t>(k), total)
                        : total;
```

As soon as the backend graph has any tombstone, query requests `total` results from Faiss instead of `k` or a bounded oversampling factor.

This is extremely expensive:

```text
normal case: fetch_k ~= k
tombstone case: fetch_k = graph ntotal
```

On dynamic workloads with delete operations, tombstones are expected. In the `gamma=1` workload, deletes are about 25% of operations. If graph-resident vectors are deleted and no timely rebuild clears tombstones, this path can dominate query time.

This behavior directly contradicts the expected GammaFresh advantage. GammaFresh should avoid paying graph rebuild/update cost for short-lived vectors, but once tombstones appear in the backend graph, each graph query can become much heavier than plain top-k HNSW.

## Root Cause 5: Benchmark Issues Measurement Inserts One At A Time

Runner path:

`bench/runner.py`

```python
self.algo.insert(
    vectors[target_id:target_id + 1],
    np.array([target_id], dtype=np.uint32),
)
```

Prefill is batched, but measurement inserts are single-vector calls. Single-vector insert is the worst case for the current GammaFresh implementation because each insert repeats:

```text
Python binding conversion
StorageManager insert
partition assignment
VectorLifeTable gamma scan
optional graph insert
metadata updates
```

This is still a valid dynamic benchmark, but it means GammaFresh needs an efficient single-vector hot path. It currently does not have one.

## Secondary Issues

### Internal Logging

GammaFresh creates a `GammaFreshLogger` in the constructor. CSV logging is disabled when `log_interval <= 0`, which is the default in the root wrapper, so this is probably not the main slowdown. However, there are still atomic metric updates and logger branches on insert/query paths. They are secondary compared with the full scans above.

### C++ Stdout From Faiss

The lower-level Faiss HNSW code prints `adding N vectors` / `adding N vectors finishes` in some variants. If this is active during measurement, it pollutes benchmark time. It is unlikely to explain the full gap, but it should be disabled for clean timing.

## Why The Current Result Contradicts The Theoretical Expectation

The theoretical GammaFresh advantage assumes:

```text
cheap buffer insert
bounded buffer scan
selective migration to graph
graph kept clean from short-lived vectors
cheap routing and metadata updates
```

The current implementation violates those assumptions:

```text
insert does global metadata scans
routing is a linear partition scan
query does partition scan plus buffer scan
graph tombstones can cause fetch-all HNSW queries
maintenance is disabled or sparse in the tested configuration
```

Therefore the benchmark is measuring an implementation with substantial control-plane overhead, not the idealized GammaFresh cost model.

## Recommended Fix Priority

### P0: Fix FaissHNSW tombstone query amplification

Replace `fetch_k = total` with bounded oversampling, for example:

```text
fetch_k = min(total, max(k * oversample_factor, k + tombstone_slack))
```

If recall needs protection under heavy tombstones, trigger rebuild/maintain earlier instead of fetching the entire graph on every query.

### P0: Make gamma statistics incremental

Avoid computing `partition_gamma()` and `global_gamma()` by scanning all live vectors during insert.

Maintain per-partition and global aggregate counters that can be updated on:

```text
insert
delete
query hit / partition scan hit
ownership change
migration
```

The insert hot path should not be O(live vectors).

### P1: Add a routing index or bounded candidate routing for partitions

Replace full partition centroid scan on insert/query with a small candidate set:

```text
centroid HNSW / IVF / heap over cached nearest centroids / grid-like routing
```

This matters more as partition count grows.

### P1: Revisit maintenance policy for benchmark fairness

With `maintain_interval=0`, GammaFresh cannot demonstrate its intended promotion/drain behavior. Compare at least two modes:

```text
maintain_interval=0: no explicit maintenance, stress buffer path
maintain_interval=N: scheduled maintenance, expected GammaFresh operating mode
```

Also record migrated vectors, buffer size, graph size, tombstones, and scanned buffer vectors.

### P2: Reduce benchmark-time output and optional telemetry

Disable C++ `printf` in Faiss HNSW add paths and keep GammaFresh CSV/internal telemetry off for performance runs.

## Immediate Diagnostics To Add

For future runs, record these per `(algorithm, gamma)`:

```text
GammaFresh:
  live vector count
  partition count
  average active partition size
  total buffered vectors
  graph live size
  graph ntotal
  graph tombstone count
  average query candidate count
  average scanned buffer vectors per query
  number of graph inserts
  number of buffer inserts
  maintain migrated vectors

FaissHNSW backend:
  query fetch_k
  tombstone count
  rebuild count
```

These metrics will make it clear whether a run is dominated by:

```text
metadata scan
partition routing
buffer scan
graph tombstones
single-vector graph insert
```

## Bottom Line

GammaFresh is slower in the current experiment because the implementation pays global metadata and partition-scan costs on ordinary operations, and the FaissHNSW backend has a severe tombstone fallback that can turn top-k search into fetch-all search. These implementation details erase the theoretical benefit of buffering short-lived vectors.

The first fixes should be:

```text
1. remove fetch_k=total under tombstones
2. replace VectorLifeTable full scans with incremental aggregates
3. bound partition routing and query candidate selection
4. run GammaFresh with explicit, measured maintenance enabled
```

