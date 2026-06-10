# Concurrency runner internals

The runner replaces the previous Go harness (removed; see git history) with
the same architecture the rest of the repository uses: C++ for everything
on the measurement path, Python for orchestration and analysis. Configs,
output formats and workload semantics are unchanged.

## Where things run

```
Python (this package)                      C++ (src/index/concurrent)
config / variants / recall / CSVs   --->   ../index/concurrent/driver/concurrent_driver.hpp
sweep / compare / simulate          <---   producers, queues, rate limiters,
                                           worker pools, latency capture
```

One call into `concurrency_native.run_benchmark` executes the entire
concurrent phase. The GIL is released for the whole call and no C++
thread touches Python, so the interpreter is out of the measurement path
(`python/test_gil_isolation.py` demonstrates this; the load generator and
data collection behave identically whether or not Python threads are
busy).

## Workload semantics

Carried over from the Go driver, verified line by line and by side-by-side
runs on identical data (throughput within 2-3%, mean latencies within 1%,
in both buffered and unbuffered queue configurations):

- worker split: `num_threads / 2` insert workers (min 1), the rest search
  workers (min 1)
- rate limiting: token bucket per queue, rate `event_rate / batch_size`
  batches/s, burst `batch_size`; rate <= 0 disables the limiter
- `queue_size: 0` is an unbuffered handoff — a producer blocks until a
  worker receives the task
- inserts may complete out of order; the committed offset advances in
  order, and a search issued at committed offset `o` runs with
  `visible_ts = o`
- search snapshot lag = points inserted but not yet visible to the search,
  tracked as avg/max
- op latency is dequeue-to-done, e2e latency is creation-to-done; insert
  creation is stamped before the rate-limiter wait
- percentiles use `sorted[int((n-1) * p)]`

Differences from the Go version, all deliberate:

- memory sampling reads VmRSS instead of jemalloc allocated bytes
- a failed replay lifts the insert gate instead of deadlocking
- the progress logger terminates when the run does, even if inserts stall
- recall is computed in-process (numpy); the C++ tools under
  `tools/` produce identical numbers from the same
  `.res` files when the external pipeline is preferred
- each worker collects latencies and search records into private buffers
  merged after the run, so collection never serializes workers or leaks
  into measured latency (the Go version appended under shared mutexes
  inside the timed path)

## Files

| File | Contents |
| --- | --- |
| `config.py` | YAML loading, sweep-axis and `rate_groups(r/w)` variant expansion, validation |
| `io.py` | aligned-bin and ground-truth binary readers/writers |
| `runner.py` | CLI entry point, single-run pipeline, stats CSV |
| `incremental_gt.py` | split ground-truth loader, recall scoring, partial-diff distribution |
| `modes.py` | throughput sweep and record/replay compare |
| `simulate.py` | sequential lag-timeline experiment with partial-insert trials (stage queries come from the driver's QuerySource via the binding) |
| `results.py` | `.res` writers, result directory scheme, diff CSV |

## Ground truth formats

Incremental ground truth is an index file (`<split> <start> <end>` per
line) pointing at binary splits: `int32 n_per_batch, int32 k,
int32 num_batches`, then per batch `uint64 offset`, optional
`uint32 query_tags[n]` (detected from file size), `float32 dists[n*k]`,
`uint32 ids[n*k]`. Overall ground truth is the DiskANN layout
(`int32 n, int32 k`, ids, optional distances).

## Tests

```
python3 test/concurrency/smoke_test.py <index_type>
python3 test/concurrency/test_replay.py
python3 test/concurrency/test_gil_isolation.py
python3 test/concurrency/test_incremental_gt.py
python3 test/concurrency/test_runner_e2e.py
python3 test/concurrency/test_modes_e2e.py
```

All synthesize their own data; no datasets needed.
