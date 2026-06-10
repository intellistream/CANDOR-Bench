#!/usr/bin/env python3
"""Prove the benchmark's concurrency is insulated from the GIL.

Two checks:

1. While run_benchmark blocks the main thread, a pure-Python thread keeps
   counting. If the extension held the GIL, the counter would freeze; if
   the benchmark's worker threads needed the GIL, QPS would collapse.
2. The benchmark's measured QPS with and without that Python load running
   must match within noise — i.e., the measurement path never contends
   with the interpreter.
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from testbase import import_native  # noqa: E402

cc = import_native()

DIM, N, BEGIN, K = 32, 60_000, 10_000, 10


def make_index(data):
    index = cc.Index("hnsw", dim=DIM, max_elements=N, M=16,
                     ef_construction=200, num_threads=8)
    index.build(data[:BEGIN])
    index.set_query_params(ef_search=100)
    return index


def run(index, data, queries):
    return cc.run_benchmark(
        index, data, queries,
        {
            "begin_num": BEGIN,
            "max_elements": N,
            "batch_size": 100,
            "num_threads": 8,
            "insert_event_rate": 40_000.0,
            "search_event_rate": 40_000.0,
            "recall_at": K,
            "query_mode": "round_robin",
            "progress_log": False,
        },
    )["stats"]


def main() -> int:
    rng = np.random.default_rng(3)
    data = rng.standard_normal((N, DIM), dtype=np.float32)
    queries = data

    # Baseline: how fast does a pure-Python counter spin on an idle GIL?
    stop = threading.Event()
    counts = []

    def spinner():
        local = 0
        t0 = time.perf_counter()
        while not stop.is_set():
            local += 1
            if local % 100_000 == 0:
                counts.append((time.perf_counter() - t0, local))

    th = threading.Thread(target=spinner)
    th.start()
    time.sleep(1.0)
    baseline_rate = counts[-1][1] / counts[-1][0]

    # Now run the benchmark on the main thread while the spinner keeps going.
    mark = len(counts)
    stats_with_load = run(make_index(data), data, queries)
    stop.set()
    th.join()

    during = [c for c in counts[mark:]]
    if len(during) < 2:
        print("FAIL: python thread made no progress during benchmark "
              "(extension is holding the GIL)")
        return 1
    (t1, c1), (t2, c2) = during[0], during[-1]
    during_rate = (c2 - c1) / (t2 - t1)

    # Control: benchmark with no Python load at all.
    stats_quiet = run(make_index(data), data, queries)

    ratio_spin = during_rate / baseline_rate
    ratio_insert = stats_with_load["insert_qps"] / stats_quiet["insert_qps"]
    ratio_search = stats_with_load["search_qps"] / stats_quiet["search_qps"]

    print(f"python spinner rate idle:          {baseline_rate:,.0f} incr/s")
    print(f"python spinner rate during bench:  {during_rate:,.0f} incr/s "
          f"({ratio_spin:.2f}x)")
    print(f"insert QPS quiet vs python-loaded: {stats_quiet['insert_qps']:,.0f} "
          f"vs {stats_with_load['insert_qps']:,.0f} ({ratio_insert:.2f}x)")
    print(f"search QPS quiet vs python-loaded: {stats_quiet['search_qps']:,.0f} "
          f"vs {stats_with_load['search_qps']:,.0f} ({ratio_search:.2f}x)")

    failures = []
    if ratio_spin < 0.3:
        failures.append(
            "python thread starved during benchmark -> GIL in the path"
        )
    if not (0.8 < ratio_insert < 1.25) or not (0.8 < ratio_search < 1.25):
        failures.append("benchmark throughput affected by python-side load")
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: GIL is not in the measurement path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
