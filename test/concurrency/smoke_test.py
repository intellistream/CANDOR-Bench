#!/usr/bin/env python3
"""Smoke test for the native concurrent driver (concurrency_native).

Builds an index on a prefix of random data, then streams the rest in
concurrently with rate-limited searches. Verifies stats and visibility
semantics look sane.

Usage: python3 smoke_test.py [index_type]   (default: hnsw)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from testbase import import_native  # noqa: E402

cc = import_native()

DIM = 32
N = 20_000
BEGIN = 2_000
K = 10


def main() -> int:
    index_type = sys.argv[1] if len(sys.argv) > 1 else "hnsw"
    rng = np.random.default_rng(42)
    data = rng.standard_normal((N, DIM), dtype=np.float32)
    # Self-queries: query i is base vector i, so a visible, well-built index
    # should return tag i among the top-k once vector i is committed.
    queries = data

    # Parlay indexes ignore ef_search; they need beam_width/visit_limit
    # (and a sane level multiplier) or search degenerates.
    extra = {}
    qparams = {"ef_search": 100}
    if index_type.startswith("parlay"):
        extra = {"level_m": 0.36, "alpha": 1.2, "visit_limit": 1000}
        qparams = {"ef_search": 100, "beam_width": 100, "visit_limit": 1000}

    index = cc.Index(
        index_type,
        dim=DIM,
        max_elements=N,
        M=16,
        ef_construction=200,
        num_threads=8,
        **extra,
    )
    index.build(data[:BEGIN])
    index.set_query_params(**qparams)

    out = cc.run_benchmark(
        index,
        data,
        queries,
        {
            "begin_num": BEGIN,
            "max_elements": N,
            "batch_size": 100,
            "queue_size": 100,
            "num_threads": 8,
            "insert_event_rate": 50_000.0,
            "search_event_rate": 20_000.0,
            "recall_at": K,
            "query_mode": "round_robin",
            "seed": 7,
            "progress_log": True,
        },
    )

    stats = out["stats"]
    records = out["records"]
    print("\n--- stats ---")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")

    n_records = len(records["query_tags"])
    print(f"\n--- records: {n_records} ---")

    failures = []
    if stats["insert_points"] != N - BEGIN:
        failures.append(
            f"insert_points {stats['insert_points']} != {N - BEGIN}"
        )
    if stats["search_points"] != n_records:
        failures.append(
            f"search_points {stats['search_points']} != records {n_records}"
        )
    if n_records == 0:
        failures.append("no search records produced")
    if stats["insert_qps"] <= 0 or stats["search_qps"] <= 0:
        failures.append("zero QPS reported")

    if n_records:
        tags = records["query_tags"]
        offsets = records["insert_offsets"]
        results = records["result_tags"]
        # Self-recall on queries whose vector was committed when searched.
        eligible = tags < offsets
        if eligible.sum() == 0:
            failures.append("no eligible self-recall queries")
        else:
            hits = (results[eligible] == tags[eligible, None]).any(axis=1)
            self_recall = hits.mean()
            print(f"self-recall@{K} (committed queries): {self_recall:.4f} "
                  f"on {int(eligible.sum())} queries")
            if self_recall < 0.5:
                failures.append(f"self-recall too low: {self_recall:.4f}")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
