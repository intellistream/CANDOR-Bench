#!/usr/bin/env python3
"""Record/replay test for the native driver (compare.go semantics).

Run A records its search schedule; run B starts from the same snapshot
and replays it. Every replayed search must execute at exactly the
committed offset the schedule dictates, with the same query tags.
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from testbase import import_native  # noqa: E402

cc = import_native()

DIM, N, BEGIN, K = 32, 20_000, 5_000, 10


def make_index():
    return cc.Index("hnsw", dim=DIM, max_elements=N, M=16,
                    ef_construction=200, num_threads=8)


def main() -> int:
    rng = np.random.default_rng(11)
    data = rng.standard_normal((N, DIM), dtype=np.float32)
    queries = rng.standard_normal((500, DIM), dtype=np.float32)

    base_cfg = {
        "begin_num": BEGIN,
        "max_elements": N,
        "batch_size": 100,
        "num_threads": 8,
        "insert_event_rate": 30_000.0,
        "recall_at": K,
        "query_mode": "round_robin",
        "progress_log": False,
    }

    index_a = make_index()
    index_a.build(data[:BEGIN])
    index_a.set_query_params(ef_search=100)
    snapshot = index_a.snapshot()

    out_a = cc.run_benchmark(
        index_a, data, queries,
        {**base_cfg, "search_event_rate": 20_000.0, "record_schedule": True},
    )
    schedule = out_a.get("schedule", [])

    failures = []
    if not schedule:
        failures.append("record run produced no schedule")
        print("FAIL:\n  - " + failures[0])
        return 1
    print(f"recorded {len(schedule)} schedule entries "
          f"(offsets {schedule[0][0]} .. {schedule[-1][0]})")

    index_b = make_index()
    index_b.restore(snapshot)
    index_b.set_query_params(ef_search=100)
    out_b = cc.run_benchmark(
        index_b, data, queries,
        {
            **base_cfg,
            "search_event_rate": 0.0,  # replay is offset-gated, not timed
            "replay_schedule": [(int(off), tags) for off, tags in schedule],
            "external_rw_lock": True,  # the wlock side of compare.go
        },
    )

    rec_b = out_b["records"]
    if out_b["stats"]["insert_points"] != N - BEGIN:
        failures.append(
            f"replay run inserted {out_b['stats']['insert_points']}, "
            f"want {N - BEGIN} (insert gate must fully lift)"
        )

    want = {int(off): sorted(tags.tolist()) for off, tags in schedule}
    got = defaultdict(list)
    for off, tag in zip(rec_b["insert_offsets"], rec_b["query_tags"]):
        got[int(off)].append(int(tag))
    got = {off: sorted(tags) for off, tags in got.items()}

    if set(got) != set(want):
        missing = set(want) - set(got)
        extra = set(got) - set(want)
        failures.append(
            f"replayed offsets mismatch: missing {sorted(missing)[:5]}, "
            f"extra {sorted(extra)[:5]}"
        )
    else:
        bad = [off for off in want if got[off] != want[off]]
        if bad:
            failures.append(f"query tags differ at offsets {bad[:5]}")
        else:
            print(f"replay executed all {len(want)} stages at exact "
                  f"offsets with matching tags")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
