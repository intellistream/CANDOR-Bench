#!/usr/bin/env python3
"""End-to-end test of the Python runner + native driver.

Synthesizes a dataset, brute-force overall ground truth, and incremental
ground truth splits for every batch offset, writes a Go-compatible YAML
config, then runs concurrency.runner and checks the recall numbers
and stats CSV. Run directly: ``python3 test_runner_e2e.py``
"""

import csv
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
sys.path.insert(0, os.path.join(REPO, "src"))

from concurrency.io import write_aligned_bin  # noqa: E402

# batch count must comfortably exceed the limiter burst (= batch_size
# tokens) or inserts finish instantly and the search producer never runs.
N, DIM, QN, K = 8_000, 16, 200, 10
BEGIN, BATCH = 2_000, 100




def brute_force_topk(queries, base, k):
    dists = (
        (queries**2).sum(1)[:, None]
        - 2 * queries @ base.T
        + (base**2).sum(1)[None, :]
    )
    idx = np.argsort(dists, axis=1, kind="stable")[:, :k]
    return idx.astype("<u4"), np.take_along_axis(dists, idx, axis=1).astype("<f4")


def main() -> int:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((N, DIM)).astype(np.float32)
    queries = rng.standard_normal((QN, DIM)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        write_aligned_bin(os.path.join(tmp, "base.bin"), data)
        write_aligned_bin(os.path.join(tmp, "query.bin"), queries)

        # Overall GT: full visibility (ids then dists, DiskANN layout).
        ids, dists = brute_force_topk(queries, data, K)
        with open(os.path.join(tmp, "overall.gt"), "wb") as f:
            f.write(struct.pack("<ii", QN, K))
            ids.tofile(f)
            dists.tofile(f)

        # Incremental GT: one split with a stage per committed batch offset.
        offsets = list(range(BEGIN, N + 1, BATCH))
        with open(os.path.join(tmp, "incr_gt_0.bin"), "wb") as f:
            f.write(struct.pack("<iii", QN, K, len(offsets)))
            for off in offsets:
                f.write(struct.pack("<Q", off))
                np.arange(QN, dtype="<u4").tofile(f)  # query tags
                s_ids, s_dists = brute_force_topk(queries, data[:off], K)
                s_dists.tofile(f)
                s_ids.tofile(f)
        with open(os.path.join(tmp, "incr_gt_index.txt"), "w") as f:
            f.write(f"incr_gt_0.bin {BEGIN} {N}\n")

        config = f"""
data:
  dataset_name: synthetic
  max_elements: {N}
  begin_num: {BEGIN}
  data_path: {tmp}/base.bin
  query_path: {tmp}/query.bin
  overall_gt_path: {tmp}/overall.gt
  incr_gt_path: {tmp}/incr_gt_index.txt
index:
  index_type: hnsw
  m: 16
  ef_construction: 200
search:
  recall_at: {K}
  ef_search: 100
  use_node_lock: true
workload:
  batch_size: {BATCH}
  num_threads: 8
  queue_size: 50
  insert_event_rate: 20000
  search_event_rate: 50000
  query_mode: round_robin
result:
  output_dir: {tmp}/results
profile:
  memory_monitor_interval: 50
  enable_memory_profile: true
"""
        cfg_path = os.path.join(tmp, "e2e.yaml")
        with open(cfg_path, "w") as f:
            f.write(config)

        proc = subprocess.run(
            [sys.executable, "-m", "concurrency.runner", "--config", cfg_path],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")},
        )
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)

        failures = []
        if proc.returncode != 0:
            failures.append(f"runner exited {proc.returncode}")
        csv_path = os.path.join(tmp, "results", "benchmark_results.csv")
        if not os.path.exists(csv_path):
            failures.append("stats CSV missing")
        else:
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if len(rows) != 1:
                failures.append(f"want 1 CSV row, got {len(rows)}")
            else:
                row = rows[0]
                recall = float(row["recall"])
                incr = float(row["incr_recall"])
                if recall < 0.85:
                    failures.append(f"overall recall too low: {recall}")
                if incr < 0.5:
                    failures.append(f"incremental recall too low: {incr}")
                if float(row["insert_qps (per-point)"]) <= 0:
                    failures.append("no insert throughput recorded")
                if float(row["peak_memory_mb"]) <= 0:
                    failures.append("memory profile produced no samples")
                print(
                    f"\noverall recall: {recall}, incr recall: {incr}, "
                    f"insert qps: {row['insert_qps (per-point)']}, "
                    f"peak mem: {row['peak_memory_mb']} MB"
                )

        if failures:
            print("FAIL:")
            for x in failures:
                print(f"  - {x}")
            return 1
        print("PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())
