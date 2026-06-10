#!/usr/bin/env python3
"""End-to-end test for throughput sweep and compare modes.

Run directly: ``python3 test_modes_e2e.py``
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

N, DIM, QN = 20_000, 32, 200




BASE_CONFIG = """
data:
  dataset_name: synthetic
  max_elements: {n}
  begin_num: 5000
  data_path: {tmp}/base.bin
  query_path: {tmp}/query.bin
index:
  index_type: hnsw
  m: 16
  ef_construction: 200
search:
  recall_at: 10
  ef_search: 100
workload:
  batch_size: 100
  num_threads: 8
  queue_size: 50
  insert_event_rate: 30000
  search_event_rate: 20000
  query_mode: round_robin
result:
  output_dir: {tmp}/results
{extra}
"""


def run_config(tmp, extra):
    cfg_path = os.path.join(tmp, "cfg.yaml")
    with open(cfg_path, "w") as f:
        f.write(BASE_CONFIG.format(n=N, tmp=tmp, extra=extra))
    return subprocess.run(
        [sys.executable, "-m", "concurrency.runner", "--config", cfg_path],
        cwd=REPO, capture_output=True, text=True, timeout=900,
        env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")},
    )


def main() -> int:
    rng = np.random.default_rng(5)
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        write_aligned_bin(
            os.path.join(tmp, "base.bin"),
            rng.standard_normal((N, DIM)).astype(np.float32),
        )
        write_aligned_bin(
            os.path.join(tmp, "query.bin"),
            rng.standard_normal((QN, DIM)).astype(np.float32),
        )

        # --- throughput sweep, 3 log-spaced steps ---
        proc = run_config(tmp, (
            "throughput_sweep:\n"
            "  enabled: true\n"
            "  start_rate: 5000\n"
            "  end_rate: 20000\n"
            "  steps: 3\n"
        ))
        curve = os.path.join(tmp, "results", "throughput_latency_curve.csv")
        if proc.returncode != 0:
            failures.append(f"sweep run failed:\n{proc.stderr[-800:]}")
        elif not os.path.exists(curve):
            failures.append("sweep curve CSV missing")
        else:
            with open(curve) as f:
                rows = list(csv.DictReader(f))
            if len(rows) != 3:
                failures.append(f"sweep: want 3 rows, got {len(rows)}")
            else:
                targets = [float(r["target_search_rate"]) for r in rows]
                actuals = [float(r["actual_search_qps"]) for r in rows]
                if targets != sorted(targets) or targets[0] != 5000:
                    failures.append(f"sweep targets wrong: {targets}")
                if any(a <= 0 for a in actuals):
                    failures.append(f"sweep produced zero QPS: {actuals}")
                print(f"sweep targets {targets} -> actual {actuals}")

        # --- simulate (lag timeline + partial-insert trial) ---
        proc = run_config(tmp, (
            "compare:\n"
            "  enabled: true\n"
            "  simulate:\n"
            "    enabled: true\n"
            "    partial_update:\n"
            "      enabled: true\n"
            "      seed: 3\n"
            "      mode: 1\n"
        ))
        files_dir = os.path.join(
            tmp, "results", "hnsw", "synthetic", "round_robin", "files"
        )
        if proc.returncode != 0:
            failures.append(f"simulate run failed:\n{proc.stderr[-800:]}")
        else:
            for fname in ("hnsw_incr.res", "hnsw_incr_lag.res",
                          "hnsw_incr_partial.res"):
                path = os.path.join(files_dir, fname)
                if not os.path.exists(path) or os.path.getsize(path) <= 8:
                    failures.append(f"simulate output missing/empty: {fname}")
            if not failures:
                print("simulate wrote consistent/lagging/partial res files")

        # --- compare (record wolock / replay wlock) ---
        proc = run_config(tmp, "compare:\n  enabled: true\n")
        cmp_csv = os.path.join(tmp, "results", "compare_stats.csv")
        if proc.returncode != 0:
            failures.append(f"compare run failed:\n{proc.stderr[-800:]}")
        elif not os.path.exists(cmp_csv):
            failures.append("compare CSV missing")
        else:
            with open(cmp_csv) as f:
                rows = {r["metric"]: r for r in csv.DictReader(f)}
            for key in ("insert_qps", "search_qps"):
                if key not in rows:
                    failures.append(f"compare CSV missing {key}")
                    continue
                a = float(rows[key]["wolock"])
                b = float(rows[key]["wlock"])
                if a <= 0 or b <= 0:
                    failures.append(f"compare {key} non-positive: {a}, {b}")
            if "search_qps" in rows:
                print(
                    f"compare search_qps wolock={rows['search_qps']['wolock']} "
                    f"wlock={rows['search_qps']['wlock']}"
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
