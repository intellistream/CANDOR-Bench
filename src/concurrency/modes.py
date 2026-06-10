"""Sweep and compare run modes.

Throughput sweep: log-spaced search rates between start_rate and end_rate,
a fresh index per step, emitting throughput_latency_curve.csv.

Compare: run A (no external RW lock) records its search schedule from a
snapshot baseline; run B (external RW lock) restores the snapshot and
replays the identical schedule, so both runs are comparable stage by
stage. Stats for both land in compare_stats.csv.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict

SWEEP_HEADER = [
    "target_search_rate", "actual_search_qps",
    "mean_search_op_latency_ms", "p95_search_op_latency_ms",
    "p99_search_op_latency_ms", "mean_search_e2e_latency_ms",
    "p95_search_e2e_latency_ms", "p99_search_e2e_latency_ms",
]


def run_throughput_sweep(cc, cfg: Dict[str, Any], run_variant) -> str:
    """run_variant(cc, name, cfg) is runner.run_variant — reused per step."""
    sweep = cfg.get("throughput_sweep", {})
    steps = int(sweep.get("steps") or 10)
    start_rate = float(sweep.get("start_rate") or 0)
    end_rate = float(sweep.get("end_rate") or 0)
    if start_rate <= 0 or end_rate <= 0:
        raise ValueError(
            "throughput_sweep: start_rate and end_rate must be positive"
        )

    output_dir = cfg["result"].get("output_dir") or "."
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "throughput_latency_curve.csv")

    log_step = (
        (math.log(end_rate) - math.log(start_rate)) / (steps - 1)
        if steps > 1
        else 0.0
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SWEEP_HEADER)
        for i in range(steps):
            target = math.exp(math.log(start_rate) + i * log_step)
            print(f"==== sweep step {i + 1}/{steps}: "
                  f"target_search_rate={target:.0f} ====")
            step_cfg = {**cfg, "workload": dict(cfg["workload"])}
            step_cfg["workload"]["search_event_rate"] = target
            step_cfg.pop("throughput_sweep", None)
            stats = run_variant(cc, f"sweep_{i + 1}", step_cfg)["stats"]
            writer.writerow([
                f"{target:.0f}",
                f"{stats['search_qps']:.2f}",
                f"{stats['mean_search_op_latency_ms']:.4f}",
                f"{stats['p95_search_op_latency_ms']:.4f}",
                f"{stats['p99_search_op_latency_ms']:.4f}",
                f"{stats['mean_search_e2e_latency_ms']:.4f}",
                f"{stats['p95_search_e2e_latency_ms']:.4f}",
                f"{stats['p99_search_e2e_latency_ms']:.4f}",
            ])
            f.flush()
    print(f"throughput-latency curve written to {csv_path}")
    return csv_path


_COMPARE_KEYS = [
    "insert_qps", "search_qps",
    "mean_insert_op_latency_ms", "p99_insert_op_latency_ms",
    "mean_search_op_latency_ms", "p99_search_op_latency_ms",
    "mean_insert_e2e_latency_ms", "mean_search_e2e_latency_ms",
    "search_lag_avg", "search_lag_max",
]


def run_compare(cc, cfg: Dict[str, Any], run_variant) -> str:
    """Record (wolock) vs replay (wlock) comparison.

    The record phase runs without the external RW lock and captures both
    its search schedule and a baseline snapshot; the replay phase
    restores the snapshot (or rebuilds when the index type has no
    snapshots) and re-issues the identical searches under the lock.

    Both phases run with one insert and one search worker, and the wlock
    phase caps the queue at one batch, so lock-cost ratios stay comparable
    with historical results.
    """
    queue = int(cfg["workload"].get("queue_size", 0))

    wolock_cfg = {**cfg, "workload": dict(cfg["workload"])}
    wolock_cfg["workload"]["with_external_rw_lock"] = False
    print("==== compare: record phase (wolock) ====")
    out_a = run_variant(
        cc, "compare_wol", wolock_cfg,
        extra_driver={"record_schedule": True, "num_threads": 2},
        take_snapshot=True,
    )
    schedule = out_a["out"].get("schedule", [])
    if not schedule:
        raise RuntimeError("compare: record phase produced no schedule")
    print(f"recorded {len(schedule)} schedule entries")

    wlock_cfg = {**cfg, "workload": dict(cfg["workload"])}
    wlock_cfg["workload"]["with_external_rw_lock"] = True
    print("==== compare: replay phase (wlock) ====")
    out_b = run_variant(
        cc,
        "compare_wl",
        wlock_cfg,
        extra_driver={
            "replay_schedule": [(int(o), t) for o, t in schedule],
            "search_event_rate": 0.0,
            "num_threads": 2,
            "queue_size": min(queue, 1),
        },
        snapshot=out_a.get("snapshot"),
    )

    output_dir = cfg["result"].get("output_dir") or "."
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "compare_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "wolock", "wlock", "wlock/wolock"])
        sa, sb = out_a["out"]["stats"], out_b["out"]["stats"]
        for key in _COMPARE_KEYS:
            a, b = float(sa[key]), float(sb[key])
            writer.writerow(
                [key, f"{a:.4f}", f"{b:.4f}",
                 f"{b / a:.4f}" if a else ""]
            )
        for label, out in (("wolock", out_a), ("wlock", out_b)):
            if out.get("incr"):
                writer.writerow(
                    [f"incr_recall_{label}", f"{out['incr']['recall']:.4f}",
                     "", ""]
                )
    print(f"comparison written to {csv_path}")
    return csv_path
