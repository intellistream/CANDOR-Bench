#!/usr/bin/env python3
"""M3 writer-phase watermark pre-experiment figure."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


def to_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_batches(path: Path) -> list[dict[str, float]]:
    seen: dict[tuple[int, int, int], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            create = int(to_float(row.get("create_raw_ns")))
            start = int(to_float(row.get("start_raw_ns")))
            finish = int(to_float(row.get("finish_raw_ns")))
            if finish <= 0:
                continue
            seen[(create, start, finish)] = {
                "create_raw_ns": float(create),
                "start_raw_ns": float(start),
                "finish_raw_ns": float(finish),
                "task_insert_offset": to_float(row.get("task_insert_offset")),
                "view_offset": to_float(row.get("view_offset")),
                "e2e_ms": max(0.0, (finish - create) / 1_000_000.0),
            }
    return sorted(seen.values(), key=lambda r: (r["create_raw_ns"], r["finish_raw_ns"]))


def load_periods(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        return [{k: to_float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def weighted_period_value(batch: dict[str, float], periods: list[dict[str, float]], col: str) -> float:
    starts = [int(r["bin_start_ns"]) for r in periods]
    begin = int(batch["create_raw_ns"])
    end = int(batch["finish_raw_ns"])
    first = max(0, bisect.bisect_right(starts, begin) - 1)
    total = 0.0
    acc = 0.0
    for row in periods[first:]:
        bin_start = int(row["bin_start_ns"])
        bin_end = int(row["bin_end_ns"])
        if bin_start >= end:
            break
        overlap = min(end, bin_end) - max(begin, bin_start)
        if overlap <= 0:
            continue
        total += overlap
        acc += overlap * row.get(col, 0.0)
    return acc / total if total > 0 else 0.0


def max_period_value(batch: dict[str, float], periods: list[dict[str, float]], col: str) -> float:
    starts = [int(r["bin_start_ns"]) for r in periods]
    begin = int(batch["create_raw_ns"])
    end = int(batch["finish_raw_ns"])
    first = max(0, bisect.bisect_right(starts, begin) - 1)
    out = 0.0
    for row in periods[first:]:
        bin_start = int(row["bin_start_ns"])
        bin_end = int(row["bin_end_ns"])
        if bin_start >= end:
            break
        overlap = min(end, bin_end) - max(begin, bin_start)
        if overlap > 0:
            out = max(out, row.get(col, 0.0))
    return out


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = max(0, min(len(vals) - 1, math.ceil(pct / 100.0 * len(vals)) - 1))
    return vals[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--xmin-k", type=float, default=0.0)
    parser.add_argument("--xmax-k", type=float, default=40.0)
    parser.add_argument("--active-threshold", type=float, default=0.5)
    args = parser.parse_args()

    target_batches = load_batches(args.target / "raw_latency" / "search_wide.csv")
    baseline_batches = load_batches(args.baseline / "raw_latency" / "search_wide.csv")
    target_periods = load_periods(args.target / "period_groups.csv")
    baseline_periods = load_periods(args.baseline / "period_groups.csv")
    if not target_batches or not baseline_batches or not target_periods:
        raise SystemExit("missing target/baseline data")

    baseline_l2_vals = [
        r["perf_l2_rqsts_miss_per_s"] / 1.0e9
        for r in baseline_periods
        if r.get("perf_l2_rqsts_miss_per_s", 0.0) > 0
    ]
    baseline_l2 = sum(baseline_l2_vals) / len(baseline_l2_vals) if baseline_l2_vals else 0.0

    rows: list[dict[str, float]] = []
    if any(r.get("view_offset", 0.0) <= 0.0 for r in target_batches):
        raise SystemExit("target search_wide.csv is missing view_offset; rerun with updated benchmark")
    if any(r.get("task_insert_offset", 0.0) <= 0.0 for r in target_batches + baseline_batches):
        raise SystemExit("search_wide.csv is missing task_insert_offset; rerun with updated benchmark")

    baseline_by_task_offset = {
        int(r["task_insert_offset"]): r for r in baseline_batches
    }
    task_offsets = sorted({int(r["task_insert_offset"]) for r in target_batches})
    task_steps = [
        b - a for a, b in zip(task_offsets, task_offsets[1:]) if b > a
    ]
    batch_step = min(task_steps) if task_steps else 200
    begin_offset = min(task_offsets) - batch_step
    for target in target_batches:
        baseline = baseline_by_task_offset.get(int(target["task_insert_offset"]))
        if baseline is None:
            continue
        rows.append(
            {
                "target_watermark_k": (target["task_insert_offset"] - begin_offset) / 1000.0,
                "latency_delta_ms": target["e2e_ms"] - baseline["e2e_ms"],
                "l2_miss_1e9": weighted_period_value(target, target_periods, "perf_l2_rqsts_miss_per_s") / 1.0e9,
                "path_activity_raw": weighted_period_value(
                    target, target_periods, "insert_path_search_active_workers"
                ),
                "path_activity_peak": max_period_value(
                    target, target_periods, "insert_path_search_active_workers"
                ),
            }
        )

    zoom = [r for r in rows if args.xmin_k <= r["target_watermark_k"] <= args.xmax_k]
    if not zoom:
        raise SystemExit("no batches in requested target-watermark range")
    for row in zoom:
        row["path_activity"] = row["path_activity_peak"]

    watermark = [r["target_watermark_k"] for r in zoom]
    latency = [r["latency_delta_ms"] for r in zoom]
    writer_pressure = [r["path_activity"] for r in zoom]

    active = [p >= args.active_threshold for p in writer_pressure]
    active_latency = [v for v, a in zip(latency, active) if a]
    inactive_latency = [v for v, a in zip(latency, active) if not a]
    active_median = percentile(active_latency, 50.0)
    inactive_median = percentile(inactive_latency, 50.0)

    fig, ax = plt.subplots(figsize=(7.2, 2.75), constrained_layout=True)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)
    inactive_x = [x for x, a in zip(watermark, active) if not a]
    inactive_y = [y for y, a in zip(latency, active) if not a]
    active_x = [x for x, a in zip(watermark, active) if a]
    active_y = [y for y, a in zip(latency, active) if a]
    ax.scatter(
        inactive_x,
        inactive_y,
        s=7,
        color="#aaaaaa",
        alpha=0.65,
        linewidths=0,
        label="no writer graph-search overlap",
    )
    ax.scatter(
        active_x,
        active_y,
        s=8,
        color="#2f6db2",
        alpha=0.85,
        linewidths=0,
        label="overlaps writer graph search",
    )

    ax.axhline(active_median, color="#2f6db2", linestyle="--", linewidth=1.0)
    ax.axhline(inactive_median, color="#777777", linestyle="--", linewidth=1.0)
    ax.text(args.xmax_k - 0.5, active_median + 3.0, f"overlap median {active_median:.1f} ms", ha="right", va="bottom", color="#2f6db2")
    ax.text(args.xmax_k - 0.5, inactive_median + 3.0, f"no-overlap median {inactive_median:.1f} ms", ha="right", va="bottom", color="#555555")
    ax.set_xlabel("search-batch target watermark since run start (K inserted points)")
    ax.set_ylabel("batch latency Δ ms")
    ax.set_title("Batch latency spikes overlap writer graph search")
    ax.set_xlim(args.xmin_k, args.xmax_k)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right", frameon=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    print(f"[m3-watermark-align] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
