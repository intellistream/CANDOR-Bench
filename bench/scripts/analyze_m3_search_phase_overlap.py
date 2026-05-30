#!/usr/bin/env python3
"""Attribute measured search op latency to exact overlapping insert phases."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path

import yaml


PHASES = [
    "insert_path_search",
    "existing_neighbor_load_scan",
    "existing_neighbor_prune",
    "existing_neighbor_undo_record",
    "existing_neighbor_rewrite",
]


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def u(value: object) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    xm = mean([x for x, _ in pairs])
    ym = mean([y for _, y in pairs])
    num = sum((x - xm) * (y - ym) for x, y in pairs)
    xd = math.sqrt(sum((x - xm) ** 2 for x, _ in pairs))
    yd = math.sqrt(sum((y - ym) ** 2 for _, y in pairs))
    if xd == 0.0 or yd == 0.0:
        return float("nan")
    return num / (xd * yd)


def load_windows(config: Path) -> list[tuple[float, float]]:
    with config.open() as file:
        doc = yaml.safe_load(file) or {}
    windows = []
    for item in (doc.get("workload") or {}).get("insert_burst_schedule") or []:
        start = f(item.get("start_ms")) / 1000.0
        end = start + f(item.get("duration_ms")) / 1000.0
        if end > start:
            windows.append((start, end))
    return windows


def window_name(t_s: float, windows: list[tuple[float, float]]) -> str:
    last = 0.0
    for idx, (start, end) in enumerate(windows, 1):
        if last <= t_s < start:
            return f"O{idx}"
        if start <= t_s < end:
            return f"I{idx}"
        last = end
    return "post"


def load_search_tasks(run_dir: Path, config: Path) -> list[dict[str, object]]:
    windows = load_windows(config)
    tasks_by_key: dict[tuple[int, int, int], dict[str, object]] = {}
    with (run_dir / "raw_latency" / "search_compact.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            if f(row.get("measured")) < 0.5:
                continue
            start_ns = u(row.get("start_raw_ns"))
            finish_ns = u(row.get("finish_raw_ns"))
            create_ns = u(row.get("create_raw_ns"))
            if finish_ns <= start_ns:
                continue
            key = (create_ns, start_ns, finish_ns)
            if key in tasks_by_key:
                continue
            finish_s = f(row.get("finish_ms")) / 1000.0
            tasks_by_key[key] = {
                "task_id": len(tasks_by_key),
                "start_ns": start_ns,
                "finish_ns": finish_ns,
                "duration_ns": finish_ns - start_ns,
                "finish_s": finish_s,
                "window": window_name(finish_s, windows),
                "op_ms": f(row.get("op_ms")),
                "queue_ms": max(0.0, f(row.get("e2e_ms")) - f(row.get("op_ms"))),
                **{f"{phase}_overlap_ns": 0.0 for phase in PHASES},
            }
    return sorted(tasks_by_key.values(), key=lambda row: int(row["start_ns"]))


def add_overlaps(run_dir: Path, tasks: list[dict[str, object]]) -> None:
    starts = [int(row["start_ns"]) for row in tasks]
    max_duration = max(int(row["duration_ns"]) for row in tasks) if tasks else 0
    trace = run_dir / "insert_phase_trace.csv"
    with trace.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            phase = row.get("phase", "")
            if phase not in PHASES:
                continue
            start = u(row.get("start_ns"))
            end = u(row.get("end_ns"))
            if end <= start:
                continue
            left = bisect.bisect_left(starts, start - max_duration)
            right = bisect.bisect_left(starts, end)
            for idx in range(left, right):
                task = tasks[idx]
                overlap = max(0, min(end, int(task["finish_ns"])) - max(start, int(task["start_ns"])))
                if overlap > 0:
                    task[f"{phase}_overlap_ns"] = float(task[f"{phase}_overlap_ns"]) + overlap


def write_tasks(path: Path, tasks: list[dict[str, object]]) -> None:
    fields = [
        "task_id",
        "window",
        "finish_s",
        "op_ms",
        "queue_ms",
        "duration_ns",
    ]
    for phase in PHASES:
        fields.append(f"{phase}_workers")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            duration = float(task["duration_ns"])
            row = {field: task.get(field, "") for field in fields}
            for phase in PHASES:
                row[f"{phase}_workers"] = float(task[f"{phase}_overlap_ns"]) / duration
            writer.writerow(row)


def summarize(tasks: list[dict[str, object]], out_path: Path) -> None:
    fields = [
        "window",
        "n",
        "op_p50",
        "op_p95",
        "op_p99",
        "queue_p95",
        "top1_n",
        "top1_op_mean",
    ]
    for phase in PHASES:
        fields += [
            f"corr_op_{phase}",
            f"top1_{phase}_workers_mean",
            f"rest_{phase}_workers_mean",
        ]
    windows = ["O1", "I1", "O2", "I2", "O3", "I3", "post"]
    rows = []
    for window in windows:
        group = [task for task in tasks if task["window"] == window]
        if not group:
            continue
        op = [float(task["op_ms"]) for task in group]
        threshold = percentile(op, 0.99)
        top = [task for task in group if float(task["op_ms"]) >= threshold]
        rest = [task for task in group if float(task["op_ms"]) < threshold]
        row: dict[str, object] = {
            "window": window,
            "n": len(group),
            "op_p50": percentile(op, 0.50),
            "op_p95": percentile(op, 0.95),
            "op_p99": percentile(op, 0.99),
            "queue_p95": percentile([float(task["queue_ms"]) for task in group], 0.95),
            "top1_n": len(top),
            "top1_op_mean": mean([float(task["op_ms"]) for task in top]),
        }
        for phase in PHASES:
            key = f"{phase}_overlap_ns"
            workers = [float(task[key]) / float(task["duration_ns"]) for task in group]
            row[f"corr_op_{phase}"] = pearson(op, workers)
            row[f"top1_{phase}_workers_mean"] = mean([
                float(task[key]) / float(task["duration_ns"]) for task in top
            ])
            row[f"rest_{phase}_workers_mean"] = mean([
                float(task[key]) / float(task["duration_ns"]) for task in rest
            ])
        rows.append(row)

    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-tasks", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_search_tasks(args.run_dir, args.config)
    add_overlaps(args.run_dir, tasks)
    write_tasks(args.out_tasks, tasks)
    summarize(tasks, args.out_summary)
    print(f"tasks={len(tasks)}")
    print(f"wrote {args.out_tasks}")
    print(f"wrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
