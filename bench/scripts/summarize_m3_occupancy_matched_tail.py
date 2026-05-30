#!/usr/bin/env python3
"""Summarize M3 measured-search tail with occupancy-matched controls.

The summary intentionally avoids comparing search and insert at the same input
rate. It reports the actual competing active-worker occupancy and picks the
background-search run whose occupancy best matches each insert run.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import yaml


SEARCH_WORKER_COLS = [
    "measured_search_workers",
    "background_search_workers",
]

INSERT_WORKER_COLS = [
    "insert_task_workers",
    "insert_path_workers",
    "existing_update_workers",
    "prune_workers",
    "undo_workers",
    "rewrite_workers",
]


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def percentile(values: Iterable[float], pct: float) -> float:
    ordered = sorted(v for v in values if math.isfinite(v))
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def load_config(path: Path) -> tuple[list[tuple[float, float]], float]:
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    windows = []
    for item in workload.get("insert_burst_schedule") or []:
        start = f(item.get("start_ms"))
        end = start + f(item.get("duration_ms"))
        if end > start:
            windows.append((start, end))
    horizon = f(workload.get("schedule_horizon_ms"))
    return windows, horizon


def scope_windows(kind: str, windows: list[tuple[float, float]], horizon: float) -> list[tuple[float, float]]:
    if kind.startswith("insert") and windows:
        return windows
    return [(0.0, horizon)]


def in_any_window(t_ms: float, windows: list[tuple[float, float]]) -> bool:
    return any(start <= t_ms < end for start, end in windows)


def duration_ms(windows: list[tuple[float, float]]) -> float:
    return sum(max(0.0, end - start) for start, end in windows)


def overlap_ms(start_ms: float, end_ms: float, windows: list[tuple[float, float]]) -> float:
    if end_ms <= start_ms:
        return 0.0
    total = 0.0
    for win_start, win_end in windows:
        total += max(0.0, min(end_ms, win_end) - max(start_ms, win_start))
    return total


def open_search_csv(run_dir: Path) -> Path:
    wide = run_dir / "raw_latency" / "search_wide.csv"
    if wide.exists():
        return wide
    compact = run_dir / "raw_latency" / "search_compact.csv"
    if compact.exists():
        return compact
    raise FileNotFoundError(f"missing search_wide.csv/search_compact.csv under {run_dir}")


def search_rows(run_dir: Path) -> list[dict[str, str]]:
    with open_search_csv(run_dir).open(newline="") as file:
        return list(csv.DictReader(file))


def search_worker_occupancy(rows: list[dict[str, str]], windows: list[tuple[float, float]]) -> dict[str, float]:
    scope_ms = duration_ms(windows)
    if scope_ms <= 0.0:
        return {col: float("nan") for col in SEARCH_WORKER_COLS}
    measured_ms = 0.0
    background_ms = 0.0
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            row.get("measured", ""),
            row.get("create_raw_ns", ""),
            row.get("start_raw_ns", ""),
            row.get("finish_raw_ns", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        finish_ms = f(row.get("finish_ms"))
        duration = max(0.0, (f(row.get("finish_raw_ns")) - f(row.get("start_raw_ns"))) / 1_000_000.0)
        start_ms = finish_ms - duration
        active = overlap_ms(start_ms, finish_ms, windows)
        if f(row.get("measured")) >= 0.5:
            measured_ms += active
        else:
            background_ms += active
    return {
        "measured_search_workers": measured_ms / scope_ms,
        "background_search_workers": background_ms / scope_ms,
    }


def measured_latency_summary(rows: list[dict[str, str]], windows: list[tuple[float, float]], queue_max_ms: float) -> dict[str, float]:
    measured = [
        row for row in rows
        if f(row.get("measured")) >= 0.5 and in_any_window(f(row.get("finish_ms")), windows)
    ]
    lowq = [
        row for row in measured
        if max(0.0, f(row.get("e2e_ms")) - f(row.get("op_ms"))) <= queue_max_ms
    ]
    return {
        "measured_rows": float(len(measured)),
        "lowq_rows": float(len(lowq)),
        "op_p50_lowq_ms": percentile((f(row.get("op_ms")) for row in lowq), 50),
        "op_p95_lowq_ms": percentile((f(row.get("op_ms")) for row in lowq), 95),
        "op_p99_lowq_ms": percentile((f(row.get("op_ms")) for row in lowq), 99),
        "op_p995_lowq_ms": percentile((f(row.get("op_ms")) for row in lowq), 99.5),
        "op_p999_lowq_ms": percentile((f(row.get("op_ms")) for row in lowq), 99.9),
        "queue_p95_all_ms": percentile((max(0.0, f(row.get("e2e_ms")) - f(row.get("op_ms"))) for row in measured), 95),
        "dist_p95_lowq": percentile((f(row.get("diag_dist_comps")) for row in lowq), 95),
        "hops_p95_lowq": percentile((f(row.get("diag_hops")) for row in lowq), 95),
    }


def insert_worker_summary(run_dir: Path, windows: list[tuple[float, float]]) -> dict[str, float]:
    groups_path = run_dir / "period_groups.csv"
    scope_ms = duration_ms(windows)
    insert_task_workers = 0.0
    insert_path = run_dir / "raw_latency" / "insert_wide.csv"
    if insert_path.exists() and scope_ms > 0.0:
        with insert_path.open(newline="") as file:
            for row in csv.DictReader(file):
                finish_ms = f(row.get("finish_ms"))
                duration = max(0.0, (f(row.get("finish_raw_ns")) - f(row.get("start_raw_ns"))) / 1_000_000.0)
                start_ms = finish_ms - duration
                insert_task_workers += overlap_ms(start_ms, finish_ms, windows) / scope_ms
    if groups_path.exists():
        buckets: dict[str, list[float]] = {col: [] for col in INSERT_WORKER_COLS}
        with groups_path.open(newline="") as file:
            for row in csv.DictReader(file):
                mid_ms = (f(row.get("bin_start_ms")) + f(row.get("bin_end_ms"))) / 2.0
                if not in_any_window(mid_ms, windows):
                    continue
                buckets["insert_task_workers"].append(insert_task_workers)
                buckets["insert_path_workers"].append(f(row.get("insert_path_search_active_workers")))
                buckets["existing_update_workers"].append(f(row.get("existing_neighbor_update_active_workers")))
                buckets["prune_workers"].append(f(row.get("existing_neighbor_prune_active_workers")))
                buckets["undo_workers"].append(f(row.get("existing_neighbor_undo_record_active_workers")))
                buckets["rewrite_workers"].append(f(row.get("existing_neighbor_rewrite_active_workers")))
        out = {col: mean(vals) for col, vals in buckets.items()}
        out["insert_total_highlevel_workers"] = f(out["insert_path_workers"]) + f(out["existing_update_workers"])
        out["occupancy_source"] = "period_groups"
        return out

    if not insert_path.exists() or scope_ms <= 0.0:
        out = {col: 0.0 for col in INSERT_WORKER_COLS}
        out["insert_total_highlevel_workers"] = 0.0
        out["occupancy_source"] = "none"
        return out
    out = {col: 0.0 for col in INSERT_WORKER_COLS}
    out["insert_task_workers"] = insert_task_workers
    out["insert_total_highlevel_workers"] = insert_task_workers
    out["occupancy_source"] = "insert_task_interval_fallback"
    return out


def benchmark_recall(run_dir: Path) -> float:
    path = run_dir / "benchmark_results.csv"
    if not path.exists():
        return float("nan")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return float("nan")
    return f(rows[-1].get("recall"))


def summarize_case(case: dict[str, str], queue_max_ms: float) -> dict[str, object]:
    run_dir = Path(case["run_dir"])
    config = Path(case["config"])
    windows, horizon = load_config(config)
    scoped = scope_windows(case["kind"], windows, horizon)
    rows = search_rows(run_dir)
    out: dict[str, object] = {
        "case": case["case"],
        "kind": case["kind"],
        "rate": case.get("rate", ""),
        "ef_construction": case.get("ef_construction", ""),
        "mutation": case.get("mutation", ""),
        "scope_ms": duration_ms(scoped),
        "run_dir": str(run_dir),
        "config": str(config),
    }
    out.update(measured_latency_summary(rows, scoped, queue_max_ms))
    out.update(search_worker_occupancy(rows, scoped))
    out.update(insert_worker_summary(run_dir, scoped))
    out["recall"] = benchmark_recall(run_dir)
    return out


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def matched_pairs(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    bg_rows = [row for row in summary if row.get("kind") == "bg_search"]
    baseline = next((row for row in summary if row.get("kind") == "baseline"), None)
    pairs: list[dict[str, object]] = []
    for insert in summary:
        if not str(insert.get("kind", "")).startswith("insert"):
            continue
        target = f(insert.get("insert_total_highlevel_workers"))
        if not bg_rows or not math.isfinite(target):
            continue
        bg = min(bg_rows, key=lambda row: abs(f(row.get("background_search_workers")) - target))
        row: dict[str, object] = {
            "insert_case": insert["case"],
            "matched_bg_case": bg["case"],
            "target_insert_total_workers": target,
            "matched_bg_workers": bg.get("background_search_workers"),
            "worker_abs_delta": abs(f(bg.get("background_search_workers")) - target),
            "insert_op_p95_lowq_ms": insert.get("op_p95_lowq_ms"),
            "bg_op_p95_lowq_ms": bg.get("op_p95_lowq_ms"),
            "insert_op_p99_lowq_ms": insert.get("op_p99_lowq_ms"),
            "bg_op_p99_lowq_ms": bg.get("op_p99_lowq_ms"),
            "insert_op_p999_lowq_ms": insert.get("op_p999_lowq_ms"),
            "bg_op_p999_lowq_ms": bg.get("op_p999_lowq_ms"),
            "insert_queue_p95_all_ms": insert.get("queue_p95_all_ms"),
            "bg_queue_p95_all_ms": bg.get("queue_p95_all_ms"),
            "insert_dist_p95_lowq": insert.get("dist_p95_lowq"),
            "bg_dist_p95_lowq": bg.get("dist_p95_lowq"),
            "insert_hops_p95_lowq": insert.get("hops_p95_lowq"),
            "bg_hops_p95_lowq": bg.get("hops_p95_lowq"),
            "insert_recall": insert.get("recall"),
            "bg_recall": bg.get("recall"),
        }
        row["op_p95_delta_vs_bg_ms"] = f(row["insert_op_p95_lowq_ms"]) - f(row["bg_op_p95_lowq_ms"])
        row["op_p99_delta_vs_bg_ms"] = f(row["insert_op_p99_lowq_ms"]) - f(row["bg_op_p99_lowq_ms"])
        row["op_p999_delta_vs_bg_ms"] = f(row["insert_op_p999_lowq_ms"]) - f(row["bg_op_p999_lowq_ms"])
        if baseline is not None:
            row["baseline_case"] = baseline["case"]
            row["baseline_op_p95_lowq_ms"] = baseline.get("op_p95_lowq_ms")
            row["baseline_op_p99_lowq_ms"] = baseline.get("op_p99_lowq_ms")
            row["op_p95_delta_vs_baseline_ms"] = f(insert.get("op_p95_lowq_ms")) - f(baseline.get("op_p95_lowq_ms"))
            row["op_p99_delta_vs_baseline_ms"] = f(insert.get("op_p99_lowq_ms")) - f(baseline.get("op_p99_lowq_ms"))
        pairs.append(row)
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-matched", type=Path, required=True)
    parser.add_argument("--queue-max-ms", type=float, default=0.1)
    args = parser.parse_args()

    cases = load_manifest(args.manifest)
    summary = [summarize_case(case, args.queue_max_ms) for case in cases]
    summary_fields = [
        "case",
        "kind",
        "rate",
        "ef_construction",
        "mutation",
        "scope_ms",
        "measured_rows",
        "lowq_rows",
        "op_p50_lowq_ms",
        "op_p95_lowq_ms",
        "op_p99_lowq_ms",
        "op_p995_lowq_ms",
        "op_p999_lowq_ms",
        "queue_p95_all_ms",
        "dist_p95_lowq",
        "hops_p95_lowq",
        "measured_search_workers",
        "background_search_workers",
        "insert_task_workers",
        "insert_path_workers",
        "existing_update_workers",
        "insert_total_highlevel_workers",
        "prune_workers",
        "undo_workers",
        "rewrite_workers",
        "occupancy_source",
        "recall",
        "run_dir",
        "config",
    ]
    write_csv(args.out_summary, summary, summary_fields)

    pairs = matched_pairs(summary)
    matched_fields = [
        "insert_case",
        "matched_bg_case",
        "target_insert_total_workers",
        "matched_bg_workers",
        "worker_abs_delta",
        "insert_op_p95_lowq_ms",
        "bg_op_p95_lowq_ms",
        "op_p95_delta_vs_bg_ms",
        "insert_op_p99_lowq_ms",
        "bg_op_p99_lowq_ms",
        "op_p99_delta_vs_bg_ms",
        "insert_op_p999_lowq_ms",
        "bg_op_p999_lowq_ms",
        "op_p999_delta_vs_bg_ms",
        "insert_queue_p95_all_ms",
        "bg_queue_p95_all_ms",
        "insert_dist_p95_lowq",
        "bg_dist_p95_lowq",
        "insert_hops_p95_lowq",
        "bg_hops_p95_lowq",
        "insert_recall",
        "bg_recall",
        "baseline_case",
        "baseline_op_p95_lowq_ms",
        "baseline_op_p99_lowq_ms",
        "op_p95_delta_vs_baseline_ms",
        "op_p99_delta_vs_baseline_ms",
    ]
    write_csv(args.out_matched, pairs, matched_fields)
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_matched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
