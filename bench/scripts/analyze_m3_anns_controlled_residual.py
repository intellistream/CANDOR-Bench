#!/usr/bin/env python3
"""Analyze M3 search tail after controlling ANN traversal work.

The analyzer intentionally separates three quantities:

1. measured-search traversal work: diag_dist_comps and diag_hops;
2. queue/start delay: e2e_ms - op_ms;
3. overlap with insert-side graph mutation phases.

This is meant for the "exclude ef/traversal" question: after lowering or
matching traversal pressure, do remaining measured-search tails line up with
ANN graph-state mutation phases?
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import yaml


PHASE_COLS = [
    "diag_phase_existing_update_samples",
    "diag_phase_link_critical_samples",
    "diag_phase_load_scan_samples",
    "diag_phase_append_samples",
    "diag_phase_prune_samples",
    "diag_phase_undo_record_samples",
    "diag_phase_rewrite_samples",
]


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def load_windows(config: Path) -> list[tuple[float, float]]:
    with config.open() as file:
        doc = yaml.safe_load(file) or {}
    windows = []
    for item in (doc.get("workload") or {}).get("insert_burst_schedule") or []:
        start = f(item.get("start_ms"))
        end = start + f(item.get("duration_ms"))
        if end > start:
            windows.append((start, end))
    return windows


def phase_total(row: dict[str, str]) -> float:
    return sum(f(row.get(col)) for col in PHASE_COLS)


def scope_for_ms(t_ms: float, windows: list[tuple[float, float]]) -> str:
    return "insert" if any(start <= t_ms <= end for start, end in windows) else "outside"


def load_rows(run_dir: Path, config: Path) -> list[dict[str, float | str]]:
    windows = load_windows(config)
    rows: list[dict[str, float | str]] = []
    with (run_dir / "raw_latency" / "search_wide.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            if f(row.get("measured")) < 0.5:
                continue
            op_ms = f(row.get("op_ms"))
            e2e_ms = f(row.get("e2e_ms"))
            item: dict[str, float | str] = {
                "scope": scope_for_ms(f(row.get("finish_ms")), windows),
                "op_ms": op_ms,
                "queue_ms": max(0.0, e2e_ms - op_ms),
                "dist": f(row.get("diag_dist_comps")),
                "hops": f(row.get("diag_hops")),
                "phase_total": phase_total(row),
                "rewrite_period_expansions": f(row.get("diag_rewrite_period_expansions")),
                "future_skip_queries": f(row.get("diag_future_skip_queries")),
            }
            for col in PHASE_COLS:
                item[col] = f(row.get(col))
            rows.append(item)
    return rows


def write_summary(path: Path, cases: dict[str, list[dict[str, float | str]]], queue_max: float) -> None:
    fields = [
        "case",
        "scope",
        "n",
        "lowq_n",
        "op_p50",
        "op_p95",
        "op_p99",
        "op_p995",
        "op_p999",
        "op_max",
        "queue_p95_all",
        "dist_p95_lowq",
        "hops_p95_lowq",
        "phase_mean_lowq",
        "phase_p95_lowq",
        "existing_update_mean_lowq",
        "prune_mean_lowq",
        "undo_mean_lowq",
        "rewrite_period_mean_lowq",
        "future_skip_mean_lowq",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for case, rows in cases.items():
            for scope in ["outside", "insert"]:
                group = [r for r in rows if r["scope"] == scope]
                lowq = [r for r in group if float(r["queue_ms"]) < queue_max]
                writer.writerow({
                    "case": case,
                    "scope": scope,
                    "n": len(group),
                    "lowq_n": len(lowq),
                    "op_p50": percentile([float(r["op_ms"]) for r in lowq], 50),
                    "op_p95": percentile([float(r["op_ms"]) for r in lowq], 95),
                    "op_p99": percentile([float(r["op_ms"]) for r in lowq], 99),
                    "op_p995": percentile([float(r["op_ms"]) for r in lowq], 99.5),
                    "op_p999": percentile([float(r["op_ms"]) for r in lowq], 99.9),
                    "op_max": max([float(r["op_ms"]) for r in lowq]) if lowq else float("nan"),
                    "queue_p95_all": percentile([float(r["queue_ms"]) for r in group], 95),
                    "dist_p95_lowq": percentile([float(r["dist"]) for r in lowq], 95),
                    "hops_p95_lowq": percentile([float(r["hops"]) for r in lowq], 95),
                    "phase_mean_lowq": mean([float(r["phase_total"]) for r in lowq]),
                    "phase_p95_lowq": percentile([float(r["phase_total"]) for r in lowq], 95),
                    "existing_update_mean_lowq": mean([float(r["diag_phase_existing_update_samples"]) for r in lowq]),
                    "prune_mean_lowq": mean([float(r["diag_phase_prune_samples"]) for r in lowq]),
                    "undo_mean_lowq": mean([float(r["diag_phase_undo_record_samples"]) for r in lowq]),
                    "rewrite_period_mean_lowq": mean([float(r["rewrite_period_expansions"]) for r in lowq]),
                    "future_skip_mean_lowq": mean([float(r["future_skip_queries"]) for r in lowq]),
                })


def write_tail(path: Path, cases: dict[str, list[dict[str, float | str]]], queue_max: float, tail_pct: float) -> None:
    fields = [
        "case",
        "group",
        "n",
        "op_mean",
        "op_p95",
        "dist_mean",
        "hops_mean",
        "phase_total_mean",
        "existing_update_mean",
        "link_critical_mean",
        "prune_mean",
        "undo_mean",
        "rewrite_mean",
        "rewrite_period_mean",
        "future_skip_mean",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for case, rows in cases.items():
            group = [
                r for r in rows
                if r["scope"] == "insert" and float(r["queue_ms"]) < queue_max
            ]
            threshold = percentile([float(r["op_ms"]) for r in group], 100.0 - tail_pct)
            parts = {
                f"tail_top_{tail_pct:g}pct": [r for r in group if float(r["op_ms"]) >= threshold],
                "rest": [r for r in group if float(r["op_ms"]) < threshold],
            }
            for name, part in parts.items():
                writer.writerow({
                    "case": case,
                    "group": name,
                    "n": len(part),
                    "op_mean": mean([float(r["op_ms"]) for r in part]),
                    "op_p95": percentile([float(r["op_ms"]) for r in part], 95),
                    "dist_mean": mean([float(r["dist"]) for r in part]),
                    "hops_mean": mean([float(r["hops"]) for r in part]),
                    "phase_total_mean": mean([float(r["phase_total"]) for r in part]),
                    "existing_update_mean": mean([float(r["diag_phase_existing_update_samples"]) for r in part]),
                    "link_critical_mean": mean([float(r["diag_phase_link_critical_samples"]) for r in part]),
                    "prune_mean": mean([float(r["diag_phase_prune_samples"]) for r in part]),
                    "undo_mean": mean([float(r["diag_phase_undo_record_samples"]) for r in part]),
                    "rewrite_mean": mean([float(r["diag_phase_rewrite_samples"]) for r in part]),
                    "rewrite_period_mean": mean([float(r["rewrite_period_expansions"]) for r in part]),
                    "future_skip_mean": mean([float(r["future_skip_queries"]) for r in part]),
                })


def write_matched(path: Path, cases: dict[str, list[dict[str, float | str]]], queue_max: float) -> None:
    fields = [
        "case",
        "band_n",
        "dist_p10",
        "dist_p90",
        "hops_p10",
        "hops_p90",
        "overlap_n",
        "no_overlap_n",
        "overlap_op_p95",
        "no_overlap_op_p95",
        "overlap_op_p99",
        "no_overlap_op_p99",
        "overlap_dist_mean",
        "no_overlap_dist_mean",
        "overlap_hops_mean",
        "no_overlap_hops_mean",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for case, rows in cases.items():
            group = [
                r for r in rows
                if r["scope"] == "insert" and float(r["queue_ms"]) < queue_max
            ]
            d10 = percentile([float(r["dist"]) for r in group], 10)
            d90 = percentile([float(r["dist"]) for r in group], 90)
            h10 = percentile([float(r["hops"]) for r in group], 10)
            h90 = percentile([float(r["hops"]) for r in group], 90)
            band = [
                r for r in group
                if d10 <= float(r["dist"]) <= d90 and h10 <= float(r["hops"]) <= h90
            ]
            overlap = [r for r in band if float(r["phase_total"]) > 0]
            no_overlap = [r for r in band if float(r["phase_total"]) == 0]
            writer.writerow({
                "case": case,
                "band_n": len(band),
                "dist_p10": d10,
                "dist_p90": d90,
                "hops_p10": h10,
                "hops_p90": h90,
                "overlap_n": len(overlap),
                "no_overlap_n": len(no_overlap),
                "overlap_op_p95": percentile([float(r["op_ms"]) for r in overlap], 95),
                "no_overlap_op_p95": percentile([float(r["op_ms"]) for r in no_overlap], 95),
                "overlap_op_p99": percentile([float(r["op_ms"]) for r in overlap], 99),
                "no_overlap_op_p99": percentile([float(r["op_ms"]) for r in no_overlap], 99),
                "overlap_dist_mean": mean([float(r["dist"]) for r in overlap]),
                "no_overlap_dist_mean": mean([float(r["dist"]) for r in no_overlap]),
                "overlap_hops_mean": mean([float(r["hops"]) for r in overlap]),
                "no_overlap_hops_mean": mean([float(r["hops"]) for r in no_overlap]),
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", nargs=3, metavar=("LABEL", "RUN_DIR", "CONFIG"), required=True)
    parser.add_argument("--queue-max-ms", type=float, default=0.1)
    parser.add_argument("--tail-pct", type=float, default=5.0)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-tail", type=Path, required=True)
    parser.add_argument("--out-matched", type=Path, required=True)
    args = parser.parse_args()

    cases = {
        label: load_rows(Path(run_dir), Path(config))
        for label, run_dir, config in args.case
    }
    write_summary(args.out_summary, cases, args.queue_max_ms)
    write_tail(args.out_tail, cases, args.queue_max_ms, args.tail_pct)
    write_matched(args.out_matched, cases, args.queue_max_ms)
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_tail}")
    print(f"wrote {args.out_matched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
