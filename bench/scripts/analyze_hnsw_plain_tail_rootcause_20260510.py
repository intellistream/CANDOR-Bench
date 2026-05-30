#!/usr/bin/env python3
"""Summarize the 2026-05-10 plain-HNSW tail root-cause runs."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
from pathlib import Path


WINDOWS = ((2500.0, 3500.0), (6000.0, 7000.0))
MODES = ("round_robin", "chasing", "peeking", "zipfian")


def quantile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def in_insert_window(finish_ms: float) -> bool:
    return any(start <= finish_ms < end for start, end in WINDOWS)


def analyze_raw(raw_dir: Path) -> dict[str, float]:
    insert_intervals: list[tuple[int, int]] = []
    with (raw_dir / "insert_wide.csv").open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: header.index(name) for name in ("start_raw_ns", "finish_raw_ns")}
        for row in reader:
            insert_intervals.append(
                (int(row[idx["start_raw_ns"]]), int(row[idx["finish_raw_ns"]]))
            )

    insert_starts = sorted(start for start, _ in insert_intervals)
    insert_finishes = sorted(end for _, end in insert_intervals)

    batches: dict[tuple[str, str], dict[str, object]] = {}
    with (raw_dir / "search_wide.csv").open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {
            name: header.index(name)
            for name in (
                "measured",
                "finish_ms",
                "start_raw_ns",
                "finish_raw_ns",
                "search_index_ms",
                "work_searchknn_ms",
                "work_distance_computations",
                "work_level0_edges_scanned",
                "work_level0_expansions",
                "work_level0_lock_wait_ms",
                "work_upper_lock_wait_ms",
            )
        }
        for row in reader:
            if row[idx["measured"]] != "1":
                continue
            if not in_insert_window(float(row[idx["finish_ms"]])):
                continue
            key = (row[idx["start_raw_ns"]], row[idx["finish_raw_ns"]])
            batch = batches.setdefault(
                key,
                {
                    "start": int(row[idx["start_raw_ns"]]),
                    "finish": int(row[idx["finish_raw_ns"]]),
                    "index": float(row[idx["search_index_ms"]]),
                    "count": 0,
                    "work": [],
                    "dist": [],
                    "edges": [],
                    "expansions": [],
                    "level0_wait": [],
                    "upper_wait": [],
                },
            )
            batch["count"] = int(batch["count"]) + 1
            batch["work"].append(float(row[idx["work_searchknn_ms"]]))  # type: ignore[index]
            batch["dist"].append(float(row[idx["work_distance_computations"]]))  # type: ignore[index]
            batch["edges"].append(float(row[idx["work_level0_edges_scanned"]]))  # type: ignore[index]
            batch["expansions"].append(float(row[idx["work_level0_expansions"]]))  # type: ignore[index]
            batch["level0_wait"].append(float(row[idx["work_level0_lock_wait_ms"]]))  # type: ignore[index]
            batch["upper_wait"].append(float(row[idx["work_upper_lock_wait_ms"]]))  # type: ignore[index]

    rows: list[dict[str, float]] = []
    for batch in batches.values():
        start = int(batch["start"])
        finish = int(batch["finish"])
        active_inserts = (
            bisect.bisect_left(insert_starts, finish)
            - bisect.bisect_right(insert_finishes, start)
        )
        work_values = batch["work"]  # type: ignore[assignment]
        batch_wall = float(batch["index"]) * int(batch["count"])
        max_work = max(work_values) if work_values else 0.0
        rows.append(
            {
                "index": float(batch["index"]),
                "batch_gap": batch_wall - max_work,
                "active_inserts": float(active_inserts),
                "max_work": max_work,
                "mean_dist": mean(batch["dist"]),  # type: ignore[arg-type]
                "mean_edges": mean(batch["edges"]),  # type: ignore[arg-type]
                "mean_expansions": mean(batch["expansions"]),  # type: ignore[arg-type]
                "mean_level0_wait": mean(batch["level0_wait"]),  # type: ignore[arg-type]
                "mean_upper_wait": mean(batch["upper_wait"]),  # type: ignore[arg-type]
            }
        )

    active0 = [row["index"] for row in rows if row["active_inserts"] == 0]
    active20 = [row["index"] for row in rows if row["active_inserts"] >= 20]
    return {
        "batch_count": float(len(rows)),
        "index_p99": quantile([row["index"] for row in rows], 0.99),
        "batch_gap_p99": quantile([row["batch_gap"] for row in rows], 0.99),
        "active_insert_p99": quantile([row["active_inserts"] for row in rows], 0.99),
        "active0_index_p99": quantile(active0, 0.99),
        "active20_index_p99": quantile(active20, 0.99),
        "active20_batches": float(len(active20)),
        "max_work_p99": quantile([row["max_work"] for row in rows], 0.99),
        "dist_mean_p99": quantile([row["mean_dist"] for row in rows], 0.99),
        "edges_mean_p99": quantile([row["mean_edges"] for row in rows], 0.99),
        "expansions_mean_p99": quantile([row["mean_expansions"] for row in rows], 0.99),
        "level0_wait_mean": mean([row["mean_level0_wait"] for row in rows]),
        "upper_wait_mean": mean([row["mean_upper_wait"] for row in rows]),
    }


def raw_path(bench_root: Path, case: str, mode: str) -> Path:
    if case == "shared" and mode == "peeking":
        return (
            bench_root
            / "result/hnsw_plain_work_rootcause_peeking_20260510"
            / mode
            / "nolock/rep1/raw_latency"
        )
    if case == "search_arena" and mode == "peeking":
        return (
            bench_root
            / "result/hnsw_plain_work_rootcause_peeking_searcharena_20260510"
            / mode
            / "nolock/rep1/raw_latency"
        )
    root_name = (
        "hnsw_plain_work_rootcause_extra_shared_20260510"
        if case == "shared"
        else "hnsw_plain_work_rootcause_extra_searcharena_20260510"
    )
    return bench_root / "result" / root_name / mode / "nolock/rep1/raw_latency"


def main() -> None:
    raise SystemExit(
        "Disabled: the first hnsw_plain_work_rootcause_*searcharena* roots were "
        "generated before the HNSW measured/non-measured arena call sites were "
        "corrected. Use paper-progress/notes/hnsw_nosteal_write_specific_2026-05-10.md."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    shared = {mode: analyze_raw(raw_path(args.bench_root, "shared", mode)) for mode in MODES}
    arena = {
        mode: analyze_raw(raw_path(args.bench_root, "search_arena", mode)) for mode in MODES
    }

    print(
        "mode,shared_p99,search_arena_p99,reduction_pct,"
        "shared_batch_gap_p99,arena_batch_gap_p99,shared_active_insert_p99,"
        "active0_p99,active20_p99,active20_batches,max_work_p99,"
        "dist_mean_p99,edges_mean_p99"
    )
    for mode in MODES:
        shared_p99 = shared[mode]["index_p99"]
        arena_p99 = arena[mode]["index_p99"]
        reduction = (shared_p99 - arena_p99) / shared_p99 * 100.0
        print(
            f"{mode},{shared_p99:.6f},{arena_p99:.6f},{reduction:.1f},"
            f"{shared[mode]['batch_gap_p99']:.6f},{arena[mode]['batch_gap_p99']:.6f},"
            f"{shared[mode]['active_insert_p99']:.1f},"
            f"{shared[mode]['active0_index_p99']:.6f},"
            f"{shared[mode]['active20_index_p99']:.6f},"
            f"{int(shared[mode]['active20_batches'])},"
            f"{shared[mode]['max_work_p99']:.6f},"
            f"{shared[mode]['dist_mean_p99']:.2f},"
            f"{shared[mode]['edges_mean_p99']:.2f}"
        )


if __name__ == "__main__":
    main()
