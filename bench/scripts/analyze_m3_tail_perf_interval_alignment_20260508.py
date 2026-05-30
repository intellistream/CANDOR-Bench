#!/usr/bin/env python3
"""Align measured-search top-tail batches with measured-CPU perf intervals."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


POST_START_MS = 2500.0


def percentile(values: list[float], q: float) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return fmean(vals) if vals else math.nan


def ratio(num: float, den: float) -> float:
    if not den or not math.isfinite(num) or not math.isfinite(den):
        return math.nan
    return num / den


def row_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def read_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def read_single_float(path: Path) -> float:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return math.nan


def bench_start_raw_ns(case_dir: Path) -> float:
    meta = read_meta(case_dir / "raw_latency" / "meta.csv")
    try:
        return float(meta.get("bench_start_raw_ns", ""))
    except ValueError:
        return math.nan


def find_case_dir(case_root: Path) -> Path | None:
    if (case_root / "raw_latency" / "search_wide.csv").exists():
        return case_root
    matches = sorted(case_root.glob("clean/*/raw_latency/search_wide.csv"))
    if matches:
        return matches[0].parents[1]
    matches = sorted(case_root.glob("**/raw_latency/search_wide.csv"))
    if matches:
        return matches[0].parents[1]
    return None


def load_batches(case_dir: Path) -> list[dict[str, float]]:
    path = case_dir / "raw_latency" / "search_wide.csv"
    batches: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("measured") != "1":
                continue
            finish_ms = row_float(row, "finish_ms")
            if finish_ms < POST_START_MS:
                continue
            key = (row["start_raw_ns"], row["finish_raw_ns"])
            item = batches.setdefault(
                key,
                {
                    "finish_ms": finish_ms,
                    "op_ms": row_float(row, "op_ms"),
                    "rows": 0.0,
                    "base_ms": 0.0,
                    "thread_ms": 0.0,
                    "dist": 0.0,
                    "edges": 0.0,
                    "visited": 0.0,
                },
            )
            item["rows"] += 1.0
            item["base_ms"] += row_float(row, "work_base_search_ms")
            item["thread_ms"] += row_float(row, "work_searchknn_thread_cpu_ms")
            item["dist"] += row_float(row, "work_distance_computations")
            item["edges"] += row_float(row, "work_level0_edges_scanned")
            item["visited"] += row_float(row, "work_visited_nodes")
    return sorted(batches.values(), key=lambda r: r["finish_ms"])


def load_perf_intervals(path: Path, bench_offset_ms: float) -> list[dict[str, object]]:
    by_end: dict[float, dict[str, float]] = defaultdict(dict)
    with path.open(errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 4:
                continue
            try:
                elapsed_ms = float(parts[0]) * 1000.0
                count = float(parts[1])
            except ValueError:
                continue
            event = parts[3]
            bench_end_ms = elapsed_ms - bench_offset_ms if math.isfinite(bench_offset_ms) else elapsed_ms
            by_end[bench_end_ms][event] = count

    intervals: list[dict[str, object]] = []
    prev_end = 0.0 - (bench_offset_ms if math.isfinite(bench_offset_ms) else 0.0)
    for end_ms in sorted(by_end):
        intervals.append(
            {
                "start_ms": prev_end,
                "end_ms": end_ms,
                "events": by_end[end_ms],
                "batches": [],
            }
        )
        prev_end = end_ms
    return intervals


def event(interval: dict[str, object], name: str) -> float:
    events = interval["events"]
    assert isinstance(events, dict)
    value = events.get(name, math.nan)
    return float(value) if isinstance(value, (int, float)) else math.nan


def event_per_cycle(interval: dict[str, object], name: str) -> float:
    cycles = event(interval, "cycles")
    return ratio(event(interval, name), cycles)


def ipc(interval: dict[str, object]) -> float:
    return ratio(event(interval, "instructions"), event(interval, "cycles"))


def llc_miss_rate(interval: dict[str, object]) -> float:
    return ratio(event(interval, "LLC-load-misses"), event(interval, "LLC-loads"))


def cache_miss_rate(interval: dict[str, object]) -> float:
    return ratio(event(interval, "cache-misses"), event(interval, "cache-references"))


def assign_batches(intervals: list[dict[str, object]], batches: list[dict[str, float]]) -> None:
    idx = 0
    for batch in batches:
        finish = batch["finish_ms"]
        while idx < len(intervals) and finish > float(intervals[idx]["end_ms"]):
            idx += 1
        if idx < len(intervals) and finish >= float(intervals[idx]["start_ms"]):
            interval_batches = intervals[idx]["batches"]
            assert isinstance(interval_batches, list)
            interval_batches.append(batch)


def summarize_case(case_root: Path) -> dict[str, object] | None:
    case_dir = find_case_dir(case_root)
    if case_dir is None:
        return None
    perf_path = case_root / "measured_cpu_perf_interval.csv"
    if not perf_path.exists():
        return None
    batches = load_batches(case_dir)
    if not batches:
        return None
    perf_start = read_single_float(case_root / "perf_start_raw_ns.txt")
    bench_start = bench_start_raw_ns(case_dir)
    bench_offset_ms = (
        (bench_start - perf_start) / 1_000_000.0
        if math.isfinite(perf_start) and math.isfinite(bench_start)
        else math.nan
    )
    intervals = load_perf_intervals(perf_path, bench_offset_ms)
    assign_batches(intervals, batches)

    p99 = percentile([b["op_ms"] for b in batches], 0.99)
    post_intervals = [
        it
        for it in intervals
        if float(it["end_ms"]) >= POST_START_MS and isinstance(it["batches"], list) and it["batches"]
    ]
    tail_intervals = [
        it
        for it in post_intervals
        if any(b["op_ms"] >= p99 for b in it["batches"])  # type: ignore[index]
    ]
    non_tail_intervals = [
        it
        for it in post_intervals
        if not any(b["op_ms"] >= p99 for b in it["batches"])  # type: ignore[index]
    ]

    def m(items: list[dict[str, object]], fn) -> float:
        return mean([fn(it) for it in items])

    def batch_metric(items: list[dict[str, object]], key: str) -> float:
        vals: list[float] = []
        for it in items:
            for b in it["batches"]:  # type: ignore[union-attr]
                vals.append(float(b[key]))
        return mean(vals)

    row: dict[str, object] = {
        "case": case_root.name,
        "run_dir": str(case_dir),
        "bench_offset_ms": bench_offset_ms,
        "post_batches": len(batches),
        "post_p99_ms": p99,
        "post_intervals": len(post_intervals),
        "tail_intervals": len(tail_intervals),
        "non_tail_intervals": len(non_tail_intervals),
        "tail_interval_pct": 100.0 * len(tail_intervals) / max(1, len(post_intervals)),
        "tail_interval_max_op_ms": m(tail_intervals, lambda it: max(b["op_ms"] for b in it["batches"])),  # type: ignore[union-attr]
        "non_tail_interval_max_op_ms": m(non_tail_intervals, lambda it: max(b["op_ms"] for b in it["batches"])),  # type: ignore[union-attr]
        "tail_batch_base_ms_mean": batch_metric(tail_intervals, "base_ms"),
        "non_tail_batch_base_ms_mean": batch_metric(non_tail_intervals, "base_ms"),
        "tail_batch_dist_mean": batch_metric(tail_intervals, "dist"),
        "non_tail_batch_dist_mean": batch_metric(non_tail_intervals, "dist"),
    }

    metrics = {
        "ipc": ipc,
        "llc_miss_per_cycle": lambda it: event_per_cycle(it, "LLC-load-misses"),
        "llc_miss_rate": llc_miss_rate,
        "cache_miss_per_cycle": lambda it: event_per_cycle(it, "cache-misses"),
        "cache_miss_rate": cache_miss_rate,
        "dtlb_miss_per_cycle": lambda it: event_per_cycle(it, "dTLB-load-misses"),
        "stalls_l1d_per_cycle": lambda it: event_per_cycle(it, "cycle_activity.stalls_l1d_miss"),
        "stalls_l2_per_cycle": lambda it: event_per_cycle(it, "cycle_activity.stalls_l2_miss"),
        "stalls_l3_per_cycle": lambda it: event_per_cycle(it, "cycle_activity.stalls_l3_miss"),
        "cycles_mem_any_per_cycle": lambda it: event_per_cycle(it, "cycle_activity.cycles_mem_any"),
        "scoreboard_per_cycle": lambda it: event_per_cycle(it, "resource_stalls.scoreboard"),
        "offcore_data_rd_cycles_per_cycle": lambda it: event_per_cycle(
            it, "offcore_requests_outstanding.cycles_with_demand_data_rd"
        ),
    }
    for name, fn in metrics.items():
        tail = m(tail_intervals, fn)
        non_tail = m(non_tail_intervals, fn)
        row[f"tail_{name}"] = tail
        row[f"non_tail_{name}"] = non_tail
        row[f"tail_over_non_tail_{name}"] = ratio(tail, non_tail)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    case_roots = [p for p in sorted(args.root.iterdir()) if p.is_dir()]
    rows = [row for p in case_roots if (row := summarize_case(p)) is not None]
    if not rows:
        raise SystemExit(f"no analyzable cases under {args.root}")
    out = args.out or (args.root / "tail_perf_interval_alignment.csv")
    write_csv(out, rows)
    for row in rows:
        print(
            f"{row['case']}: p99={float(row['post_p99_ms']):.3f}ms "
            f"tail_intervals={row['tail_intervals']}/{row['post_intervals']} "
            f"ipc={float(row['tail_over_non_tail_ipc']):.2f}x "
            f"l1d_stall={float(row['tail_over_non_tail_stalls_l1d_per_cycle']):.2f}x "
            f"l3_stall={float(row['tail_over_non_tail_stalls_l3_per_cycle']):.2f}x "
            f"llc_miss={float(row['tail_over_non_tail_llc_miss_rate']):.2f}x"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
