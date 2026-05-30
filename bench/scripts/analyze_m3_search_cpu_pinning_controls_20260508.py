#!/usr/bin/env python3
"""Analyze pinned measured-search CPU controls for the M3 tail question."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


INSERT_WINDOWS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]
STABLE_OUTSIDE = [(3500.0, 6000.0), (7000.0, 9500.0), (10500.0, 12000.0)]
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


def in_ranges(ms: float, ranges: list[tuple[float, float]]) -> bool:
    return any(start <= ms < end for start, end in ranges)


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return fmean(vals) if vals else math.nan


def read_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def row_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def load_batches(case_dir: Path) -> list[dict[str, float]]:
    path = case_dir / "raw_latency" / "search_wide.csv"
    if not path.exists():
        return []
    batches: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("measured") != "1":
                continue
            try:
                finish_ms = float(row["finish_ms"])
                op_ms = float(row["op_ms"])
            except (KeyError, ValueError):
                continue
            key = (row["start_raw_ns"], row["finish_raw_ns"])
            item = batches.setdefault(
                key,
                {
                    "finish_ms": finish_ms,
                    "op_ms": op_ms,
                    "search_index_ms": row_float(row, "search_index_ms"),
                    "sum_searchknn_ms": 0.0,
                    "sum_thread_cpu_ms": 0.0,
                    "sum_base_ms": 0.0,
                    "sum_dist": 0.0,
                    "sum_edges": 0.0,
                    "sum_l0_lock_ms": 0.0,
                    "cpu_migrations": 0.0,
                    "rows": 0.0,
                },
            )
            item["sum_searchknn_ms"] += row_float(row, "work_searchknn_ms")
            item["sum_thread_cpu_ms"] += row_float(row, "work_searchknn_thread_cpu_ms")
            item["sum_base_ms"] += row_float(row, "work_base_search_ms")
            item["sum_dist"] += row_float(row, "work_distance_computations")
            item["sum_edges"] += row_float(row, "work_level0_edges_scanned")
            item["sum_l0_lock_ms"] += row_float(row, "work_level0_lock_wait_ms")
            start_cpu = row.get("work_start_cpu", "")
            end_cpu = row.get("work_end_cpu", "")
            if start_cpu and end_cpu and start_cpu != end_cpu:
                item["cpu_migrations"] += 1.0
            item["rows"] += 1.0
    return sorted(batches.values(), key=lambda r: r["finish_ms"])


def summarize_batches(batches: list[dict[str, float]]) -> dict[str, float]:
    post = [b for b in batches if b["finish_ms"] >= POST_START_MS]
    insert = [b for b in post if in_ranges(b["finish_ms"], INSERT_WINDOWS)]
    outside = [b for b in post if in_ranges(b["finish_ms"], STABLE_OUTSIDE)]
    ops = [b["op_ms"] for b in post]
    threshold = percentile(ops, 0.99)
    top = [b for b in post if b["op_ms"] >= threshold]
    rest = [b for b in post if b["op_ms"] < threshold]

    def mean(items: list[dict[str, float]], key: str) -> float:
        return finite_mean([b[key] for b in items])

    def ratio(key: str) -> float:
        denom = mean(rest, key)
        return mean(top, key) / denom if denom and math.isfinite(denom) else math.nan

    return {
        "batches_post": float(len(post)),
        "insert_batches_post": float(len(insert)),
        "outside_batches_post": float(len(outside)),
        "post_p50_ms": percentile(ops, 0.50),
        "post_p95_ms": percentile(ops, 0.95),
        "post_p99_ms": percentile(ops, 0.99),
        "insert_p99_ms": percentile([b["op_ms"] for b in insert], 0.99),
        "outside_p99_ms": percentile([b["op_ms"] for b in outside], 0.99),
        "top1_insert_pct": 100.0 * sum(1 for b in top if in_ranges(b["finish_ms"], INSERT_WINDOWS)) / max(1, len(top)),
        "top1_base_time_ratio": ratio("sum_base_ms"),
        "top1_dist_ratio": ratio("sum_dist"),
        "top1_edges_ratio": ratio("sum_edges"),
        "top1_l0_lock_ratio": ratio("sum_l0_lock_ms"),
        "top1_thread_cpu_ratio": ratio("sum_thread_cpu_ms"),
        "top1_searchknn_wall_ratio": ratio("sum_searchknn_ms"),
        "top1_wall_minus_cpu_ms": mean(top, "sum_searchknn_ms") - mean(top, "sum_thread_cpu_ms"),
        "rest_wall_minus_cpu_ms": mean(rest, "sum_searchknn_ms") - mean(rest, "sum_thread_cpu_ms"),
        "cpu_migration_query_pct": 100.0
        * sum(b["cpu_migrations"] for b in post)
        / max(1.0, sum(b["rows"] for b in post)),
    }


def read_single_float(path: Path) -> float:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return math.nan


def bench_start_raw_ns(case_run_dir: Path) -> float:
    meta = read_meta(case_run_dir / "raw_latency" / "meta.csv")
    try:
        return float(meta.get("bench_start_raw_ns", ""))
    except ValueError:
        return math.nan


def parse_perf_interval(path: Path, bench_offset_ms: float) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    events: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
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
            bench_elapsed_ms = elapsed_ms - bench_offset_ms if math.isfinite(bench_offset_ms) else elapsed_ms
            events[event].append((bench_elapsed_ms, count, 0.0))

    out: dict[str, dict[str, float]] = {}
    for event, rows in events.items():
        rows.sort()
        enriched: list[tuple[float, float, float]] = []
        prev_ms = 0.0
        for elapsed_ms, count, _ in rows:
            dur_s = max((elapsed_ms - prev_ms) / 1000.0, 1e-9)
            enriched.append((elapsed_ms, count, count / dur_s))
            prev_ms = elapsed_ms
        post = [r for r in enriched if r[0] >= POST_START_MS]
        insert = [r for r in post if in_ranges(r[0], INSERT_WINDOWS)]
        outside = [r for r in post if in_ranges(r[0], STABLE_OUTSIDE)]
        out[event] = {
            "post_rate": finite_mean([r[2] for r in post]),
            "insert_rate": finite_mean([r[2] for r in insert]),
            "outside_rate": finite_mean([r[2] for r in outside]),
        }
    return out


def flatten_perf(perf: dict[str, dict[str, float]], bench_offset_ms: float) -> dict[str, float]:
    out: dict[str, float] = {}
    out["perf_bench_offset_ms"] = bench_offset_ms
    for event, stats in perf.items():
        safe = event.replace(".", "_").replace("-", "_")
        for key, value in stats.items():
            out[f"perf_{safe}_{key}"] = value
    cycles = perf.get("cycles", {}).get("post_rate", math.nan)
    instr = perf.get("instructions", {}).get("post_rate", math.nan)
    llc = perf.get("LLC-load-misses", {}).get("post_rate", math.nan)
    llc_loads = perf.get("LLC-loads", {}).get("post_rate", math.nan)
    rfo = perf.get("l2_rqsts.rfo_miss", {}).get("post_rate", math.nan)
    if cycles and math.isfinite(cycles):
        out["perf_post_ipc"] = instr / cycles if math.isfinite(instr) else math.nan
        out["perf_post_llc_miss_per_cycle"] = llc / cycles if math.isfinite(llc) else math.nan
        out["perf_post_rfo_miss_per_cycle"] = rfo / cycles if math.isfinite(rfo) else math.nan
    if llc_loads and math.isfinite(llc_loads):
        out["perf_post_llc_load_miss_rate"] = llc / llc_loads if math.isfinite(llc) else math.nan
    return out


def find_case_dir(case_root: Path) -> Path | None:
    matches = sorted(case_root.glob("clean/*/raw_latency/search_wide.csv"))
    if not matches:
        return None
    return matches[0].parents[1]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row.get(key, "")
                out[key] = f"{value:.6g}" if isinstance(value, float) else value
            writer.writerow(out)


def write_readme(path: Path, rows: list[dict[str, object]]) -> None:
    order = ["searchonly", "noop", "cpu", "annread", "real"]
    by_case = {str(r.get("case")): r for r in rows}
    lines = [
        "# M3 Pinned Measured-Search CPU Controls",
        "",
        "Question: for post-warmup measured search tail, is the same-work slowdown visible on measured-search CPUs, and which insert-side control reproduces it?",
        "",
        "Current scope: one SIFT200K rep by default; use this as directional evidence, not a final paper claim.",
        "",
        "| case | post P99 ms | insert P99 ms | outside P99 ms | top1 base ratio | top1 dist ratio | top1 thread CPU ratio | IPC | LLC miss/cycle | RFO miss/cycle | CPU migration % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in order:
        row = by_case.get(case)
        if not row:
            continue
        f = lambda key: row.get(key, math.nan)
        lines.append(
            "| {case} | {post:.4f} | {insert:.4f} | {outside:.4f} | {base:.2f} | {dist:.2f} | {cpu:.2f} | {ipc:.3f} | {llc:.3g} | {rfo:.3g} | {mig:.2f} |".format(
                case=case,
                post=float(f("post_p99_ms")),
                insert=float(f("insert_p99_ms")),
                outside=float(f("outside_p99_ms")),
                base=float(f("top1_base_time_ratio")),
                dist=float(f("top1_dist_ratio")),
                cpu=float(f("top1_thread_cpu_ratio")),
                ipc=float(f("perf_post_ipc")),
                llc=float(f("perf_post_llc_miss_per_cycle")),
                rfo=float(f("perf_post_rfo_miss_per_cycle")),
                mig=float(f("cpu_migration_query_pct")),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation guardrails:",
            "",
            "- `perf` is sampled with `perf stat -C` on the measured-search CPU list, so it is closer to search-side evidence than process-level perf, but still includes OS noise and any task that lands on those CPUs.",
            "- Perf intervals are shifted by the measured `perf_start_raw_ns -> bench_start_raw_ns` offset before insert/outside segmentation.",
            "- A control only explains the tail if it reproduces both latency and the same-work ratios under the pinned setup.",
            "- RFO/coherence still needs interval or top-tail alignment before being promoted from plausible to confirmed.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for case_root in sorted(args.root.iterdir()):
        if not case_root.is_dir():
            continue
        run_dir = find_case_dir(case_root)
        if run_dir is None:
            continue
        meta = read_meta(case_root / "case_meta.csv")
        row: dict[str, object] = {
            "case": case_root.name,
            "run_dir": str(run_dir),
            "dummy_mode": meta.get("dummy_mode", ""),
            "measured_cpus": meta.get("measured_cpus", ""),
            "background_cpus": meta.get("background_cpus", ""),
            "insert_cpus": meta.get("insert_cpus", ""),
            "case_insert_workers": meta.get("case_insert_workers", ""),
        }
        row.update(summarize_batches(load_batches(run_dir)))
        perf_start = read_single_float(case_root / "perf_start_raw_ns.txt")
        bench_start = bench_start_raw_ns(run_dir)
        bench_offset_ms = (bench_start - perf_start) / 1_000_000.0 if math.isfinite(perf_start) and math.isfinite(bench_start) else math.nan
        row.update(flatten_perf(parse_perf_interval(case_root / "measured_cpu_perf_interval.csv", bench_offset_ms), bench_offset_ms))
        rows.append(row)

    order = {"searchonly": 0, "noop": 1, "cpu": 2, "annread": 3, "real": 4}
    rows.sort(key=lambda r: order.get(str(r.get("case")), 99))
    write_csv(args.out, rows)
    write_readme(args.readme, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
