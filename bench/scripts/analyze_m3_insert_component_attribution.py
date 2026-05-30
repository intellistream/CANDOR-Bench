#!/usr/bin/env python3
"""Attribute M3 search overhead to insert-side controls and phase counters."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from bisect import bisect_right
from pathlib import Path


BASELINE = "readonly_noop"
FULL = "full_insert"

COMPONENT_METRICS = [
    ("insert_path_search_active_workers", "path-search workers"),
    ("existing_neighbor_update_active_workers", "neighbor-update workers"),
    ("existing_neighbor_load_scan_active_workers", "load-scan workers"),
    ("existing_neighbor_prune_active_workers", "prune workers"),
    ("existing_neighbor_rewrite_active_workers", "rewrite workers"),
    ("insert_path_edges_scanned", "path edges scanned"),
    ("insert_path_dist_comps", "path distance comps"),
    ("existing_neighbor_visits", "neighbor visits"),
    ("existing_neighbor_edges_loaded", "neighbor edges loaded"),
    ("existing_neighbor_prune_candidates", "prune candidates"),
    ("existing_neighbor_prune_dist_comps", "prune distance comps"),
    ("existing_neighbor_edges_written", "edges written"),
    ("existing_neighbor_edges_pruned", "edges pruned"),
    ("perf_llc_load_misses_per_s", "LLC load misses/s"),
    ("perf_dtlb_load_misses_walk_completed_per_s", "DTLB walks/s"),
]


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * pct)]


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def load_search_batches(path: Path, batch_size: int) -> dict[int, dict[str, float]]:
    batches: dict[int, dict[str, list[float] | float]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            idx = int(f(row.get("sample_idx"))) // batch_size
            batch = batches.setdefault(idx, {"op": [], "e2e": [], "finish": 0.0})
            batch["op"].append(f(row.get("op_ms")))  # type: ignore[union-attr]
            batch["e2e"].append(f(row.get("e2e_ms")))  # type: ignore[union-attr]
            batch["finish"] = max(float(batch["finish"]), f(row.get("finish_raw_ns")))
    return {
        idx: {
            "op_ms": statistics.mean(batch["op"]),  # type: ignore[arg-type]
            "e2e_ms": statistics.mean(batch["e2e"]),  # type: ignore[arg-type]
            "finish_raw_ns": float(batch["finish"]),
        }
        for idx, batch in batches.items()
    }


def load_periods(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row.get("bin_start_ns") == "bin_start_ns":
                continue
            rows.append({key: f(value) for key, value in row.items()})
    rows.sort(key=lambda row: row["bin_start_ns"])
    return rows


def period_for_finish(periods: list[dict[str, float]], starts: list[float], finish_ns: float) -> dict[str, float] | None:
    idx = bisect_right(starts, finish_ns) - 1
    if idx < 0 or idx >= len(periods):
        return None
    row = periods[idx]
    if finish_ns > row["bin_end_ns"]:
        return None
    return row


def load_control_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--case", default="high_writer_concurrency/sift1m_b200_round_robin_w40000_r5000_nolock")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_search_batches(args.root / BASELINE / args.case / "raw_latency/search_wide.csv", args.batch_size)
    full = load_search_batches(args.root / FULL / args.case / "raw_latency/search_wide.csv", args.batch_size)
    periods = load_periods(args.root / FULL / args.case / "period_groups.csv")
    starts = [row["bin_start_ns"] for row in periods]

    joined: list[dict[str, float]] = []
    for idx in sorted(set(baseline) & set(full)):
        period = period_for_finish(periods, starts, full[idx]["finish_raw_ns"])
        if period is None:
            continue
        op_delta = full[idx]["op_ms"] - baseline[idx]["op_ms"]
        e2e_delta = full[idx]["e2e_ms"] - baseline[idx]["e2e_ms"]
        joined.append(
            {
                "batch_idx": float(idx),
                "e2e_overhead_ms": e2e_delta,
                "op_overhead_ms": op_delta,
                "wait_overhead_ms": e2e_delta - op_delta,
                **period,
            }
        )

    low_cut = percentile([row["e2e_overhead_ms"] for row in joined], 0.25)
    high_cut = percentile([row["e2e_overhead_ms"] for row in joined], 0.75)
    low = [row for row in joined if row["e2e_overhead_ms"] <= low_cut]
    high = [row for row in joined if row["e2e_overhead_ms"] >= high_cut]

    rows: list[dict[str, float | str]] = []
    for key, label in COMPONENT_METRICS:
        values = [row.get(key, 0.0) for row in joined]
        high_mean = statistics.mean(row.get(key, 0.0) for row in high) if high else 0.0
        low_mean = statistics.mean(row.get(key, 0.0) for row in low) if low else 0.0
        rows.append(
            {
                "metric": key,
                "label": label,
                "corr_with_e2e_overhead": corr(values, [row["e2e_overhead_ms"] for row in joined]),
                "corr_with_wait_overhead": corr(values, [row["wait_overhead_ms"] for row in joined]),
                "corr_with_op_overhead": corr(values, [row["op_overhead_ms"] for row in joined]),
                "high_e2e_mean": high_mean,
                "low_e2e_mean": low_mean,
                "high_over_low_ratio": high_mean / low_mean if low_mean else 0.0,
            }
        )

    write_csv(args.out_csv, rows)

    e2e = [row["e2e_overhead_ms"] for row in joined]
    op = [row["op_overhead_ms"] for row in joined]
    wait = [row["wait_overhead_ms"] for row in joined]
    control_path = args.root / "figures/m3_cache_causality_control_summary.csv"
    controls = load_control_summary(control_path) if control_path.exists() else []

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with args.out_md.open("w") as file:
        file.write("# M3 Insert Component Attribution\n\n")
        file.write("Scope: SIFT1M steady read/write run, full insert compared against matched read-only replay.\n\n")
        file.write("## Search Overhead Decomposition\n\n")
        file.write(f"- matched search batches: {len(joined)}\n")
        file.write(f"- e2e overhead mean / p95: {statistics.mean(e2e):.3f} ms / {percentile(e2e, 0.95):.3f} ms\n")
        file.write(f"- wait overhead mean / p95: {statistics.mean(wait):.3f} ms / {percentile(wait, 0.95):.3f} ms\n")
        file.write(f"- search-op overhead mean / p95: {statistics.mean(op):.3f} ms / {percentile(op, 0.95):.3f} ms\n\n")
        if controls:
            file.write("## Cross-Mode Causal Control\n\n")
            file.write("| mode | e2e overhead mean ms | wait overhead mean ms | op overhead mean ms | LLC M/s | DTLB M/s | insert QPS |\n")
            file.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for row in controls:
                file.write(
                    f"| {row['label']} | {f(row['e2e_overhead_mean_ms']):.3f} | {f(row['wait_overhead_mean_ms']):.3f} | "
                    f"{f(row['op_overhead_mean_ms']):.3f} | {f(row['llc_load_miss_rate_mean']) / 1e6:.1f} | "
                    f"{f(row['dtlb_walk_rate_mean']) / 1e6:.1f} | {f(row['insert_qps']):.0f} |\n"
                )
            file.write("\n")
        file.write("## Same-Bin Component Correlation\n\n")
        file.write("| insert-side metric | corr with e2e | corr with wait | corr with op | high/low e2e-bin ratio |\n")
        file.write("|---|---:|---:|---:|---:|\n")
        for row in sorted(rows, key=lambda item: abs(float(item["corr_with_e2e_overhead"])), reverse=True):
            file.write(
                f"| {row['label']} | {float(row['corr_with_e2e_overhead']):.3f} | "
                f"{float(row['corr_with_wait_overhead']):.3f} | {float(row['corr_with_op_overhead']):.3f} | "
                f"{float(row['high_over_low_ratio']):.2f} |\n"
            )
        file.write("\n")
        file.write("## Mechanism Read\n\n")
        file.write(
            "The dominant full-insert effect is search wait/dispatch delay, not a slower search operator. "
            "Cache-heavy writer-side ANN traversal alone is not sufficient: the ANN-search dummy has higher "
            "LLC/DTLB pressure than full insert but very small search e2e overhead. Insert backlog alone is "
            "also not sufficient: the ANN-search dummy completes fewer insert points per second than full "
            "insert under the same release rate, but reader overhead stays low. Path-only insert is also near "
            "the read-only baseline. The current evidence therefore points to the full live graph-mutation / "
            "existing-node repair path as the necessary insert-side condition that turns write pressure into "
            "reader-visible wait. Same-bin phase counters do not isolate one micro-step by themselves, so the "
            "remaining proof gap is a matched-writer-pressure control that keeps writer pressure comparable "
            "while toggling the existing-node repair/mutation path.\n"
        )

    print(f"[m3-insert-attribution] wrote {args.out_csv}")
    print(f"[m3-insert-attribution] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
