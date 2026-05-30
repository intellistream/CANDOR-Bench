#!/usr/bin/env python3
"""Plot matched search e2e latency overhead against insert-side pressure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": True,
        "axes.spines.right": True,
    }
)


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as file:
        return [{key: f(value) for key, value in row.items()} for row in csv.DictReader(file)]


def load_config(path: Path) -> tuple[float, int, float]:
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    data = doc.get("data") or {}
    insert_rate = f(workload.get("insert_event_rate"))
    if insert_rate <= 0:
        groups = workload.get("rate_groups(r/w)") or workload.get("rate_groups") or []
        if groups:
            first = groups[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                insert_rate = f(first[1])
            elif isinstance(first, dict):
                insert_rate = f(first.get("write") or first.get("insert") or first.get("insert_event_rate"))
    batch_size = int(f(workload.get("batch_size")) or 1)
    insert_work = f(data.get("max_elements")) - f(data.get("begin_num"))
    return insert_rate, batch_size, insert_work


def load_start_raw_ns(run_dir: Path) -> float:
    with (run_dir / "raw_latency" / "meta.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            if row.get("key") == "bench_start_raw_ns":
                return f(row.get("value"))
    return 0.0


def load_finish_times(path: Path) -> list[float]:
    out: list[float] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            finish = f(row.get("finish_raw_ns"))
            if finish > 0:
                out.append(finish)
    return sorted(out)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * pct)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--min-matched-batches", type=int, default=2)
    parser.add_argument("--title", default="Search e2e latency rises with insert backlog")
    args = parser.parse_args()

    rows = load_rows(args.timeline_csv)
    insert_rate, batch_size, insert_work = load_config(args.config)
    start_raw_ns = load_start_raw_ns(args.full_run)
    insert_finishes = load_finish_times(args.full_run / "raw_latency" / "insert_wide.csv")

    xs: list[float] = []
    ys: list[float] = []
    colors: list[float] = []
    graph: list[float] = []
    completed_idx = 0
    for row in rows:
        if row["matched_search_batches"] < args.min_matched_batches:
            continue
        elapsed = row["elapsed_s"]
        raw_cutoff = start_raw_ns + elapsed * 1e9
        while completed_idx < len(insert_finishes) and insert_finishes[completed_idx] <= raw_cutoff:
            completed_idx += 1
        released = min(elapsed * insert_rate, insert_work)
        completed = min(completed_idx * batch_size, insert_work)
        backlog_k = max(0.0, released - completed) / 1000.0
        xs.append(backlog_k)
        ys.append(row["observed_delta_ms_mean"])
        colors.append(elapsed)
        graph.append(row["insert_graph_traversal_occupancy_worker_equiv"])

    if not xs:
        raise SystemExit("no matched bins to plot")

    # Equal-width bins over the observed backlog range for a readable trend.
    bin_count = 10
    lo, hi = min(xs), max(xs)
    width = (hi - lo) / bin_count if hi > lo else 1.0
    trend_x: list[float] = []
    trend_y: list[float] = []
    trend_p95: list[float] = []
    for i in range(bin_count):
        start = lo + i * width
        end = hi + 1e-9 if i == bin_count - 1 else start + width
        vals = [y for x, y in zip(xs, ys) if start <= x < end]
        if not vals:
            continue
        trend_x.append((start + end) / 2)
        trend_y.append(sum(vals) / len(vals))
        trend_p95.append(percentile(vals, 0.95))

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    fig.patch.set_facecolor("white")
    ax.grid(axis="both", color="#e5e7eb", linewidth=0.75)
    ax.set_axisbelow(True)
    sizes = [18 + min(g, 16.0) * 1.8 for g in graph]
    scatter = ax.scatter(xs, ys, c=colors, s=sizes, cmap="viridis", alpha=0.78, linewidth=0)
    ax.plot(trend_x, trend_y, color="#111827", linewidth=1.8, label="binned mean")
    ax.plot(trend_x, trend_p95, color="#6b7280", linewidth=1.15, linestyle="--", label="binned p95")
    ax.set_title(args.title, loc="left", pad=8)
    ax.set_xlabel("insert backlog: released minus completed (k points)")
    ax.set_ylabel("matched search e2e latency overhead (ms)")
    ax.legend(frameon=False, loc="upper left")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("elapsed time (s)")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240)

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_out.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["backlog_k_points_bin_center", "e2e_overhead_mean_ms", "e2e_overhead_p95_ms"])
            writer.writerows(zip(trend_x, trend_y, trend_p95))
    print(f"[m3-latency-vs-insert-pressure] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
