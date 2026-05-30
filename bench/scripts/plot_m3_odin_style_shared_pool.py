#!/usr/bin/env python3
"""Plot OdinANN-style shared-pool mixed workload evidence without a memory panel."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


MODES = [
    ("cpu_dummy", "CPU-hold", "#64748b"),
    ("path" + "_only_cpu_delay", "repair-disabled", "#2563eb"),
    ("full_insert", "full insert", "#dc2626"),
]


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_config(path: Path) -> tuple[list[tuple[float, float]], float, float]:
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    windows = []
    for item in workload.get("insert_burst_schedule") or []:
        start = f(item.get("start_ms")) / 1000.0
        end = start + f(item.get("duration_ms")) / 1000.0
        if end > start:
            windows.append((start, end))
    return windows, f(workload.get("schedule_horizon_ms")) / 1000.0, f(workload.get("insert_event_rate")) / 1000.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def case_dirs(root: Path, mode: str, case_prefix: str, label: str, reps: int) -> list[Path]:
    if reps == 1:
        return [root / mode / case_prefix / label]
    return [root / mode / f"{case_prefix}_rep{rep:02d}" / label for rep in range(1, reps + 1)]


def bin_latency(root: Path, mode: str, case_prefix: str, label: str, reps: int, horizon: float, bin_s: float) -> tuple[list[float], list[float], list[float]]:
    nbins = int(math.ceil(horizon / bin_s)) + 1
    e2e_buckets: list[list[float]] = [[] for _ in range(nbins)]
    op_buckets: list[list[float]] = [[] for _ in range(nbins)]
    for case_dir in case_dirs(root, mode, case_prefix, label, reps):
        with (case_dir / "raw_latency" / "search_wide.csv").open(newline="") as file:
            for row in csv.DictReader(file):
                if "measured" in row and f(row["measured"]) < 0.5:
                    continue
                elapsed = f(row["finish_ms"]) / 1000.0
                if elapsed < 0 or elapsed > horizon:
                    continue
                idx = min(nbins - 1, int(elapsed / bin_s))
                e2e_buckets[idx].append(f(row["e2e_ms"]))
                op_buckets[idx].append(f(row["op_ms"]))
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    e2e = [percentile(bucket, 0.95) for bucket in e2e_buckets]
    op = [percentile(bucket, 0.95) for bucket in op_buckets]
    return xs, e2e, op


def bin_period(root: Path, mode: str, case_prefix: str, label: str, reps: int, horizon: float, bin_s: float, columns: list[str], scale: float = 1.0) -> tuple[list[float], list[float]]:
    nbins = int(math.ceil(horizon / bin_s)) + 1
    buckets: list[list[float]] = [[] for _ in range(nbins)]
    for case_dir in case_dirs(root, mode, case_prefix, label, reps):
        with (case_dir / "period_groups.csv").open(newline="") as file:
            for row in csv.DictReader(file):
                elapsed = (f(row.get("bin_start_ms")) + f(row.get("bin_end_ms"))) / 2000.0
                if elapsed < 0 or elapsed > horizon:
                    continue
                value = sum(f(row.get(column)) for column in columns) * scale
                idx = min(nbins - 1, int(elapsed / bin_s))
                buckets[idx].append(value)
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    ys = [statistics.mean(bucket) if bucket else float("nan") for bucket in buckets]
    return xs, ys


def scheduled_rate(windows: list[tuple[float, float]], horizon: float, bin_s: float, rate_kpts: float) -> tuple[list[float], list[float]]:
    nbins = int(math.ceil(horizon / bin_s)) + 1
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    ys = [rate_kpts if any(start <= x <= end for start, end in windows) else 0.0 for x in xs]
    return xs, ys


def mark_windows(ax: plt.Axes, windows: list[tuple[float, float]], annotate: bool = False) -> None:
    for idx, (start, end) in enumerate(windows):
        ax.axvspan(start, end, color="#e5e7eb", alpha=0.45, lw=0)
        ax.axvline(start, color="#6b7280", lw=0.45)
        ax.axvline(end, color="#6b7280", lw=0.45)
        if annotate and idx == 0:
            ymax = ax.get_ylim()[1]
            ax.annotate("begin\ninsert", xy=(start, ymax * 0.48), xytext=(start - 0.8, ymax * 0.76), arrowprops={"arrowstyle": "->", "lw": 0.45}, ha="right", fontsize=6.3)
            ax.annotate("end\ninsert", xy=(end, ymax * 0.34), xytext=(end + 0.55, ymax * 0.60), arrowprops={"arrowstyle": "->", "lw": 0.45}, ha="left", fontsize=6.3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--case-prefix", default="shared_worker_pool")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--bin-s", type=float, default=0.20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    windows, horizon, insert_rate = load_config(args.config)
    plt.rcParams.update({"font.size": 7.0, "axes.titlesize": 8.0, "axes.labelsize": 7.4, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.5})
    fig, axes = plt.subplots(3, 1, figsize=(6.9, 4.7), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax_lat, ax_insert, ax_hw = axes

    for mode, label, color in MODES:
        xs, e2e, op = bin_latency(args.root, mode, args.case_prefix, args.label, args.reps, horizon, args.bin_s)
        ax_lat.plot(xs, e2e, color=color, lw=1.1, label=f"{label} e2e p95")
        if mode == "full_insert":
            ax_lat.plot(xs, op, color=color, lw=0.85, ls="--", alpha=0.8, label="full execution p95")
    ax_lat.set_title("(a) Search p95 latency under shared-pool read/write load", loc="left", pad=1)
    ax_lat.set_ylabel("Latency\n(ms)")
    ax_lat.set_ylim(bottom=0)
    mark_windows(ax_lat, windows)
    ax_lat.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.53, 1.0))

    graph_insert_cols = [
        "insert_path_search_active_workers",
        "existing_neighbor_load_scan_active_workers",
        "existing_neighbor_prune_active_workers",
        "existing_neighbor_undo_record_active_workers",
        "existing_neighbor_rewrite_active_workers",
    ]
    search_cols = ["search_e2e_active_workers"]
    xs, ys = bin_period(args.root, "full_insert", args.case_prefix, args.label, args.reps, horizon, args.bin_s, graph_insert_cols)
    ax_insert.plot(xs, ys, color="#dc2626", lw=1.05, label="full graph-mutation workers")
    for mode, label, color in MODES:
        xs, ys = bin_period(args.root, mode, args.case_prefix, args.label, args.reps, horizon, args.bin_s, search_cols)
        ax_insert.plot(xs, ys, color=color, lw=0.9, ls=":", alpha=0.85, label=f"{label} search occupancy")
    ax_rate = ax_insert.twinx()
    xs_rate, ys_rate = scheduled_rate(windows, horizon, args.bin_s, insert_rate)
    ax_rate.plot(xs_rate, ys_rate, color="#111827", lw=0.9, ls="--", label="insert release")
    ax_insert.set_title("(b) Shared-pool occupancy and insert release", loc="left", pad=1)
    ax_insert.set_ylabel("Active\nworkers")
    ax_rate.set_ylabel("Release\n(k pts/s)")
    mark_windows(ax_insert, windows, annotate=True)
    lines, labels = ax_insert.get_legend_handles_labels()
    r_lines, r_labels = ax_rate.get_legend_handles_labels()
    ax_insert.legend(lines + r_lines, labels + r_labels, frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.52, 1.0))

    hw_cols = ["perf_llc_load_misses_per_s"]
    for mode, label, color in MODES:
        xs, ys = bin_period(args.root, mode, args.case_prefix, args.label, args.reps, horizon, args.bin_s, hw_cols, scale=1e-6)
        ax_hw.plot(xs, ys, color=color, lw=1.05, label=f"{label} LLC")
    ax_hw.set_title("(c) Hardware pressure during the same intervals", loc="left", pad=1)
    ax_hw.set_ylabel("LLC miss\n(M/s)")
    ax_hw.set_xlabel("Elapsed time (s)")
    mark_windows(ax_hw, windows)
    ax_hw.legend(frameon=False, ncol=3, loc="upper right")

    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, horizon)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(f"[m3-odin-shared-pool] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
