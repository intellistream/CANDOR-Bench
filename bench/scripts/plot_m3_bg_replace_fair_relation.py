#!/usr/bin/env python3
"""Plot fair background-replacement evidence with latency/worker/cache alignment."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def to_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


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


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    xmean = sum(x for x, _ in pairs) / len(pairs)
    ymean = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - xmean) * (y - ymean) for x, y in pairs)
    xden = math.sqrt(sum((x - xmean) ** 2 for x, _ in pairs))
    yden = math.sqrt(sum((y - ymean) ** 2 for _, y in pairs))
    if xden == 0.0 or yden == 0.0:
        return float("nan")
    return num / (xden * yden)


def norm(values: list[float]) -> list[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [float("nan")] * len(values)
    lo = min(finite)
    hi = max(finite)
    if hi <= lo:
        return [0.0 if math.isfinite(v) else float("nan") for v in values]
    return [(v - lo) / (hi - lo) if math.isfinite(v) else float("nan") for v in values]


def load_config(path: Path) -> tuple[list[tuple[float, float]], float]:
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    windows = []
    for item in workload.get("insert_burst_schedule") or []:
        start = to_float(item.get("start_ms")) / 1000.0
        end = start + to_float(item.get("duration_ms")) / 1000.0
        if end > start:
            windows.append((start, end))
    return windows, to_float(workload.get("schedule_horizon_ms")) / 1000.0


def mark_windows(ax: plt.Axes, windows: list[tuple[float, float]]) -> None:
    for start, end in windows:
        ax.axvspan(start, end, color="#e5e7eb", alpha=0.45, lw=0)
        ax.axvline(start, color="#6b7280", lw=0.45)
        ax.axvline(end, color="#6b7280", lw=0.45)


def add_interval(buckets: list[float], start_s: float, end_s: float, bin_s: float, horizon: float) -> None:
    if end_s <= 0.0 or start_s >= horizon or end_s <= start_s:
        return
    start_s = max(0.0, start_s)
    end_s = min(horizon, end_s)
    first = max(0, int(start_s / bin_s))
    last = min(len(buckets) - 1, int((end_s - 1e-12) / bin_s))
    for idx in range(first, last + 1):
        bin_start = idx * bin_s
        bin_end = bin_start + bin_s
        overlap = max(0.0, min(end_s, bin_end) - max(start_s, bin_start))
        buckets[idx] += overlap / bin_s


def load_search(run_dir: Path, horizon: float, bin_s: float) -> dict[str, list[float]]:
    nbins = int(math.ceil(horizon / bin_s))
    measured_op: list[list[float]] = [[] for _ in range(nbins)]
    measured_queue: list[list[float]] = [[] for _ in range(nbins)]
    measured_workers = [0.0] * nbins
    background_workers = [0.0] * nbins
    seen_tasks: set[tuple[str, str, str, str]] = set()
    path = run_dir / "raw_latency" / "search_compact.csv"
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            finish_s = to_float(row.get("finish_ms")) / 1000.0
            if finish_s < 0.0 or finish_s >= horizon:
                continue
            measured = to_float(row.get("measured")) >= 0.5
            if measured:
                idx = min(nbins - 1, int(finish_s / bin_s))
                op_ms = to_float(row.get("op_ms"))
                e2e_ms = to_float(row.get("e2e_ms"))
                measured_op[idx].append(op_ms)
                measured_queue[idx].append(max(0.0, e2e_ms - op_ms))

            key = (
                row.get("measured", ""),
                row.get("create_raw_ns", ""),
                row.get("start_raw_ns", ""),
                row.get("finish_raw_ns", ""),
            )
            if key in seen_tasks:
                continue
            seen_tasks.add(key)
            duration_s = max(0.0, (to_float(row.get("finish_raw_ns")) - to_float(row.get("start_raw_ns"))) / 1e9)
            start_s = finish_s - duration_s
            add_interval(measured_workers if measured else background_workers, start_s, finish_s, bin_s, horizon)

    return {
        "op_p95": [percentile(bucket, 0.95) for bucket in measured_op],
        "op_p99": [percentile(bucket, 0.99) for bucket in measured_op],
        "queue_p95": [percentile(bucket, 0.95) for bucket in measured_queue],
        "measured_workers": measured_workers,
        "background_workers": background_workers,
    }


def load_periods(run_dir: Path, horizon: float, bin_s: float) -> dict[str, list[float]]:
    nbins = int(math.ceil(horizon / bin_s))
    path_workers: list[list[float]] = [[] for _ in range(nbins)]
    with (run_dir / "period_groups.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            mid_s = (to_float(row.get("bin_start_ms")) + to_float(row.get("bin_end_ms"))) / 2000.0
            if mid_s < 0.0 or mid_s >= horizon:
                continue
            idx = min(nbins - 1, int(mid_s / bin_s))
            path_workers[idx].append(to_float(row.get("insert_path_search_active_workers")))
    return {
        "path_workers": [
            sum(bucket) / len(bucket) if bucket else float("nan")
            for bucket in path_workers
        ]
    }


def load_perf(run_dir: Path, horizon: float, bin_s: float) -> dict[str, list[float]]:
    nbins = int(math.ceil(horizon / bin_s))
    event_counts = {
        "l2_rqsts.miss": [0.0] * nbins,
        "LLC-load-misses": [0.0] * nbins,
    }
    with (run_dir / "perf_interval.csv").open() as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            elapsed_s = to_float(parts[0])
            event = parts[3]
            if event not in event_counts or elapsed_s < 0.0 or elapsed_s >= horizon:
                continue
            idx = min(nbins - 1, int(elapsed_s / bin_s))
            event_counts[event][idx] += to_float(parts[1])
    return {
        "l2_miss_mps": [value / bin_s / 1e6 for value in event_counts["l2_rqsts.miss"]],
        "llc_miss_mps": [value / bin_s / 1e6 for value in event_counts["LLC-load-misses"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bin-s", type=float, default=0.20)
    args = parser.parse_args()

    windows, horizon = load_config(args.config)
    xs = [(idx + 0.5) * args.bin_s for idx in range(int(math.ceil(horizon / args.bin_s)))]
    search = load_search(args.run_dir, horizon, args.bin_s)
    periods = load_periods(args.run_dir, horizon, args.bin_s)
    perf = load_perf(args.run_dir, horizon, args.bin_s)

    corr_path = pearson(search["op_p99"], periods["path_workers"])
    corr_l2 = pearson(search["op_p99"], perf["l2_miss_mps"])
    corr_llc = pearson(search["op_p99"], perf["llc_miss_mps"])

    plt.rcParams.update({
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.5,
    })
    fig, axes = plt.subplots(4, 1, figsize=(7.1, 5.8), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    ax.plot(xs, norm(search["op_p99"]), color="#dc2626", lw=1.15, label="measured search op p99")
    ax.plot(xs, norm(periods["path_workers"]), color="#2563eb", lw=1.05, label="insert path-search workers")
    ax.plot(xs, norm(perf["l2_miss_mps"]), color="#111827", lw=1.0, label="L2 misses/s")
    mark_windows(ax, windows)
    ax.set_title("(a) Aligned signals, min-max normalized per series", loc="left", pad=1)
    ax.set_ylabel("0..1")
    ax.text(
        0.995,
        0.97,
        f"corr(op p99, pathW)={corr_path:+.2f}\n"
        f"corr(op p99, L2)={corr_l2:+.2f}\n"
        f"corr(op p99, LLC)={corr_llc:+.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "linewidth": 0.5, "alpha": 0.86},
    )
    ax.legend(frameon=False, loc="upper left", ncol=3)

    ax = axes[1]
    ax.plot(xs, search["op_p95"], color="#f97316", lw=1.0, label="op p95")
    ax.plot(xs, search["op_p99"], color="#dc2626", lw=1.15, label="op p99")
    ax.plot(xs, search["queue_p95"], color="#64748b", lw=0.9, ls="--", alpha=0.82, label="queue p95")
    mark_windows(ax, windows)
    ax.set_title("(b) Measured-search latency", loc="left", pad=1)
    ax.set_ylabel("ms")
    ax.legend(frameon=False, loc="upper left", ncol=3)

    ax = axes[2]
    ax.plot(xs, search["measured_workers"], color="#16a34a", lw=0.95, label="measured search workers")
    ax.plot(xs, search["background_workers"], color="#64748b", lw=0.95, label="background search workers")
    ax.plot(xs, periods["path_workers"], color="#2563eb", lw=1.15, label="insert path-search workers")
    mark_windows(ax, windows)
    ax.set_title("(c) Worker occupancy; one insert phase signal only", loc="left", pad=1)
    ax.set_ylabel("active\nworkers")
    ax.legend(frameon=False, loc="upper left", ncol=3)

    ax = axes[3]
    ax.plot(xs, perf["l2_miss_mps"], color="#111827", lw=1.0, label="L2 misses/s")
    ax.plot(xs, perf["llc_miss_mps"], color="#7c3aed", lw=1.0, label="LLC load misses/s")
    mark_windows(ax, windows)
    ax.set_title("(d) Hardware counters, same bins", loc="left", pad=1)
    ax.set_ylabel("M misses/s")
    ax.set_xlabel("Elapsed time (s)")
    ax.legend(frameon=False, loc="upper left", ncol=2)

    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, horizon)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"corr_op_p99_path_workers={corr_path:.4f}")
    print(f"corr_op_p99_l2_miss_mps={corr_l2:.4f}")
    print(f"corr_op_p99_llc_miss_mps={corr_llc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
