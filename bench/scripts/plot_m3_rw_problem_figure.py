#!/usr/bin/env python3
"""Build the main M3 read/write interference figure."""

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
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 7.5,
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


def load_repeat_rows(path: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row.get("rep") == "all":
                out[row["window"]] = {key: f(value) for key, value in row.items()}
    return out


def load_config(path: Path) -> tuple[float, list[tuple[float, float, float]], float]:
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    read_rate = f(workload.get("search_event_rate")) / 1000.0
    default_insert_rate = f(workload.get("insert_event_rate")) / 1000.0
    windows: list[tuple[float, float, float]] = []
    for item in workload.get("insert_burst_schedule") or []:
        start = f(item.get("start_ms")) / 1000.0
        end = start + f(item.get("duration_ms")) / 1000.0
        rate = f(item.get("insert_event_rate")) / 1000.0
        if rate <= 0:
            rate = default_insert_rate
        if end > start:
            windows.append((start, end, rate))
    horizon = f(workload.get("schedule_horizon_ms")) / 1000.0
    return read_rate, windows, horizon


def normalize(values: list[float]) -> list[float]:
    positive = sorted(v for v in values if v > 0)
    if not positive:
        return [0.0 for _ in values]
    scale = positive[int((len(positive) - 1) * 0.95)]
    if scale <= 0:
        scale = max(positive)
    return [min(v / scale, 1.25) for v in values]


def shade_windows(axes: list[plt.Axes], windows: list[tuple[float, float, float]]) -> None:
    for ax in axes:
        for idx, (start, end, _) in enumerate(windows):
            ax.axvspan(
                start,
                end,
                color="#d1d5db",
                alpha=0.20,
                lw=0,
                label="insert release window" if idx == 0 and ax is axes[0] else None,
            )
            ax.axvline(start, color="#9ca3af", linewidth=0.55, alpha=0.55)
            ax.axvline(end, color="#9ca3af", linewidth=0.55, alpha=0.55)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--repeat-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-matched-batches", type=int, default=2)
    parser.add_argument("--latency-ymax-ms", type=float, default=140.0)
    parser.add_argument("--title", default="Controlled insert windows expose read-latency interference")
    args = parser.parse_args()

    rows = load_rows(args.timeline_csv)
    repeat = load_repeat_rows(args.repeat_csv)
    read_rate, windows, horizon = load_config(args.config)

    x = [row["elapsed_s"] for row in rows]
    graph = [row["insert_graph_traversal_occupancy_worker_equiv"] for row in rows]
    l2 = normalize([row["l2_miss_rate"] for row in rows])
    llc = normalize([row["llc_load_miss_rate"] for row in rows])
    dtlb = normalize([row["dtlb_walk_rate"] for row in rows])
    op_x: list[float] = []
    op_mean: list[float] = []
    op_max: list[float] = []
    for row in rows:
        if row["matched_search_batches"] < args.min_matched_batches:
            continue
        op_x.append(row["elapsed_s"])
        op_mean.append(row["execution_delta_ms_mean"])
        op_max.append(min(row["execution_delta_ms_max"], args.latency_ymax_ms))

    fig = plt.figure(figsize=(12.2, 6.2), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, width_ratios=[4.5, 1.55], height_ratios=[0.85, 1.0, 1.1], wspace=0.26, hspace=0.28)
    ax_work = fig.add_subplot(gs[0, 0])
    ax_pressure = fig.add_subplot(gs[1, 0], sharex=ax_work)
    ax_latency = fig.add_subplot(gs[2, 0], sharex=ax_work)
    ax_bar = fig.add_subplot(gs[:, 1])
    timeline_axes = [ax_work, ax_pressure, ax_latency]
    fig.patch.set_facecolor("white")

    for ax in timeline_axes + [ax_bar]:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.75)
        ax.set_axisbelow(True)

    shade_windows(timeline_axes, windows)

    ax_work.plot([0, horizon], [read_rate, read_rate], color="#2563eb", linewidth=1.7, label="continuous read release rate")
    for idx, (start, end, rate) in enumerate(windows):
        ax_work.fill_between([start, end], [0, 0], [rate, rate], color="#dc2626", alpha=0.80, step="post", label="insert release rate" if idx == 0 else None)
    ax_work.set_ylim(0, max([read_rate] + [rate for _, _, rate in windows]) * 1.22)
    ax_work.set_ylabel("scheduled work\n(k points/s)")
    ax_work.set_title(args.title, loc="left", pad=8)
    ax_work.legend(frameon=False, loc="upper right", ncol=3, handlelength=1.8, columnspacing=1.0)

    ax_pressure.plot(x, graph, color="#dc2626", linewidth=1.2, label="insert graph traversal")
    ax_pressure.set_ylabel("writer graph\noccupancy")
    ax_pressure_cache = ax_pressure.twinx()
    ax_pressure_cache.plot(x, l2, color="#059669", linewidth=0.9, label="L2 miss rate")
    ax_pressure_cache.plot(x, llc, color="#0891b2", linewidth=0.9, label="LLC load miss rate")
    ax_pressure_cache.plot(x, dtlb, color="#9333ea", linewidth=0.9, label="DTLB walk rate")
    ax_pressure_cache.set_ylabel("cache/TLB rate\n(normalized)")
    ax_pressure_cache.set_ylim(0, 1.3)
    lines, labels = ax_pressure.get_legend_handles_labels()
    lines2, labels2 = ax_pressure_cache.get_legend_handles_labels()
    ax_pressure.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right", ncol=2, handlelength=1.7, columnspacing=1.0)

    ax_latency.plot(op_x, op_max, color="#111827", linewidth=0.9, alpha=0.78, linestyle="--", label="execution overhead (max, clipped)")
    ax_latency.plot(op_x, op_mean, color="#2563eb", linewidth=1.35, label="execution overhead (mean)")
    ax_latency.set_ylabel("matched read-latency\noverhead (ms)")
    ax_latency.set_xlabel("elapsed time (s)")
    ax_latency.legend(frameon=False, loc="upper right", ncol=2, handlelength=1.8, columnspacing=1.0)
    ax_latency.set_ylim(-8, args.latency_ymax_ms)

    for label, ax in zip(["(a)", "(b)", "(c)"], timeline_axes):
        ax.text(0.008, 0.86, label, transform=ax.transAxes, ha="left", va="top", color="#374151")
    if horizon > 0:
        ax_latency.set_xlim(0, horizon)
    else:
        ax_latency.set_xlim(min(x), max(x))

    groups = [("release", "release"), ("release_plus_drain", "release+drain"), ("quiet", "quiet")]
    xs = list(range(len(groups)))
    means = [repeat[key]["execution_overhead_mean_ms"] for key, _ in groups]
    p95s = [repeat[key]["execution_overhead_p95_ms"] for key, _ in groups]
    width = 0.34
    ax_bar.bar([i - width / 2 for i in xs], means, width=width, color="#2563eb", label="mean")
    ax_bar.bar([i + width / 2 for i in xs], p95s, width=width, color="#111827", label="p95")
    ax_bar.set_title("3-run aggregate", loc="left", pad=8)
    ax_bar.set_ylabel("execution overhead (ms)")
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels([label for _, label in groups], rotation=22, ha="right")
    ax_bar.legend(frameon=False, loc="upper right")
    ax_bar.text(0.02, 0.96, "(d)", transform=ax_bar.transAxes, ha="left", va="top", color="#374151")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240)
    print(f"[m3-problem-figure] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
