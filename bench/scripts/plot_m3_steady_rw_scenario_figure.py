#!/usr/bin/env python3
"""Plot the steady read/write scenario figure for M3."""

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
    rows = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows.append({key: f(value) for key, value in row.items()})
    return rows


def load_config(path: Path) -> tuple[float, float, float, int, float]:
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    data = doc.get("data") or {}

    insert_rate = f(workload.get("insert_event_rate"))
    search_rate = f(workload.get("search_event_rate"))
    if insert_rate <= 0 or search_rate <= 0:
        groups = workload.get("rate_groups(r/w)") or workload.get("rate_groups") or []
        if groups:
            first = groups[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                search_rate = f(first[0])
                insert_rate = f(first[1])
            elif isinstance(first, dict):
                search_rate = f(first.get("read") or first.get("search") or first.get("search_event_rate"))
                insert_rate = f(first.get("write") or first.get("insert") or first.get("insert_event_rate"))

    insert_work = f(data.get("max_elements")) - f(data.get("begin_num"))
    release_horizon = insert_work / insert_rate if insert_rate > 0 and insert_work > 0 else 0.0
    batch_size = int(f(workload.get("batch_size")) or 1)
    return search_rate, insert_rate, release_horizon, batch_size, insert_work


def load_start_raw_ns(run_dir: Path) -> float:
    meta = run_dir / "raw_latency" / "meta.csv"
    if not meta.exists():
        return 0.0
    with meta.open(newline="") as file:
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


def normalize(values: list[float]) -> list[float]:
    positive = sorted(v for v in values if v > 0)
    if not positive:
        return [0.0 for _ in values]
    scale = positive[int((len(positive) - 1) * 0.95)]
    if scale <= 0:
        scale = max(positive)
    return [min(v / scale, 1.25) for v in values]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * pct)]


def active_ranges(x: list[float], active: list[bool], x_end: float) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    start: float | None = None
    step = x[1] - x[0] if len(x) > 1 else 0.1
    half = step / 2
    for t, flag in zip(x, active):
        if t > x_end:
            break
        if flag and start is None:
            start = max(0.0, t - half)
        elif not flag and start is not None:
            ranges.append((start, max(start, t - half)))
            start = None
    if start is not None:
        ranges.append((start, x_end))
    merged: list[tuple[float, float]] = []
    for start, end in ranges:
        if end - start < 0.25:
            continue
        if merged and start - merged[-1][1] <= 0.35:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-matched-batches", type=int, default=2)
    parser.add_argument("--latency-ymax-ms", type=float, default=180.0)
    parser.add_argument("--title", default="Steady read/write workload: continuous request release")
    args = parser.parse_args()

    rows = load_rows(args.timeline_csv)
    read_rate, insert_rate, release_horizon, batch_size, insert_work = load_config(args.config)
    start_raw_ns = load_start_raw_ns(args.full_run)
    insert_finishes = load_finish_times(args.full_run / "raw_latency" / "insert_wide.csv")
    search_finishes = load_finish_times(args.full_run / "raw_latency" / "search_wide.csv")
    read_work = len(search_finishes)

    x = [row["elapsed_s"] for row in rows]
    graph = [row["insert_graph_traversal_occupancy_worker_equiv"] for row in rows]
    l2 = normalize([row["l2_miss_rate"] for row in rows])
    llc = normalize([row["llc_load_miss_rate"] for row in rows])
    dtlb = normalize([row["dtlb_walk_rate"] for row in rows])

    lat_x: list[float] = []
    e2e_mean: list[float] = []
    for row in rows:
        if row["matched_search_batches"] < args.min_matched_batches:
            continue
        lat_x.append(row["elapsed_s"])
        e2e_mean.append(row["observed_delta_ms_mean"])
    x_end = max(lat_x) + 0.35 if lat_x else min(max(x), release_horizon)

    fig, axes = plt.subplots(3, 1, figsize=(10.8, 6.6), sharex=True, gridspec_kw={"height_ratios": [0.85, 1.05, 1.1]})
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.75)
        ax.set_axisbelow(True)

    pressure_threshold = percentile([g for t, g in zip(x, graph) if t <= x_end], 0.70)
    pressure_ranges = active_ranges(x, [g >= pressure_threshold for g in graph], x_end)
    for ax in axes:
        for idx, (start, end) in enumerate(pressure_ranges):
            ax.axvspan(start, end, color="#fee2e2", alpha=0.34, lw=0, label="high writer pressure" if ax is axes[0] and idx == 0 else None)

    insert_backlog_k = []
    read_backlog_k = []
    insert_idx = 0
    search_idx = 0
    for t in x:
        raw_cutoff = start_raw_ns + t * 1e9
        while insert_idx < len(insert_finishes) and insert_finishes[insert_idx] <= raw_cutoff:
            insert_idx += 1
        while search_idx < len(search_finishes) and search_finishes[search_idx] <= raw_cutoff:
            search_idx += 1
        insert_released = min(t * insert_rate, insert_work)
        insert_completed = min(insert_idx * batch_size, insert_work)
        read_released = min(t * read_rate, read_work)
        read_completed = min(search_idx, read_work)
        insert_backlog_k.append(max(0.0, insert_released - insert_completed) / 1000.0)
        read_backlog_k.append(max(0.0, read_released - read_completed) / 1000.0)

    axes[0].plot(x, insert_backlog_k, color="#dc2626", linewidth=1.65, label="insert backlog")
    axes[0].plot(x, read_backlog_k, color="#2563eb", linewidth=1.25, alpha=0.85, label="read backlog")
    axes[0].axvline(release_horizon, color="#6b7280", linestyle="--", linewidth=0.9, alpha=0.9, label="last scheduled release")
    axes[0].set_ylim(0, max(1.0, max(insert_backlog_k + read_backlog_k) * 1.12))
    axes[0].set_ylabel("backlog\n(k points)")
    axes[0].set_title(args.title, loc="left", pad=8)
    axes[0].legend(frameon=False, loc="upper left", ncol=4, handlelength=1.8, columnspacing=1.0)

    axes[1].plot(x, graph, color="#dc2626", linewidth=1.15, label="insert graph traversal")
    axes[1].axhline(pressure_threshold, color="#dc2626", linewidth=0.8, linestyle=":", alpha=0.75)
    axes[1].set_ylabel("writer graph\noccupancy")
    ax_cache = axes[1].twinx()
    ax_cache.plot(x, l2, color="#059669", linewidth=0.9, label="L2 miss rate")
    ax_cache.plot(x, llc, color="#0891b2", linewidth=0.9, label="LLC load miss rate")
    ax_cache.plot(x, dtlb, color="#9333ea", linewidth=0.9, label="DTLB walk rate")
    ax_cache.set_ylabel("cache/TLB rate\n(normalized)")
    ax_cache.set_ylim(0, 1.3)
    lines, labels = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax_cache.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, frameon=False, loc="upper right", ncol=2, handlelength=1.7, columnspacing=1.0)

    axes[2].plot(lat_x, e2e_mean, color="#2563eb", linewidth=1.35, label="e2e latency overhead (mean)")
    axes[2].axhline(0, color="#6b7280", linewidth=0.8, alpha=0.7)
    axes[2].set_ylabel("matched search e2e\nlatency overhead (ms)")
    axes[2].set_xlabel("elapsed time (s)")
    axes[2].set_ylim(-8, args.latency_ymax_ms)
    axes[2].legend(frameon=False, loc="upper right", handlelength=1.8, columnspacing=1.0)

    for ax in axes[1:]:
        ax.axvline(release_horizon, color="#6b7280", linestyle="--", linewidth=0.9, alpha=0.75)
    for label, ax in zip(["(a)", "(b)", "(c)"], axes):
        ax.text(0.008, 0.86, label, transform=ax.transAxes, ha="left", va="top", color="#374151")
    axes[-1].set_xlim(0, x_end)

    fig.tight_layout(h_pad=0.9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240)
    print(f"[m3-steady-scenario] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
