#!/usr/bin/env python3
"""Plot an Odin-style timeline: reader latency vs writer graph-search period."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def to_int(value: object) -> int:
    try:
        return int(float(value or 0.0))
    except (TypeError, ValueError):
        return 0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * pct)
    return ordered[idx]


def load_period_bins(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "start_ns": to_int(row["bin_start_ns"]),
                    "end_ns": to_int(row["bin_end_ns"]),
                    "graph_workers": to_float(row["insert_path_search_active_workers"]),
                    "repair_workers": to_float(row.get("existing_neighbor_update_active_workers")),
                    "prune_workers": to_float(row.get("existing_neighbor_prune_active_workers")),
                    "rewrite_workers": to_float(row.get("existing_neighbor_rewrite_active_workers")),
                }
            )
    return rows


def average_workers(bins: list[dict[str, float]], start_ns: int, end_ns: int, key: str) -> float:
    duration = max(1, end_ns - start_ns)
    weighted = 0.0
    for row in bins:
        if row["end_ns"] <= start_ns:
            continue
        if row["start_ns"] >= end_ns:
            break
        overlap = max(0, min(end_ns, row["end_ns"]) - max(start_ns, row["start_ns"]))
        weighted += overlap * row[key]
    return weighted / duration


def load_search_batches(path: Path, bins: list[dict[str, float]]) -> list[dict[str, float]]:
    batches: list[dict[str, float]] = []
    seen: set[tuple[int, int, int]] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            create_ns = to_int(row["create_raw_ns"])
            start_ns = to_int(row["start_raw_ns"])
            finish_ns = to_int(row["finish_raw_ns"])
            key = (create_ns, start_ns, finish_ns)
            if key in seen:
                continue
            seen.add(key)
            batches.append(
                {
                    "idx": float(len(batches)),
                    "e2e_ms": (finish_ns - create_ns) / 1e6,
                    "op_ms": (finish_ns - start_ns) / 1e6,
                    "start_delay_ms": (start_ns - create_ns) / 1e6,
                    "graph_workers": average_workers(bins, start_ns, finish_ns, "graph_workers"),
                    "repair_workers": average_workers(bins, start_ns, finish_ns, "repair_workers"),
                    "prune_workers": average_workers(bins, start_ns, finish_ns, "prune_workers"),
                    "rewrite_workers": average_workers(bins, start_ns, finish_ns, "rewrite_workers"),
                }
            )
    return batches


def active_ranges(active: list[bool]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(active):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            ranges.append((start, idx - 1))
            start = None
    if start is not None:
        ranges.append((start, len(active) - 1))
    return ranges


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing raw_latency/search_wide.csv and periods.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--active-threshold", type=float, default=1.0)
    parser.add_argument("--title", default="Reader latency spikes when search overlaps writer graph search")
    args = parser.parse_args()

    search_path = args.run_dir / "raw_latency" / "search_wide.csv"
    period_path = args.run_dir / "periods.csv"
    if not search_path.exists():
        raise FileNotFoundError(search_path)
    if not period_path.exists():
        raise FileNotFoundError(period_path)

    bins = load_period_bins(period_path)
    batches = load_search_batches(search_path, bins)
    active = [b["graph_workers"] > args.active_threshold for b in batches]
    active_batches = [b for b, flag in zip(batches, active) if flag]
    inactive_batches = [b for b, flag in zip(batches, active) if not flag]

    x = [b["idx"] for b in batches]
    e2e = [b["e2e_ms"] for b in batches]
    op = [b["op_ms"] for b in batches]
    graph = [b["graph_workers"] for b in batches]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.8, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0, 0.8]},
    )
    fig.patch.set_facecolor("white")

    for ax in axes:
        for start, end in active_ranges(active):
            ax.axvspan(start - 0.5, end + 0.5, color="#fee2e2", alpha=0.75, lw=0)
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)

    colors = ["#c2410c" if flag else "#64748b" for flag in active]
    axes[0].scatter(x, e2e, c=colors, s=14, linewidth=0, alpha=0.9)
    axes[0].plot(x, e2e, color="#94a3b8", linewidth=0.8, alpha=0.45)
    axes[0].set_ylabel("search e2e\nlatency (ms)")
    axes[0].set_title(args.title, loc="left", fontsize=14)

    axes[1].scatter(x, op, c=colors, s=14, linewidth=0, alpha=0.9)
    axes[1].plot(x, op, color="#94a3b8", linewidth=0.8, alpha=0.45)
    axes[1].set_ylabel("search op\nlatency (ms)")

    axes[2].bar(x, graph, color="#dc2626", width=0.85, alpha=0.82, label="writer graph-search overlap")
    axes[2].set_ylabel("active writer\nworkers")
    axes[2].set_xlabel("search batch")

    active_e2e = [b["e2e_ms"] for b in active_batches]
    inactive_e2e = [b["e2e_ms"] for b in inactive_batches]
    active_op = [b["op_ms"] for b in active_batches]
    inactive_op = [b["op_ms"] for b in inactive_batches]
    summary = (
        f"writer graph-search overlap: n={len(active_batches)}, mean e2e={mean(active_e2e):.1f} ms, "
        f"mean op={mean(active_op):.1f} ms\n"
        f"no writer graph-search overlap: n={len(inactive_batches)}, mean e2e={mean(inactive_e2e):.1f} ms, "
        f"mean op={mean(inactive_op):.1f} ms"
    )
    axes[0].text(
        0.012,
        0.96,
        summary,
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92, "boxstyle": "round,pad=0.35"},
    )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#c2410c", markersize=7, label="writer graph-search overlap"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#64748b", markersize=7, label="no writer graph-search overlap"),
    ]
    axes[0].legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=8)

    max_e2e = max(e2e) if e2e else 0.0
    max_op = max(op) if op else 0.0
    axes[0].set_ylim(0, max(max_e2e * 1.08, percentile(e2e, 0.95) * 1.3, 1.0))
    axes[1].set_ylim(0, max(max_op * 1.08, percentile(op, 0.95) * 1.3, 1.0))
    axes[2].set_ylim(0, max(max(graph) * 1.2 if graph else 0, 1.0))

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    print(f"[m3-writer-period-timeline] wrote {args.out}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
