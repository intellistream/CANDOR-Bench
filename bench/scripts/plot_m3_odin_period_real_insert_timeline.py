#!/usr/bin/env python3
"""OdinANN-style M3 period timeline for real full insert only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bool_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def segments_from_mask(frame: pd.DataFrame, mask_col: str) -> list[tuple[float, float]]:
    mask = bool_mask(frame[mask_col]).to_numpy()
    bins = frame["bin_ms"].to_numpy(dtype=float)
    if len(bins) == 0:
        return []
    step = float(np.median(np.diff(bins))) if len(bins) > 1 else 50.0
    segments: list[tuple[float, float]] = []
    start: float | None = None
    last = bins[0]
    for t, active in zip(bins, mask, strict=False):
        if active and start is None:
            start = float(t)
        if not active and start is not None:
            segments.append((start / 1000.0, (float(t) - start) / 1000.0))
            start = None
        last = float(t)
    if start is not None:
        segments.append((start / 1000.0, (last + step - start) / 1000.0))
    return segments


def outside_segments(insert_segments: list[tuple[float, float]], horizon_s: float) -> list[tuple[float, float]]:
    outside: list[tuple[float, float]] = []
    cursor = 0.0
    for start, width in insert_segments:
        if start > cursor:
            outside.append((cursor, start - cursor))
        cursor = max(cursor, start + width)
    if cursor < horizon_s:
        outside.append((cursor, horizon_s - cursor))
    return outside


def summary_value(summary: pd.DataFrame, phase: str, col: str) -> float:
    row = summary[summary["phase"] == phase]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][col])


def add_period_shading(ax: plt.Axes, outside: list[tuple[float, float]], insert: list[tuple[float, float]]) -> None:
    for start, width in outside:
        ax.axvspan(start, start + width, color="#f3f4f6", alpha=0.55, lw=0)
    for start, width in insert:
        ax.axvspan(start, start + width, color="#fee2e2", alpha=0.80, lw=0)


def period_step(
    insert_segments: list[tuple[float, float]],
    horizon_s: float,
    outside_value: float,
    insert_value: float,
) -> tuple[list[float], list[float]]:
    boundaries = [0.0, horizon_s]
    for start, width in insert_segments:
        boundaries.extend([start, start + width])
    boundaries = sorted(set(boundaries))
    xs: list[float] = []
    ys: list[float] = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=False):
        mid = (left + right) / 2.0
        in_insert = any(start <= mid < start + width for start, width in insert_segments)
        value = insert_value if in_insert else outside_value
        xs.extend([left, right])
        ys.extend([value, value])
    return xs, ys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    timeline = pd.read_csv(args.timeline)
    timeline["elapsed_s"] = timeline["bin_ms"] / 1000.0
    summary = pd.read_csv(args.summary)
    evidence = pd.read_csv(args.evidence)

    horizon_s = float((timeline["bin_ms"].max() + 50.0) / 1000.0)
    insert_segments = segments_from_mask(timeline, "in_insert_issue_window")
    outside = outside_segments(insert_segments, horizon_s)

    outside_p95 = summary_value(summary, "outside", "lowq_op_p95_ms")
    outside_p99 = summary_value(summary, "outside", "lowq_op_p99_ms")
    insert_p95 = summary_value(summary, "insert_active", "lowq_op_p95_ms")
    insert_p99 = summary_value(summary, "insert_active", "lowq_op_p99_ms")
    queue_p95 = summary_value(summary, "insert_active", "lowq_queue_p95_ms")

    evidence = evidence.copy()
    evidence["source"] = "odin_period_real_insert_timeline"
    evidence["latency_p99_insert_over_outside"] = insert_p99 / outside_p99
    evidence["latency_p95_insert_over_outside"] = insert_p95 / outside_p95
    evidence["queue_p95_ms"] = queue_p95
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(args.csv_out, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.8, 6.7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 0.76, 1.65]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "M3: real graph-mutating inserts raise measured-search tail under a fixed workload envelope",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )

    for ax in axes:
        add_period_shading(ax, outside, insert_segments)
        ax.set_xlim(0, horizon_s)
        ax.grid(axis="y", color="#e5e7eb", lw=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axes[0]
    x99, y99 = period_step(insert_segments, horizon_s, outside_p99, insert_p99)
    x95, y95 = period_step(insert_segments, horizon_s, outside_p95, insert_p95)
    ax.plot(x99, y99, color="#dc2626", lw=2.7, label="measured-search p99")
    ax.plot(x95, y95, color="#991b1b", lw=1.7, ls="--", label="measured-search p95")
    ax.plot(
        timeline["elapsed_s"],
        timeline["queue_p95_ms"],
        color="#64748b",
        lw=1.1,
        alpha=0.85,
        label="queue p95",
    )
    ax.set_title("(a) measured-search latency rises only during real-insert periods; queue stays flat", loc="left")
    ax.set_ylabel("op latency (ms)")
    ax.set_ylim(0, max(insert_p99, outside_p99) * 1.22)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.text(
        0.985,
        0.88,
        f"insert p99 / bg-search p99 = {insert_p99 / outside_p99:.2f}x",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#7f1d1d",
    )

    ax = axes[1]
    ax.broken_barh(outside, (8, 8), facecolors="#94a3b8", alpha=0.78, label="8 background-search slots")
    ax.broken_barh(insert_segments, (8, 8), facecolors="#ef4444", alpha=0.72, label="8 real-insert slots")
    ax.broken_barh([(0.0, horizon_s)], (0, 8), facecolors="#111827", alpha=0.84, label="8 measured-search slots")
    ax.set_title("(b) OdinANN-style period substitution: measured search is fixed; competing work changes", loc="left")
    ax.set_ylabel("worker slots")
    ax.set_ylim(0, 16)
    ax.set_yticks([4, 12], ["measured", "competing"])
    ax.legend(frameon=False, ncol=3, loc="upper left")

    ax = axes[2]
    x = timeline["elapsed_s"].to_numpy()
    phase_parts = [
        ("insert_path_search_active_workers", "#9ca3af", "insert traversal"),
        ("existing_neighbor_load_scan_active_workers", "#60a5fa", "load/scan"),
        ("existing_neighbor_prune_active_workers", "#f97316", "prune"),
        ("existing_neighbor_undo_record_active_workers", "#a855f7", "undo record"),
        ("existing_neighbor_rewrite_active_workers", "#dc2626", "rewrite/writeback"),
    ]
    bottom = np.zeros(len(timeline))
    for col, color, label in phase_parts:
        vals = timeline[col].fillna(0.0).to_numpy()
        ax.fill_between(x, bottom, bottom + vals, color=color, alpha=0.78, lw=0, label=label)
        bottom += vals
    ax.set_title("(c) real insert phase density aligns with the red periods; hardware counters are context, not the proof", loc="left")
    ax.set_ylabel("worker-equiv")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("elapsed time (s); red bands are real graph-mutating insert periods")

    ax_llc = ax.twinx()
    ax_llc.plot(
        x,
        timeline["perf_llc_load_misses_per_s"].fillna(0.0).to_numpy() / 1e6,
        color="#111827",
        lw=1.1,
        alpha=0.72,
        label="LLC load misses/s",
    )
    ax_llc.set_ylabel("LLC misses/s (M)")
    ax_llc.spines["top"].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_llc.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, ncol=3, loc="upper left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=260, bbox_inches="tight")
    print(args.out)
    print(args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
