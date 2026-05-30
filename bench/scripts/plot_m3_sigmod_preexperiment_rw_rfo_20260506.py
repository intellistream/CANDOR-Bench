#!/usr/bin/env python3
"""Paper-facing M3 pre-experiment figure: write periods hurt read tail via RFO."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


WRITE_PERIODS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]
RFO_COL = "l2_rqsts.rfo_miss_per_s"


def write_mask(frame: pd.DataFrame, time_col: str = "time_ms") -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for start, end in WRITE_PERIODS:
        mask |= (frame[time_col] >= start) & (frame[time_col] < end)
    return mask


def shade_write(ax: plt.Axes) -> None:
    for start, end in WRITE_PERIODS:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#fee2e2", alpha=0.72, lw=0, zorder=0)


def read_periods(max_ms: float) -> list[tuple[float, float]]:
    periods: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in WRITE_PERIODS:
        if start > cursor:
            periods.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < max_ms:
        periods.append((cursor, max_ms))
    return periods


def draw_worker_lanes(ax: plt.Axes, max_ms: float) -> None:
    measured = "#26313f"
    read = "#d7dee8"
    write = "#e45757"
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks([])
    ax.set_xlim(0.0, max_ms / 1000.0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    ax.grid(False)

    ax.broken_barh([(0.0, max_ms / 1000.0)], (0.24, 0.36), facecolors=measured, edgecolors="none")
    for start, end in read_periods(max_ms):
        ax.broken_barh([(start / 1000.0, (end - start) / 1000.0)], (1.06, 0.36), facecolors=read, edgecolors="none")
    for start, end in WRITE_PERIODS:
        ax.broken_barh([(start / 1000.0, (end - start) / 1000.0)], (1.06, 0.36), facecolors=write, edgecolors="none")

    ax.text(
        0.018,
        0.42,
        "8 measured reads",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        color="white",
        fontsize=8.8,
        fontweight="bold",
    )
    ax.text(
        0.018,
        1.24,
        "40 competing workers",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        color="#111827",
        fontsize=8.8,
        fontweight="bold",
    )

    ax.legend(
        handles=[
            Patch(facecolor=measured, edgecolor="none", label="measured read"),
            Patch(facecolor=read, edgecolor="none", label="competing read"),
            Patch(facecolor=write, edgecolor="none", label="competing write"),
        ],
        loc="center right",
        ncol=3,
        frameon=False,
        handlelength=1.25,
        columnspacing=1.0,
        borderpad=0.1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--max-ms", type=float, default=12000.0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    lat = pd.read_csv(args.run / "latency_binned.csv")
    perf = pd.read_csv(args.run / "perf_binned.csv")
    joined = pd.merge_asof(
        lat.sort_values("time_ms"),
        perf.sort_values("time_ms"),
        on="time_ms",
        direction="nearest",
        tolerance=30,
    ).dropna(subset=["p99_us", RFO_COL])
    joined = joined[(joined["time_ms"] >= 0.0) & (joined["time_ms"] <= args.max_ms)].copy()
    joined.to_csv(args.out.with_suffix(".csv"), index=False)

    mask = write_mask(joined)
    read = joined[~mask]
    write = joined[mask]
    read_p99 = float(read["p99_us"].median())
    write_p99 = float(write["p99_us"].median())
    read_rfo = float(read[RFO_COL].median())
    write_rfo = float(write[RFO_COL].median())
    tail_ratio = write_p99 / read_p99
    rfo_ratio = write_rfo / read_rfo

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.2,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )

    blue = "#2563eb"
    red = "#d73030"
    text = "#334155"

    fig = plt.figure(figsize=(11.2, 5.15), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.52, 1.0, 1.0], width_ratios=[1.70, 0.72])
    ax_lanes = fig.add_subplot(gs[0, 0])
    ax_latency = fig.add_subplot(gs[1, 0], sharex=ax_lanes)
    ax_rfo = fig.add_subplot(gs[2, 0], sharex=ax_lanes)
    ax_bar = fig.add_subplot(gs[1:, 1])

    draw_worker_lanes(ax_lanes, args.max_ms)

    for ax in [ax_latency, ax_rfo]:
        shade_write(ax)
        ax.grid(True, axis="y", color="#e8edf3", linewidth=0.75)
        ax.set_xlim(0.0, args.max_ms / 1000.0)
        ax.tick_params(axis="both", length=3, color="#94a3b8")

    x = joined["time_ms"] / 1000.0
    ax_latency.plot(x, joined["p99_us"], color=blue, lw=2.05)
    ax_latency.set_ylabel("read p99 (us)")
    ax_latency.text(
        0.020,
        0.82,
        f"{read_p99:.2f}us -> {write_p99:.2f}us  ({tail_ratio:.2f}x)",
        transform=ax_latency.transAxes,
        ha="left",
        va="center",
        color=text,
        fontsize=8.9,
        bbox=dict(facecolor="white", edgecolor="#dbe3ec", boxstyle="round,pad=0.22", alpha=0.94),
    )
    plt.setp(ax_latency.get_xticklabels(), visible=False)

    ax_rfo.plot(x, joined[RFO_COL], color=red, lw=1.95)
    ax_rfo.set_yscale("log")
    ax_rfo.set_ylabel("RFO misses/s")
    ax_rfo.set_xlabel("Elapsed time (s)")
    ax_rfo.text(
        0.020,
        0.82,
        f"{read_rfo:.2e}/s -> {write_rfo:.2e}/s  ({rfo_ratio:.0f}x)",
        transform=ax_rfo.transAxes,
        ha="left",
        va="center",
        color=text,
        fontsize=8.9,
        bbox=dict(facecolor="white", edgecolor="#dbe3ec", boxstyle="round,pad=0.22", alpha=0.94),
    )

    labels = ["read\np99", "RFO\nmisses"]
    values = [tail_ratio, rfo_ratio]
    colors = [blue, red]
    ax_bar.bar(labels, values, color=colors, width=0.58)
    ax_bar.axhline(1.0, color="#94a3b8", lw=0.9)
    ax_bar.set_yscale("log")
    ax_bar.set_ylim(0.75, max(values) * 2.6)
    ax_bar.grid(True, axis="y", color="#e8edf3", linewidth=0.75)
    ax_bar.set_ylabel("write/read")
    for idx, value in enumerate(values):
        label = f"{value:.2f}x" if value < 10 else f"{value:.0f}x"
        ax_bar.text(idx, value * 1.18, label, ha="center", va="bottom", color=text, fontsize=8.9)

    ax_lanes.text(-0.055, 1.04, "(a)", transform=ax_lanes.transAxes, ha="left", va="bottom", fontweight="bold")
    ax_latency.text(-0.055, 1.03, "(b)", transform=ax_latency.transAxes, ha="left", va="bottom", fontweight="bold")
    ax_rfo.text(-0.055, 1.03, "(c)", transform=ax_rfo.transAxes, ha="left", va="bottom", fontweight="bold")
    ax_bar.text(-0.12, 1.03, "(d)", transform=ax_bar.transAxes, ha="left", va="bottom", fontweight="bold")

    fig.align_ylabels([ax_latency, ax_rfo])
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    if args.pdf:
        fig.savefig(args.pdf, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
