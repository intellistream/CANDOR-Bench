#!/usr/bin/env python3
"""Plot the read/write coherence evidence for the M3 hardware pre-experiment."""

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
RFO_MISS = "l2_rqsts.rfo_miss_per_s"
RFO_WAIT = "offcore_requests_outstanding.cycles_with_demand_rfo_per_s"
READ_HITM = "ocr.demand_data_rd.l3_hit.snoop_hitm_per_s"
RFO_HITM = "ocr.demand_rfo.l3_hit.snoop_hitm_per_s"
LLC_LOAD = "LLC-load-misses_per_s"


def write_mask(frame: pd.DataFrame, time_col: str = "time_ms", edge_ms: float = 0.0) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for start, end in WRITE_PERIODS:
        mask |= (frame[time_col] >= start + edge_ms) & (frame[time_col] < end - edge_ms)
    return mask


def boundary_mask(frame: pd.DataFrame, time_col: str = "time_ms", edge_ms: float = 100.0) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for start, end in WRITE_PERIODS:
        mask |= ((frame[time_col] >= start - edge_ms) & (frame[time_col] <= start + edge_ms)) | (
            (frame[time_col] >= end - edge_ms) & (frame[time_col] <= end + edge_ms)
        )
    return mask


def shade_write(ax: plt.Axes) -> None:
    for start, end in WRITE_PERIODS:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#fee2e2", alpha=0.72, lw=0, zorder=0)


def read_periods(max_ms: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in WRITE_PERIODS:
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < max_ms:
        out.append((cursor, max_ms))
    return out


def draw_lanes(ax: plt.Axes, max_ms: float) -> None:
    measured = "#26313f"
    read = "#d7dee8"
    write = "#e45757"
    ax.set_ylim(0.0, 2.0)
    ax.set_xlim(0.0, max_ms / 1000.0)
    ax.set_yticks([0.42, 1.24])
    ax.set_yticklabels(["measured reads (8)", "competing workers (40)"])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(False)

    ax.broken_barh([(0.0, max_ms / 1000.0)], (0.24, 0.36), facecolors=measured, edgecolors="none")
    for start, end in read_periods(max_ms):
        ax.broken_barh([(start / 1000.0, (end - start) / 1000.0)], (1.06, 0.36), facecolors=read, edgecolors="none")
    for start, end in WRITE_PERIODS:
        ax.broken_barh([(start / 1000.0, (end - start) / 1000.0)], (1.06, 0.36), facecolors=write, edgecolors="none")

    ax.legend(
        handles=[
            Patch(facecolor=measured, edgecolor="none", label="measured read"),
            Patch(facecolor=read, edgecolor="none", label="competing read"),
            Patch(facecolor=write, edgecolor="none", label="competing write"),
        ],
        loc="upper right",
        ncol=3,
        frameon=False,
        handlelength=1.25,
        columnspacing=1.0,
        borderpad=0.0,
        bbox_to_anchor=(1.0, 1.10),
    )


def medians(frame: pd.DataFrame) -> dict[str, float]:
    steady = frame[~boundary_mask(frame)].copy()
    wmask = write_mask(steady, edge_ms=100.0)
    read = steady[~wmask]
    write = steady[wmask]
    keys = ["p99_us", RFO_MISS, RFO_WAIT, READ_HITM, RFO_HITM, LLC_LOAD]
    out: dict[str, float] = {}
    for key in keys:
        if key not in frame.columns:
            continue
        out[f"read_{key}"] = float(read[key].median())
        out[f"write_{key}"] = float(write[key].median())
        out[f"ratio_{key}"] = out[f"write_{key}"] / out[f"read_{key}"] if out[f"read_{key}"] else np.nan
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--max-ms", type=float, default=12000.0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    joined = pd.read_csv(args.run / "coherence_joined.csv")
    joined = joined[(joined["time_ms"] >= 0.0) & (joined["time_ms"] <= args.max_ms)].copy()
    stats = medians(joined)
    joined.to_csv(args.out.with_suffix(".csv"), index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.2,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )

    blue = "#2563eb"
    red = "#d73030"
    purple = "#7c3aed"
    orange = "#ea580c"
    gray = "#64748b"
    text = "#334155"

    fig = plt.figure(figsize=(11.8, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[0.46, 1.0, 1.0, 1.0], width_ratios=[1.76, 0.72])
    ax_lanes = fig.add_subplot(gs[0, :])
    ax_latency = fig.add_subplot(gs[1, 0], sharex=ax_lanes)
    ax_write = fig.add_subplot(gs[2, 0], sharex=ax_lanes)
    ax_read = fig.add_subplot(gs[3, 0], sharex=ax_lanes)
    ax_bar = fig.add_subplot(gs[1:, 1])

    draw_lanes(ax_lanes, args.max_ms)
    for ax in [ax_latency, ax_write, ax_read]:
        shade_write(ax)
        ax.grid(True, axis="y", color="#e8edf3", linewidth=0.75)
        ax.set_xlim(0.0, args.max_ms / 1000.0)
        ax.tick_params(axis="both", length=3, color="#94a3b8")

    x = joined["time_ms"] / 1000.0
    ax_latency.plot(x, joined["p99_us"], color=blue, lw=2.0)
    ax_latency.set_ylabel("read p99 (us)")
    ax_latency.text(
        0.020, 0.82,
        f"{stats['read_p99_us']:.2f}us -> {stats['write_p99_us']:.2f}us  ({stats['ratio_p99_us']:.2f}x)",
        transform=ax_latency.transAxes, ha="left", va="center", color=text, fontsize=8.9,
        bbox=dict(facecolor="white", edgecolor="#dbe3ec", boxstyle="round,pad=0.22", alpha=0.94),
    )
    plt.setp(ax_latency.get_xticklabels(), visible=False)

    ax_write.plot(x, joined[RFO_WAIT], color=red, lw=1.85, label="RFO outstanding cycles/s")
    ax_write.plot(x, joined[RFO_HITM], color=orange, lw=1.65, label="RFO HITM snoops/s")
    ax_write.set_yscale("log")
    ax_write.set_ylabel("write-side\nownership")
    ax_write.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.7, columnspacing=0.8)
    ax_write.text(
        0.020, 0.82,
        f"RFO wait {stats['ratio_' + RFO_WAIT]:.0f}x; RFO HITM {stats['ratio_' + RFO_HITM]:.0f}x",
        transform=ax_write.transAxes, ha="left", va="center", color=text, fontsize=8.7,
        bbox=dict(facecolor="white", edgecolor="#dbe3ec", boxstyle="round,pad=0.22", alpha=0.94),
    )
    plt.setp(ax_write.get_xticklabels(), visible=False)

    ax_read.plot(x, joined[READ_HITM], color=purple, lw=1.9, label="read HITM snoops/s")
    ax_read.plot(x, joined[LLC_LOAD], color=gray, lw=1.45, ls="--", label="LLC load misses/s")
    ax_read.set_yscale("log")
    ax_read.set_ylabel("read-side\nsignals")
    ax_read.set_xlabel("Elapsed time (s)")
    ax_read.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.7, columnspacing=0.8)
    ax_read.text(
        0.020, 0.82,
        f"read HITM {stats['ratio_' + READ_HITM]:.0f}x; LLC load miss {stats['ratio_' + LLC_LOAD]:.2f}x",
        transform=ax_read.transAxes, ha="left", va="center", color=text, fontsize=8.7,
        bbox=dict(facecolor="white", edgecolor="#dbe3ec", boxstyle="round,pad=0.22", alpha=0.94),
    )

    labels = ["read\np99", "RFO\nmiss", "RFO\nwait", "read\nHITM", "LLC\nload miss"]
    values = [
        stats["ratio_p99_us"],
        stats["ratio_" + RFO_MISS],
        stats["ratio_" + RFO_WAIT],
        stats["ratio_" + READ_HITM],
        stats["ratio_" + LLC_LOAD],
    ]
    colors = [blue, red, red, purple, gray]
    ax_bar.bar(labels, values, color=colors, width=0.62)
    ax_bar.axhline(1.0, color="#94a3b8", lw=0.9)
    ax_bar.set_yscale("log")
    ax_bar.set_ylim(0.06, max(values) * 2.7)
    ax_bar.grid(True, axis="y", color="#e8edf3", linewidth=0.75)
    ax_bar.set_ylabel("write/read")
    for idx, value in enumerate(values):
        if value >= 100:
            label = f"{value:.0f}x"
        elif value >= 10:
            label = f"{value:.1f}x"
        else:
            label = f"{value:.2f}x"
        ax_bar.text(idx, value * (1.18 if value >= 1 else 1.35), label,
                    ha="center", va="bottom", color=text, fontsize=8.6)

    ax_lanes.text(-0.055, 1.04, "(a)", transform=ax_lanes.transAxes, ha="left", va="bottom", fontweight="bold")
    ax_latency.text(-0.055, 1.03, "(b)", transform=ax_latency.transAxes, ha="left", va="bottom", fontweight="bold")
    ax_write.text(-0.055, 1.03, "(c)", transform=ax_write.transAxes, ha="left", va="bottom", fontweight="bold")
    ax_read.text(-0.055, 1.03, "(d)", transform=ax_read.transAxes, ha="left", va="bottom", fontweight="bold")
    ax_bar.text(-0.12, 1.03, "(e)", transform=ax_bar.transAxes, ha="left", va="bottom", fontweight="bold")

    fig.align_ylabels([ax_latency, ax_write, ax_read])
    fig.savefig(args.out, dpi=420, bbox_inches="tight")
    if args.pdf:
        fig.savefig(args.pdf, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
