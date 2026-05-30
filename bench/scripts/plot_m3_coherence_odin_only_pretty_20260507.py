#!/usr/bin/env python3
"""Draw a readable Odin-style standalone M3 coherence pre-experiment figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


WRITE_PERIODS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]
RFO_OUTSTANDING = "offcore_requests_outstanding.cycles_with_demand_rfo_per_s"
RFO_HITM = "ocr.demand_rfo.l3_hit.snoop_hitm_per_s"
READ_HITM = "ocr.demand_data_rd.l3_hit.snoop_hitm_per_s"


def write_mask(frame: pd.DataFrame, *, time_col: str = "time_ms", edge_ms: float = 0.0) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for start, end in WRITE_PERIODS:
        mask |= (frame[time_col] >= start + edge_ms) & (frame[time_col] < end - edge_ms)
    return mask


def boundary_mask(frame: pd.DataFrame, *, time_col: str = "time_ms", edge_ms: float = 100.0) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for start, end in WRITE_PERIODS:
        mask |= ((frame[time_col] >= start - edge_ms) & (frame[time_col] <= start + edge_ms)) | (
            (frame[time_col] >= end - edge_ms) & (frame[time_col] <= end + edge_ms)
        )
    return mask


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


def medians(frame: pd.DataFrame) -> dict[str, float]:
    steady = frame[~boundary_mask(frame)].copy()
    wmask = write_mask(steady, edge_ms=100.0)
    read = steady[~wmask]
    write = steady[wmask]
    keys = ["p99_us", RFO_OUTSTANDING, RFO_HITM, READ_HITM]
    stats: dict[str, float] = {}
    for key in keys:
        stats[f"read_{key}"] = float(read[key].median())
        stats[f"write_{key}"] = float(write[key].median())
        stats[f"ratio_{key}"] = stats[f"write_{key}"] / stats[f"read_{key}"] if stats[f"read_{key}"] else np.nan
    return stats


def shade_write(ax: plt.Axes) -> None:
    for start, end in WRITE_PERIODS:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#f9dada", alpha=0.55, lw=0, zorder=0)


def draw_lanes(ax: plt.Axes, max_ms: float) -> None:
    measured = "#273241"
    read = "#e2e8f0"
    write = "#e45f59"
    text = "#334155"

    ax.set_xlim(0.0, max_ms / 1000.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
    ax.grid(False)

    max_s = max_ms / 1000.0
    lane_h = 0.055
    worker_ys = np.linspace(0.70, 0.88, 5)
    measured_ys = np.linspace(0.18, 0.34, 3)

    def rect_data(x0: float, y: float, width: float, height: float, color: str, alpha: float = 1.0) -> None:
        ax.add_patch(
            Rectangle(
                (x0, y),
                width,
                height,
                transform=ax.transData,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                clip_on=False,
            )
        )

    ax.text(
        -0.038,
        0.78,
        "Background",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color=text,
        fontsize=6.4,
        fontweight="normal",
    )
    ax.text(
        -0.038,
        0.26,
        "Measured\nreads",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color=text,
        fontsize=6.4,
        fontweight="normal",
        linespacing=1.0,
    )

    for row, y in enumerate(worker_ys):
        for start, end in read_periods(max_ms):
            x0 = start / 1000.0
            x1 = end / 1000.0
            rect_data(x0, y - lane_h / 2.0, x1 - x0, lane_h, read, alpha=0.70)
            step = 0.34 + 0.025 * (row % 3)
            xs = np.arange(x0 + 0.12 + row * 0.025, x1 - 0.06, step)
            if len(xs):
                ax.scatter(xs, np.full_like(xs, y), s=5.5, color="#64748b", linewidths=0, alpha=0.95, zorder=2)
        for start, end in WRITE_PERIODS:
            x0 = start / 1000.0
            x1 = end / 1000.0
            rect_data(x0, y - lane_h / 2.0, x1 - x0, lane_h, write, alpha=0.92)

    for start, end in WRITE_PERIODS:
        ax.text(
            (start + end) / 2000.0,
            0.955,
            "Insert",
            ha="center",
            va="center",
            color=write,
            fontsize=6.8,
            fontweight="normal",
        )

    for y in measured_ys:
        rect_data(0.0, y - lane_h / 2.0, max_s, lane_h, measured, alpha=0.98)
        xs = np.arange(0.18, max_s - 0.05, 0.42)
        ax.scatter(xs, np.full_like(xs, y), s=5.2, color="#f8fafc", linewidths=0, alpha=0.95, zorder=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--max-ms", type=float, default=12000.0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.run / "coherence_joined.csv")
    frame = frame[(frame["time_ms"] >= 0.0) & (frame["time_ms"] <= args.max_ms)].copy()
    frame.to_csv(args.out.with_suffix(".csv"), index=False)
    stats = medians(frame)

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.6,
            "axes.labelsize": 7.8,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "axes.linewidth": 0.72,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    blue = "#2f6fdd"
    red = "#d95f59"
    orange = "#d97706"
    purple = "#7048e8"
    text = "#334155"
    grid = "#E3E3E3"

    fig = plt.figure(figsize=(3.35, 2.58), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.35, 0.93, 1.10], hspace=0.16)
    ax_lanes = fig.add_subplot(gs[0, 0])
    ax_latency = fig.add_subplot(gs[1, 0], sharex=ax_lanes)
    ax_signal = fig.add_subplot(gs[2, 0], sharex=ax_lanes)

    draw_lanes(ax_lanes, args.max_ms)

    x = frame["time_ms"] / 1000.0
    for ax in (ax_latency, ax_signal):
        shade_write(ax)
        ax.set_xlim(0.0, args.max_ms / 1000.0)
        ax.grid(True, axis="y", color=grid, linewidth=0.45)
        ax.tick_params(axis="both", length=2.4, width=0.6, color="#303030")
        ax.spines["left"].set_linewidth(0.72)
        ax.spines["bottom"].set_linewidth(0.72)

    ax_latency.plot(x, frame["p99_us"], color=blue, lw=1.45, solid_capstyle="round")
    ax_latency.set_ylabel("Read P99\n(us)")
    ax_latency.set_ylim(0.0, max(36.0, float(frame["p99_us"].max()) * 1.08))
    ax_latency.set_yticks([0, 10, 20, 30])
    ax_latency.tick_params(axis="x", labelbottom=False)
    ax_latency.text(
        0.018,
        0.86,
        f"P99 {stats['read_p99_us']:.2f}->{stats['write_p99_us']:.2f} us",
        transform=ax_latency.transAxes,
        ha="left",
        va="center",
        color=text,
        fontsize=6.7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.8),
    )

    ax_signal.plot(x, frame[RFO_OUTSTANDING], color=red, lw=1.35, label="RFO outstanding")
    ax_signal.plot(x, frame[RFO_HITM], color=orange, lw=1.15, label="RFO HITM")
    ax_signal.plot(x, frame[READ_HITM], color=purple, lw=1.15, label="Read HITM")
    ax_signal.set_yscale("log")
    ax_signal.set_ylabel("Hardware events/s\n(log)")
    ax_signal.set_xlabel("Elapsed Time (s)", labelpad=2)
    ax_signal.set_ylim(2.5e4, 3.0e11)
    ax_signal.set_xticks(np.arange(0.0, args.max_ms / 1000.0 + 0.1, 2.0))
    ax_signal.legend(
        loc="upper right",
        frameon=False,
        ncol=3,
        handlelength=1.6,
        columnspacing=0.55,
        bbox_to_anchor=(1.0, 1.10),
    )

    fig.subplots_adjust(left=0.205, right=0.992, top=0.975, bottom=0.145)
    fig.savefig(args.out, dpi=260, bbox_inches="tight", pad_inches=0.015)
    if args.pdf:
        fig.savefig(args.pdf, bbox_inches="tight", pad_inches=0.015)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
