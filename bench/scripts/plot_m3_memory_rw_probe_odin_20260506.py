#!/usr/bin/env python3
"""OdinANN-style hardware pre-experiment figure for read/write memory interference."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PERIODS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]
LOAD_WAIT = "offcore_requests_outstanding.l3_miss_demand_data_rd_per_s"
RFO = "l2_rqsts.rfo_miss_per_s"


def period_mask(frame: pd.DataFrame, time_col: str = "time_ms") -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for start, end in PERIODS:
        mask |= (frame[time_col] >= start) & (frame[time_col] < end)
    return mask


def scale01(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo = float(vals.quantile(0.05))
    hi = float(vals.quantile(0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return vals * 0.0
    return ((vals - lo) / (hi - lo)).clip(0.0, 1.0)


def complement_periods(max_ms: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in PERIODS:
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < max_ms:
        out.append((cursor, max_ms))
    return out


def shade_write(ax: plt.Axes) -> None:
    for start, end in PERIODS:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#f6b6b6", alpha=0.30, lw=0)


def draw_layout(ax: plt.Axes, max_ms: float) -> None:
    measured_dark = "#1f2937"
    read_gray = "#cbd5e1"
    write_red = "#ef4444"
    ax.set_ylim(0, 2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)
    ax.grid(False)
    ax.broken_barh([(0, max_ms / 1000.0)], (0.24, 0.38), facecolors=measured_dark, edgecolors="none")
    for start, end in complement_periods(max_ms):
        ax.broken_barh([(start / 1000.0, (end - start) / 1000.0)], (1.07, 0.38), facecolors=read_gray, edgecolors="none")
    for start, end in PERIODS:
        ax.broken_barh([(start / 1000.0, (end - start) / 1000.0)], (1.07, 0.38), facecolors=write_red, edgecolors="none")
    ax.text(0.01, 0.43, "8 measured random-read workers", transform=ax.get_yaxis_transform(),
            ha="left", va="center", color="white", fontsize=9.3, fontweight="bold")
    ax.text(0.01, 1.26, "40 competing pull-based workers", transform=ax.get_yaxis_transform(),
            ha="left", va="center", color="#111827", fontsize=9.3, fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor=measured_dark, edgecolor="none", label="measured read"),
            Patch(facecolor=read_gray, edgecolor="none", label="competing read"),
            Patch(facecolor=write_red, edgecolor="none", label="competing write"),
        ],
        loc="upper right",
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.3,
    )


def metric_ratios(joined: pd.DataFrame) -> dict[str, float]:
    mask = period_mask(joined)
    read = joined[~mask]
    write = joined[mask]
    out = {
        "read_p99": float(read["p99_us"].median()),
        "write_p99": float(write["p99_us"].median()),
        "read_load": float(read[LOAD_WAIT].median()),
        "write_load": float(write[LOAD_WAIT].median()),
        "read_rfo": float(read[RFO].median()),
        "write_rfo": float(write[RFO].median()),
    }
    out["p99_ratio"] = out["write_p99"] / out["read_p99"]
    out["load_ratio"] = out["write_load"] / out["read_load"]
    out["rfo_ratio"] = out["write_rfo"] / out["read_rfo"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--max-ms", type=float, default=12000.0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    latency = pd.read_csv(args.run / "latency_binned.csv")
    perf = pd.read_csv(args.run / "perf_binned.csv")
    joined = pd.merge_asof(
        latency.sort_values("time_ms"),
        perf.sort_values("time_ms"),
        on="time_ms",
        direction="nearest",
        tolerance=30,
    ).dropna(subset=[LOAD_WAIT, RFO, "p99_us"])
    joined.to_csv(args.out.with_suffix(".binned.csv"), index=False)
    ratios = metric_ratios(joined)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.0,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )

    blue = "#2563eb"
    green = "#059669"
    red = "#dc2626"
    gray = "#64748b"
    text_gray = "#475569"

    fig = plt.figure(figsize=(12.6, 7.05), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.50, 1.05, 1.05], width_ratios=[1.42, 0.98])
    ax_layout = fig.add_subplot(gs[0, :])
    ax_lat = fig.add_subplot(gs[1, :], sharex=ax_layout)
    ax_hw = fig.add_subplot(gs[2, 0], sharex=ax_layout)
    ax_bar = fig.add_subplot(gs[2, 1])
    fig.patch.set_facecolor("white")

    draw_layout(ax_layout, args.max_ms)
    ax_layout.set_title("Thread roles in the hardware pre-experiment", loc="left", pad=2, fontweight="bold")

    for ax in [ax_lat, ax_hw]:
        shade_write(ax)
        ax.grid(True, axis="y", color="#e8edf3", linewidth=0.75)
        ax.set_xlim(0, args.max_ms / 1000.0)
        ax.tick_params(axis="both", length=3, color="#94a3b8")

    ax_lat.plot(joined["time_ms"] / 1000.0, joined["p99_us"], color=blue, lw=2.1, label="measured read p99")
    ax_lat.set_ylabel("batch p99 (us)")
    ax_lat.set_title("Measured read tail rises during competing writes", loc="left", pad=7, fontweight="bold")
    ax_lat.legend(loc="upper right", frameon=False)
    ax_lat.text(
        0.015,
        0.86,
        f"write/read p99 = {ratios['p99_ratio']:.2f}x  ({ratios['read_p99']:.2f}us -> {ratios['write_p99']:.2f}us)",
        transform=ax_lat.transAxes,
        ha="left",
        va="center",
        color=text_gray,
        fontsize=9.2,
        bbox=dict(facecolor="white", edgecolor="#dbe3ec", boxstyle="round,pad=0.25", alpha=0.88),
    )

    ax_hw.plot(joined["time_ms"] / 1000.0, scale01(joined[RFO]), color=red, lw=2.1, label="RFO miss")
    ax_hw.plot(
        joined["time_ms"] / 1000.0,
        scale01(joined[LOAD_WAIT]),
        color=gray,
        lw=1.7,
        ls="--",
        label="L3-miss load wait",
    )
    ax_hw.set_ylabel("Normalized")
    ax_hw.set_xlabel("Elapsed time (s)")
    ax_hw.set_title("Write periods expose cache-line ownership; load wait is not the main signal", loc="left", pad=7, fontweight="bold")
    ax_hw.legend(loc="upper right", frameon=False, ncol=2, handlelength=2.0, columnspacing=1.2)

    labels = ["read p99", "RFO miss", "load wait"]
    values = [ratios["p99_ratio"], ratios["rfo_ratio"], ratios["load_ratio"]]
    colors = [blue, red, gray]
    ax_bar.bar(labels, values, color=colors, width=0.62)
    ax_bar.axhline(1.0, color="#94a3b8", lw=0.9)
    ax_bar.set_yscale("log")
    ax_bar.set_ylabel("write period / read period")
    ax_bar.set_title("Period contrast", loc="left", pad=7, fontweight="bold")
    ax_bar.grid(True, axis="y", color="#e8edf3", linewidth=0.75)
    for idx, value in enumerate(values):
        ax_bar.text(idx, value * 1.10, f"{value:.1f}x", ha="center", va="bottom", fontsize=9.2, color=text_gray)

    fig.suptitle(
        "Hardware pre-experiment: 8 measured reads + 40 competing pull-based workers",
        x=0.01,
        ha="left",
        y=1.02,
        fontsize=13.5,
        fontweight="bold",
    )
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    if args.pdf:
        fig.savefig(args.pdf, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
