#!/usr/bin/env python3
"""Plot lightweight rwlock-probe evidence for M3 read tail spikes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


WINDOWS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]


def shade_insert(ax: plt.Axes, windows: list[tuple[float, float]]) -> None:
    for start, end in windows:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#f4d8d7", alpha=0.58, lw=0, zorder=0)


def draw_lanes(ax: plt.Axes, horizon_s: float, windows: list[tuple[float, float]]) -> None:
    navy = "#263445"
    red = "#d95f59"
    pale = "#edf2f7"
    dot = "#6f7f91"
    text = "#253244"

    ax.set_xlim(0.0, horizon_s)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    lane_h = 0.27
    bg_y = 0.62
    rd_y = 0.18

    ax.text(-0.024, bg_y + lane_h / 2, "Background\nSearch", transform=ax.transAxes,
            ha="right", va="center", color=text, fontsize=7.2, linespacing=0.95)
    ax.text(-0.024, rd_y + lane_h / 2, "Measured\nReads", transform=ax.transAxes,
            ha="right", va="center", color=text, fontsize=7.2, linespacing=0.95)

    cursor = 0.0
    for start_ms, end_ms in windows + [(horizon_s * 1000.0, horizon_s * 1000.0)]:
        start_s = start_ms / 1000.0
        if start_s > cursor:
            ax.add_patch(Rectangle((cursor, bg_y), start_s - cursor, lane_h, facecolor=pale, edgecolor="none"))
            xs = np.arange(cursor + 0.12, start_s - 0.04, 0.32)
            if len(xs):
                ys = bg_y + lane_h * (0.40 + 0.24 * (np.arange(len(xs)) % 2))
                ax.scatter(xs, ys, s=8, color=dot, linewidths=0)
        if end_ms > start_ms:
            end_s = end_ms / 1000.0
            ax.add_patch(Rectangle((start_s, bg_y), end_s - start_s, lane_h, facecolor=red, edgecolor="none"))
            ax.text((start_s + end_s) / 2, bg_y + lane_h / 2, "Insert",
                    ha="center", va="center", color="white", fontsize=7.6)
            cursor = end_s

    ax.add_patch(Rectangle((0.0, rd_y), horizon_s, lane_h, facecolor=navy, edgecolor="none"))
    xs = np.arange(0.18, horizon_s - 0.03, 0.38)
    for j, frac in enumerate([0.32, 0.50, 0.68]):
        shifted = xs + j * 0.035
        shifted = shifted[shifted < horizon_s]
        ax.scatter(shifted, np.full_like(shifted, rd_y + lane_h * frac), s=5.0, color="#f8fafc", linewidths=0)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, frame in [("Insert-active", df[df["insert_period"]]), ("Outside", df[~df["insert_period"]])]:
        rows.append(
            {
                "phase": phase,
                "read_p99_max_ms": frame["op_p99"].max(),
                "rdlock_wait_max_ms": frame["rdlock_wait_max_ms"].max(),
                "rdlock_wait_sum_ms": frame["rdlock_wait_sum_ms"].sum(),
                "rdlock_slow_count": int(frame["rdlock_slow_count"].sum()),
            }
        )
    insert = df[df["insert_period"]].copy()
    rows.append(
        {
            "phase": "Insert-active correlation",
            "read_p99_max_ms": "",
            "rdlock_wait_max_ms": insert["rdlock_wait_max_ms"].corr(insert["op_p99"], method="pearson"),
            "rdlock_wait_sum_ms": insert["rdlock_wait_sum_ms"].corr(insert["op_p99"], method="pearson"),
            "rdlock_slow_count": "",
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[(df["start_ms"] >= 0.0) & (df["start_ms"] < 12000.0) & (df["n"] > 0)].copy()
    df["elapsed_s"] = (df["start_ms"] + 25.0) / 1000.0
    df["insert_period"] = df["insert_period"].astype(bool)

    out_summary = args.summary or args.out.with_name(args.out.stem + "_summary.csv")
    out_png = args.png or args.out.with_suffix(".png")
    for path in (args.out, out_png, out_summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    summarize(df).to_csv(out_summary, index=False)

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 7.7,
            "axes.labelsize": 8.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.72,
        }
    )

    blue = "#2f6fdd"
    red = "#d95f59"
    dark = "#263445"
    gray = "#7a8493"
    grid = "#e6e9ef"

    fig = plt.figure(figsize=(6.25, 2.95), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.48, 1.15, 0.92],
        width_ratios=[2.65, 1.0],
        hspace=0.17,
        wspace=0.34,
    )
    ax_lane = fig.add_subplot(gs[0, 0])
    ax_lat = fig.add_subplot(gs[1, 0], sharex=ax_lane)
    ax_wait = fig.add_subplot(gs[2, 0], sharex=ax_lane)
    ax_scatter = fig.add_subplot(gs[1:, 1])

    horizon_s = 12.0
    draw_lanes(ax_lane, horizon_s, WINDOWS)
    for ax in (ax_lat, ax_wait):
        shade_insert(ax, WINDOWS)
        ax.grid(True, axis="y", color=grid, linewidth=0.58)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=2.4, width=0.62)
        ax.set_xlim(0.0, horizon_s)

    x = df["elapsed_s"]
    ax_lat.plot(x, df["op_p99"], color=blue, lw=1.55, label="Read P99")
    top_insert = df[df["insert_period"]].sort_values("op_p99", ascending=False).head(2)
    ax_lat.scatter(top_insert["elapsed_s"], top_insert["op_p99"], s=22, color=blue, edgecolor="white", linewidth=0.55, zorder=4)
    ax_lat.set_ylabel("Read P99\n(ms)")
    ax_lat.set_ylim(0.0, max(7.6, float(df["op_p99"].quantile(0.995)) * 1.06))
    ax_lat.tick_params(axis="x", labelbottom=False)
    ax_lat.legend(loc="upper left", frameon=False, handlelength=1.6)

    ax_wait.fill_between(x, 0, df["rdlock_wait_max_ms"], color=red, alpha=0.18, linewidth=0)
    ax_wait.plot(x, df["rdlock_wait_max_ms"], color=red, lw=1.32, label="Read-Lock Wait")
    ax_wait.set_ylabel("Max Wait\n(ms)")
    ax_wait.set_xlabel("Time (s)")
    ax_wait.set_ylim(0.0, max(145.0, float(df["rdlock_wait_max_ms"].max()) * 1.08))
    ax_wait.set_yticks([0, 60, 120])
    ax_wait.legend(loc="upper left", frameon=False, handlelength=1.6)

    insert = df[df["insert_period"]]
    outside = df[~df["insert_period"]]
    ax_scatter.scatter(
        outside["rdlock_wait_max_ms"],
        outside["op_p99"],
        s=15,
        color="#aab3bf",
        alpha=0.55,
        linewidths=0,
        label="Outside",
    )
    ax_scatter.scatter(
        insert["rdlock_wait_max_ms"],
        insert["op_p99"],
        s=18,
        color=red,
        alpha=0.86,
        linewidths=0,
        label="Insert-Active",
    )
    if len(insert) > 2 and insert["rdlock_wait_max_ms"].max() > 0:
        coef = np.polyfit(insert["rdlock_wait_max_ms"], insert["op_p99"], 1)
        xs = np.linspace(0.0, insert["rdlock_wait_max_ms"].max(), 80)
        ax_scatter.plot(xs, coef[0] * xs + coef[1], color=dark, lw=1.0)
    r = insert["rdlock_wait_max_ms"].corr(insert["op_p99"], method="pearson")
    ax_scatter.text(
        0.05,
        0.94,
        f"Insert bins\nr = {r:.2f}",
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
        color=dark,
        fontsize=7.6,
        linespacing=1.0,
    )
    ax_scatter.grid(True, color=grid, linewidth=0.58)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)
    ax_scatter.tick_params(axis="both", length=2.4, width=0.62)
    ax_scatter.set_xlabel("Read-Lock Wait (ms)")
    ax_scatter.set_ylabel("Read P99 (ms)")
    ax_scatter.set_xlim(left=0.0)
    ax_scatter.set_ylim(bottom=0.0)
    ax_scatter.legend(loc="lower right", frameon=False, handletextpad=0.35, borderpad=0.1)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.985, bottom=0.16)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.018)
    fig.savefig(out_png, dpi=260, bbox_inches="tight", pad_inches=0.018)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
