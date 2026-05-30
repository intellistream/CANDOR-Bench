#!/usr/bin/env python3
"""Draw the M3 read/write coherence evidence with a compact path hierarchy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


WRITE_PERIODS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]
RFO_MISS = "l2_rqsts.rfo_miss_per_s"
RFO_WAIT = "offcore_requests_outstanding.cycles_with_demand_rfo_per_s"
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
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in WRITE_PERIODS:
        if start > cursor:
            out.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < max_ms:
        out.append((cursor, max_ms))
    return out


def shade_write(ax: plt.Axes) -> None:
    for start, end in WRITE_PERIODS:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#f9dada", alpha=0.50, lw=0, zorder=0)


def draw_lanes(ax: plt.Axes, max_ms: float) -> None:
    measured = "#273241"
    read = "#e2e8f0"
    write = "#e4645d"
    ax.set_xlim(0.0, max_ms / 1000.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, labelbottom=False)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.grid(False)

    label_x = 0.018
    label_w = 0.230
    track_x = 0.272
    track_w = 0.708
    bar_h = 0.340
    measured_y = 0.080
    background_y = 0.590

    def add_rect(x: float, y: float, w: float, h: float, color: str) -> None:
        ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, facecolor=color, edgecolor="none", clip_on=False))

    def add_track_segment(start_ms: float, end_ms: float, y: float, color: str) -> tuple[float, float]:
        x0 = track_x + (start_ms / max_ms) * track_w
        width = ((end_ms - start_ms) / max_ms) * track_w
        add_rect(x0, y, width, bar_h, color)
        return x0, width

    add_rect(label_x, measured_y, label_w, bar_h, measured)
    add_rect(label_x, background_y, label_w, bar_h, "#f8fafc")
    add_rect(track_x, measured_y, track_w, bar_h, measured)
    for start, end in read_periods(max_ms):
        add_track_segment(start, end, background_y, read)
    for start, end in WRITE_PERIODS:
        x0, width = add_track_segment(start, end, background_y, write)
        ax.text(
            x0 + width / 2.0,
            background_y + bar_h / 2.0,
        "Write",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=8.8,
            fontweight="normal",
        )

    ax.text(
        label_x + 0.020,
        measured_y + bar_h / 2.0,
        "Measured Read",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color="white",
        fontsize=9.0,
        fontweight="normal",
    )
    ax.text(
        label_x + 0.020,
        background_y + bar_h / 2.0,
        "Background Search",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color="#111827",
        fontsize=9.0,
        fontweight="normal",
    )


def medians(frame: pd.DataFrame) -> dict[str, float]:
    steady = frame[~boundary_mask(frame)].copy()
    wmask = write_mask(steady, edge_ms=100.0)
    read = steady[~wmask]
    write = steady[wmask]
    keys = ["p99_us", RFO_WAIT, RFO_HITM, READ_HITM]
    out: dict[str, float] = {}
    for key in keys:
        out[f"read_{key}"] = float(read[key].median())
        out[f"write_{key}"] = float(write[key].median())
        out[f"ratio_{key}"] = out[f"write_{key}"] / out[f"read_{key}"] if out[f"read_{key}"] else np.nan
    return out


def draw_chip(
    ax: plt.Axes,
    cx: float,
    cy: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    *,
    face: str,
    edge: str,
) -> None:
    x = cx - w / 2.0
    y = cy - h / 2.0
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.026",
        linewidth=1.05,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    for pin_y in np.linspace(y + h * 0.24, y + h * 0.76, 3):
        ax.add_patch(Rectangle((x - 0.024, pin_y - 0.006), 0.024, 0.012, facecolor=edge, edgecolor="none", alpha=0.85))
        ax.add_patch(Rectangle((x + w, pin_y - 0.006), 0.024, 0.012, facecolor=edge, edgecolor="none", alpha=0.85))
    ax.text(cx, cy + h * 0.12, title, ha="center", va="center", fontsize=8.8, color="#111827", fontweight="normal")
    ax.text(cx, cy - h * 0.18, subtitle, ha="center", va="center", fontsize=8.0, color="#334155", fontweight="normal")


def draw_cache_line(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    outer = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.030",
        linewidth=1.1,
        edgecolor="#d97706",
        facecolor="#fff7ed",
    )
    ax.add_patch(outer)
    cells = 6
    gap = 0.009
    cell_w = (w - gap * (cells + 1)) / cells
    for i in range(cells):
        cx = x + gap + i * (cell_w + gap)
        face = "#fed7aa" if i in (2, 3) else "#ffedd5"
        ax.add_patch(
            FancyBboxPatch(
                (cx, y + h * 0.32),
                cell_w,
                h * 0.36,
                boxstyle="round,pad=0.002,rounding_size=0.010",
                linewidth=0.55,
                edgecolor="#fdba74",
                facecolor=face,
            )
        )
    ax.text(x + w / 2.0, y + h * 0.78, "Cache Line", ha="center", va="center", fontsize=8.6, color="#111827")
    ax.scatter([x + w * 0.56], [y + h * 0.12], s=58, color="#d97706", edgecolors="white", linewidths=0.8, zorder=4)
    ax.text(x + w * 0.69, y + h * 0.12, "Owner", ha="left", va="center", fontsize=7.6, color="#9a3412")


def draw_l1_line(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str,
    fill: str,
    owner: bool = False,
    invalid: bool = False,
) -> None:
    cells = 5
    gap = 0.006
    cell_w = (w - gap * (cells - 1)) / cells
    for i in range(cells):
        cx = x + i * (cell_w + gap)
        face = fill if i in (1, 2) else "white"
        ax.add_patch(
            FancyBboxPatch(
                (cx, y),
                cell_w,
                h,
                boxstyle="round,pad=0.002,rounding_size=0.010",
                linewidth=0.75,
                edgecolor=edge,
                facecolor=face,
            )
        )
    if owner:
        ax.scatter([x + w * 0.56], [y + h * 0.50], s=54, color="#d97706", edgecolors="white", linewidths=0.8, zorder=5)
        ax.text(x + w * 0.66, y + h * 0.50, "Owner", ha="left", va="center", fontsize=7.2, color="#9a3412")
    if invalid:
        ax.plot([x + w * 0.10, x + w * 0.90], [y + h * 0.18, y + h * 0.82], color="#b91c1c", lw=1.25)
        ax.plot([x + w * 0.10, x + w * 0.90], [y + h * 0.82, y + h * 0.18], color="#b91c1c", lw=1.25)
        ax.text(x + w * 0.50, y - h * 0.42, "Invalid L1 Copy", ha="center", va="center", fontsize=7.1, color="#b91c1c")


def draw_core_with_l1(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    action: str,
    edge: str,
    face: str,
    fill: str,
    owner: bool = False,
    invalid: bool = False,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.030",
        linewidth=1.15,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    for pin_y in np.linspace(y + h * 0.25, y + h * 0.75, 3):
        ax.add_patch(Rectangle((x - 0.020, pin_y - 0.005), 0.020, 0.010, facecolor=edge, edgecolor="none", alpha=0.82))
        ax.add_patch(Rectangle((x + w, pin_y - 0.005), 0.020, 0.010, facecolor=edge, edgecolor="none", alpha=0.82))
    ax.text(x + w * 0.50, y + h * 0.78, title, ha="center", va="center", fontsize=8.9, color="#111827")
    ax.text(x + w * 0.50, y + h * 0.56, action, ha="center", va="center", fontsize=7.8, color="#334155")
    ax.text(x + w * 0.18, y + h * 0.30, "L1 Line", ha="center", va="center", fontsize=7.1, color="#475569")
    draw_l1_line(ax, x + w * 0.33, y + h * 0.20, w * 0.48, h * 0.16, edge=edge, fill=fill, owner=owner, invalid=invalid)


def draw_hierarchy(ax: plt.Axes, stats: dict[str, float]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.05, 0.965, "(b)", ha="left", va="top", fontsize=9.6, fontweight="semibold")

    draw_core_with_l1(
        ax,
        0.17,
        0.720,
        0.68,
        0.160,
        title="Writer Core",
        action="Store",
        face="#fff1f0",
        edge="#d95f59",
        fill="#fee2e2",
        owner=True,
    )
    draw_cache_line(ax, 0.21, 0.455, 0.60, 0.175)
    draw_core_with_l1(
        ax,
        0.17,
        0.145,
        0.68,
        0.170,
        title="Reader Core",
        action="Load",
        face="#eff6ff",
        edge="#3867d6",
        fill="#dbeafe",
        invalid=True,
    )

    ax.annotate("", xy=(0.51, 0.645), xytext=(0.51, 0.720), arrowprops=dict(arrowstyle="-|>", lw=1.55, color="#b91c1c"))
    ax.text(0.70, 0.675, "RFO: Exclusive Copy", ha="center", va="center", fontsize=7.8, color="#b91c1c")

    ax.annotate("", xy=(0.51, 0.332), xytext=(0.51, 0.455), arrowprops=dict(arrowstyle="-|>", lw=1.55, color="#7048e8"))
    ax.text(0.70, 0.382, "Snoop / HITM Response", ha="center", va="center", fontsize=7.8, color="#5b35d5")

    clock = plt.Circle((0.23, 0.090), 0.038, facecolor="white", edgecolor="#3867d6", linewidth=1.1)
    ax.add_patch(clock)
    ax.plot([0.23, 0.23], [0.090, 0.113], color="#3867d6", lw=1.0)
    ax.plot([0.23, 0.250], [0.090, 0.078], color="#3867d6", lw=1.0)
    ax.text(0.55, 0.090, "Read Load Wait", ha="center", va="center", fontsize=8.2, color="#1d4ed8")


def draw_l1_cells_compact(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str,
    fill: str,
    mode: str,
) -> None:
    cells = 5
    gap = 0.006
    cell_w = (w - gap * (cells - 1)) / cells
    for i in range(cells):
        cx = x + i * (cell_w + gap)
        active = i in (2, 3)
        ax.add_patch(
            FancyBboxPatch(
                (cx, y),
                cell_w,
                h,
                boxstyle="round,pad=0.002,rounding_size=0.008",
                linewidth=0.72,
                edgecolor=edge,
                facecolor=fill if active else "white",
            )
        )
    if mode == "owner":
        ax.scatter([x + w * 0.64], [y + h * 0.50], s=38, color="#d97706", edgecolors="white", linewidths=0.7, zorder=5)
        ax.text(x + w * 0.78, y + h * 0.50, "Owner", ha="left", va="center", fontsize=6.7, color="#9a3412")
    elif mode == "invalid":
        ax.plot([x + w * 0.34, x + w * 0.74], [y + h * 0.20, y + h * 0.80], color="#b91c1c", lw=1.10)
        ax.plot([x + w * 0.34, x + w * 0.74], [y + h * 0.80, y + h * 0.20], color="#b91c1c", lw=1.10)
        ax.text(x + w * 0.92, y + h * 0.50, "Invalid", ha="left", va="center", fontsize=6.7, color="#b91c1c")


def draw_core_tile_compact(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    action: str,
    face: str,
    edge: str,
    line_fill: str,
    mode: str,
) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.026",
        linewidth=1.05,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w * 0.09, y + h * 0.66, title, ha="left", va="center", fontsize=8.4, color="#111827")
    action_box = FancyBboxPatch(
        (x + w * 0.08, y + h * 0.18),
        w * 0.20,
        h * 0.30,
        boxstyle="round,pad=0.006,rounding_size=0.014",
        linewidth=0.75,
        edgecolor=edge,
        facecolor="white",
    )
    ax.add_patch(action_box)
    ax.text(x + w * 0.18, y + h * 0.33, action, ha="center", va="center", fontsize=7.2, color="#334155")
    ax.text(x + w * 0.43, y + h * 0.66, "L1 Line", ha="left", va="center", fontsize=7.0, color="#475569")
    line_x = x + w * 0.43
    line_y = y + h * 0.23
    line_w = w * 0.42
    line_h = h * 0.25
    draw_l1_cells_compact(ax, line_x, line_y, line_w, line_h, edge=edge, fill=line_fill, mode=mode)
    return line_x, line_y, line_w, line_h


def draw_shared_line_compact(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.024",
        linewidth=1.05,
        edgecolor="#d97706",
        facecolor="#fff7ed",
    )
    ax.add_patch(patch)
    ax.text(x + w * 0.07, y + h * 0.68, "Cache Line", ha="left", va="center", fontsize=8.0, color="#111827")
    draw_l1_cells_compact(ax, x + w * 0.36, y + h * 0.30, w * 0.44, h * 0.27, edge="#d97706", fill="#fed7aa", mode="owner")


def draw_hierarchy(ax: plt.Axes, stats: dict[str, float]) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.05, 0.965, "(b)", ha="left", va="top", fontsize=9.6, fontweight="semibold")
    panel = FancyBboxPatch(
        (0.11, 0.105),
        0.80,
        0.780,
        boxstyle="round,pad=0.014,rounding_size=0.030",
        linewidth=0.75,
        edgecolor="#e2e8f0",
        facecolor="#fbfdff",
    )
    ax.add_patch(panel)

    draw_core_tile_compact(
        ax,
        0.17,
        0.705,
        0.68,
        0.135,
        title="Writer Core",
        action="Store",
        face="#fff5f4",
        edge="#d95f59",
        line_fill="#fee2e2",
        mode="owner",
    )
    draw_shared_line_compact(ax, 0.21, 0.460, 0.60, 0.120)
    draw_core_tile_compact(
        ax,
        0.17,
        0.185,
        0.68,
        0.140,
        title="Reader Core",
        action="Load",
        face="#eff6ff",
        edge="#3867d6",
        line_fill="#dbeafe",
        mode="invalid",
    )

    ax.annotate("", xy=(0.51, 0.590), xytext=(0.51, 0.705), arrowprops=dict(arrowstyle="-|>", lw=1.35, color="#b91c1c"))
    ax.text(0.66, 0.640, "RFO Gets Ownership", ha="center", va="center", fontsize=7.4, color="#b91c1c")

    ax.annotate("", xy=(0.51, 0.335), xytext=(0.51, 0.460), arrowprops=dict(arrowstyle="-|>", lw=1.35, color="#7048e8"))
    ax.text(0.66, 0.395, "Snoop / HITM", ha="center", va="center", fontsize=7.4, color="#5b35d5")

    clock = plt.Circle((0.25, 0.105), 0.036, facecolor="white", edgecolor="#3867d6", linewidth=1.05)
    ax.add_patch(clock)
    ax.plot([0.25, 0.25], [0.105, 0.126], color="#3867d6", lw=1.0)
    ax.plot([0.25, 0.268], [0.105, 0.095], color="#3867d6", lw=1.0)
    ax.text(0.55, 0.105, "Read Load Wait", ha="center", va="center", fontsize=8.0, color="#1d4ed8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--right-png", type=Path)
    parser.add_argument("--max-ms", type=float, default=12000.0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    joined = pd.read_csv(args.run / "coherence_joined.csv")
    joined = joined[(joined["time_ms"] >= 0.0) & (joined["time_ms"] <= args.max_ms)].copy()
    joined.to_csv(args.out.with_suffix(".csv"), index=False)
    stats = medians(joined)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Times New Roman", "Times"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 7.9,
            "ytick.labelsize": 7.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )

    blue = "#2f6fdd"
    red = "#d95f59"
    orange = "#d97706"
    purple = "#7048e8"
    text = "#334155"

    fig = plt.figure(figsize=(10.70, 4.42), constrained_layout=True)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.40, 1.0, 1.0],
        width_ratios=[1.58, 0.92],
        wspace=0.06,
    )
    ax_lanes = fig.add_subplot(gs[0, 0])
    ax_latency = fig.add_subplot(gs[1, 0], sharex=ax_lanes)
    ax_signal = fig.add_subplot(gs[2, 0], sharex=ax_lanes)
    ax_hierarchy = fig.add_subplot(gs[:, 1])

    draw_lanes(ax_lanes, args.max_ms)
    ax_lanes.text(
        0.0,
        1.03,
        "(a)",
        transform=ax_lanes.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="semibold",
    )
    for ax in [ax_latency, ax_signal]:
        shade_write(ax)
        ax.set_xlim(0.0, args.max_ms / 1000.0)
        ax.grid(True, axis="y", color="#e8edf3", linewidth=0.72)
        ax.tick_params(axis="both", length=3, color="#94a3b8")

    x = joined["time_ms"] / 1000.0
    ax_latency.plot(x, joined["p99_us"], color=blue, lw=2.0)
    ax_latency.set_ylabel("Read P99 (us)")
    ax_latency.text(
        0.020,
        0.82,
        f"{stats['read_p99_us']:.2f}us -> {stats['write_p99_us']:.2f}us  ({stats['ratio_p99_us']:.2f}x)",
        transform=ax_latency.transAxes,
        ha="left",
        va="center",
        color=text,
        fontsize=7.0,
        bbox=dict(facecolor="white", edgecolor="#dbe3ec", boxstyle="round,pad=0.16", alpha=0.94),
    )
    plt.setp(ax_latency.get_xticklabels(), visible=False)

    ax_signal.plot(x, joined[RFO_WAIT], color=red, lw=1.75, label="RFO Wait")
    ax_signal.plot(x, joined[RFO_HITM], color=orange, lw=1.55, label="RFO HITM")
    ax_signal.plot(x, joined[READ_HITM], color=purple, lw=1.75, label="Read HITM")
    ax_signal.set_yscale("log")
    ax_signal.set_ylabel("Coherence\nSignals")
    ax_signal.set_xlabel("Elapsed Time (s)")
    ax_signal.legend(loc="upper right", frameon=False, ncol=3, handlelength=1.45, columnspacing=0.7)
    ax_signal.text(
        0.020,
        0.82,
        f"RFO Wait {stats['ratio_' + RFO_WAIT]:.0f}x; RFO HITM {stats['ratio_' + RFO_HITM]:.0f}x; Read HITM {stats['ratio_' + READ_HITM]:.0f}x",
        transform=ax_signal.transAxes,
        ha="left",
        va="center",
        color=text,
        fontsize=6.8,
        bbox=dict(facecolor="white", edgecolor="#dbe3ec", boxstyle="round,pad=0.16", alpha=0.94),
    )

    right_png = args.right_png or (args.out.parent / "02_m3_coherence_right_drawio.png")
    if right_png.exists():
        ax_hierarchy.imshow(plt.imread(right_png))
        ax_hierarchy.set_axis_off()
    else:
        draw_hierarchy(ax_hierarchy, stats)

    fig.align_ylabels([ax_latency, ax_signal])
    fig.savefig(args.out, dpi=420, bbox_inches="tight")
    if args.pdf:
        fig.savefig(args.pdf, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
