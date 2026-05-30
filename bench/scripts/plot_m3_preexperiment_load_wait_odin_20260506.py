#!/usr/bin/env python3
"""OdinANN-style pre-experiment figure for M3 load-wait evidence."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def percentile(series: pd.Series, pct: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return math.nan
    return float(np.percentile(vals.to_numpy(), pct))


def scale01(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    lo = float(vals.quantile(0.05))
    hi = float(vals.quantile(0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return vals * 0.0
    return ((vals - lo) / (hi - lo)).clip(0.0, 1.0)


def load_search(root: Path, bin_ms: float, max_ms: float) -> pd.DataFrame:
    frame = pd.read_csv(root / "raw_latency" / "search_wide.csv")
    frame = frame[frame["measured"] == 1].copy()
    frame["mid_ms"] = frame["finish_ms"] - (
        frame["finish_raw_ns"] - (frame["start_raw_ns"] + frame["finish_raw_ns"]) / 2.0
    ) / 1_000_000.0
    frame = frame[(frame["mid_ms"] >= 0.0) & (frame["mid_ms"] <= max_ms)].copy()
    frame["bin_start_ms"] = np.floor(frame["mid_ms"] / bin_ms) * bin_ms
    rows: list[dict[str, float]] = []
    for bin_start, group in frame.groupby("bin_start_ms", sort=True):
        rows.append(
            {
                "bin_start_ms": float(bin_start),
                "time_ms": float(bin_start + bin_ms / 2.0),
                "search_p99_ms": percentile(group["work_searchknn_ms"], 99),
                "search_cpu_p99_ms": percentile(group["work_searchknn_thread_cpu_ms"], 99),
                "distance_computations_mean": float(
                    pd.to_numeric(group["work_distance_computations"], errors="coerce").mean()
                ),
                "level0_edges_mean": float(
                    pd.to_numeric(group["work_level0_edges_scanned"], errors="coerce").mean()
                ),
                "rows": float(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ["search_p99_ms", "search_cpu_p99_ms", "distance_computations_mean", "level0_edges_mean"]:
        out[f"{col}_smooth"] = out[col].rolling(3, center=True, min_periods=1).median()
    return out


def load_periods(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "period_groups.csv")
    frame["time_ms"] = (frame["bin_start_ms"] + frame["bin_end_ms"]) / 2.0
    cycles = pd.to_numeric(frame["perf_cycles"], errors="coerce").replace(0, np.nan)
    instr = pd.to_numeric(frame["perf_instructions"], errors="coerce")
    frame["ipc"] = instr / cycles
    return frame


def load_insert_windows(root: Path, gap_ms: float = 10.0) -> list[tuple[float, float]]:
    inserts = pd.read_csv(root / "raw_latency" / "insert_wide.csv")
    windows: list[tuple[float, float]] = []
    for start, end, finish_ms in inserts.sort_values("start_raw_ns")[
        ["start_raw_ns", "finish_raw_ns", "finish_ms"]
    ].itertuples(index=False):
        start = float(start)
        end = float(end)
        finish_ms = float(finish_ms)
        start_ms = finish_ms - (end - start) / 1_000_000.0
        if not windows or start_ms > windows[-1][1] + gap_ms:
            windows.append((start_ms, finish_ms))
        else:
            windows[-1] = (windows[-1][0], max(windows[-1][1], finish_ms))
    return windows


def shade_insert(ax: plt.Axes, windows: list[tuple[float, float]]) -> None:
    for start, end in windows:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#f6b6b6", alpha=0.30, lw=0)


def mask_windows(frame: pd.DataFrame, windows: list[tuple[float, float]], time_col: str = "time_ms") -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    if time_col not in frame.columns:
        return mask
    time = pd.to_numeric(frame[time_col], errors="coerce")
    for start, end in windows:
        mask |= (time >= start) & (time <= end)
    return mask


def clipped_windows(windows: list[tuple[float, float]], max_ms: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in windows:
        clipped_start = max(0.0, start)
        clipped_end = min(max_ms, end)
        if clipped_end > clipped_start:
            out.append((clipped_start, clipped_end))
    return out


def complement_windows(windows: list[tuple[float, float]], max_ms: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in clipped_windows(sorted(windows), max_ms):
        if start > cursor:
            spans.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < max_ms:
        spans.append((cursor, max_ms))
    return spans


def draw_worker_layout(
    ax: plt.Axes,
    windows: list[tuple[float, float]],
    max_ms: float,
    measured_color: str,
    competing_search_color: str,
    competing_insert_color: str,
) -> None:
    ax.set_ylim(0.0, 2.0)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0, labelbottom=False)

    max_s = max_ms / 1000.0
    measured_lane = (0.24, 0.38)
    competing_lane = (1.07, 0.38)
    ax.broken_barh([(0.0, max_s)], measured_lane, facecolors=measured_color, edgecolors="none")

    for start, end in complement_windows(windows, max_ms):
        ax.broken_barh(
            [(start / 1000.0, (end - start) / 1000.0)],
            competing_lane,
            facecolors=competing_search_color,
            edgecolors="none",
        )
    for start, end in clipped_windows(windows, max_ms):
        ax.broken_barh(
            [(start / 1000.0, (end - start) / 1000.0)],
            competing_lane,
            facecolors=competing_insert_color,
            edgecolors="none",
        )

    ax.text(
        0.01,
        measured_lane[0] + measured_lane[1] / 2.0,
        "8 measured-search threads",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        color="white",
        fontsize=9.3,
        fontweight="bold",
    )
    ax.text(
        0.01,
        competing_lane[0] + competing_lane[1] / 2.0,
        "40 competing pull-based workers",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        color="#111827",
        fontsize=9.3,
        fontweight="bold",
    )
    if windows:
        start, end = clipped_windows(windows, max_ms)[0]
        ax.text(
            (start + end) / 2000.0,
            competing_lane[0] + competing_lane[1] + 0.13,
            "insert period: start to commit",
            ha="center",
            va="bottom",
            color="#991b1b",
            fontsize=9.0,
            fontweight="bold",
        )


def joined_signal(search: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "bin_start_ms",
        "perf_offcore_requests_outstanding_l3_miss_demand_data_rd_per_s",
        "perf_mem_load_retired_l3_miss_per_s",
        "perf_llc_load_misses_per_s",
        "perf_l2_rqsts_rfo_miss_per_s",
        "ipc",
    ]
    present = [col for col in cols if col in periods.columns]
    joined = search.merge(periods[present], on="bin_start_ms", how="inner")
    joined = joined[joined["rows"] >= 20].copy()
    return joined


def corr(frame: pd.DataFrame, x: str, y: str = "search_p99_ms") -> float:
    data = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 4 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return math.nan
    return float(data[x].corr(data[y], method="spearman"))


def work_ratio(background: pd.DataFrame, real_insert: pd.DataFrame) -> pd.DataFrame:
    joined = real_insert.merge(
        background,
        on="bin_start_ms",
        suffixes=("_insert", "_background"),
        how="inner",
    )
    for metric in ["distance_computations_mean_smooth", "level0_edges_mean_smooth"]:
        denom = pd.to_numeric(joined[f"{metric}_background"], errors="coerce").replace(0, np.nan)
        joined[f"{metric}_ratio"] = pd.to_numeric(joined[f"{metric}_insert"], errors="coerce") / denom
    joined["time_ms"] = joined["time_ms_insert"]
    return joined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--real-insert", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--bin-ms", type=float, default=50.0)
    parser.add_argument("--max-ms", type=float, default=5200.0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    bg_search = load_search(args.background, args.bin_ms, args.max_ms)
    insert_search = load_search(args.real_insert, args.bin_ms, args.max_ms)
    insert_periods = load_periods(args.real_insert)
    windows = load_insert_windows(args.real_insert)
    joined = joined_signal(insert_search, insert_periods)
    joined.to_csv(args.out.with_suffix(".binned.csv"), index=False)

    load_wait_col = "perf_offcore_requests_outstanding_l3_miss_demand_data_rd_per_s"
    active = joined[mask_windows(joined, windows)].copy() if windows else joined
    load_wait_r = corr(joined, load_wait_col, "search_p99_ms_smooth")
    load_wait_insert_r = corr(active, load_wait_col, "search_p99_ms_smooth")
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
    ratio = work_ratio(bg_search, insert_search)
    ratio_active = ratio[mask_windows(ratio, windows)].copy() if windows else ratio
    dist_ratio = float(pd.to_numeric(ratio_active["distance_computations_mean_smooth_ratio"], errors="coerce").median())
    edge_ratio = float(pd.to_numeric(ratio_active["level0_edges_mean_smooth_ratio"], errors="coerce").median())

    fig = plt.figure(figsize=(12.6, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.50, 1.05, 1.05],
        width_ratios=[1.45, 0.95],
    )
    ax_layout = fig.add_subplot(gs[0, :])
    ax_tail = fig.add_subplot(gs[1, :], sharex=ax_layout)
    ax_wait = fig.add_subplot(gs[2, 0], sharex=ax_layout)
    ax_scatter = fig.add_subplot(gs[2, 1])
    fig.patch.set_facecolor("white")

    blue = "#2563eb"
    gray = "#64748b"
    green = "#059669"
    text_gray = "#475569"
    measured_dark = "#1f2937"
    insert_red = "#ef4444"
    competing_search = "#cbd5e1"

    draw_worker_layout(
        ax_layout,
        windows,
        args.max_ms,
        measured_color=measured_dark,
        competing_search_color=competing_search,
        competing_insert_color=insert_red,
    )
    ax_layout.set_title("Thread roles in this pre-experiment", loc="left", pad=2, fontweight="bold")
    ax_layout.legend(
        handles=[
            Patch(facecolor=measured_dark, edgecolor="none", label="measured search"),
            Patch(facecolor=competing_search, edgecolor="none", label="competing search"),
            Patch(facecolor=insert_red, edgecolor="none", label="competing insert"),
        ],
        loc="upper right",
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.3,
    )

    for ax in [ax_tail, ax_wait]:
        shade_insert(ax, windows)
        ax.grid(True, axis="y", color="#e8edf3", linewidth=0.75)
        ax.set_xlim(0, args.max_ms / 1000.0)
        ax.tick_params(axis="both", length=3, color="#94a3b8")

    ax_scatter.grid(True, color="#e8edf3", linewidth=0.75)
    ax_scatter.tick_params(axis="both", length=3, color="#94a3b8")

    ax_tail.plot(
        bg_search["time_ms"] / 1000.0,
        bg_search["search_p99_ms_smooth"],
        color=gray,
        lw=2.0,
        label="Background: competing workers search",
    )
    ax_tail.plot(
        insert_search["time_ms"] / 1000.0,
        insert_search["search_p99_ms_smooth"],
        color=blue,
        lw=2.4,
        label="Periodic run: competing insert periods",
    )
    ax_tail.set_ylabel("C++ HNSW p99 (ms)")
    ax_tail.set_title("Measured-search tail under competing insert", loc="left", pad=7, fontweight="bold")
    lines, labels = ax_tail.get_legend_handles_labels()
    ax_tail.legend(
        handles=[Patch(facecolor="#f6b6b6", edgecolor="none", alpha=0.40, label="insert period"), *lines],
        labels=["insert period", *labels],
        loc="upper right",
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
    )
    ax_tail.text(
        0.015,
        0.86,
        f"ANN work unchanged: distance {dist_ratio:.2f}x, L0 edges {edge_ratio:.2f}x",
        transform=ax_tail.transAxes,
        ha="left",
        va="center",
        color=text_gray,
        fontsize=9.2,
        bbox=dict(facecolor="white", edgecolor="#dbe3ec", boxstyle="round,pad=0.25", alpha=0.88),
    )

    load_wait_smooth = pd.to_numeric(joined[load_wait_col], errors="coerce").rolling(3, center=True, min_periods=1).median()
    search_norm = scale01(joined["search_p99_ms_smooth"])
    wait_norm = scale01(load_wait_smooth)
    ax_wait.plot(
        joined["time_ms"] / 1000.0,
        search_norm,
        color=blue,
        lw=2.3,
        label="Search p99",
    )
    ax_wait.plot(
        joined["time_ms"] / 1000.0,
        wait_norm,
        color=green,
        lw=2.2,
        label="L3-miss load wait",
    )
    ax_wait.set_ylabel("Normalized")
    ax_wait.set_title("Same time bins: measured-search p99 and load wait", loc="left", pad=7, fontweight="bold")
    ax_wait.legend(loc="upper right", frameon=False, ncol=2, handlelength=2.0, columnspacing=1.4)
    ax_wait.set_xlabel("Elapsed time (s)")

    scatter = joined[["time_ms", "search_p99_ms_smooth", load_wait_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    scatter["load_wait_norm"] = scale01(pd.to_numeric(scatter[load_wait_col], errors="coerce"))
    scatter_mask = mask_windows(scatter, windows)
    ax_scatter.scatter(
        scatter.loc[~scatter_mask, "load_wait_norm"],
        scatter.loc[~scatter_mask, "search_p99_ms_smooth"],
        s=20,
        color=gray,
        alpha=0.55,
        edgecolors="none",
        label="competing search",
    )
    ax_scatter.scatter(
        scatter.loc[scatter_mask, "load_wait_norm"],
        scatter.loc[scatter_mask, "search_p99_ms_smooth"],
        s=24,
        color=insert_red,
        alpha=0.75,
        edgecolors="none",
        label="insert period",
    )
    fit = scatter[["load_wait_norm", "search_p99_ms_smooth"]].dropna()
    if len(fit) >= 4 and fit["load_wait_norm"].nunique() >= 2:
        xvals = fit["load_wait_norm"].to_numpy()
        yvals = fit["search_p99_ms_smooth"].to_numpy()
        slope, intercept = np.polyfit(xvals, yvals, 1)
        line_x = np.linspace(float(np.nanmin(xvals)), float(np.nanmax(xvals)), 50)
        ax_scatter.plot(line_x, slope * line_x + intercept, color="#111827", lw=1.4, alpha=0.80)
    ax_scatter.text(
        0.03,
        0.95,
        f"all-bin r={load_wait_r:+.2f}\ninsert-bin r={load_wait_insert_r:+.2f}",
        transform=ax_scatter.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        color=text_gray,
        bbox=dict(facecolor="white", edgecolor="#dbe3ec", boxstyle="round,pad=0.25", alpha=0.90),
    )
    ax_scatter.set_title("Per-bin relation", loc="left", pad=7, fontweight="bold")
    ax_scatter.set_xlabel("L3-miss load wait (normalized)")
    ax_scatter.set_ylabel("Search p99 (ms)")
    ax_scatter.legend(loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2)

    fig.suptitle(
        "SIFT m16/efc200/efs50: 8 measured-search threads + 40 competing pull-based workers",
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
