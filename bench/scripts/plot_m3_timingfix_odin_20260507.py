#!/usr/bin/env python3
"""Draw the M3 timing-fix validation as a compact Odin-style figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def pctl(series: pd.Series, pct: float) -> float:
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(np.percentile(vals, pct))


def load_origin(run_dir: Path) -> int:
    meta = pd.read_csv(run_dir / "raw_latency" / "meta.csv").set_index("key")["value"]
    return int(meta["bench_start_raw_ns"])


def load_active_windows(run_dir: Path) -> list[tuple[float, float]]:
    path = run_dir / "closed_loop_odin_tail_insert_boundaries.csv"
    frame = pd.read_csv(path)
    return list(zip(frame["active_start_ms"].astype(float), frame["active_end_ms"].astype(float)))


def in_windows(values: pd.Series, windows: list[tuple[float, float]]) -> pd.Series:
    mask = pd.Series(False, index=values.index)
    for start, end in windows:
        mask |= (values >= start) & (values < end)
    return mask


def load_search(run_dir: Path, origin_raw_ns: int) -> pd.DataFrame:
    available = set(pd.read_csv(run_dir / "raw_latency" / "search_wide.csv", nrows=0).columns)
    wanted = [
        "measured",
        "start_raw_ns",
        "finish_raw_ns",
        "op_ms",
        "e2e_ms",
        "search_index_ms",
        "search_record_lock_wait_ms",
    ]
    missing = [col for col in wanted if col not in available]
    if missing:
        raise ValueError(f"missing columns in search_wide.csv: {', '.join(missing)}")
    search = pd.read_csv(run_dir / "raw_latency" / "search_wide.csv", usecols=wanted)
    search["start_ms"] = (search["start_raw_ns"] - origin_raw_ns) / 1_000_000.0
    search["finish_raw_ms"] = (search["finish_raw_ns"] - origin_raw_ns) / 1_000_000.0
    search["mid_ms"] = (search["start_ms"] + search["finish_raw_ms"]) / 2.0
    search["queue_ms"] = search["e2e_ms"] - search["op_ms"]
    return search


def binned_search(
    search: pd.DataFrame,
    *,
    horizon_ms: float,
    bin_ms: float,
    window_ms: float,
    queue_threshold_ms: float,
) -> pd.DataFrame:
    measured = search[(search["measured"] == 1) & (search["queue_ms"] < queue_threshold_ms)].copy()
    rows = []
    half = window_ms / 2.0
    for bin_start in np.arange(0.0, horizon_ms, bin_ms):
        center = bin_start + bin_ms / 2.0
        part = measured[(measured["mid_ms"] >= center - half) & (measured["mid_ms"] < center + half)]
        rows.append(
            {
                "bin_ms": bin_start,
                "elapsed_s": (bin_start + bin_ms / 2.0) / 1000.0,
                "measured_reads": int(len(part)),
                "read_op_p95_ms": pctl(part["op_ms"], 95),
                "read_op_p99_ms": pctl(part["op_ms"], 99),
                "index_stage_p99_ms": pctl(part["search_index_ms"], 99),
                "recording_wait_p99_ms": pctl(part["search_record_lock_wait_ms"], 99),
            }
        )
    return pd.DataFrame(rows)


def phase_summary(search: pd.DataFrame, windows: list[tuple[float, float]], queue_threshold_ms: float) -> pd.DataFrame:
    measured = search[(search["measured"] == 1) & (search["queue_ms"] < queue_threshold_ms)].copy()
    measured["insert_period"] = in_windows(measured["mid_ms"], windows)
    rows = []
    for label, mask in [("outside", ~measured["insert_period"]), ("insert_period", measured["insert_period"])]:
        part = measured[mask]
        rows.append(
            {
                "phase": label,
                "measured_reads": int(len(part)),
                "read_op_p95_ms": pctl(part["op_ms"], 95),
                "read_op_p99_ms": pctl(part["op_ms"], 99),
                "index_stage_p99_ms": pctl(part["search_index_ms"], 99),
                "recording_wait_p99_ms": pctl(part["search_record_lock_wait_ms"], 99),
            }
        )
    return pd.DataFrame(rows)


def shade_insert(ax: plt.Axes, windows: list[tuple[float, float]]) -> None:
    for start, end in windows:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#f5d7d7", alpha=0.50, lw=0, zorder=0)


def read_segments(windows: list[tuple[float, float]], horizon_ms: float) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in sorted(windows):
        if start > cursor:
            segments.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < horizon_ms:
        segments.append((cursor, horizon_ms))
    return segments


def draw_lanes(ax: plt.Axes, windows: list[tuple[float, float]], horizon_ms: float) -> None:
    navy = "#263445"
    blue = "#3569d4"
    red = "#df625d"
    pale = "#edf2f7"
    text = "#263445"

    ax.set_xlim(0.0, horizon_ms / 1000.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    bg_y, read_y = 0.64, 0.18
    lane_h = 0.28
    label_x = -0.045

    ax.text(label_x, bg_y + lane_h / 2.0, "Background\nSearch", transform=ax.transAxes, ha="right", va="center", color=text, fontsize=8.2, linespacing=0.95)
    ax.text(label_x, read_y + lane_h / 2.0, "Measured\nReads", transform=ax.transAxes, ha="right", va="center", color=text, fontsize=8.2, linespacing=0.95)

    for start, end in read_segments(windows, horizon_ms):
        ax.add_patch(Rectangle((start / 1000.0, bg_y), (end - start) / 1000.0, lane_h, facecolor=pale, edgecolor="none"))
        xs = np.arange(start / 1000.0 + 0.10, end / 1000.0 - 0.05, 0.28)
        if len(xs):
            ys = bg_y + lane_h * (0.35 + 0.28 * ((np.arange(len(xs)) % 2)))
            ax.scatter(xs, ys, s=10, color="#6b7c92", linewidths=0, zorder=2)

    for start, end in windows:
        ax.add_patch(Rectangle((start / 1000.0, bg_y), (end - start) / 1000.0, lane_h, facecolor=red, edgecolor="none"))
        ax.text((start + end) / 2000.0, bg_y + lane_h / 2.0, "Insert", ha="center", va="center", color="white", fontsize=8.5)

    ax.add_patch(Rectangle((0.0, read_y), horizon_ms / 1000.0, lane_h, facecolor=navy, edgecolor="none"))
    xs = np.arange(0.18, horizon_ms / 1000.0 - 0.05, 0.36)
    for row, offset in enumerate([0.33, 0.50, 0.67]):
        shifted = xs + row * 0.035
        shifted = shifted[shifted < horizon_ms / 1000.0 - 0.05]
        ax.scatter(shifted, np.full_like(shifted, read_y + lane_h * offset), s=6.0, color="#f8fafc", linewidths=0, zorder=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--bin-ms", type=float, default=50.0)
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--queue-threshold-ms", type=float, default=0.1)
    args = parser.parse_args()

    origin = load_origin(args.run)
    windows = load_active_windows(args.run)
    search = load_search(args.run, origin)
    horizon_ms = float(max(search["finish_raw_ms"].max(), max(end for _, end in windows)))
    binned = binned_search(
        search,
        horizon_ms=horizon_ms,
        bin_ms=args.bin_ms,
        window_ms=args.window_ms,
        queue_threshold_ms=args.queue_threshold_ms,
    )
    binned["insert_period"] = in_windows(binned["bin_ms"], windows)
    summary = phase_summary(search, windows, args.queue_threshold_ms)

    csv_out = args.csv or args.out.with_suffix(".csv")
    summary_out = args.summary or args.out.with_name(args.out.stem + "_summary.csv")
    for path in [args.out, csv_out, summary_out, args.pdf]:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    binned.to_csv(csv_out, index=False)
    summary.to_csv(summary_out, index=False)

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 8.4,
            "axes.labelsize": 8.8,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.linewidth": 0.75,
        }
    )

    blue = "#2f6fdd"
    gray = "#6b7280"
    grid = "#e5e7eb"

    fig = plt.figure(figsize=(4.15, 3.05), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 1.24, 0.86], hspace=0.16)
    ax_lanes = fig.add_subplot(gs[0, 0])
    ax_lat = fig.add_subplot(gs[1, 0], sharex=ax_lanes)
    ax_audit = fig.add_subplot(gs[2, 0], sharex=ax_lanes)

    draw_lanes(ax_lanes, windows, horizon_ms)
    for ax in (ax_lat, ax_audit):
        shade_insert(ax, windows)
        ax.grid(True, axis="y", color=grid, lw=0.58)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=2.5, width=0.65)
        ax.set_xlim(0.0, horizon_ms / 1000.0)

    x = binned["elapsed_s"]
    ax_lat.plot(x, binned["read_op_p99_ms"], color=blue, lw=1.55, solid_capstyle="round", label="Read P99")
    ymax = max(4.0, float(np.nanpercentile(binned["read_op_p99_ms"], 99)) * 1.14)
    ax_lat.set_ylim(0.0, ymax)
    ax_lat.set_ylabel("P99 Latency\n(ms)")
    ax_lat.tick_params(axis="x", labelbottom=False)
    ax_lat.legend(loc="upper left", frameon=False, handlelength=1.5)

    ax_audit.plot(
        x,
        binned["recording_wait_p99_ms"],
        color=gray,
        lw=1.15,
        ls=(0, (3.0, 2.0)),
        solid_capstyle="round",
        label="Benchmark Recording Wait",
    )
    ax_audit.set_ylim(0.0, max(2.0, float(np.nanpercentile(binned["recording_wait_p99_ms"], 99)) * 1.12))
    ax_audit.set_ylabel("P99 Wait\n(ms)")
    ax_audit.set_xlabel("Elapsed Time (s)")
    ax_audit.legend(loc="upper left", frameon=False, handlelength=1.7)
    ax_audit.set_xticks(np.arange(0.0, horizon_ms / 1000.0 + 0.1, 2.0))

    fig.tight_layout(pad=0.28)
    fig.savefig(args.out, dpi=260, bbox_inches="tight")
    if args.pdf is not None:
        fig.savefig(args.pdf, bbox_inches="tight")

    print(args.out)
    print(csv_out)
    print(summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
