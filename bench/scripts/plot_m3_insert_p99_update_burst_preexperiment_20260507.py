#!/usr/bin/env python3
"""Draw the M3 insert-period P99 and graph-update burst pre-experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def pctl(values: list[float], pct: float) -> float:
    vals = sorted(v for v in values if np.isfinite(v))
    if not vals:
        return float("nan")
    return vals[min(len(vals) - 1, int(len(vals) * pct / 100.0))]


def truth(value: object) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def read_origin(run_dir: Path) -> int:
    meta = pd.read_csv(run_dir / "raw_latency" / "meta.csv").set_index("key")["value"]
    return int(meta["bench_start_raw_ns"])


def read_windows(run_dir: Path) -> list[tuple[float, float]]:
    path = run_dir / "closed_loop_odin_tail_insert_boundaries.csv"
    frame = pd.read_csv(path)
    return list(zip(frame["active_start_ms"].astype(float), frame["active_end_ms"].astype(float)))


def read_timeline(run_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "closed_loop_odin_tail.csv")
    if "elapsed_s" not in frame:
        frame["elapsed_s"] = (frame["bin_ms"] + 25.0) / 1000.0
    if "lowq_op_p99_ms" in frame:
        frame["read_p99_ms"] = frame["lowq_op_p99_ms"]
    elif "read_op_p99_ms" in frame:
        frame["read_p99_ms"] = frame["read_op_p99_ms"]
    else:
        raise ValueError(f"{run_dir} timeline has no P99 latency column")
    return frame


def insert_period_mask(frame: pd.DataFrame, windows: list[tuple[float, float]]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for start, end in windows:
        mask |= (frame["bin_ms"] >= start) & (frame["bin_ms"] < end)
    return mask


def read_insert_spans(run_dir: Path) -> pd.DataFrame:
    origin = read_origin(run_dir)
    rows: list[dict[str, float]] = []
    with (run_dir / "raw_latency" / "insert_wide.csv").open() as file:
        for row in csv.DictReader(file):
            start_ms = (int(row["start_raw_ns"]) - origin) / 1_000_000.0
            finish_ms = (int(row["finish_raw_ns"]) - origin) / 1_000_000.0
            rows.append(
                {
                    "start_ms": start_ms,
                    "finish_ms": finish_ms,
                    "op_ms": float(row.get("op_ms", 0.0) or 0.0),
                    "loaded_edges": float(row.get("graph_existing_neighbor_loaded_edges", 0.0) or 0.0),
                    "written_edges": float(row.get("graph_existing_neighbor_edges_written", 0.0) or 0.0),
                    "visits": float(row.get("graph_existing_neighbor_visits", 0.0) or 0.0),
                }
            )
    return pd.DataFrame(rows)


def attach_update_burst(frame: pd.DataFrame, spans: pd.DataFrame, threshold_ms: float) -> pd.DataFrame:
    out = frame.copy()
    long_counts: list[int] = []
    long_loaded: list[float] = []
    long_written: list[float] = []
    max_op: list[float] = []
    for row in out.itertuples(index=False):
        center = float(row.bin_ms) + 25.0
        active = spans[(spans["start_ms"] <= center) & (spans["finish_ms"] >= center)]
        long = active[active["op_ms"] >= threshold_ms]
        long_counts.append(int(len(long)))
        long_loaded.append(float(long["loaded_edges"].sum()))
        long_written.append(float(long["written_edges"].sum()))
        max_op.append(float(active["op_ms"].max()) if len(active) else 0.0)
    out["long_update_batches"] = long_counts
    out["long_update_loaded_edges"] = long_loaded
    out["long_update_written_edges"] = long_written
    out["max_active_insert_ms"] = max_op
    return out


def draw_lanes(ax: plt.Axes, windows: list[tuple[float, float]], horizon_s: float) -> None:
    text = "#263445"
    navy = "#263445"
    red = "#df625d"
    pale = "#edf2f7"
    dot = "#6b7c92"
    ax.set_xlim(0.0, horizon_s)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    bg_y, read_y, lane_h = 0.62, 0.17, 0.28
    label_x = -0.05
    ax.text(label_x, bg_y + lane_h / 2.0, "Background\nSearch", transform=ax.transAxes, ha="right", va="center", color=text, fontsize=7.6, linespacing=0.94)
    ax.text(label_x, read_y + lane_h / 2.0, "Measured\nReads", transform=ax.transAxes, ha="right", va="center", color=text, fontsize=7.6, linespacing=0.94)

    cursor = 0.0
    read_segments: list[tuple[float, float]] = []
    for start_ms, end_ms in windows:
        start_s, end_s = start_ms / 1000.0, end_ms / 1000.0
        if start_s > cursor:
            read_segments.append((cursor, start_s))
        cursor = max(cursor, end_s)
    if cursor < horizon_s:
        read_segments.append((cursor, horizon_s))

    for start_s, end_s in read_segments:
        ax.add_patch(Rectangle((start_s, bg_y), end_s - start_s, lane_h, facecolor=pale, edgecolor="none"))
        xs = np.arange(start_s + 0.10, end_s - 0.04, 0.29)
        if len(xs):
            ys = bg_y + lane_h * (0.38 + 0.28 * (np.arange(len(xs)) % 2))
            ax.scatter(xs, ys, s=8.0, color=dot, linewidths=0)
    for start_ms, end_ms in windows:
        start_s, end_s = start_ms / 1000.0, end_ms / 1000.0
        ax.add_patch(Rectangle((start_s, bg_y), end_s - start_s, lane_h, facecolor=red, edgecolor="none"))
        ax.text((start_s + end_s) / 2.0, bg_y + lane_h / 2.0, "Insert", ha="center", va="center", color="white", fontsize=8.0)

    ax.add_patch(Rectangle((0.0, read_y), horizon_s, lane_h, facecolor=navy, edgecolor="none"))
    xs = np.arange(0.18, horizon_s - 0.04, 0.36)
    for idx, frac in enumerate([0.33, 0.50, 0.67]):
        shifted = xs + idx * 0.035
        shifted = shifted[shifted < horizon_s - 0.04]
        ax.scatter(shifted, np.full_like(shifted, read_y + lane_h * frac), s=5.2, color="#f8fafc", linewidths=0)


def shade_insert(ax: plt.Axes, windows: list[tuple[float, float]]) -> None:
    for start, end in windows:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#f6dada", alpha=0.52, lw=0, zorder=0)


def write_summary(path: Path, real: pd.DataFrame, bg: pd.DataFrame, windows: list[tuple[float, float]]) -> None:
    real = real.copy()
    real["insert_period"] = insert_period_mask(real, windows)
    top_rows: list[dict[str, object]] = []
    ordered = real.sort_values("read_p99_ms", ascending=False)
    for topn in [10, 20, 50]:
        top = ordered.head(topn)
        top_rows.append(
            {
                "metric": f"top{topn}_p99_bins_in_insert_period",
                "value": int(top["insert_period"].sum()),
                "denominator": int(len(top)),
            }
        )
    for label, start, end in [("window1", 2500.0, 3500.0), ("window2", 6000.0, 7000.0), ("window3", 9500.0, 10500.0)]:
        real_vals = real[(real["bin_ms"] >= start) & (real["bin_ms"] < end)]["read_p99_ms"].dropna().tolist()
        bg_vals = bg[(bg["bin_ms"] >= start) & (bg["bin_ms"] < end)]["read_p99_ms"].dropna().tolist()
        top_rows.append({"metric": f"{label}_real_insert_p99", "value": pctl(real_vals, 99), "denominator": ""})
        top_rows.append({"metric": f"{label}_search_only_control_p99", "value": pctl(bg_vals, 99), "denominator": ""})
    cause = real[(real["bin_ms"] >= 9500.0) & (real["bin_ms"] < 10500.0)]
    top_cause = cause.loc[cause["read_p99_ms"].idxmax()]
    top_rows.extend(
        [
            {"metric": "third_window_peak_bin_ms", "value": float(top_cause["bin_ms"]), "denominator": ""},
            {"metric": "third_window_peak_p99_ms", "value": float(top_cause["read_p99_ms"]), "denominator": ""},
            {"metric": "third_window_peak_long_update_batches", "value": int(top_cause["long_update_batches"]), "denominator": 40},
            {"metric": "third_window_peak_max_active_insert_ms", "value": float(top_cause["max_active_insert_ms"]), "denominator": ""},
            {"metric": "third_window_peak_loaded_edges", "value": float(top_cause["long_update_loaded_edges"]), "denominator": ""},
            {"metric": "third_window_peak_written_edges", "value": float(top_cause["long_update_written_edges"]), "denominator": ""},
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(top_rows).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--long-update-ms", type=float, default=100.0)
    args = parser.parse_args()

    real = read_timeline(args.real_run)
    control = read_timeline(args.control_run)
    windows = read_windows(args.real_run)
    horizon_s = max(float(real["elapsed_s"].max()), float(control["elapsed_s"].max()))
    spans = read_insert_spans(args.real_run)
    real = attach_update_burst(real, spans, args.long_update_ms)
    control = control[["bin_ms", "elapsed_s", "read_p99_ms"]].rename(columns={"read_p99_ms": "control_read_p99_ms"})
    joined = real.merge(control, on=["bin_ms", "elapsed_s"], how="left")

    csv_out = args.csv or args.out.with_suffix(".csv")
    summary_out = args.summary or args.out.with_name(args.out.stem + "_summary.csv")
    for path in [args.out, args.png, csv_out, summary_out]:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(csv_out, index=False)
    write_summary(summary_out, real, control.rename(columns={"control_read_p99_ms": "read_p99_ms"}), windows)

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "legend.fontsize": 7.1,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.72,
        }
    )
    blue = "#2f6fdd"
    gray = "#7a8493"
    red = "#d95f59"
    grid = "#e5e7eb"

    fig = plt.figure(figsize=(4.15, 3.05), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.48, 1.20, 0.90], hspace=0.15)
    ax_lanes = fig.add_subplot(gs[0, 0])
    ax_lat = fig.add_subplot(gs[1, 0], sharex=ax_lanes)
    ax_cause = fig.add_subplot(gs[2, 0], sharex=ax_lanes)

    draw_lanes(ax_lanes, windows, horizon_s)
    for ax in (ax_lat, ax_cause):
        shade_insert(ax, windows)
        ax.grid(True, axis="y", color=grid, linewidth=0.56)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=2.3, width=0.62)
        ax.set_xlim(0.0, horizon_s)

    x = joined["elapsed_s"]
    ax_lat.plot(x, joined["read_p99_ms"], color=blue, lw=1.55, label="Real Insert")
    ax_lat.plot(x, joined["control_read_p99_ms"], color=gray, lw=1.10, ls=(0, (3.0, 2.0)), label="Search Control")
    ymax = max(7.8, float(np.nanpercentile(joined[["read_p99_ms", "control_read_p99_ms"]].to_numpy(), 99.5)) * 1.08)
    ax_lat.set_ylim(0.0, ymax)
    ax_lat.set_ylabel("Read P99\n(ms)")
    ax_lat.tick_params(axis="x", labelbottom=False)
    ax_lat.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.5, columnspacing=0.9)

    ax_cause.fill_between(x, 0, joined["long_update_batches"], color=red, alpha=0.20, linewidth=0)
    ax_cause.plot(x, joined["long_update_batches"], color=red, lw=1.30, label="Long Graph-Update Batches")
    ax_cause.set_ylim(0.0, 42.0)
    ax_cause.set_yticks([0, 20, 40])
    ax_cause.set_ylabel("Active\nBatches")
    ax_cause.set_xlabel("Elapsed Time (s)")
    ax_cause.set_xticks(np.arange(0.0, horizon_s + 0.1, 2.0))
    ax_cause.legend(loc="upper left", frameon=False, handlelength=1.6)

    fig.tight_layout(pad=0.30)
    fig.savefig(args.out, bbox_inches="tight")
    if args.png:
        fig.savefig(args.png, dpi=260, bbox_inches="tight")
    print(args.out)
    if args.png:
        print(args.png)
    print(csv_out)
    print(summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
