#!/usr/bin/env python3
"""Plot guarded M3 existing-neighbor update confirmation as Odin-style timelines."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = [
    (
        "existing_neighbor_update_read_rewrite",
        "existing-neighbor update: read + rewrite",
        "#dc2626",
    ),
    (
        "existing_neighbor_update_read_scan_only",
        "existing-neighbor update: read scan only",
        "#7c3aed",
    ),
    (
        "existing_neighbor_update_disabled",
        "existing-neighbor update disabled",
        "#2563eb",
    ),
]

VISUAL_CONDITIONS = [
    (
        "existing_neighbor_update_read_rewrite",
        "existing-neighbor update: read + rewrite",
        "#dc2626",
    ),
    (
        "existing_neighbor_update_disabled",
        "existing-neighbor update disabled",
        "#2563eb",
    ),
]


def bool_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def segments_from_mask(frame: pd.DataFrame, mask_col: str) -> list[tuple[float, float]]:
    mask = bool_mask(frame[mask_col]).to_numpy()
    times = frame["elapsed_s"].to_numpy(dtype=float)
    if len(times) == 0:
        return []
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.05
    segments: list[tuple[float, float]] = []
    start: float | None = None
    last = float(times[0])
    for t, active in zip(times, mask, strict=False):
        t = float(t)
        if active and start is None:
            start = t - step / 2.0
        if not active and start is not None:
            segments.append((start, t - step / 2.0 - start))
            start = None
        last = t
    if start is not None:
        segments.append((start, last + step / 2.0 - start))
    return [(max(0.0, start), max(0.0, width)) for start, width in segments]


def active_segments_from_boundaries(rep_dir: Path) -> list[tuple[float, float]]:
    path = rep_dir / "closed_loop_odin_tail_insert_boundaries.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    boundaries = pd.read_csv(path)
    segments: list[tuple[float, float]] = []
    for row in boundaries.itertuples(index=False):
        start_s = float(row.active_start_ms) / 1000.0
        end_s = float(row.active_end_ms) / 1000.0
        if end_s > start_s:
            segments.append((start_s, end_s - start_s))
    return segments


def find_rep_dir(root: Path, condition_id: str, rep: int) -> Path:
    rep_dir = root / f"{condition_id}_rep{rep}"
    if rep_dir.exists():
        return rep_dir
    matches = sorted(root.glob(f"{condition_id}_rep*"))
    if not matches:
        raise FileNotFoundError(f"no rep directories found for {condition_id} in {root}")
    available = ", ".join(path.name for path in matches)
    raise FileNotFoundError(f"{rep_dir} does not exist; available: {available}")


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


def center_mask(frame: pd.DataFrame, insert_segments: list[tuple[float, float]]) -> np.ndarray:
    times = frame["elapsed_s"].to_numpy(dtype=float)
    return np.array(
        [any(start <= t <= start + width for start, width in insert_segments) for t in times],
        dtype=bool,
    )


def plot_period_colored_line(
    ax: plt.Axes,
    frame: pd.DataFrame,
    value_col: str,
    insert_segments: list[tuple[float, float]],
    *,
    insert_color: str,
    label: str,
) -> None:
    x = frame["elapsed_s"].to_numpy(dtype=float)
    values = frame[value_col].to_numpy(dtype=float)
    inside = center_mask(frame, insert_segments)
    ax.plot(x, np.where(inside, np.nan, values), color="#64748b", lw=1.15, alpha=0.78, label="outside")
    ax.plot(x, np.where(inside, values, np.nan), color=insert_color, lw=1.95, label=label)


def load_condition(root: Path, condition_id: str, condition_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for rep_dir in sorted(root.glob(f"{condition_id}_rep*")):
        match = re.search(r"_rep(\d+)$", rep_dir.name)
        rep = int(match.group(1)) if match else len(frames) + 1
        timeline_path = rep_dir / "closed_loop_odin_tail.csv"
        summary_path = rep_dir / "closed_loop_odin_tail_summary.csv"
        if not timeline_path.exists() or not summary_path.exists():
            continue
        frame = pd.read_csv(timeline_path)
        frame["condition"] = condition_label
        frame["rep"] = rep
        frame["in_insert_issue_window"] = bool_mask(frame["in_insert_issue_window"])
        frames.append(frame)

        summary = pd.read_csv(summary_path)
        outside = summary[summary["phase"] == "outside"].iloc[0]
        insert = summary[summary["phase"] == "insert_active"].iloc[0]
        summary_rows.append(
            {
                "condition": condition_label,
                "rep": rep,
                "outside_p99_ms": float(outside["lowq_op_p99_ms"]),
                "insert_p99_ms": float(insert["lowq_op_p99_ms"]),
                "delta_p99_ms": float(insert["lowq_op_p99_ms"]) - float(outside["lowq_op_p99_ms"]),
                "ratio_p99": float(insert["lowq_op_p99_ms"]) / max(float(outside["lowq_op_p99_ms"]), 1e-12),
                "outside_p95_ms": float(outside["lowq_op_p95_ms"]),
                "insert_p95_ms": float(insert["lowq_op_p95_ms"]),
                "insert_traversal_phase_time_per_bin": float(insert["insert_path_workers_mean"]),
                "existing_neighbor_update_phase_time_per_bin": float(insert["existing_update_workers_mean"]),
            }
        )
    if not frames:
        raise FileNotFoundError(f"no rep timelines found for condition {condition_label} in {root}")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summary_rows)


def aggregate(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    grouped = frame.groupby("elapsed_s", as_index=False)[col].agg(
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    )
    return grouped.sort_values("elapsed_s")


def summary_for_rep(summary: pd.DataFrame, condition_label: str, rep: int) -> pd.Series:
    part = summary[(summary["condition"] == condition_label) & (summary["rep"] == rep)]
    if part.empty:
        raise KeyError(f"missing summary for {condition_label} rep{rep}")
    return part.iloc[0]


def median_summary(summary: pd.DataFrame, condition_label: str) -> tuple[float, int]:
    part = summary[summary["condition"] == condition_label]
    return float(part["delta_p99_ms"].median()), int((part["delta_p99_ms"] > 0).sum())


def lookup_condition_label(condition_id: str) -> str:
    for cid, label, _ in CONDITIONS:
        if cid == condition_id:
            return label
    raise KeyError(condition_id)


def shade_periods(ax: plt.Axes, outside: list[tuple[float, float]], insert: list[tuple[float, float]]) -> None:
    for start, width in outside:
        ax.axvspan(start, start + width, color="#e5e7eb", alpha=0.20, lw=0)
    for start, width in insert:
        ax.axvspan(start, start + width, color="#fee2e2", alpha=0.72, lw=0)


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
        inside = any(start <= mid < start + width for start, width in insert_segments)
        value = insert_value if inside else outside_value
        xs.extend([left, right])
        ys.extend([value, value])
    return xs, ys


def bin_step_s(frame: pd.DataFrame) -> float:
    bins_s = frame["bin_ms"].to_numpy(dtype=float) / 1000.0
    if len(bins_s) <= 1:
        return 0.05
    return float(np.median(np.diff(bins_s)))


def clipped_bin_bars(
    frame: pd.DataFrame,
    value_col: str,
    insert_segments: list[tuple[float, float]],
) -> tuple[list[float], list[float], list[float]]:
    """Return bar pieces clipped to the exact insert start-to-commit spans."""
    step = bin_step_s(frame)
    lefts: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    for row in frame.itertuples(index=False):
        height = float(getattr(row, value_col))
        if height <= 0:
            continue
        bin_left = float(row.bin_ms) / 1000.0
        bin_right = bin_left + step
        for seg_start, seg_width in insert_segments:
            seg_right = seg_start + seg_width
            left = max(bin_left, seg_start)
            right = min(bin_right, seg_right)
            if right > left:
                lefts.append(left)
                widths.append(right - left)
                heights.append(height)
    return lefts, widths, heights


def plot_clipped_phase_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    insert_segments: list[tuple[float, float]],
    parts: list[tuple[str, str, str]],
) -> None:
    bottoms: dict[tuple[float, float], float] = {}
    for col, color, label in parts:
        lefts, widths, heights = clipped_bin_bars(frame, col, insert_segments)
        stacked_bottom: list[float] = []
        for left, width in zip(lefts, widths, strict=False):
            key = (round(left, 9), round(width, 9))
            stacked_bottom.append(bottoms.get(key, 0.0))
        ax.bar(lefts, heights, width=widths, bottom=stacked_bottom, align="edge", color=color, alpha=0.82, lw=0, label=label)
        for left, width, bottom, height in zip(lefts, widths, stacked_bottom, heights, strict=False):
            key = (round(left, 9), round(width, 9))
            bottoms[key] = bottom + height


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--timeline-rep", type=int, default=1)
    args = parser.parse_args()

    condition_frames: dict[str, pd.DataFrame] = {}
    summaries: list[pd.DataFrame] = []
    for condition_id, condition_label, _ in CONDITIONS:
        frame, summary = load_condition(args.root, condition_id, condition_label)
        condition_frames[condition_id] = frame
        summaries.append(summary)
    summary_df = pd.concat(summaries, ignore_index=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.csv_out, index=False)

    read_id = "existing_neighbor_update_read_rewrite"
    disabled_id = "existing_neighbor_update_disabled"
    read_label = lookup_condition_label(read_id)
    disabled_label = lookup_condition_label(disabled_id)
    read = condition_frames[read_id][condition_frames[read_id]["rep"] == args.timeline_rep].copy()
    disabled = condition_frames[disabled_id][condition_frames[disabled_id]["rep"] == args.timeline_rep].copy()
    if read.empty or disabled.empty:
        raise KeyError(f"timeline rep{args.timeline_rep} not found for both visual conditions")

    read_insert = active_segments_from_boundaries(find_rep_dir(args.root, read_id, args.timeline_rep))
    disabled_insert = active_segments_from_boundaries(find_rep_dir(args.root, disabled_id, args.timeline_rep))
    horizon_s = max(float(read["elapsed_s"].max() + 0.05), float(disabled["elapsed_s"].max() + 0.05))
    read_outside = outside_segments(read_insert, horizon_s)
    disabled_outside = outside_segments(disabled_insert, horizon_s)
    read_run = summary_for_rep(summary_df, read_label, args.timeline_rep)
    disabled_run = summary_for_rep(summary_df, disabled_label, args.timeline_rep)
    read_median_delta, read_positive = median_summary(summary_df, read_label)
    disabled_median_delta, disabled_positive = median_summary(summary_df, disabled_label)

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
        4,
        1,
        figsize=(10.8, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.45, 1.35, 0.82]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "M3 guarded confirmation: insert-period tail is tied to existing-neighbor update",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )

    for ax in axes[:2]:
        shade_periods(ax, read_outside, read_insert)
    shade_periods(axes[2], disabled_outside, disabled_insert)
    shade_periods(axes[3], read_outside, read_insert)
    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axes[0]
    plot_period_colored_line(
        ax,
        read,
        "lowq_op_p99_ms",
        read_insert,
        insert_color="#dc2626",
        label="insert active",
    )
    ax.set_title("(a) existing-neighbor update present: measured-search p99 timeline", loc="left")
    ax.set_ylabel("p99 latency\n(ms)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.985,
        0.96,
        f"shown run delta: {float(read_run['delta_p99_ms']):+.3f} ms\n"
        f"3-run median delta: {read_median_delta:+.3f} ms\n"
        f"positive runs: {read_positive}/3",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#e5e7eb", "alpha": 0.92},
    )

    ax = axes[1]
    update_parts = [
        ("existing_neighbor_load_scan_active_workers", "#60a5fa", "load/scan"),
        ("existing_neighbor_prune_active_workers", "#f97316", "prune"),
        ("existing_neighbor_undo_record_active_workers", "#a855f7", "undo record"),
        ("existing_neighbor_rewrite_active_workers", "#dc2626", "rewrite/writeback"),
    ]
    plot_clipped_phase_bars(ax, read, read_insert, update_parts)
    ax.set_title("(b) same run: existing-neighbor update phase-time occurs inside the insert active periods", loc="left")
    ax.set_ylabel("existing-neighbor\nupdate phase-time/bin")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.text(
        0.985,
        0.92,
        "red = insert start to commit\nbars use the same run and boundaries",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#e5e7eb", "alpha": 0.92},
    )

    ax = axes[2]
    disabled_parts = [
        ("insert_path_search_active_workers", "#2563eb", "insert traversal"),
    ]
    plot_clipped_phase_bars(ax, disabled, disabled_insert, disabled_parts)
    ax.set_title("(c) existing-neighbor update disabled: insert still runs; update phase-time is zero", loc="left")
    ax.set_ylabel("phase-time\nper bin")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.985,
        0.92,
        f"period p99 delta, shown run: {float(disabled_run['delta_p99_ms']):+.3f} ms\n"
        f"3-run median period delta: {disabled_median_delta:+.3f} ms\n"
        f"positive runs: {disabled_positive}/3\n"
        "update phase-time/bin: 0",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#e5e7eb", "alpha": 0.92},
    )

    ax = axes[3]
    ax.broken_barh(read_outside, (8, 8), facecolors="#94a3b8", alpha=0.78, label="8 competing search threads")
    ax.broken_barh(read_insert, (8, 8), facecolors="#ef4444", alpha=0.72, label="8 competing insert threads")
    ax.broken_barh([(0.0, horizon_s)], (0, 8), facecolors="#111827", alpha=0.84, label="8 measured-search threads")
    ax.set_title("(d) pull-based closed-loop schedule for the shown run", loc="left")
    ax.set_ylabel("threads")
    ax.set_yticks([4, 12], ["measured", "competing"])
    ax.set_ylim(0, 16)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_xlabel("elapsed time (s); red bands are insert active periods, start to commit")
    ax.set_xlim(0, horizon_s)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(args.out)
    print(args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
