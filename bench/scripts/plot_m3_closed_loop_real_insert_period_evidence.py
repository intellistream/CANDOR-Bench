#!/usr/bin/env python3
"""Plot fair period evidence using only unmodified real insert."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bool_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def segments_from_mask(frame: pd.DataFrame, mask_col: str) -> list[tuple[float, float]]:
    mask = bool_mask(frame[mask_col]).to_numpy()
    bins = frame["bin_ms"].to_numpy(dtype=float)
    if len(bins) == 0:
        return []
    step = float(np.median(np.diff(bins))) if len(bins) > 1 else 50.0
    segments: list[tuple[float, float]] = []
    start: float | None = None
    last = bins[0]
    for t, active in zip(bins, mask, strict=False):
        if active and start is None:
            start = float(t)
        if not active and start is not None:
            segments.append((start / 1000.0, (float(t) - start) / 1000.0))
            start = None
        last = float(t)
    if start is not None:
        segments.append((start / 1000.0, (last + step - start) / 1000.0))
    return segments


def active_segments_from_boundaries(path: Path) -> list[tuple[float, float]]:
    boundaries = pd.read_csv(path)
    segments: list[tuple[float, float]] = []
    for row in boundaries.itertuples(index=False):
        start_s = float(row.active_start_ms) / 1000.0
        end_s = float(row.active_end_ms) / 1000.0
        if end_s > start_s:
            segments.append((start_s, end_s - start_s))
    return segments


def default_boundaries_for_timeline(timeline: Path) -> Path:
    stem_specific = timeline.with_name(f"{timeline.stem}_insert_boundaries.csv")
    if stem_specific.exists():
        return stem_specific
    return timeline.with_name("closed_loop_odin_tail_insert_boundaries.csv")


def outside_segments(insert_segments: list[tuple[float, float]], horizon_s: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for start, width in insert_segments:
        if start > cursor:
            out.append((cursor, start - cursor))
        cursor = max(cursor, start + width)
    if cursor < horizon_s:
        out.append((cursor, horizon_s - cursor))
    return out


def center_mask(frame: pd.DataFrame, insert_segments: list[tuple[float, float]]) -> np.ndarray:
    times = frame["elapsed_s"].to_numpy(dtype=float)
    return np.array(
        [any(start <= t <= start + width for start, width in insert_segments) for t in times],
        dtype=bool,
    )


def plot_period_colored_line(
    ax: plt.Axes,
    frame: pd.DataFrame,
    values: np.ndarray,
    insert_segments: list[tuple[float, float]],
    *,
    outside_color: str = "#64748b",
    insert_color: str = "#dc2626",
    label: str = "measured-search p99 timeline",
) -> np.ndarray:
    x = frame["elapsed_s"].to_numpy(dtype=float)
    inside = center_mask(frame, insert_segments)
    ax.plot(x, np.where(inside, np.nan, values), color=outside_color, lw=1.25, alpha=0.82, label="No Insert")
    ax.plot(x, np.where(inside, values, np.nan), color=insert_color, lw=2.0, label=label)
    return inside


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
    """Return bin bars clipped to exact insert start-to-commit spans."""
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


def normalized_signal(series: pd.Series) -> pd.Series:
    vals = series.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    positive = vals[vals > 0]
    if positive.empty:
        return vals
    denom = float(np.percentile(positive, 95))
    if not np.isfinite(denom) or denom <= 0:
        denom = float(positive.max())
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    return vals / denom


def mean_if_present(frame: pd.DataFrame, mask: np.ndarray, col: str) -> float:
    if col not in frame.columns:
        return float("nan")
    return float(frame.loc[mask, col].mean())


def summary_value(summary: pd.DataFrame, phase: str, col: str) -> float:
    row = summary[summary["phase"] == phase]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][col])


def period_step(
    issue_segments: list[tuple[float, float]],
    horizon_s: float,
    outside_value: float,
    insert_value: float,
) -> tuple[list[float], list[float]]:
    boundaries = [0.0, horizon_s]
    for start, width in issue_segments:
        boundaries.extend([start, start + width])
    boundaries = sorted(set(boundaries))
    xs: list[float] = []
    ys: list[float] = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=False):
        mid = (left + right) / 2.0
        inside = any(start <= mid < start + width for start, width in issue_segments)
        value = insert_value if inside else outside_value
        xs.extend([left, right])
        ys.extend([value, value])
    return xs, ys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--latency-window-ms", type=float, default=200.0)
    parser.add_argument("--sensitivity-runs", type=Path)
    parser.add_argument("--sensitivity-windows", type=Path)
    args = parser.parse_args()

    timeline = pd.read_csv(args.timeline)
    if "elapsed_s" not in timeline.columns:
        bins = timeline["bin_ms"].to_numpy(dtype=float)
        step_ms = float(np.median(np.diff(bins))) if len(bins) > 1 else 50.0
        timeline["elapsed_s"] = (timeline["bin_ms"] + step_ms / 2.0) / 1000.0
    summary = pd.read_csv(args.summary)
    overlap = pd.read_csv(args.overlap)

    step_ms = bin_step_s(timeline) * 1000.0
    horizon_s = float((timeline["bin_ms"].max() + step_ms) / 1000.0)
    boundaries = args.boundaries or default_boundaries_for_timeline(args.timeline)
    issue_segments = active_segments_from_boundaries(boundaries)
    outside = outside_segments(issue_segments, horizon_s)

    outside_p99 = summary_value(summary, "outside", "lowq_op_p99_ms")
    insert_p99 = summary_value(summary, "insert_active", "lowq_op_p99_ms")
    outside_p95 = summary_value(summary, "outside", "lowq_op_p95_ms")
    insert_p95 = summary_value(summary, "insert_active", "lowq_op_p95_ms")
    queue_p95 = summary_value(summary, "insert_active", "lowq_queue_p95_ms")
    outside_llc = summary_value(summary, "outside", "llc_load_misses_per_s_mean")
    insert_llc = summary_value(summary, "insert_active", "llc_load_misses_per_s_mean")
    inside_for_stats = center_mask(timeline, issue_segments)

    memory_cols = {
        "l3_miss_stall_cycles_per_s": "perf_cycle_activity_stalls_l3_miss_per_s",
        "demand_rfo_cycles_per_s": "perf_offcore_requests_outstanding_cycles_with_demand_rfo_per_s",
        "snoop_hitm_per_s": "perf_ocr_demand_data_rd_l3_hit_snoop_hitm_per_s",
    }

    out_row = overlap[overlap["group"] == "outside_all"].iloc[0]
    ins_row = overlap[overlap["group"] == "insert_all"].iloc[0]
    evidence_rows = [
        {
            "phase": "outside",
            "lowq_p95_ms": outside_p95,
            "lowq_p99_ms": outside_p99,
            "lowq_queue_p95_ms": queue_p95,
            "dist_p95": float(out_row["dist_p95"]),
            "hops_p95": float(out_row["hops_p95"]),
            "llc_load_misses_per_s": outside_llc,
        },
        {
            "phase": "insert_active",
            "lowq_p95_ms": insert_p95,
            "lowq_p99_ms": insert_p99,
            "lowq_queue_p95_ms": queue_p95,
            "dist_p95": float(ins_row["dist_p95"]),
            "hops_p95": float(ins_row["hops_p95"]),
            "llc_load_misses_per_s": insert_llc,
        },
    ]
    for out in evidence_rows:
        mask = ~inside_for_stats if out["phase"] == "outside" else inside_for_stats
        for out_col, source_col in memory_cols.items():
            out[out_col] = mean_if_present(timeline, mask, source_col)
    evidence = pd.DataFrame(evidence_rows)
    csv_out = args.csv_out or args.out.with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(csv_out, index=False)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.8, 7.8),
        gridspec_kw={"height_ratios": [1.05, 1.55, 1.65]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    for ax in axes[:2]:
        for start, width in outside:
            ax.axvspan(start, start + width, color="#e5e7eb", alpha=0.25, lw=0)
        for start, width in issue_segments:
            ax.axvspan(start, start + width, color="#fee2e2", alpha=0.75, lw=0)
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axes[0]
    measured_lane = (1.0, 0.72)
    corun_lane = (2.05, 0.72)
    ax.broken_barh([(0.0, horizon_s)], measured_lane, facecolors="#64748b", alpha=0.62)
    ax.broken_barh(outside, corun_lane, facecolors="#cbd5e1", alpha=0.82)
    ax.broken_barh(issue_segments, corun_lane, facecolors="#fca5a5", alpha=0.82)
    ax.text(
        horizon_s * 0.015,
        measured_lane[0] + measured_lane[1] / 2,
        "Measured Search: 8 Threads",
        color="#0f172a",
        va="center",
        ha="left",
        fontsize=11,
    )
    for start, width in outside:
        if width >= 0.7:
            ax.text(start + width / 2, corun_lane[0] + corun_lane[1] / 2, "Search", color="#334155", va="center", ha="center", fontsize=11)
    for start, width in issue_segments:
        if width >= 0.7:
            ax.text(start + width / 2, corun_lane[0] + corun_lane[1] / 2, "Insert", color="#7f1d1d", va="center", ha="center", fontsize=11)
    ax.set_title("(a) Fixed-Thread Streaming Schedule", loc="left")
    ax.set_ylabel("Threads", fontsize=13)
    ax.set_yticks(
        [measured_lane[0] + measured_lane[1] / 2, corun_lane[0] + corun_lane[1] / 2],
        ["Measured", "Co-run"],
    )
    ax.set_ylim(0.65, 3.05)

    ax = axes[1]
    x = timeline["elapsed_s"].to_numpy()
    p99 = timeline["lowq_op_p99_ms"].to_numpy()
    inside_bins = plot_period_colored_line(ax, timeline, p99, issue_segments, label="Insert Active")
    ax.set_title(f"(b) Measured-Search Execution Latency: P99 of op_ms over {args.latency_window_ms:.0f} ms windows", loc="left")
    ax.set_ylabel("Search op_ms\nP99 (ms)")
    ax.set_xlim(0, horizon_s)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)
    ax.text(
        0.985,
        0.92,
        f"No Insert: {outside_p99:.3f} ms\nInsert Active: {insert_p99:.3f} ms\n"
        f"Delta: {insert_p99 - outside_p99:+.3f} ms",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#e5e7eb", "alpha": 0.92},
    )
    ax.set_xlabel("Elapsed Time (s); Red Bands = Insert Active, Start To Commit", fontsize=13)

    ax = axes[2]
    if "lowq_batch_search_arena_p99" in timeline.columns:
        component = timeline["lowq_batch_search_arena_p99"].to_numpy()
        plot_period_colored_line(
            ax,
            timeline,
            component,
            issue_segments,
            outside_color="#64748b",
            insert_color="#dc2626",
            label="Insert Active",
        )
        if "lowq_batch_search_knn_p99" in timeline.columns:
            ax.plot(
                x,
                timeline["lowq_batch_search_knn_p99"].to_numpy(),
                color="#2563eb",
                lw=1.05,
                alpha=0.55,
                label="SearchKNN P99",
            )
        ax.set_title("(c) Search Execution Component: BatchSearch Wall Time", loc="left")
        ax.set_ylabel("Component P99\n(ms)")
        ax.set_xlim(0, horizon_s)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)
    elif args.sensitivity_runs and args.sensitivity_windows:
        runs = pd.read_csv(args.sensitivity_runs)
        windows = pd.read_csv(args.sensitivity_windows)
        relation = runs.merge(windows[["efc", "rep", "batches_per_active_s"]], on=["efc", "rep"], how="inner")
        colors_by_efc = {35: "#dc2626", 100: "#f59e0b", 200: "#2563eb"}
        for efc, part in relation.groupby("efc"):
            ax.scatter(
                part["batches_per_active_s"],
                part["ratio_p99"],
                s=62,
                color=colors_by_efc.get(int(efc), "#64748b"),
                alpha=0.78,
                label=f"efc={int(efc)}",
            )
        ax.axhline(1.0, color="#64748b", lw=1.0, ls="--")
        ax.set_title("(c) Across Runs: Insert Density vs Search Tail", loc="left")
        ax.set_xlabel("Insert Batches / Active s", fontsize=13)
        ax.set_ylabel("Insert / No-Insert\nSearch P99", fontsize=13)
        ax.legend(frameon=False, ncol=3, loc="upper left")
    else:
        outside_evidence = evidence[evidence["phase"] == "outside"].iloc[0]
        insert_evidence = evidence[evidence["phase"] == "insert_active"].iloc[0]
        ratio_items = [
            ("Search\nP99", insert_evidence["lowq_p99_ms"] / outside_evidence["lowq_p99_ms"], "#dc2626"),
        ]
        if np.isfinite(outside_evidence.get("demand_rfo_cycles_per_s", float("nan"))) and outside_evidence["demand_rfo_cycles_per_s"] > 0:
            ratio_items.append(
                (
                    "Demand\nRFO",
                    insert_evidence["demand_rfo_cycles_per_s"] / outside_evidence["demand_rfo_cycles_per_s"],
                    "#b45309",
                )
            )
        if np.isfinite(outside_evidence.get("snoop_hitm_per_s", float("nan"))) and outside_evidence["snoop_hitm_per_s"] > 0:
            ratio_items.append(
                (
                    "Snoop\nHITM",
                    insert_evidence["snoop_hitm_per_s"] / outside_evidence["snoop_hitm_per_s"],
                    "#7c3aed",
                )
            )
        if np.isfinite(outside_evidence.get("l3_miss_stall_cycles_per_s", float("nan"))) and outside_evidence["l3_miss_stall_cycles_per_s"] > 0:
            ratio_items.append(
                (
                    "L3-Miss\nStall",
                    insert_evidence["l3_miss_stall_cycles_per_s"] / outside_evidence["l3_miss_stall_cycles_per_s"],
                    "#64748b",
                )
            )

        labels = [item[0] for item in ratio_items]
        values = [float(item[1]) for item in ratio_items]
        colors = [item[2] for item in ratio_items]
        xpos = np.arange(len(ratio_items))
        ax.bar(xpos, values, color=colors, alpha=0.78, width=0.58)
        ax.axhline(1.0, color="#64748b", lw=1.0, ls="--")
        for pos, value in zip(xpos, values, strict=False):
            ax.text(pos, value + max(values) * 0.04, f"{value:.2f}x", ha="center", va="bottom", fontsize=10, color="#0f172a")
        ax.set_title("(c) Phase-Level Ratio", loc="left")
        ax.set_ylabel("Insert / No-Insert\nRatio", fontsize=13)
        ax.set_xticks(xpos, labels)
        ax.set_ylim(0, max(values + [1.0]) * 1.24)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(args.out)
    print(csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
