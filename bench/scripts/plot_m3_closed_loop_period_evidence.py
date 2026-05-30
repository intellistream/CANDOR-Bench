#!/usr/bin/env python3
"""Plot the fair closed-loop period evidence for M3 search-tail root cause."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASES = [
    ("full", "full update", "#dc2626"),
    ("skip_undo", "skip undo", "#f97316"),
    ("update_readonly", "read-only update", "#2563eb"),
    ("skip_update", "skip update", "#16a34a"),
]


def read_case(root: Path, key: str) -> pd.DataFrame:
    path = root / f"closed_loop_odin_{key}_tail.csv"
    frame = pd.read_csv(path)
    frame["elapsed_s"] = frame["bin_ms"] / 1000.0
    return frame


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


def phase_write_workers(frame: pd.DataFrame) -> pd.Series:
    cols = [
        "existing_neighbor_prune_active_workers",
        "existing_neighbor_undo_record_active_workers",
        "existing_neighbor_rewrite_active_workers",
    ]
    total = pd.Series(0.0, index=frame.index)
    for col in cols:
        if col in frame:
            total += frame[col].fillna(0.0)
    return total


def read_summary(root: Path, key: str) -> pd.DataFrame:
    return pd.read_csv(root / f"closed_loop_odin_{key}_tail_summary.csv")


def summary_value(summary: pd.DataFrame, phase: str, col: str) -> float:
    row = summary[summary["phase"] == phase]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][col])


def summary_row(root: Path, key: str, label: str, frame: pd.DataFrame) -> dict[str, float | str]:
    case_summary = read_summary(root, key)
    insert = bool_mask(frame["in_insert_active_window"])
    outside_p99 = summary_value(case_summary, "outside", "lowq_op_p99_ms")
    insert_p99 = summary_value(case_summary, "insert_active", "lowq_op_p99_ms")
    return {
        "case": key,
        "label": label,
        "outside_lowq_p99_ms": outside_p99,
        "insert_lowq_p99_ms": insert_p99,
        "delta_lowq_p99_ms": insert_p99 - outside_p99,
        "insert_write_phase_workers_mean": float(phase_write_workers(frame.loc[insert]).mean()),
        "insert_existing_update_workers_mean": float(
            frame.loc[insert, "existing_neighbor_update_active_workers"].mean()
        ),
        "insert_llc_load_misses_per_s_mean": float(
            frame.loc[insert, "perf_llc_load_misses_per_s"].mean()
        ),
        "queue_p95_max_ms": float(frame["queue_p95_ms"].max()),
        "measured_workers_median": float(frame["measured_search_active_workers"].median()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    frames = {key: read_case(args.root, key) for key, _, _ in CASES}
    full = frames["full"]
    horizon_s = float((full["bin_ms"].max() + 50.0) / 1000.0)
    issue_segments = segments_from_mask(full, "in_insert_issue_window")
    outside = outside_segments(issue_segments, horizon_s)

    rows = [summary_row(args.root, key, label, frames[key]) for key, label, _ in CASES]
    summary = pd.DataFrame(rows)
    csv_out = args.csv_out or args.out.with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_out, index=False)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
        }
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.6, 8.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.25, 1.5, 1.35]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    for ax in axes:
        for start, width in outside:
            ax.axvspan(start, start + width, color="#e5e7eb", alpha=0.25, lw=0)
        for start, width in issue_segments:
            ax.axvspan(start, start + width, color="#fee2e2", alpha=0.75, lw=0)
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axes[0]
    for key, label, color in CASES:
        case_summary = read_summary(args.root, key)
        outside_p99 = summary_value(case_summary, "outside", "lowq_op_p99_ms")
        insert_p99 = summary_value(case_summary, "insert_active", "lowq_op_p99_ms")
        xs: list[float] = []
        ys: list[float] = []
        boundaries = [0.0]
        for start, width in issue_segments:
            boundaries.extend([start, start + width])
        boundaries.append(horizon_s)
        boundaries = sorted(set(boundaries))
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=False):
            mid = (left + right) / 2.0
            inside = any(start <= mid < start + width for start, width in issue_segments)
            value = insert_p99 if inside else outside_p99
            xs.extend([left, right])
            ys.extend([value, value])
        ax.plot(xs, ys, color=color, lw=2.2, label=label)
    ax.set_title("(a) period-level measured-search p99 under identical fixed-worker schedule", loc="left")
    ax.set_ylabel("period p99 op (ms)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=4, loc="upper left")

    ax = axes[1]
    ax.broken_barh(outside, (8, 8), facecolors="#94a3b8", alpha=0.78, label="8 bg-search workers")
    ax.broken_barh(issue_segments, (8, 8), facecolors="#ef4444", alpha=0.72, label="8 insert workers")
    ax.broken_barh([(0.0, horizon_s)], (0, 8), facecolors="#111827", alpha=0.82, label="8 measured-search workers")
    ax.set_title("(b) fairness: fixed closed-loop worker capacity, only competing period changes", loc="left")
    ax.set_ylabel("worker slots")
    ax.set_yticks([4, 12], ["measured", "competing"])
    ax.set_ylim(0, 16)
    ax.legend(frameon=False, ncol=3, loc="upper left")

    ax = axes[2]
    for key, label, color in CASES:
        frame = frames[key]
        ax.plot(
            frame["elapsed_s"],
            phase_write_workers(frame),
            color=color,
            lw=1.6,
            label=label,
        )
    ax.set_title("(c) write-side graph mutation occupancy: prune + undo + rewrite", loc="left")
    ax.set_ylabel("worker-equiv")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=4, loc="upper left")

    ax = axes[3]
    for key, label, color in [
        ("full", "full update", "#dc2626"),
        ("skip_update", "skip update", "#16a34a"),
        ("update_readonly", "read-only update", "#2563eb"),
    ]:
        frame = frames[key]
        ax.plot(
            frame["elapsed_s"],
            frame["perf_llc_load_misses_per_s"] / 1e6,
            color=color,
            lw=1.35,
            label=label,
        )
    ax.set_title("(d) LLC is not sufficient: skip-update can have high LLC without p99 tail", loc="left")
    ax.set_ylabel("LLC misses/s (M)")
    ax.set_xlabel("elapsed time (s); red bands are insert periods")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=3, loc="upper left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(args.out)
    print(csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
