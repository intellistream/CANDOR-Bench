#!/usr/bin/env python3
"""Polished OdinANN-style pre-experiment figure for real full insert only."""

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


def segments_from_mask(frame: pd.DataFrame, col: str) -> list[tuple[float, float]]:
    mask = bool_mask(frame[col]).to_numpy()
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


def summary_value(summary: pd.DataFrame, phase: str, col: str) -> float:
    row = summary[summary["phase"] == phase]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][col])


def add_period_shading(ax: plt.Axes, outside: list[tuple[float, float]], insert: list[tuple[float, float]]) -> None:
    for start, width in outside:
        ax.axvspan(start, start + width, color="#f3f4f6", alpha=0.80, lw=0)
    for start, width in insert:
        ax.axvspan(start, start + width, color="#fee2e2", alpha=0.85, lw=0)


def draw_period_latency(
    ax: plt.Axes,
    insert_segments: list[tuple[float, float]],
    horizon_s: float,
    p95_out: float,
    p95_in: float,
    p99_out: float,
    p99_in: float,
) -> None:
    ax.hlines(p99_out, 0, horizon_s, color="#64748b", lw=2.0, label="outside p99")
    ax.hlines(p95_out, 0, horizon_s, color="#94a3b8", lw=1.5, ls="--", label="outside p95")
    first = True
    for start, width in insert_segments:
        label99 = "insert-period p99" if first else None
        label95 = "insert-period p95" if first else None
        ax.hlines(p99_in, start, start + width, color="#dc2626", lw=5.0, label=label99)
        ax.hlines(p95_in, start, start + width, color="#991b1b", lw=2.4, ls="--", label=label95)
        ax.plot([start, start + width], [p99_in, p99_in], color="#dc2626", marker="|", ms=12, lw=0)
        first = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--rep-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    timeline = pd.read_csv(args.timeline)
    timeline["elapsed_s"] = timeline["bin_ms"] / 1000.0
    summary = pd.read_csv(args.summary)
    reps = pd.read_csv(args.rep_summary)
    reps["rep_idx"] = np.arange(1, len(reps) + 1)

    horizon_s = float((timeline["bin_ms"].max() + 50.0) / 1000.0)
    insert_segments = segments_from_mask(timeline, "in_insert_issue_window")
    outside = outside_segments(insert_segments, horizon_s)

    metrics = {
        "amortized_op_p95_ms": (
            summary_value(summary, "outside", "lowq_op_p95_ms"),
            summary_value(summary, "insert_active", "lowq_op_p95_ms"),
        ),
        "amortized_op_p99_ms": (
            summary_value(summary, "outside", "lowq_op_p99_ms"),
            summary_value(summary, "insert_active", "lowq_op_p99_ms"),
        ),
        "queue_p95_ms": (
            summary_value(summary, "outside", "lowq_queue_p95_ms"),
            summary_value(summary, "insert_active", "lowq_queue_p95_ms"),
        ),
        "dist_p95_per_query": (
            float(reps["outside_dist_p95"].median()),
            float(reps["insert_dist_p95"].median()),
        ),
        "hops_p95_per_query": (
            float(reps["outside_hops_p95"].median()),
            float(reps["insert_hops_p95"].median()),
        ),
    }
    rows = []
    for name, (outside_value, insert_value) in metrics.items():
        rows.append(
            {
                "metric": name,
                "outside": outside_value,
                "insert": insert_value,
                "insert_over_outside": insert_value / outside_value if outside_value else np.nan,
                "insert_minus_outside": insert_value - outside_value,
            }
        )
    rows.extend(
        [
            {
                "metric": "batch_wall_p95_est_ms",
                "outside": metrics["amortized_op_p95_ms"][0] * args.batch_size,
                "insert": metrics["amortized_op_p95_ms"][1] * args.batch_size,
                "insert_over_outside": metrics["amortized_op_p95_ms"][1] / metrics["amortized_op_p95_ms"][0],
                "insert_minus_outside": (metrics["amortized_op_p95_ms"][1] - metrics["amortized_op_p95_ms"][0]) * args.batch_size,
            },
            {
                "metric": "batch_wall_p99_est_ms",
                "outside": metrics["amortized_op_p99_ms"][0] * args.batch_size,
                "insert": metrics["amortized_op_p99_ms"][1] * args.batch_size,
                "insert_over_outside": metrics["amortized_op_p99_ms"][1] / metrics["amortized_op_p99_ms"][0],
                "insert_minus_outside": (metrics["amortized_op_p99_ms"][1] - metrics["amortized_op_p99_ms"][0]) * args.batch_size,
            },
            {
                "metric": "replicate_median_delta_amortized_op_p99_ms",
                "outside": float(reps["outside_p99_ms"].median()),
                "insert": float(reps["insert_p99_ms"].median()),
                "insert_over_outside": float((reps["insert_p99_ms"] / reps["outside_p99_ms"]).median()),
                "insert_minus_outside": float(reps["delta_p99_ms"].median()),
            },
        ]
    )
    pd.DataFrame(rows).to_csv(args.summary_out, index=False)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
        }
    )

    fig = plt.figure(figsize=(11.0, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.55, 0.72, 1.35, 1.38])
    ax_lat = fig.add_subplot(gs[0, :])
    ax_sched = fig.add_subplot(gs[1, :], sharex=ax_lat)
    ax_phase = fig.add_subplot(gs[2, :], sharex=ax_lat)
    ax_ctrl = fig.add_subplot(gs[3, 0])
    ax_rep = fig.add_subplot(gs[3, 1])
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "M3 pre-experiment: real inserts raise measured-search tail under fixed workers",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )

    for ax in [ax_lat, ax_sched, ax_phase]:
        add_period_shading(ax, outside, insert_segments)
        ax.set_xlim(0, horizon_s)

    p99_out, p99_in = metrics["amortized_op_p99_ms"]
    p95_out, p95_in = metrics["amortized_op_p95_ms"]
    draw_period_latency(ax_lat, insert_segments, horizon_s, p95_out, p95_in, p99_out, p99_in)
    ax_lat.set_title("(a) period aggregate: measured-search tail rises during real-insert periods", loc="left")
    ax_lat.set_ylabel("amortized per-query op (ms)")
    ax_lat.set_ylim(0, max(p99_in, p99_out) * 1.18)
    ax_lat.legend(frameon=False, ncol=4, loc="upper left")

    ax_sched.broken_barh(outside, (8, 8), facecolors="#94a3b8", alpha=0.78, label="8 bg-search workers")
    ax_sched.broken_barh(insert_segments, (8, 8), facecolors="#ef4444", alpha=0.72, label="8 real-insert workers")
    ax_sched.broken_barh([(0.0, horizon_s)], (0, 8), facecolors="#111827", alpha=0.84, label="8 measured-search workers")
    ax_sched.set_title("(b) fair fixed-worker envelope: competing operation alternates", loc="left")
    ax_sched.set_ylabel("worker slots")
    ax_sched.set_ylim(0, 16)
    ax_sched.set_yticks([4, 12], ["measured", "competing"])
    ax_sched.legend(frameon=False, ncol=3, loc="upper left")

    x = timeline["elapsed_s"].to_numpy()
    phase_parts = [
        ("existing_neighbor_load_scan_active_workers", "#60a5fa", "load/scan"),
        ("existing_neighbor_prune_active_workers", "#f97316", "prune"),
        ("existing_neighbor_undo_record_active_workers", "#a855f7", "undo record"),
        ("existing_neighbor_rewrite_active_workers", "#dc2626", "rewrite/writeback"),
    ]
    bottom = np.zeros(len(timeline))
    for col, color, label in phase_parts:
        vals = timeline[col].fillna(0.0).to_numpy()
        ax_phase.fill_between(x, bottom, bottom + vals, color=color, alpha=0.82, lw=0, label=label)
        bottom += vals
    ax_phase.set_title("(c) real insert exposes existing-neighbor update work on the same periods", loc="left")
    ax_phase.set_ylabel("worker-equiv")
    ax_phase.set_xlabel("elapsed time (s); red bands are real-insert periods")
    ax_phase.set_ylim(bottom=0)
    ax_phase.legend(frameon=False, ncol=4, loc="upper left")

    labels = ["p99 latency", "dist p95", "hops p95"]
    ratios = [
        p99_in / p99_out,
        metrics["dist_p95_per_query"][1] / metrics["dist_p95_per_query"][0],
        metrics["hops_p95_per_query"][1] / metrics["hops_p95_per_query"][0],
    ]
    ax_ctrl.axhline(1.0, color="#6b7280", lw=1.0)
    ax_ctrl.bar(np.arange(len(labels)), ratios, color=["#dc2626", "#2563eb", "#60a5fa"], alpha=0.9)
    ax_ctrl.set_title("(d) amortized latency rises while search traversal work drops", loc="left")
    ax_ctrl.set_ylabel("insert / outside")
    ax_ctrl.set_xticks(np.arange(len(labels)), labels)

    ax_rep.plot(reps["rep_idx"], reps["outside_p99_ms"], marker="o", color="#64748b", lw=1.8, label="outside")
    ax_rep.plot(reps["rep_idx"], reps["insert_p99_ms"], marker="o", color="#dc2626", lw=1.8, label="real insert")
    for row in reps.itertuples(index=False):
        ax_rep.plot([row.rep_idx, row.rep_idx], [row.outside_p99_ms, row.insert_p99_ms], color="#ef4444", alpha=0.35, lw=1.1)
    ax_rep.set_title("(e) reproduced in 5 full-insert-only runs", loc="left")
    ax_rep.set_xlabel("replicate")
    ax_rep.set_ylabel("p99 amortized op (ms)")
    ax_rep.set_xticks(reps["rep_idx"])
    ax_rep.legend(frameon=False, ncol=2)

    for ax in [ax_lat, ax_sched, ax_phase, ax_ctrl, ax_rep]:
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=260, bbox_inches="tight")
    print(args.out)
    print(args.summary_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
