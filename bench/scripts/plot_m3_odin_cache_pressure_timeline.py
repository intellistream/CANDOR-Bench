#!/usr/bin/env python3
"""Diagnostic OdinANN-style M3 cache-pressure timeline from fair-replace bins."""

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
    times = frame["elapsed_s"].to_numpy(dtype=float)
    if len(times) == 0:
        return []
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.05
    segments: list[tuple[float, float]] = []
    start: float | None = None
    last = times[0]
    for t, active in zip(times, mask, strict=False):
        if active and start is None:
            start = float(t - step / 2)
        if not active and start is not None:
            segments.append((start, float(t - step / 2) - start))
            start = None
        last = float(t)
    if start is not None:
        segments.append((start, last + step / 2 - start))
    return [(max(0.0, start), max(0.0, width)) for start, width in segments]


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


def normalize(series: pd.Series) -> pd.Series:
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    denom = float(np.nanpercentile(finite, 95)) if len(finite) else 1.0
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    return series / denom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.timeline)
    if "elapsed_s" not in df:
        df["elapsed_s"] = (df["bin_ms"] + 25.0) / 1000.0
    df = df.sort_values("elapsed_s").copy()
    df["in_insert_window"] = bool_mask(df["in_insert_window"])
    df["llc_norm"] = normalize(df["perf_llc_load_misses_per_s"])
    df["cache_norm"] = normalize(df["perf_cache_misses_per_s"]) if "perf_cache_misses_per_s" in df else np.nan
    df["neighbor_mutation_occupancy"] = df[
        [
            "existing_neighbor_update_active_workers",
            "existing_neighbor_prune_active_workers",
            "existing_neighbor_undo_record_active_workers",
            "existing_neighbor_rewrite_active_workers",
        ]
    ].fillna(0.0).sum(axis=1)

    insert_segments = segments_from_mask(df, "in_insert_window")
    horizon_s = float(df["elapsed_s"].max() + 0.05)
    outside = outside_segments(insert_segments, horizon_s)

    summary_rows = []
    for name, mask in [("background_search_period", ~df["in_insert_window"]), ("real_insert_period", df["in_insert_window"])]:
        part = df[mask]
        summary_rows.append(
            {
                "period": name,
                "bins": int(len(part)),
                "op_p95_median_ms": float(part["lowq_op_p95_ms"].median()),
                "op_p99_median_ms": float(part["lowq_op_p99_ms"].median()),
                "op_p95_mean_ms": float(part["lowq_op_p95_ms"].mean()),
                "op_p99_mean_ms": float(part["lowq_op_p99_ms"].mean()),
                "queue_p95_median_ms": float(part["queue_p95_ms"].median()),
                "llc_miss_rate_median_per_s": float(part["perf_llc_load_misses_per_s"].median()),
                "llc_miss_rate_mean_per_s": float(part["perf_llc_load_misses_per_s"].mean()),
                "insert_traversal_worker_equiv_mean": float(part["insert_path_search_active_workers"].mean()),
                "neighbor_mutation_worker_equiv_mean": float(part["neighbor_mutation_occupancy"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    bg = summary[summary["period"] == "background_search_period"].iloc[0]
    ins = summary[summary["period"] == "real_insert_period"].iloc[0]
    summary["insert_over_bg_op_p99_median"] = ins["op_p99_median_ms"] / bg["op_p99_median_ms"]
    summary["insert_over_bg_llc_median"] = ins["llc_miss_rate_median_per_s"] / bg["llc_miss_rate_median_per_s"]
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.csv_out, index=False)

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
        3,
        1,
        figsize=(11.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 0.78, 1.45]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "M3 diagnostic timeline: older fair-replace run aligns insert with search tail and cache pressure",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )

    for ax in axes:
        for start, width in outside:
            ax.axvspan(start, start + width, color="#f3f4f6", alpha=0.55, lw=0)
        for start, width in insert_segments:
            ax.axvspan(start, start + width, color="#fee2e2", alpha=0.82, lw=0)
        ax.set_xlim(0, horizon_s)
        ax.grid(axis="y", color="#e5e7eb", lw=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    x = df["elapsed_s"].to_numpy()

    ax = axes[0]
    ax.plot(x, df["lowq_op_p99_ms"], color="#dc2626", lw=1.6, label="measured-search op p99")
    ax.plot(x, df["lowq_op_p95_ms"], color="#f97316", lw=1.35, label="measured-search op p95")
    ax.set_title("(a) real 200ms sliding-window low-queue op latency; no period-aggregate step function", loc="left")
    ax.set_ylabel("op latency (ms)")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.text(
        0.985,
        0.88,
        f"median p99: insert/bg = {ins['op_p99_median_ms'] / bg['op_p99_median_ms']:.2f}x",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#7f1d1d",
    )

    ax = axes[1]
    measured_threads = float(df["measured_thread_budget"].dropna().median()) if "measured_thread_budget" in df else 8.0
    competing_threads = float(
        max(df["background_search_thread_budget"].dropna().max(), df["insert_thread_budget"].dropna().max())
    )
    ax.broken_barh(outside, (measured_threads, competing_threads), facecolors="#94a3b8", alpha=0.78, label="background-search competing slots")
    ax.broken_barh(insert_segments, (measured_threads, competing_threads), facecolors="#ef4444", alpha=0.72, label="real-insert competing slots")
    ax.broken_barh([(0.0, horizon_s)], (0, measured_threads), facecolors="#111827", alpha=0.84, label="fixed measured-search slots")
    ax.set_title("(b) OdinANN-style period substitution: measured search fixed, competing operation changes", loc="left")
    ax.set_ylabel("thread slots")
    ax.set_ylim(0, measured_threads + competing_threads)
    ax.set_yticks([measured_threads / 2, measured_threads + competing_threads / 2], ["measured", "competing"])
    ax.legend(frameon=False, ncol=3, loc="upper left")

    ax = axes[2]
    ax.plot(x, df["insert_path_search_active_workers"], color="#2563eb", lw=1.35, label="insert traversal occupancy")
    ax.plot(x, df["neighbor_mutation_occupancy"], color="#dc2626", lw=1.35, label="neighbor mutation occupancy")
    ax.set_title("(c) insert work and LLC pressure rise in the same red periods", loc="left")
    ax.set_ylabel("worker-equiv\noccupancy")
    ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.plot(x, df["llc_norm"], color="#111827", lw=1.1, alpha=0.72, label="LLC load misses normalized")
    ax2.fill_between(x, 0.0, df["llc_norm"], color="#111827", alpha=0.10, linewidth=0)
    ax2.set_ylabel("LLC miss norm")
    ax2.spines["top"].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, ncol=3, loc="upper left")
    ax.set_xlabel("elapsed time (s); red bands are real graph-mutating insert periods")
    ax.text(
        0.985,
        0.90,
        f"median LLC: insert/bg = {ins['llc_miss_rate_median_per_s'] / bg['llc_miss_rate_median_per_s']:.2f}x",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#111827",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=260, bbox_inches="tight")
    print(args.out)
    print(args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
