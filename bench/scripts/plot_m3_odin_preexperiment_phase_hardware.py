#!/usr/bin/env python3
"""Build the M3 Odin-style search-tail pre-experiment figure.

The plot aligns three signals on the same elapsed-time axis:

1. low-queue measured-search execution latency;
2. real insert phase occupancy from the phase trace bins;
3. perf cache/TLB pressure from interval counters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEARCH_COLS = [
    "measured",
    "finish_ms",
    "op_ms",
    "e2e_ms",
    "diag_dist_comps",
    "diag_hops",
]

PHASE_COLS = [
    "insert_path_search_active_workers",
    "existing_neighbor_update_active_workers",
    "existing_neighbor_prune_active_workers",
    "existing_neighbor_undo_record_active_workers",
    "existing_neighbor_rewrite_active_workers",
]

HW_COLS = [
    "perf_l2_rqsts_miss_per_s",
    "perf_llc_load_misses_per_s",
    "perf_cache_misses_per_s",
    "perf_dtlb_load_misses_walk_completed_per_s",
]


def parse_insert_window(value: str) -> tuple[float, float]:
    """Parse START_MS:DURATION_MS into an inclusive/exclusive ms window."""

    try:
        start_raw, duration_raw = value.split(":", 1)
        start = float(start_raw)
        duration = float(duration_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected START_MS:DURATION_MS, got {value!r}"
        ) from exc
    if duration <= 0:
        raise argparse.ArgumentTypeError("insert window duration must be positive")
    return start, start + duration


def pctl(series: pd.Series, pct: float) -> float:
    if series.empty:
        return float("nan")
    return float(np.percentile(series, pct))


def mark_insert_window(frame: pd.DataFrame, time_col: str, windows: list[tuple[float, float]]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for start, end in windows:
        mask |= (frame[time_col] >= start) & (frame[time_col] < end)
    return mask


def summarize_rows(frame: pd.DataFrame, label: str) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for phase, mask in [("outside", ~frame["in_insert"]), ("insert", frame["in_insert"])]:
        part = frame[mask]
        rows.append(
            {
                "filter": label,
                "phase": phase,
                "n": int(len(part)),
                "op_p50_ms": pctl(part["op_ms"], 50),
                "op_p95_ms": pctl(part["op_ms"], 95),
                "op_p99_ms": pctl(part["op_ms"], 99),
                "queue_p95_ms": pctl(part["queue_ms"], 95),
                "dist_p95": pctl(part["diag_dist_comps"], 95),
                "hops_p95": pctl(part["diag_hops"], 95),
            }
        )
    return rows


def load_search_bins(
    search_wide: Path,
    bin_ms: float,
    windows: list[tuple[float, float]],
    queue_threshold_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    search = pd.read_csv(search_wide, usecols=SEARCH_COLS)
    search = search[search["measured"] == 1].copy()
    search["queue_ms"] = search["e2e_ms"] - search["op_ms"]
    search["bin_ms"] = np.floor(search["finish_ms"] / bin_ms) * bin_ms
    search["in_insert"] = mark_insert_window(search, "finish_ms", windows)

    lowq = search[search["queue_ms"] < queue_threshold_ms].copy()
    bins = (
        lowq.groupby("bin_ms")
        .agg(
            search_count=("op_ms", "size"),
            search_op_p50_ms=("op_ms", lambda s: pctl(s, 50)),
            search_op_p95_ms=("op_ms", lambda s: pctl(s, 95)),
            search_op_p99_ms=("op_ms", lambda s: pctl(s, 99)),
            search_op_mean_ms=("op_ms", "mean"),
            search_queue_p95_ms=("queue_ms", lambda s: pctl(s, 95)),
            search_dist_p95=("diag_dist_comps", lambda s: pctl(s, 95)),
            search_hops_p95=("diag_hops", lambda s: pctl(s, 95)),
        )
        .reset_index()
    )
    summary = pd.DataFrame(
        summarize_rows(search, "all_measured")
        + summarize_rows(lowq, f"queue_lt_{queue_threshold_ms:g}ms")
    )
    return bins, summary


def load_period_bins(period_groups: Path, bin_ms: float) -> pd.DataFrame:
    groups = pd.read_csv(period_groups)
    groups["bin_ms"] = np.floor(groups["bin_start_ms"] / bin_ms) * bin_ms
    return groups.groupby("bin_ms")[PHASE_COLS + HW_COLS].mean().reset_index()


def add_normalized_hw(df: pd.DataFrame) -> None:
    for col in HW_COLS:
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        finite = vals.dropna()
        denom = float(np.nanpercentile(finite, 95)) if len(finite) else 1.0
        if not np.isfinite(denom) or denom <= 0:
            denom = float(np.nanmax(finite)) if len(finite) else 1.0
        if not np.isfinite(denom) or denom <= 0:
            denom = 1.0
        df[f"{col}_norm"] = vals / denom


def write_plot(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    out: Path,
    windows: list[tuple[float, float]],
    title: str,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8.6,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.8, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.18, 1.08, 1.0]},
    )
    ax0, ax1, ax2 = axes

    for ax in axes:
        for idx, (start, end) in enumerate(windows):
            ax.axvspan(
                start / 1000.0,
                end / 1000.0,
                color="#e5e7eb",
                alpha=0.62,
                lw=0,
                label="insert window" if idx == 0 else None,
            )
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    x = df["elapsed_s"]
    ax0.plot(x, df["search_op_p95_ms"], color="#1d4ed8", lw=1.9, label="op p95, low queue")
    ax0.plot(x, df["search_op_mean_ms"], color="#60a5fa", lw=1.1, label="op mean, low queue")
    ax0.plot(
        x,
        df["search_queue_p95_ms"],
        color="#6b7280",
        lw=1.0,
        ls="--",
        label="queue p95 within filtered rows",
    )
    ax0.set_ylabel("latency (ms)")
    ax0.set_title("Measured search execution tail remains after removing queued rows")
    ax0.legend(loc="upper left", ncol=3, frameon=False)

    lowq_label = next(label for label in summary["filter"].unique() if label.startswith("queue_lt_"))
    qsum = summary[summary["filter"] == lowq_label].set_index("phase")
    outside = qsum.loc["outside"]
    inside = qsum.loc["insert"]
    ratio = inside["op_p95_ms"] / outside["op_p95_ms"] if outside["op_p95_ms"] > 0 else np.nan
    annotation = (
        f"outside p95={outside['op_p95_ms']:.3f} ms\n"
        f"insert p95={inside['op_p95_ms']:.3f} ms ({ratio:.1f}x)\n"
        f"queue p95={inside['queue_p95_ms']:.3f} ms"
    )
    ax0.text(
        0.985,
        0.94,
        annotation,
        ha="right",
        va="top",
        transform=ax0.transAxes,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#d1d5db", alpha=0.92),
        fontsize=8.8,
    )

    mutation_labels = [
        ("existing_neighbor_update_active_workers", "existing-neighbor update", "#dc2626"),
        ("existing_neighbor_prune_active_workers", "prune", "#7c3aed"),
        ("existing_neighbor_undo_record_active_workers", "undo record", "#b45309"),
        ("existing_neighbor_rewrite_active_workers", "rewrite", "#374151"),
    ]
    for col, label, color in mutation_labels:
        width = 1.95 if col == "existing_neighbor_update_active_workers" else 1.65
        ax1.plot(x, df[col], lw=width, color=color, label=label)
    ax1.set_ylabel("mutation active workers")
    ax1.set_title("Insert mutation phases align with the same latency windows")

    ax1b = ax1.twinx()
    ax1b.plot(
        x,
        df["insert_path_search_active_workers"],
        color="#2563eb",
        lw=1.15,
        alpha=0.58,
        label="insert path search",
    )
    ax1b.set_ylabel("path-search workers")
    ax1b.spines["top"].set_visible(False)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left", ncol=3, frameon=False)

    hw_labels = [
        ("perf_l2_rqsts_miss_per_s_norm", "L2 miss rate", "#2563eb"),
        ("perf_llc_load_misses_per_s_norm", "LLC load miss rate", "#dc2626"),
        ("perf_cache_misses_per_s_norm", "cache miss rate", "#f97316"),
        ("perf_dtlb_load_misses_walk_completed_per_s_norm", "DTLB walk rate", "#16a34a"),
    ]
    for col, label, color in hw_labels:
        ax2.plot(x, df[col], lw=1.35, color=color, label=label)
    ax2.set_ylabel("normalized rate")
    ax2.set_xlabel("elapsed time (s)")
    ax2.set_title("Cache/TLB pressure co-rises with real insert bursts")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper left", ncol=4, frameon=False)

    fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing period_groups.csv and raw_latency/search_wide.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--bin-ms", type=float, default=100.0)
    parser.add_argument("--queue-threshold-ms", type=float, default=0.1)
    parser.add_argument(
        "--insert-window",
        action="append",
        type=parse_insert_window,
        default=[],
        help="Insert window as START_MS:DURATION_MS. Repeat for multiple bursts.",
    )
    parser.add_argument(
        "--title",
        default="M3 Odin-style pre-experiment: annchor-m1, batch size 20, measured search lanes 8",
    )
    args = parser.parse_args()

    windows = args.insert_window or [
        (3000.0, 4500.0),
        (8000.0, 9500.0),
        (13000.0, 14500.0),
    ]
    search_wide = args.run_dir / "raw_latency" / "search_wide.csv"
    period_groups = args.run_dir / "period_groups.csv"
    if not search_wide.exists():
        raise FileNotFoundError(search_wide)
    if not period_groups.exists():
        raise FileNotFoundError(period_groups)

    search_bins, summary = load_search_bins(search_wide, args.bin_ms, windows, args.queue_threshold_ms)
    period_bins = load_period_bins(period_groups, args.bin_ms)
    df = search_bins.merge(period_bins, on="bin_ms", how="outer").sort_values("bin_ms")
    df["elapsed_s"] = df["bin_ms"] / 1000.0
    df["in_insert_window"] = mark_insert_window(df, "bin_ms", windows)
    add_normalized_hw(df)

    csv_out = args.csv_out or args.out.with_suffix(".csv")
    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.csv")
    df.to_csv(csv_out, index=False)
    summary.to_csv(summary_out, index=False)
    write_plot(df, summary, args.out, windows, args.title)

    print(args.out)
    print(csv_out)
    print(summary_out)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
