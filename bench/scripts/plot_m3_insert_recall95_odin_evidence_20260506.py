#!/usr/bin/env python3
"""OdinANN-style evidence timeline for the recall-95 insert probe."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def percentile(series: pd.Series, pct: float) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return math.nan
    return float(np.percentile(values.to_numpy(), pct))


def load_search(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "raw_latency" / "search_wide.csv")
    frame = frame[frame["measured"] == 1].copy()
    frame["mid_ms"] = frame["finish_ms"] - (frame["finish_raw_ns"] - (frame["start_raw_ns"] + frame["finish_raw_ns"]) / 2.0) / 1_000_000.0
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


def bin_search(frame: pd.DataFrame, bin_ms: float, max_ms: float) -> pd.DataFrame:
    bins = np.arange(0.0, max_ms + bin_ms, bin_ms)
    frame = frame.copy()
    frame["bin"] = pd.cut(frame["mid_ms"], bins=bins, include_lowest=True, right=False)
    rows: list[dict[str, float]] = []
    for interval, group in frame.groupby("bin", observed=True):
        if group.empty:
            continue
        rows.append(
            {
                "time_ms": float(interval.left + (interval.right - interval.left) / 2.0),
                "search_work_p99_ms": percentile(group["work_searchknn_ms"], 99),
                "search_cpu_p99_ms": percentile(group["work_searchknn_thread_cpu_ms"], 99),
                "distance_computations_mean": float(pd.to_numeric(group["work_distance_computations"], errors="coerce").mean()),
                "level0_edges_mean": float(pd.to_numeric(group["work_level0_edges_scanned"], errors="coerce").mean()),
                "rows": float(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        for col in ["search_work_p99_ms", "search_cpu_p99_ms"]:
            out[f"{col}_smooth"] = out[col].rolling(3, center=True, min_periods=1).median()
    return out


def load_periods(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "period_groups.csv")
    frame["time_ms"] = (frame["bin_start_ms"] + frame["bin_end_ms"]) / 2.0
    slots = pd.to_numeric(frame["perf_topdown_slots"], errors="coerce").replace(0, np.nan)
    cycles = pd.to_numeric(frame["perf_cycles"], errors="coerce").replace(0, np.nan)
    instr = pd.to_numeric(frame["perf_instructions"], errors="coerce")
    frame["ipc"] = instr / cycles
    frame["memory_bound_frac"] = pd.to_numeric(frame["perf_topdown_mem_bound"], errors="coerce") / slots
    frame["backend_bound_frac"] = pd.to_numeric(frame["perf_topdown_be_bound"], errors="coerce") / slots
    return frame


def shade_insert(ax: plt.Axes, windows: list[tuple[float, float]]) -> None:
    for start, end in windows:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#f6b6b6", alpha=0.38, lw=0)


def plot(args: argparse.Namespace) -> None:
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    bg_search = bin_search(load_search(args.background), args.bin_ms, args.max_ms)
    insert_search = bin_search(load_search(args.real_insert), args.bin_ms, args.max_ms)
    bg_period = load_periods(args.background)
    insert_period = load_periods(args.real_insert)
    windows = load_insert_windows(args.real_insert)

    merged = (
        insert_search.add_prefix("insert_")
        .merge(bg_search.add_prefix("background_"), left_on="insert_time_ms", right_on="background_time_ms", how="outer")
        .rename(columns={"insert_time_ms": "time_ms"})
    )
    merged.to_csv(out.with_suffix(".binned.csv"), index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(4, 1, figsize=(12.0, 8.2), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")

    for ax in axes:
        shade_insert(ax, windows)
        ax.grid(True, axis="y", color="#e9edf2", linewidth=0.8)
        ax.set_xlim(0, args.max_ms / 1000.0)

    x_bg = bg_search["time_ms"] / 1000.0
    x_ins = insert_search["time_ms"] / 1000.0
    axes[0].plot(x_bg, bg_search["search_work_p99_ms_smooth"], color="#6b7280", lw=1.8, label="Pure-search background")
    axes[0].plot(x_ins, insert_search["search_work_p99_ms_smooth"], color="#1f77b4", lw=2.0, label="Real insert")
    axes[0].set_ylabel("C++ HNSW\np99 (ms)")
    axes[0].legend(loc="upper right", ncol=2, frameon=False)

    axes[1].plot(
        insert_period["time_ms"] / 1000.0,
        insert_period["insert_path_search_active_workers"] + insert_period["existing_neighbor_update_active_workers"],
        color="#b91c1c",
        lw=1.8,
        label="Insert active threads",
    )
    axes[1].set_ylabel("Insert active\nthreads")
    axes[1].legend(loc="upper right", frameon=False)

    axes[2].plot(
        bg_period["time_ms"] / 1000.0,
        bg_period["perf_l2_rqsts_rfo_miss_per_s"] / 1e6,
        color="#9ca3af",
        lw=1.5,
        label="RFO miss/s, pure-search",
    )
    axes[2].plot(
        insert_period["time_ms"] / 1000.0,
        insert_period["perf_l2_rqsts_rfo_miss_per_s"] / 1e6,
        color="#dc2626",
        lw=1.8,
        label="RFO miss/s, real insert",
    )
    axes[2].set_ylabel("RFO misses\n(M/s)")
    axes[2].legend(loc="upper right", ncol=2, frameon=False)

    axes[3].plot(
        bg_period["time_ms"] / 1000.0,
        bg_period["perf_llc_load_misses_per_s"] / 1e6,
        color="#6b7280",
        lw=1.5,
        label="LLC load miss/s, pure-search",
    )
    axes[3].plot(
        insert_period["time_ms"] / 1000.0,
        insert_period["perf_llc_load_misses_per_s"] / 1e6,
        color="#2563eb",
        lw=1.7,
        label="LLC load miss/s, real insert",
    )
    ax2 = axes[3].twinx()
    shade_insert(ax2, windows)
    ax2.plot(
        bg_period["time_ms"] / 1000.0,
        bg_period["ipc"],
        color="#111827",
        lw=1.2,
        ls="--",
        label="IPC, pure-search",
    )
    ax2.plot(
        insert_period["time_ms"] / 1000.0,
        insert_period["ipc"],
        color="#059669",
        lw=1.4,
        ls="--",
        label="IPC, real insert",
    )
    axes[3].set_ylabel("LLC load\nmisses (M/s)")
    ax2.set_ylabel("IPC")
    lines, labels = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[3].legend(lines + lines2, labels + labels2, loc="upper right", ncol=2, frameon=False)
    axes[3].set_xlabel("Elapsed time (s)")

    fig.suptitle("Recall-95 SIFT, 48 Threads: Insert Period and Search/HW Signals", x=0.01, ha="left", y=1.02)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    if args.pdf:
        fig.savefig(args.pdf, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--real-insert", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--bin-ms", type=float, default=100.0)
    parser.add_argument("--max-ms", type=float, default=5200.0)
    args = parser.parse_args()
    plot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
