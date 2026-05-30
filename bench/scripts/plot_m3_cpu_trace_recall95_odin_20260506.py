#!/usr/bin/env python3
"""OdinANN-style CPU-placement evidence for the recall-95 insert probe."""

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


def socket_id(cpu: float) -> int:
    try:
        value = int(cpu)
    except (TypeError, ValueError):
        return -1
    if 0 <= value <= 23 or 48 <= value <= 71:
        return 0
    if 24 <= value <= 47 or 72 <= value <= 95:
        return 1
    return -1


def load_search(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "raw_latency" / "search_wide.csv")
    frame = frame[frame["measured"] == 1].copy()
    frame["mid_ms"] = frame["finish_ms"] - (
        frame["finish_raw_ns"] - (frame["start_raw_ns"] + frame["finish_raw_ns"]) / 2.0
    ) / 1_000_000.0
    frame["work_socket"] = frame["work_start_cpu"].map(socket_id)
    frame["work_migrated"] = frame["work_start_cpu"] != frame["work_end_cpu"]
    return frame


def load_insert_windows(root: Path, gap_ms: float = 10.0) -> list[tuple[float, float]]:
    path = root / "raw_latency" / "insert_wide.csv"
    if not path.exists():
        return []
    inserts = pd.read_csv(path)
    if inserts.empty:
        return []
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


def first_window(windows: list[tuple[float, float]], warmup_ms: float) -> tuple[float, float]:
    if not windows:
        return warmup_ms, math.inf
    start, end = windows[0]
    return max(start, warmup_ms), end


def in_window(frame: pd.DataFrame, window: tuple[float, float], col: str = "mid_ms") -> pd.DataFrame:
    start, end = window
    return frame[(frame[col] >= start) & (frame[col] <= end)].copy()


def bin_search(frame: pd.DataFrame, bin_ms: float, max_ms: float) -> pd.DataFrame:
    bins = np.arange(0.0, max_ms + bin_ms, bin_ms)
    frame = frame.copy()
    frame["bin"] = pd.cut(frame["mid_ms"], bins=bins, include_lowest=True, right=False)
    rows: list[dict[str, float]] = []
    for interval, group in frame.groupby("bin", observed=True):
        if group.empty:
            continue
        socket1 = group[group["work_socket"] == 1]
        rows.append(
            {
                "time_ms": float(interval.left + (interval.right - interval.left) / 2.0),
                "work_searchknn_p99_ms": percentile(group["work_searchknn_ms"], 99),
                "work_searchknn_thread_cpu_p99_ms": percentile(
                    group["work_searchknn_thread_cpu_ms"], 99
                ),
                "socket1_pct": 100.0 * len(socket1) / len(group),
                "migration_pct": 100.0 * float(group["work_migrated"].mean()),
                "rows": float(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["work_searchknn_p99_ms_smooth"] = (
            out["work_searchknn_p99_ms"].rolling(3, center=True, min_periods=1).median()
        )
        out["socket1_pct_smooth"] = out["socket1_pct"].rolling(3, center=True, min_periods=1).median()
    return out


def summarize_search(frame: pd.DataFrame, case: str, window: tuple[float, float]) -> dict[str, float | str]:
    matched = in_window(frame, window)
    p99 = percentile(matched["work_searchknn_ms"], 99)
    slow = matched[matched["work_searchknn_ms"] >= p99]
    return {
        "case": case,
        "rows": float(len(matched)),
        "work_searchknn_ms_mean": float(pd.to_numeric(matched["work_searchknn_ms"]).mean()),
        "work_searchknn_ms_p95": percentile(matched["work_searchknn_ms"], 95),
        "work_searchknn_ms_p99": p99,
        "work_searchknn_thread_cpu_ms_p99": percentile(
            matched["work_searchknn_thread_cpu_ms"], 99
        ),
        "op_ms_p99": percentile(matched["op_ms"], 99),
        "migration_pct": 100.0 * float(matched["work_migrated"].mean()),
        "socket0_pct": 100.0 * float((matched["work_socket"] == 0).mean()),
        "socket1_pct": 100.0 * float((matched["work_socket"] == 1).mean()),
        "slow_socket0_pct": 100.0 * float((slow["work_socket"] == 0).mean()) if len(slow) else math.nan,
        "slow_socket1_pct": 100.0 * float((slow["work_socket"] == 1).mean()) if len(slow) else math.nan,
    }


def slow_cpu_counts(frame: pd.DataFrame, case: str, window: tuple[float, float]) -> pd.DataFrame:
    matched = in_window(frame, window)
    p99 = percentile(matched["work_searchknn_ms"], 99)
    slow = matched[matched["work_searchknn_ms"] >= p99].copy()
    if slow.empty:
        return pd.DataFrame(columns=["case", "work_start_cpu", "socket", "slow_queries"])
    out = (
        slow.groupby(["work_start_cpu", "work_socket"], as_index=False)
        .size()
        .rename(columns={"size": "slow_queries", "work_socket": "socket"})
    )
    out["case"] = case
    return out[["case", "work_start_cpu", "socket", "slow_queries"]].sort_values(
        ["case", "slow_queries"], ascending=[True, False]
    )


def load_periods(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "period_groups.csv")
    frame["time_ms"] = (frame["bin_start_ms"] + frame["bin_end_ms"]) / 2.0
    cycles = pd.to_numeric(frame["perf_cycles"], errors="coerce").replace(0, np.nan)
    instr = pd.to_numeric(frame["perf_instructions"], errors="coerce")
    slots = pd.to_numeric(frame["perf_topdown_slots"], errors="coerce").replace(0, np.nan)
    frame["ipc"] = instr / cycles
    frame["memory_bound_frac"] = pd.to_numeric(frame["perf_topdown_mem_bound"], errors="coerce") / slots
    frame["backend_bound_frac"] = pd.to_numeric(frame["perf_topdown_be_bound"], errors="coerce") / slots
    return frame


def summarize_periods(frame: pd.DataFrame, case: str, window: tuple[float, float]) -> dict[str, float | str]:
    start, end = window
    sub = frame[(frame["bin_end_ms"] > start) & (frame["bin_start_ms"] < end)].copy()
    if sub.empty:
        return {"case": case}
    weights = (
        np.minimum(sub["bin_end_ms"].to_numpy(), end) - np.maximum(sub["bin_start_ms"].to_numpy(), start)
    ).clip(min=0.0)
    dur_s = float(weights.sum() / 1000.0)

    def weighted_sum(col: str) -> float:
        return float((pd.to_numeric(sub[col], errors="coerce").to_numpy() * (weights / 50.0)).sum())

    cycles = weighted_sum("perf_cycles")
    instructions = weighted_sum("perf_instructions")
    slots = weighted_sum("perf_topdown_slots")
    return {
        "case": case,
        "duration_s": dur_s,
        "ipc": instructions / cycles if cycles else math.nan,
        "memory_bound_frac": weighted_sum("perf_topdown_mem_bound") / slots if slots else math.nan,
        "backend_bound_frac": weighted_sum("perf_topdown_be_bound") / slots if slots else math.nan,
        "cache_misses_per_s": weighted_sum("perf_cache_misses") / dur_s if dur_s else math.nan,
        "llc_load_misses_per_s": weighted_sum("perf_llc_load_misses") / dur_s if dur_s else math.nan,
        "dtlb_load_misses_per_s": weighted_sum("perf_dtlb_load_misses") / dur_s if dur_s else math.nan,
        "rfo_all_per_s": weighted_sum("perf_l2_rqsts_all_rfo") / dur_s if dur_s else math.nan,
        "rfo_miss_per_s": weighted_sum("perf_l2_rqsts_rfo_miss") / dur_s if dur_s else math.nan,
        "remote_hitm_per_s": weighted_sum("perf_mem_load_l3_miss_retired_remote_hitm") / dur_s
        if dur_s
        else math.nan,
    }


def shade_insert(ax: plt.Axes, windows: list[tuple[float, float]]) -> None:
    for start, end in windows:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#f6b6b6", alpha=0.34, lw=0)


def plot(
    background_search: pd.DataFrame,
    insert_search: pd.DataFrame,
    background_periods: pd.DataFrame,
    insert_periods: pd.DataFrame,
    windows: list[tuple[float, float]],
    out_png: Path,
    out_pdf: Path | None,
    bin_ms: float,
    max_ms: float,
) -> None:
    bg_bins = bin_search(background_search, bin_ms, max_ms)
    insert_bins = bin_search(insert_search, bin_ms, max_ms)

    out_png.parent.mkdir(parents=True, exist_ok=True)
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
        ax.set_xlim(0, max_ms / 1000.0)

    axes[0].plot(
        bg_bins["time_ms"] / 1000.0,
        bg_bins["work_searchknn_p99_ms_smooth"],
        color="#6b7280",
        lw=1.8,
        label="Pure-search background",
    )
    axes[0].plot(
        insert_bins["time_ms"] / 1000.0,
        insert_bins["work_searchknn_p99_ms_smooth"],
        color="#1f77b4",
        lw=2.0,
        label="Real insert",
    )
    axes[0].set_ylabel("C++ HNSW\np99 (ms)")
    axes[0].legend(loc="upper right", ncol=2, frameon=False)

    axes[1].plot(
        bg_bins["time_ms"] / 1000.0,
        bg_bins["socket1_pct_smooth"],
        color="#9ca3af",
        lw=1.7,
        label="Pure-search background",
    )
    axes[1].plot(
        insert_bins["time_ms"] / 1000.0,
        insert_bins["socket1_pct_smooth"],
        color="#0f766e",
        lw=1.9,
        label="Real insert",
    )
    axes[1].set_ylabel("Measured search\non socket 1 (%)")
    axes[1].legend(loc="upper right", ncol=2, frameon=False)

    axes[2].plot(
        background_periods["time_ms"] / 1000.0,
        background_periods["perf_l2_rqsts_rfo_miss_per_s"] / 1e6,
        color="#9ca3af",
        lw=1.5,
        label="Pure-search background",
    )
    axes[2].plot(
        insert_periods["time_ms"] / 1000.0,
        insert_periods["perf_l2_rqsts_rfo_miss_per_s"] / 1e6,
        color="#dc2626",
        lw=1.8,
        label="Real insert",
    )
    axes[2].set_ylabel("RFO misses\n(M/s)")
    axes[2].legend(loc="upper right", ncol=2, frameon=False)

    axes[3].plot(
        background_periods["time_ms"] / 1000.0,
        background_periods["perf_llc_load_misses_per_s"] / 1e6,
        color="#6b7280",
        lw=1.5,
        label="LLC load miss/s, pure-search",
    )
    axes[3].plot(
        insert_periods["time_ms"] / 1000.0,
        insert_periods["perf_llc_load_misses_per_s"] / 1e6,
        color="#2563eb",
        lw=1.7,
        label="LLC load miss/s, real insert",
    )
    ax2 = axes[3].twinx()
    shade_insert(ax2, windows)
    ax2.plot(
        background_periods["time_ms"] / 1000.0,
        background_periods["ipc"],
        color="#111827",
        lw=1.2,
        ls="--",
        label="IPC, pure-search",
    )
    ax2.plot(
        insert_periods["time_ms"] / 1000.0,
        insert_periods["ipc"],
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

    fig.suptitle(
        "Recall-95 SIFT, 48 Threads: CPU Placement and Hardware Signals",
        x=0.01,
        ha="left",
        y=1.02,
    )
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--real-insert", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--warmup-ms", type=float, default=1000.0)
    parser.add_argument("--bin-ms", type=float, default=100.0)
    parser.add_argument("--max-ms", type=float, default=5200.0)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    tables = args.out_root / "tables"
    figures = args.out_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    background_search = load_search(args.background)
    insert_search = load_search(args.real_insert)
    windows = load_insert_windows(args.real_insert)
    window = first_window(windows, args.warmup_ms)
    pd.DataFrame([{"window_start_ms": window[0], "window_end_ms": window[1]}]).to_csv(
        tables / "matched_window.csv", index=False
    )

    search_summary = pd.DataFrame(
        [
            summarize_search(background_search, "pure-search background", window),
            summarize_search(insert_search, "real insert", window),
        ]
    )
    search_summary.to_csv(tables / "matched_search_cpu_summary.csv", index=False)
    pd.concat(
        [
            slow_cpu_counts(background_search, "pure-search background", window),
            slow_cpu_counts(insert_search, "real insert", window),
        ],
        ignore_index=True,
    ).to_csv(tables / "slow_query_cpu_counts.csv", index=False)

    background_periods = load_periods(args.background)
    insert_periods = load_periods(args.real_insert)
    pd.DataFrame(
        [
            summarize_periods(background_periods, "pure-search background", window),
            summarize_periods(insert_periods, "real insert", window),
        ]
    ).to_csv(tables / "matched_hardware_summary.csv", index=False)

    plot(
        background_search,
        insert_search,
        background_periods,
        insert_periods,
        windows,
        figures / "odin_cpu_trace_topdown_recall95.png",
        figures / "odin_cpu_trace_topdown_recall95.pdf",
        args.bin_ms,
        args.max_ms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
