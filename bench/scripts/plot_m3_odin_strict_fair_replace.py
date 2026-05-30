#!/usr/bin/env python3
"""Plot strict fair replacement: measured search + background search vs insert."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


HW_COLS = [
    "perf_l2_rqsts_miss_per_s",
    "perf_llc_load_misses_per_s",
    "perf_cache_misses_per_s",
    "perf_dtlb_load_misses_walk_completed_per_s",
]


def pctl(series: pd.Series, pct: float) -> float:
    if series.empty:
        return float("nan")
    return float(np.percentile(series, pct))


def load_windows(config: Path) -> tuple[list[tuple[float, float]], float, float, float, float]:
    with config.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    windows: list[tuple[float, float]] = []
    for item in workload.get("insert_burst_schedule") or []:
        start = float(item.get("start_ms", 0.0))
        end = start + float(item.get("duration_ms", 0.0))
        if end > start:
            windows.append((start, end))
    horizon = float(workload.get("schedule_horizon_ms", 0.0))
    batch_size = float(workload.get("batch_size", 1.0))
    insert_rate = float(workload.get("insert_event_rate", 0.0))
    search_rate = float(workload.get("search_event_rate", 0.0))
    return windows, horizon, batch_size, insert_rate, search_rate


def mark_insert(frame: pd.DataFrame, col: str, windows: list[tuple[float, float]]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for start, end in windows:
        mask |= (frame[col] >= start) & (frame[col] < end)
    return mask


def load_insert_active_windows(
    run_dir: Path,
    origin_raw_ns: int,
    issue_windows: list[tuple[float, float]],
    batch_size: float,
) -> tuple[list[tuple[float, float]], pd.DataFrame]:
    insert_path = run_dir / "raw_latency" / "insert_wide.csv"
    commit_path = run_dir / "raw_latency" / "commit_events.csv"
    if not insert_path.exists() or not commit_path.exists():
        rows = [
            {
                "issue_start_ms": start,
                "issue_end_ms": end,
                "active_start_ms": start,
                "active_end_ms": end,
                "created_batches": 0,
                "created_points": 0,
                "commit_spill_ms": 0.0,
                "fallback": 1,
            }
            for start, end in issue_windows
        ]
        return issue_windows, pd.DataFrame(rows)

    inserts = pd.read_csv(
        insert_path,
        usecols=["create_raw_ns", "start_raw_ns", "finish_raw_ns", "finish_ms"],
    )
    commits = pd.read_csv(commit_path, usecols=["commit_ms"])
    inserts = inserts.iloc[: len(commits)].copy()
    inserts["commit_ms"] = commits["commit_ms"].iloc[: len(inserts)].to_numpy()
    inserts["create_ms"] = (inserts["create_raw_ns"] - origin_raw_ns) / 1_000_000.0
    inserts["start_ms"] = (inserts["start_raw_ns"] - origin_raw_ns) / 1_000_000.0
    inserts["finish_raw_ms"] = (inserts["finish_raw_ns"] - origin_raw_ns) / 1_000_000.0

    merged_segments: list[tuple[float, float]] = []
    for start_ms, commit_ms in inserts.sort_values("start_ms")[["start_ms", "commit_ms"]].itertuples(index=False):
        start = float(start_ms)
        end = float(commit_ms)
        if not merged_segments or start > merged_segments[-1][1] + 10.0:
            merged_segments.append((start, end))
        else:
            prev_start, prev_end = merged_segments[-1]
            merged_segments[-1] = (prev_start, max(prev_end, end))

    active_windows: list[tuple[float, float]] = []
    rows = []
    for issue_start_ms, issue_end_ms in issue_windows:
        overlapping = [
            (start, end)
            for start, end in merged_segments
            if end >= issue_start_ms and start <= issue_end_ms
        ]
        if overlapping:
            active_start_ms = min(start for start, _ in overlapping)
            active_end_ms = max(end for _, end in overlapping)
            part = inserts[
                (inserts["commit_ms"] >= active_start_ms)
                & (inserts["start_ms"] <= active_end_ms)
            ]
        else:
            part = inserts[
                (inserts["create_ms"] >= issue_start_ms)
                & (inserts["create_ms"] < issue_end_ms)
            ]
        if part.empty:
            active_start_ms = issue_start_ms
            active_end_ms = issue_end_ms
            first_create_ms = last_create_ms = float("nan")
            first_start_ms = last_start_ms = float("nan")
            first_finish_ms = last_finish_ms = float("nan")
            first_commit_ms = last_commit_ms = float("nan")
        else:
            active_start_ms = float(part["start_ms"].min())
            active_end_ms = float(part["commit_ms"].max())
            first_create_ms = float(part["create_ms"].min())
            last_create_ms = float(part["create_ms"].max())
            first_start_ms = active_start_ms
            last_start_ms = float(part["start_ms"].max())
            first_finish_ms = float(part["finish_raw_ms"].min())
            last_finish_ms = float(part["finish_raw_ms"].max())
            first_commit_ms = float(part["commit_ms"].min())
            last_commit_ms = active_end_ms
        active_windows.append((active_start_ms, active_end_ms))
        created_batches = int(len(part))
        rows.append(
            {
                "issue_start_ms": issue_start_ms,
                "issue_end_ms": issue_end_ms,
                "active_start_ms": active_start_ms,
                "active_end_ms": active_end_ms,
                "first_create_ms": first_create_ms,
                "last_create_ms": last_create_ms,
                "first_start_ms": first_start_ms,
                "last_start_ms": last_start_ms,
                "first_finish_ms": first_finish_ms,
                "last_finish_ms": last_finish_ms,
                "first_commit_ms": first_commit_ms,
                "last_commit_ms": last_commit_ms,
                "created_batches": created_batches,
                "created_points": int(round(created_batches * batch_size)),
                "commit_spill_ms": active_end_ms - issue_end_ms,
                "fallback": 0,
            }
        )
    return active_windows, pd.DataFrame(rows)


def normalize(series: pd.Series) -> pd.Series:
    vals = series.replace([np.inf, -np.inf], np.nan)
    finite = vals.dropna()
    denom = float(np.nanpercentile(finite, 95)) if len(finite) else 1.0
    if not np.isfinite(denom) or denom <= 0:
        denom = float(np.nanmax(finite)) if len(finite) else 1.0
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    return vals / denom


def windowed_latency(
    fixed_bins: pd.DataFrame,
    measured: pd.DataFrame,
    lowq: pd.DataFrame,
    time_col: str,
    bin_ms: float,
    window_ms: float,
) -> pd.DataFrame:
    rows = []
    half_window_ms = window_ms / 2.0
    measured_time = measured[time_col]
    lowq_time = lowq[time_col]
    for bin_start_ms in fixed_bins["bin_ms"]:
        center_ms = float(bin_start_ms) + bin_ms / 2.0
        start_ms = center_ms - half_window_ms
        end_ms = center_ms + half_window_ms
        part = measured[(measured_time >= start_ms) & (measured_time < end_ms)]
        low = lowq[(lowq_time >= start_ms) & (lowq_time < end_ms)]
        rows.append(
            {
                "bin_ms": float(bin_start_ms),
                "latency_window_start_ms": start_ms,
                "latency_window_end_ms": end_ms,
                "search_count": int(len(part)),
                "op_p95_ms": pctl(part["op_ms"], 95),
                "op_p99_ms": pctl(part["op_ms"], 99),
                "op_mean_ms": float(part["op_ms"].mean()) if len(part) else float("nan"),
                "queue_p95_ms": pctl(part["queue_ms"], 95),
                "dist_p95": pctl(part["diag_dist_comps"], 95),
                "hops_p95": pctl(part["diag_hops"], 95),
                "lowq_search_count": int(len(low)),
                "lowq_op_p95_ms": pctl(low["op_ms"], 95),
                "lowq_op_p99_ms": pctl(low["op_ms"], 99),
            }
        )
    return pd.DataFrame(rows)


def period_step_series(
    windows: list[tuple[float, float]],
    horizon_ms: float,
    inside_value: float,
    outside_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    boundaries = {0.0, horizon_ms}
    for start_ms, end_ms in windows:
        boundaries.add(max(0.0, min(horizon_ms, start_ms)))
        boundaries.add(max(0.0, min(horizon_ms, end_ms)))
    x_ms = np.array(sorted(boundaries), dtype=float)
    y = []
    for idx, boundary_ms in enumerate(x_ms):
        if idx + 1 < len(x_ms):
            probe_ms = boundary_ms + 1e-6
        else:
            probe_ms = max(0.0, boundary_ms - 1e-6)
        in_window = any(start <= probe_ms < end for start, end in windows)
        y.append(inside_value if in_window else outside_value)
    return x_ms / 1000.0, np.array(y, dtype=float)


def outside_window_segments(
    windows: list[tuple[float, float]],
    horizon_ms: float,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    cursor = 0.0
    for start_ms, end_ms in sorted(windows):
        if start_ms > cursor:
            segments.append((cursor / 1000.0, (start_ms - cursor) / 1000.0))
        cursor = max(cursor, end_ms)
    if cursor < horizon_ms:
        segments.append((cursor / 1000.0, (horizon_ms - cursor) / 1000.0))
    return segments


def window_segments(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(start_ms / 1000.0, (end_ms - start_ms) / 1000.0) for start_ms, end_ms in windows]


def label_segments(
    ax: plt.Axes,
    segments: list[tuple[float, float]],
    y: float,
    label: str,
    color: str,
) -> None:
    for start_s, duration_s in segments:
        if duration_s < 0.75:
            continue
        ax.text(
            start_s + duration_s / 2.0,
            y,
            label,
            ha="center",
            va="center",
            color=color,
            fontsize=8.5,
            clip_on=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--boundaries-out", type=Path)
    parser.add_argument("--bin-ms", type=float, default=50.0)
    parser.add_argument("--latency-window-ms", type=float, default=200.0)
    parser.add_argument("--queue-threshold-ms", type=float, default=0.1)
    parser.add_argument("--measured-arena-threads", type=float, default=8.0)
    parser.add_argument("--competing-arena-threads", type=float, default=40.0)
    parser.add_argument(
        "--latency-time",
        choices=["create", "start", "mid", "finish"],
        default="mid",
        help="Timestamp used to align measured-search latency. mid aligns long C++ executions better than finish.",
    )
    parser.add_argument(
        "--title",
        default="M3 strict fair replacement: measured search fixed, background search replaced by insert",
    )
    args = parser.parse_args()

    windows, horizon_ms, batch_size, insert_rate, search_rate = load_windows(args.config)
    meta = pd.read_csv(args.run_dir / "raw_latency" / "meta.csv").set_index("key")["value"]
    origin_raw_ns = int(meta["bench_start_raw_ns"])
    active_windows, boundary_summary = load_insert_active_windows(
        args.run_dir,
        origin_raw_ns,
        windows,
        batch_size,
    )

    search = pd.read_csv(
        args.run_dir / "raw_latency" / "search_wide.csv",
        usecols=[
            "measured",
            "finish_ms",
            "create_raw_ns",
            "start_raw_ns",
            "finish_raw_ns",
            "op_ms",
            "e2e_ms",
            "diag_dist_comps",
            "diag_hops",
        ],
    )
    search["queue_ms"] = search["e2e_ms"] - search["op_ms"]
    search["create_ms"] = (search["create_raw_ns"] - origin_raw_ns) / 1_000_000.0
    search["start_ms"] = (search["start_raw_ns"] - origin_raw_ns) / 1_000_000.0
    search["finish_raw_ms"] = (search["finish_raw_ns"] - origin_raw_ns) / 1_000_000.0
    search["mid_ms"] = (search["start_ms"] + search["finish_raw_ms"]) / 2.0
    latency_ms_col = {
        "create": "create_ms",
        "start": "start_ms",
        "mid": "mid_ms",
        "finish": "finish_ms",
    }[args.latency_time]
    search["create_bin_ms"] = np.floor(search["create_ms"] / args.bin_ms) * args.bin_ms
    search["in_insert_finish"] = mark_insert(search, "finish_ms", windows)
    search["in_insert_create"] = mark_insert(search, "create_ms", windows)
    search["in_insert_mid"] = mark_insert(search, "mid_ms", windows)

    measured = search[search["measured"] == 1].copy()
    lowq = measured[measured["queue_ms"] < args.queue_threshold_ms].copy()
    fixed_bins = pd.DataFrame({"bin_ms": np.arange(0.0, horizon_ms, args.bin_ms)})
    lat = windowed_latency(
        fixed_bins,
        measured,
        lowq,
        latency_ms_col,
        args.bin_ms,
        args.latency_window_ms,
    )

    rates = (
        search.groupby(["create_bin_ms", "measured"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"create_bin_ms": "bin_ms", 0: "background_search_qps", 1: "measured_search_qps"})
    )
    if "background_search_qps" not in rates:
        rates["background_search_qps"] = 0
    if "measured_search_qps" not in rates:
        rates["measured_search_qps"] = 0
    rates["background_search_qps"] = rates["background_search_qps"] / (args.bin_ms / 1000.0)
    rates["measured_search_qps"] = rates["measured_search_qps"] / (args.bin_ms / 1000.0)
    rates["insert_issue_qps"] = 0.0
    for start, end in windows:
        rates.loc[(rates["bin_ms"] >= start) & (rates["bin_ms"] < end), "insert_issue_qps"] = insert_rate

    groups = pd.read_csv(args.run_dir / "period_groups.csv")
    groups["bin_ms"] = np.floor(groups["bin_start_ms"] / args.bin_ms) * args.bin_ms
    phase_hw_cols = [
        "insert_path_search_active_workers",
        "existing_neighbor_update_active_workers",
        "existing_neighbor_prune_active_workers",
        "existing_neighbor_undo_record_active_workers",
        "existing_neighbor_rewrite_active_workers",
    ] + HW_COLS
    groups = groups.groupby("bin_ms")[phase_hw_cols].mean().reset_index()

    df = (
        fixed_bins.merge(lat, on="bin_ms", how="left")
        .merge(rates, on="bin_ms", how="left")
        .merge(groups, on="bin_ms", how="left")
    )
    df = df.sort_values("bin_ms")
    df["elapsed_s"] = (df["bin_ms"] + args.bin_ms / 2.0) / 1000.0
    df["latency_time"] = args.latency_time
    df["latency_window_ms"] = args.latency_window_ms
    df["in_insert_window"] = mark_insert(df, "bin_ms", windows)
    for col in ["background_search_qps", "measured_search_qps", "insert_issue_qps"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    df["measured_thread_budget"] = args.measured_arena_threads
    df["background_search_thread_budget"] = np.where(
        df["in_insert_window"], 0.0, args.competing_arena_threads
    )
    df["insert_thread_budget"] = np.where(
        df["in_insert_window"], args.competing_arena_threads, 0.0
    )
    for col in HW_COLS:
        df[col + "_norm"] = normalize(df[col])
    mutation_cols = [
        "existing_neighbor_update_active_workers",
        "existing_neighbor_prune_active_workers",
        "existing_neighbor_undo_record_active_workers",
        "existing_neighbor_rewrite_active_workers",
    ]
    df["neighbor_mutation_occupancy"] = df[mutation_cols].sum(axis=1)

    summary_rows = []
    for basis, col in [("create", "in_insert_create"), ("mid", "in_insert_mid"), ("finish", "in_insert_finish")]:
        for phase, mask in [("outside", ~search[col]), ("insert", search[col])]:
            part = search[mask]
            m = part[part["measured"] == 1]
            bg = part[part["measured"] == 0]
            low = m[m["queue_ms"] < args.queue_threshold_ms]
            summary_rows.append(
                {
                    "basis": basis,
                    "phase": phase,
                    "measured_rows": int(len(m)),
                    "background_rows": int(len(bg)),
                    "lowq_measured_rows": int(len(low)),
                    "lowq_op_p95_ms": pctl(low["op_ms"], 95),
                    "lowq_op_p99_ms": pctl(low["op_ms"], 99),
                    "lowq_queue_p95_ms": pctl(low["queue_ms"], 95),
                    "lowq_dist_p95": pctl(low["diag_dist_comps"], 95),
                    "lowq_hops_p95": pctl(low["diag_hops"], 95),
                }
            )
    summary = pd.DataFrame(summary_rows)

    out_csv = args.csv_out or args.out.with_suffix(".csv")
    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.csv")
    boundaries_out = args.boundaries_out or args.out.with_name(args.out.stem + "_insert_boundaries.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    summary.to_csv(summary_out, index=False)
    boundary_summary.to_csv(boundaries_out, index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 8.0), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0, 1.05]})
    ax0, ax1, ax2 = axes
    for ax in axes:
        for idx, (start, end) in enumerate(active_windows):
            ax.axvspan(start / 1000.0, end / 1000.0, color="#e5e7eb", alpha=0.62, lw=0, label="insert active-to-commit" if idx == 0 else None)
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, max(horizon_ms / 1000.0, float(df["elapsed_s"].max())))

    x = df["elapsed_s"]
    latency_label = f"{args.latency_time}, {args.latency_window_ms:.0f}ms window"
    ax0.plot(x, df["op_p95_ms"], color="#f97316", lw=1.5, label=f"measured op p95 ({latency_label})")
    ax0.plot(x, df["op_p99_ms"], color="#dc2626", lw=1.5, label=f"measured op p99 ({latency_label})")
    ax0.set_title(
        f"Measured search execution tail, {args.bin_ms:.0f}ms stride, "
        f"{args.latency_window_ms:.0f}ms window aligned by {args.latency_time}"
    )
    ax0.set_ylabel("latency (ms)")
    ax0.legend(frameon=False, loc="upper left", ncol=2)

    measured_segments = [(0.0, horizon_ms / 1000.0)]
    background_segments = outside_window_segments(active_windows, horizon_ms)
    insert_segments = window_segments(active_windows)
    period_x, background_thread_budget = period_step_series(
        active_windows, horizon_ms, 0.0, args.competing_arena_threads
    )
    _, insert_thread_budget = period_step_series(
        active_windows, horizon_ms, args.competing_arena_threads, 0.0
    )
    measured_thread_budget = np.full_like(period_x, args.measured_arena_threads)
    competing_base = measured_thread_budget
    ax1.fill_between(
        period_x,
        0.0,
        measured_thread_budget,
        step="post",
        color="#16a34a",
        alpha=0.86,
        linewidth=0,
        label=f"measured search ({args.measured_arena_threads:.0f}t fixed)",
    )
    ax1.fill_between(
        period_x,
        competing_base,
        competing_base + background_thread_budget,
        step="post",
        color="#64748b",
        alpha=0.76,
        linewidth=0,
        label=f"background search ({args.competing_arena_threads:.0f}t)",
    )
    ax1.fill_between(
        period_x,
        competing_base,
        competing_base + insert_thread_budget,
        step="post",
        color="#2563eb",
        alpha=0.82,
        linewidth=0,
        label=f"real insert ({args.competing_arena_threads:.0f}t)",
    )
    ax1.axhline(args.measured_arena_threads, color="white", lw=1.0, alpha=0.9)
    label_segments(
        ax1,
        measured_segments,
        args.measured_arena_threads / 2.0,
        f"measured search {args.measured_arena_threads:.0f}t fixed",
        "white",
    )
    label_segments(
        ax1,
        background_segments,
        args.measured_arena_threads + args.competing_arena_threads / 2.0,
        f"bg search: {args.competing_arena_threads:.0f}t",
        "white",
    )
    label_segments(
        ax1,
        insert_segments,
        args.measured_arena_threads + args.competing_arena_threads / 2.0,
        f"insert: {args.competing_arena_threads:.0f}t",
        "white",
    )
    ax1.set_title("C++ thread allocation envelope: 8 measured threads + insert active-to-commit")
    ax1.set_ylabel("threads")
    ax1.set_yticks([0, args.measured_arena_threads, args.measured_arena_threads + args.competing_arena_threads])
    ax1.set_ylim(0.0, (args.measured_arena_threads + args.competing_arena_threads) * 1.08)
    ax1.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)

    ax2.plot(
        x,
        df["insert_path_search_active_workers"],
        color="#2563eb",
        lw=1.4,
        label="insert traversal occupancy (read/search phase)",
    )
    ax2.plot(
        x,
        df["neighbor_mutation_occupancy"],
        color="#dc2626",
        lw=1.45,
        label="neighbor mutation occupancy (update+prune+undo+rewrite)",
    )
    ax2b = ax2.twinx()
    ax2b.fill_between(
        x,
        0.0,
        df["perf_llc_load_misses_per_s_norm"],
        color="#111827",
        alpha=0.14,
        linewidth=0,
        label="_nolegend_",
    )
    ax2b.plot(
        x,
        df["perf_llc_load_misses_per_s_norm"],
        color="#111827",
        lw=1.0,
        ls="-",
        alpha=0.72,
        label="LLC miss rate norm (100ms perf avg)",
    )
    ax2.set_title("Insert traversal vs mutation occupancy, plus LLC pressure")
    ax2.set_ylabel("worker-equiv occupancy")
    ax2b.set_ylabel("LLC miss rate norm")
    ax2.set_xlabel("elapsed time (s)")
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left", ncol=3)

    fig.suptitle(args.title, y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")

    print(args.out)
    print(out_csv)
    print(summary_out)
    print(boundaries_out)
    print(boundary_summary.to_string(index=False))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
