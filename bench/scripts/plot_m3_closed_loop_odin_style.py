#!/usr/bin/env python3
"""Plot no-rate closed-loop fixed-worker data in OdinANN-style timeline form."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def pctl(series: pd.Series, pct: float) -> float:
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(np.percentile(vals, pct))


def normalize(series: pd.Series) -> pd.Series:
    vals = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    finite = vals[vals > 0]
    if finite.empty:
        return vals
    denom = float(np.percentile(finite, 95))
    if not np.isfinite(denom) or denom <= 0:
        denom = float(finite.max())
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    return vals / denom


def load_windows(config: Path) -> tuple[list[tuple[float, float]], float]:
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
    return windows, horizon


def mark_windows(frame: pd.DataFrame, col: str, windows: list[tuple[float, float]]) -> pd.Series:
    mask = pd.Series(False, index=frame.index)
    for start, end in windows:
        mask |= (frame[col] >= start) & (frame[col] < end)
    return mask


def load_origin(run_dir: Path) -> int:
    meta = pd.read_csv(run_dir / "raw_latency" / "meta.csv").set_index("key")["value"]
    return int(meta["bench_start_raw_ns"])


def load_insert_active_windows(
    run_dir: Path,
    origin_raw_ns: int,
    issue_windows: list[tuple[float, float]],
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
                "commit_spill_ms": 0.0,
                "fallback": 1,
            }
            for start, end in issue_windows
        ]
        return issue_windows, pd.DataFrame(rows)

    inserts = pd.read_csv(
        insert_path,
        usecols=["create_raw_ns", "start_raw_ns", "finish_raw_ns"],
    )
    commits = pd.read_csv(commit_path, usecols=["commit_ms"])
    inserts = inserts.iloc[: len(commits)].copy()
    inserts["commit_ms"] = commits["commit_ms"].iloc[: len(inserts)].to_numpy()
    inserts["create_ms"] = (inserts["create_raw_ns"] - origin_raw_ns) / 1_000_000.0
    inserts["start_ms"] = (inserts["start_raw_ns"] - origin_raw_ns) / 1_000_000.0
    inserts["finish_raw_ms"] = (inserts["finish_raw_ns"] - origin_raw_ns) / 1_000_000.0

    merged: list[tuple[float, float]] = []
    for start_ms, commit_ms in inserts.sort_values("start_ms")[["start_ms", "commit_ms"]].itertuples(index=False):
        start = float(start_ms)
        end = float(commit_ms)
        if not merged or start > merged[-1][1] + 10.0:
            merged.append((start, end))
        else:
            old_start, old_end = merged[-1]
            merged[-1] = (old_start, max(old_end, end))

    active_windows: list[tuple[float, float]] = []
    rows = []
    for issue_start_ms, issue_end_ms in issue_windows:
        overlapping = [(s, e) for s, e in merged if e >= issue_start_ms and s <= issue_end_ms]
        if overlapping:
            active_start_ms = min(s for s, _ in overlapping)
            active_end_ms = max(e for _, e in overlapping)
            part = inserts[(inserts["commit_ms"] >= active_start_ms) & (inserts["start_ms"] <= active_end_ms)]
        else:
            part = inserts[(inserts["create_ms"] >= issue_start_ms) & (inserts["create_ms"] < issue_end_ms)]
            active_start_ms = issue_start_ms if part.empty else float(part["start_ms"].min())
            active_end_ms = issue_end_ms if part.empty else float(part["commit_ms"].max())
        active_windows.append((active_start_ms, active_end_ms))
        rows.append(
            {
                "issue_start_ms": issue_start_ms,
                "issue_end_ms": issue_end_ms,
                "active_start_ms": active_start_ms,
                "active_end_ms": active_end_ms,
                "created_batches": int(len(part)),
                "commit_spill_ms": active_end_ms - issue_end_ms,
                "fallback": 0 if len(part) else 1,
            }
        )
    return active_windows, pd.DataFrame(rows)


def load_search(run_dir: Path, origin_raw_ns: int) -> pd.DataFrame:
    available = set(pd.read_csv(run_dir / "raw_latency" / "search_wide.csv", nrows=0).columns)
    wanted = [
        "measured",
        "finish_ms",
        "create_raw_ns",
        "start_raw_ns",
        "finish_raw_ns",
        "op_ms",
        "e2e_ms",
        "diag_batch_search_arena_ms",
        "diag_batch_search_knn_ms",
        "diag_batch_search_copy_ms",
        "diag_dist_comps",
        "diag_hops",
    ]
    search = pd.read_csv(
        run_dir / "raw_latency" / "search_wide.csv",
        usecols=[col for col in wanted if col in available],
    )
    search["queue_ms"] = search["e2e_ms"] - search["op_ms"]
    search["create_ms"] = (search["create_raw_ns"] - origin_raw_ns) / 1_000_000.0
    search["start_ms"] = (search["start_raw_ns"] - origin_raw_ns) / 1_000_000.0
    search["finish_raw_ms"] = (search["finish_raw_ns"] - origin_raw_ns) / 1_000_000.0
    search["mid_ms"] = (search["start_ms"] + search["finish_raw_ms"]) / 2.0
    return search


def windowed_latency(
    bins: pd.DataFrame,
    measured: pd.DataFrame,
    lowq: pd.DataFrame,
    bin_ms: float,
    window_ms: float,
) -> pd.DataFrame:
    rows = []
    half = window_ms / 2.0
    for bin_start in bins["bin_ms"]:
        center = float(bin_start) + bin_ms / 2.0
        start = center - half
        end = center + half
        part = measured[(measured["mid_ms"] >= start) & (measured["mid_ms"] < end)]
        low = lowq[(lowq["mid_ms"] >= start) & (lowq["mid_ms"] < end)]
        row = {
            "bin_ms": float(bin_start),
            "search_count": int(len(part)),
            "op_p95_ms": pctl(part["op_ms"], 95),
            "op_p99_ms": pctl(part["op_ms"], 99),
            "queue_p95_ms": pctl(part["queue_ms"], 95),
            "lowq_search_count": int(len(low)),
            "lowq_op_p95_ms": pctl(low["op_ms"], 95),
            "lowq_op_p99_ms": pctl(low["op_ms"], 99),
        }
        for source_col, prefix in [
            ("diag_batch_search_arena_ms", "batch_search_arena"),
            ("diag_batch_search_knn_ms", "batch_search_knn"),
            ("diag_batch_search_copy_ms", "batch_search_copy"),
            ("diag_dist_comps", "dist_comps"),
            ("diag_hops", "hops"),
        ]:
            if source_col in low.columns:
                row[f"lowq_{prefix}_p95"] = pctl(low[source_col], 95)
                row[f"lowq_{prefix}_p99"] = pctl(low[source_col], 99)
                row[f"lowq_{prefix}_mean"] = float(low[source_col].mean()) if len(low) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def allocate_workers(search: pd.DataFrame, bin_ms: float, horizon_ms: float) -> pd.DataFrame:
    bin_count = int(np.ceil(horizon_ms / bin_ms))
    measured_ns = np.zeros(bin_count, dtype=float)
    bg_ns = np.zeros(bin_count, dtype=float)
    tasks = search.drop_duplicates(["measured", "start_raw_ns", "finish_raw_ns"])
    for row in tasks.itertuples(index=False):
        start = max(0.0, float(row.start_ms))
        end = min(horizon_ms, float(row.finish_raw_ms))
        if end <= start:
            continue
        first = int(start // bin_ms)
        last = int((end - 1e-9) // bin_ms)
        for idx in range(max(0, first), min(bin_count - 1, last) + 1):
            b0 = idx * bin_ms
            b1 = b0 + bin_ms
            overlap = min(end, b1) - max(start, b0)
            if overlap <= 0:
                continue
            if int(row.measured) == 1:
                measured_ns[idx] += overlap * 1_000_000.0
            else:
                bg_ns[idx] += overlap * 1_000_000.0
    return pd.DataFrame(
        {
            "bin_ms": np.arange(bin_count, dtype=float) * bin_ms,
            "measured_search_active_workers": measured_ns / (bin_ms * 1_000_000.0),
            "background_search_active_workers": bg_ns / (bin_ms * 1_000_000.0),
        }
    )


def load_periods(run_dir: Path, bins: pd.DataFrame, period_groups: Path | None = None) -> pd.DataFrame:
    path = period_groups or (run_dir / "period_groups.csv")
    base_cols = [
        "bin_start_ms",
        "insert_path_search_active_workers",
        "existing_neighbor_update_active_workers",
        "existing_neighbor_load_scan_active_workers",
        "existing_neighbor_prune_active_workers",
        "existing_neighbor_undo_record_active_workers",
        "existing_neighbor_rewrite_active_workers",
        "perf_llc_load_misses_per_s",
        "perf_dtlb_load_misses_walk_completed_per_s",
    ]
    if not path.exists():
        out = bins.copy()
        for col in base_cols[1:]:
            out[col] = 0.0
        return out
    periods = pd.read_csv(path)
    rename = {"bin_start_ms": "bin_ms"}
    periods = periods.rename(columns=rename)
    for col in base_cols[1:]:
        if col not in periods.columns:
            periods[col] = 0.0
    extra_perf_cols = [
        col
        for col in periods.columns
        if col.startswith("perf_") and col.endswith("_per_s") and col not in base_cols
    ]
    return periods[["bin_ms"] + base_cols[1:] + extra_perf_cols]


def period_step_series(
    windows: list[tuple[float, float]],
    horizon_ms: float,
    inside_value: float,
    outside_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    boundaries = {0.0, horizon_ms}
    for start, end in windows:
        boundaries.add(max(0.0, min(horizon_ms, start)))
        boundaries.add(max(0.0, min(horizon_ms, end)))
    xs = np.array(sorted(boundaries), dtype=float)
    ys = []
    for idx, x in enumerate(xs):
        probe = x + 1e-6 if idx + 1 < len(xs) else max(0.0, x - 1e-6)
        inside = any(start <= probe < end for start, end in windows)
        ys.append(inside_value if inside else outside_value)
    return xs / 1000.0, np.array(ys, dtype=float)


def window_segments(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(start / 1000.0, (end - start) / 1000.0) for start, end in windows]


def outside_segments(windows: list[tuple[float, float]], horizon_ms: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in sorted(windows):
        if start > cursor:
            out.append((cursor / 1000.0, (start - cursor) / 1000.0))
        cursor = max(cursor, end)
    if cursor < horizon_ms:
        out.append((cursor / 1000.0, (horizon_ms - cursor) / 1000.0))
    return out


def label_segments(ax: plt.Axes, segments: list[tuple[float, float]], y: float, text: str) -> None:
    for start, duration in segments:
        if duration < 0.55:
            continue
        ax.text(start + duration / 2.0, y, text, ha="center", va="center", color="white", fontsize=8.0)


def summarize(df: pd.DataFrame, search: pd.DataFrame, active_windows: list[tuple[float, float]], queue_threshold_ms: float) -> pd.DataFrame:
    search = search.copy()
    search["in_insert_active"] = mark_windows(search, "mid_ms", active_windows)
    df = df.copy()
    df["in_insert_active"] = mark_windows(df, "bin_ms", active_windows)
    rows = []
    for phase, mask in [("outside", ~search["in_insert_active"]), ("insert_active", search["in_insert_active"])]:
        measured = search[mask & (search["measured"] == 1)]
        lowq = measured[measured["queue_ms"] < queue_threshold_ms]
        period = df[df["in_insert_active"] == (phase == "insert_active")]
        rows.append(
            {
                "phase": phase,
                "measured_queries": int(len(measured)),
                "lowq_measured_queries": int(len(lowq)),
                "lowq_op_p95_ms": pctl(lowq["op_ms"], 95),
                "lowq_op_p99_ms": pctl(lowq["op_ms"], 99),
                "lowq_queue_p95_ms": pctl(lowq["queue_ms"], 95),
                "bg_search_active_workers_mean": float(period["background_search_active_workers"].mean()),
                "insert_path_workers_mean": float(period["insert_path_search_active_workers"].mean()),
                "existing_update_workers_mean": float(period["existing_neighbor_update_active_workers"].mean()),
                "llc_load_misses_per_s_mean": float(period["perf_llc_load_misses_per_s"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--boundaries-out", type=Path)
    parser.add_argument("--period-groups", type=Path)
    parser.add_argument("--bin-ms", type=float, default=50.0)
    parser.add_argument("--latency-window-ms", type=float, default=200.0)
    parser.add_argument("--queue-threshold-ms", type=float, default=0.1)
    parser.add_argument("--measured-workers", type=float, default=8.0)
    parser.add_argument("--competing-workers", type=float, default=40.0)
    parser.add_argument("--title", default="M3 no-rate closed-loop OdinANN-style timeline")
    args = parser.parse_args()

    issue_windows, horizon_ms = load_windows(args.config)
    origin = load_origin(args.run_dir)
    active_windows, boundaries = load_insert_active_windows(args.run_dir, origin, issue_windows)
    search = load_search(args.run_dir, origin)
    horizon_ms = max(horizon_ms, float(search["finish_raw_ms"].max()), max((e for _, e in active_windows), default=0.0))

    bins = pd.DataFrame({"bin_ms": np.arange(0.0, horizon_ms, args.bin_ms)})
    measured = search[search["measured"] == 1].copy()
    lowq = measured[measured["queue_ms"] < args.queue_threshold_ms].copy()
    lat = windowed_latency(bins, measured, lowq, args.bin_ms, args.latency_window_ms)
    workers = allocate_workers(search, args.bin_ms, horizon_ms)
    periods = load_periods(args.run_dir, bins, args.period_groups)
    df = (
        bins.merge(lat, on="bin_ms", how="left")
        .merge(workers, on="bin_ms", how="left")
        .merge(periods, on="bin_ms", how="left")
        .sort_values("bin_ms")
    )
    for col in [
        "background_search_active_workers",
        "measured_search_active_workers",
        "insert_path_search_active_workers",
        "existing_neighbor_update_active_workers",
        "existing_neighbor_load_scan_active_workers",
        "existing_neighbor_prune_active_workers",
        "existing_neighbor_undo_record_active_workers",
        "existing_neighbor_rewrite_active_workers",
        "perf_llc_load_misses_per_s",
        "perf_dtlb_load_misses_walk_completed_per_s",
    ]:
        df[col] = df[col].fillna(0.0)
    for col in [c for c in df.columns if c.startswith("perf_") and c.endswith("_per_s")]:
        df[col] = df[col].fillna(0.0)
    df["elapsed_s"] = (df["bin_ms"] + args.bin_ms / 2.0) / 1000.0
    df["in_insert_issue_window"] = mark_windows(df, "bin_ms", issue_windows)
    df["in_insert_active_window"] = mark_windows(df, "bin_ms", active_windows)

    csv_out = args.csv_out or args.out.with_suffix(".csv")
    summary_out = args.summary_out or args.out.with_name(args.out.stem + "_summary.csv")
    boundaries_out = args.boundaries_out or args.out.with_name(args.out.stem + "_insert_boundaries.csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    summarize(df, search, active_windows, args.queue_threshold_ms).to_csv(summary_out, index=False)
    boundaries.to_csv(boundaries_out, index=False)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8.4,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.8, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0, 1.05]},
    )
    ax0, ax1, ax2 = axes
    for ax in axes:
        for idx, (start, end) in enumerate(active_windows):
            ax.axvspan(
                start / 1000.0,
                end / 1000.0,
                color="#e5e7eb",
                alpha=0.62,
                lw=0,
                label="insert active-to-commit" if idx == 0 else None,
            )
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0.0, horizon_ms / 1000.0)

    x = df["elapsed_s"]
    ax0.plot(x, df["lowq_op_p99_ms"], color="#dc2626", lw=1.7, label="measured search op p99")
    ax0.set_title(
        f"Measured search execution tail, {args.bin_ms:.0f}ms stride, "
        f"{args.latency_window_ms:.0f}ms window aligned by execution mid"
    )
    ax0.set_ylabel("latency (ms)")
    ax0.legend(frameon=False, loc="upper left", ncol=3)

    period_x, bg_budget = period_step_series(active_windows, horizon_ms, 0.0, args.competing_workers)
    _, insert_budget = period_step_series(active_windows, horizon_ms, args.competing_workers, 0.0)
    measured_budget = np.full_like(period_x, args.measured_workers)
    ax1.fill_between(period_x, 0.0, measured_budget, step="post", color="#16a34a", alpha=0.86, linewidth=0)
    ax1.fill_between(
        period_x,
        measured_budget,
        measured_budget + bg_budget,
        step="post",
        color="#64748b",
        alpha=0.76,
        linewidth=0,
    )
    ax1.fill_between(
        period_x,
        measured_budget,
        measured_budget + insert_budget,
        step="post",
        color="#2563eb",
        alpha=0.82,
        linewidth=0,
    )
    ax1.axhline(args.measured_workers, color="white", lw=1.0, alpha=0.9)
    label_segments(ax1, [(0.0, horizon_ms / 1000.0)], args.measured_workers / 2.0, f"measured search {args.measured_workers:.0f}t fixed")
    label_segments(ax1, outside_segments(active_windows, horizon_ms), args.measured_workers + args.competing_workers / 2.0, f"bg search: {args.competing_workers:.0f}t")
    label_segments(ax1, window_segments(active_windows), args.measured_workers + args.competing_workers / 2.0, f"insert: {args.competing_workers:.0f}t")
    ax1.set_title(
        f"C++ thread allocation envelope: {args.measured_workers:.0f} measured threads + "
        f"{args.competing_workers:.0f} competing threads, no input rate"
    )
    ax1.set_ylabel("threads")
    ax1.set_yticks([0, args.measured_workers, args.measured_workers + args.competing_workers])
    ax1.set_ylim(0.0, (args.measured_workers + args.competing_workers) * 1.08)

    ax2.plot(
        x,
        df["existing_neighbor_update_active_workers"],
        color="#dc2626",
        lw=1.45,
        label="existing-neighbor update occupancy",
    )
    ax2b = ax2.twinx()
    llc_norm = normalize(df["perf_llc_load_misses_per_s"])
    ax2b.plot(x, llc_norm, color="#111827", lw=1.2, alpha=0.82, label="LLC-load-miss rate norm")
    ax2.set_title("Insert mutation occupancy vs LLC pressure")
    ax2.set_ylabel("worker-equiv occupancy")
    ax2b.set_ylabel("LLC norm")
    ax2.set_xlabel("elapsed time (s)")
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left", ncol=2)

    fig.suptitle(args.title, y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")

    print(args.out)
    print(csv_out)
    print(summary_out)
    print(boundaries_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
