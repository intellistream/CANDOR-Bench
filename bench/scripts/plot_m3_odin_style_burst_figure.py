#!/usr/bin/env python3
"""Plot an OdinANN-Figure-2-style M3 burst mechanism figure."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


LABEL = "sift1m_b200_round_robin_burst_w40000_r5000_nolock"
REPS = (1, 2, 3)
FULL = "full_insert"
PATH = "path_only_cpu_delay"


plt.rcParams.update(
    {
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.75,
    }
)


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_config(path: Path) -> tuple[list[tuple[float, float]], float, float]:
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    windows = []
    for item in workload.get("insert_burst_schedule") or []:
        start = f(item.get("start_ms")) / 1000.0
        end = start + f(item.get("duration_ms")) / 1000.0
        if end > start:
            windows.append((start, end))
    horizon = f(workload.get("schedule_horizon_ms")) / 1000.0
    insert_rate = f(workload.get("insert_event_rate")) / 1000.0
    return windows, horizon, insert_rate


def load_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="") as file:
        return [{key: f(value) if key not in {"mode"} else value for key, value in row.items()} for row in csv.DictReader(file)]


def load_summary(path: Path) -> dict[tuple[str, str], dict[str, float | str]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return {(row["mode"], row["window"]): {key: f(value) if key not in {"mode", "window"} else value for key, value in row.items()} for row in rows}


def load_pressure(path: Path) -> dict[str, dict[str, float | str]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return {row["mode"]: {key: f(value) if key != "mode" else value for key, value in row.items()} for row in rows}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def batch_points(rows: list[dict[str, float | str]], mode: str, horizon: float, batch_size: int) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if row["mode"] != mode:
            continue
        elapsed = float(row["elapsed_s"])
        if elapsed < 0 or elapsed > horizon:
            continue
        xs.append(elapsed)
        ys.append(float(row["execution_overhead_ms"]) * batch_size)
    return xs, ys


def clipped(values: list[float], cap: float) -> list[float]:
    return [min(value, cap) if value == value else value for value in values]


def bin_batch_rows(rows: list[dict[str, float | str]], mode: str, bin_s: float, horizon: float, batch_size: int) -> tuple[list[float], list[float]]:
    nbins = int(horizon / bin_s) + 1
    buckets: list[list[float]] = [[] for _ in range(nbins)]
    for row in rows:
        if row["mode"] != mode:
            continue
        elapsed = float(row["elapsed_s"])
        if elapsed < 0 or elapsed > horizon:
            continue
        idx = min(nbins - 1, int(elapsed / bin_s))
        buckets[idx].append(float(row["execution_overhead_ms"]) * batch_size)
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    ys = [percentile(bucket, 0.95) if bucket else float("nan") for bucket in buckets]
    return xs, ys


def load_period_bins(root: Path, mode: str, bin_s: float, horizon: float, case_prefix: str) -> tuple[list[float], list[float]]:
    nbins = int(horizon / bin_s) + 1
    buckets: list[list[float]] = [[] for _ in range(nbins)]
    for rep in REPS:
        path = root / mode / f"{case_prefix}_rep{rep:02d}" / LABEL / "period_groups.csv"
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                elapsed = (f(row.get("bin_start_ms")) + f(row.get("bin_end_ms"))) / 2000.0
                if elapsed < 0 or elapsed > horizon:
                    continue
                idx = min(nbins - 1, int(elapsed / bin_s))
                buckets[idx].append(f(row.get("existing_neighbor_update_active_workers")))
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    ys = [statistics.mean(bucket) if bucket else 0.0 for bucket in buckets]
    return xs, ys


def load_period_metric_bins(
    root: Path, mode: str, column: str, bin_s: float, horizon: float, case_prefix: str, scale: float = 1.0
) -> tuple[list[float], list[float]]:
    nbins = int(horizon / bin_s) + 1
    buckets: list[list[float]] = [[] for _ in range(nbins)]
    for rep in REPS:
        path = root / mode / f"{case_prefix}_rep{rep:02d}" / LABEL / "period_groups.csv"
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                elapsed = (f(row.get("bin_start_ms")) + f(row.get("bin_end_ms"))) / 2000.0
                if elapsed < 0 or elapsed > horizon:
                    continue
                value = f(row.get(column))
                if value <= 0:
                    continue
                idx = min(nbins - 1, int(elapsed / bin_s))
                buckets[idx].append(value * scale)
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    ys = [statistics.mean(bucket) if bucket else float("nan") for bucket in buckets]
    return xs, ys


def scheduled_rate_series(windows: list[tuple[float, float]], horizon: float, bin_s: float, insert_rate: float) -> tuple[list[float], list[float]]:
    nbins = int(horizon / bin_s) + 1
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    ys = []
    for x in xs:
        ys.append(insert_rate if any(start <= x <= end for start, end in windows) else 0.0)
    return xs, ys


def annotate_window(ax: plt.Axes, windows: list[tuple[float, float]], labels: bool = True) -> None:
    for idx, (start, end) in enumerate(windows):
        ax.axvspan(start, end, color="#e5e7eb", alpha=0.55, lw=0)
        ax.axvline(start, color="#6b7280", lw=0.45)
        ax.axvline(end, color="#6b7280", lw=0.45)
        if labels and idx == 0:
            ymax = ax.get_ylim()[1]
            ax.annotate("begin\ninsert", xy=(start, ymax * 0.55), xytext=(start - 0.8, ymax * 0.78), arrowprops={"arrowstyle": "->", "lw": 0.45}, ha="right", fontsize=6.3)
            ax.annotate("end\ninsert", xy=(end, ymax * 0.34), xytext=(end + 0.55, ymax * 0.61), arrowprops={"arrowstyle": "->", "lw": 0.45}, ha="left", fontsize=6.3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--bin-s", type=float, default=0.20)
    parser.add_argument("--case-prefix", default="shared_worker_pool")
    args = parser.parse_args()

    windows, horizon, insert_rate = load_config(args.config)
    batch_rows = load_rows(args.control_root / "m3_sift1m_burst_matched_control_batches.csv")
    summary = load_summary(args.control_root / "m3_sift1m_burst_matched_control_summary.csv")
    pressure = load_pressure(args.control_root / "m3_sift1m_burst_matched_control_pressure.csv")

    x_full, y_full = bin_batch_rows(batch_rows, FULL, args.bin_s, horizon, args.batch_size)
    x_path, y_path = bin_batch_rows(batch_rows, PATH, args.bin_s, horizon, args.batch_size)
    x_full_pts, y_full_pts = batch_points(batch_rows, FULL, horizon, args.batch_size)
    x_path_pts, y_path_pts = batch_points(batch_rows, PATH, horizon, args.batch_size)
    x_phase_full, y_phase_full = load_period_bins(args.full_root, FULL, args.bin_s, horizon, args.case_prefix)
    x_phase_path, y_phase_path = load_period_bins(args.path_root, PATH, args.bin_s, horizon, args.case_prefix)
    x_llc_full, y_llc_full = load_period_metric_bins(
        args.full_root, FULL, "perf_llc_load_misses_per_s", args.bin_s, horizon, args.case_prefix, scale=1e-6
    )
    x_llc_path, y_llc_path = load_period_metric_bins(
        args.path_root, PATH, "perf_llc_load_misses_per_s", args.bin_s, horizon, args.case_prefix, scale=1e-6
    )
    x_dtlb_full, y_dtlb_full = load_period_metric_bins(
        args.full_root, FULL, "perf_dtlb_load_misses_walk_completed_per_s", args.bin_s, horizon, args.case_prefix, scale=1e-6
    )
    x_dtlb_path, y_dtlb_path = load_period_metric_bins(
        args.path_root, PATH, "perf_dtlb_load_misses_walk_completed_per_s", args.bin_s, horizon, args.case_prefix, scale=1e-6
    )
    x_sched, y_sched = scheduled_rate_series(windows, horizon, args.bin_s, insert_rate)

    fig = plt.figure(figsize=(6.9, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.0, 1.0], hspace=0.08)
    ax_lat = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[1, 0], sharex=ax_lat)
    ax_hw = fig.add_subplot(gs[2, 0], sharex=ax_lat)
    fig.patch.set_facecolor("white")

    latency_cap = 120.0
    clipped_full = sum(1 for value in y_full_pts if value > latency_cap)
    clipped_path = sum(1 for value in y_path_pts if value > latency_cap)
    ax_lat.scatter(x_full_pts, clipped(y_full_pts, latency_cap), s=4.5, color="#4f46e5", alpha=0.18, linewidths=0)
    ax_lat.scatter(x_path_pts, clipped(y_path_pts, latency_cap), s=4.5, color="#6b7280", alpha=0.14, linewidths=0)
    ax_lat.plot(x_full, clipped(y_full, latency_cap), color="#4f46e5", lw=1.25, label="full insert p95/bin")
    ax_lat.plot(x_path, clipped(y_path, latency_cap), color="#6b7280", lw=1.05, label="repair-disabled control p95/bin")
    ax_lat.set_title("(a) Search execution overhead: per-batch points + p95/bin", loc="left", pad=1)
    ax_lat.set_ylabel("Latency\n(ms, cap 120)")
    ax_lat.set_ylim(0, latency_cap + 20.0)
    ax_lat.axhline(latency_cap, color="#9ca3af", lw=0.6, ls=":", zorder=1)
    annotate_window(ax_lat, windows, labels=False)
    full_release = float(summary[(FULL, "release")]["batch_equiv_execution_overhead_mean_ms"])
    path_release = float(summary[(PATH, "release")]["batch_equiv_execution_overhead_mean_ms"])
    ax_lat.text(
        0.012,
        0.98,
        f"48 shared foreground workers\n"
        f"release mean: full {full_release:.1f} ms vs control {path_release:.1f} ms\n"
        f"clipped above 120 ms: full {clipped_full}, control {clipped_path}",
        transform=ax_lat.transAxes,
        fontsize=6.4,
        color="#111827",
        va="top",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#e5e7eb", "alpha": 0.88},
    )
    ax_lat.text(0.68, 0.73, "full insert p95/bin", transform=ax_lat.transAxes, color="#4f46e5", fontsize=6.5)
    ax_lat.text(0.68, 0.63, "control p95/bin", transform=ax_lat.transAxes, color="#6b7280", fontsize=6.5)

    ax_phase.plot(x_phase_full, y_phase_full, color="#dc2626", lw=1.05, label="full existing-node repair")
    ax_phase.plot(x_phase_path, y_phase_path, color="#6b7280", lw=0.95, label="repair-disabled control")
    ax_sched2 = ax_phase.twinx()
    ax_sched2.plot(x_sched, y_sched, color="#111827", lw=0.85, ls="--", label="insert release")
    ax_phase.set_title("(b) Insert mutation period", loc="left", pad=1)
    ax_phase.set_ylabel("Repair\nworkers")
    ax_sched2.set_ylabel("Release\n(k pts/s)")
    ax_phase.set_xlabel("Elapsed time (s)")
    ax_phase.set_ylim(0, max(1.0, max(y_phase_full) * 1.25))
    ax_sched2.set_ylim(0, max(1.0, insert_rate * 1.35))
    annotate_window(ax_phase, windows)
    lines, labels = ax_phase.get_legend_handles_labels()
    lines2, labels2 = ax_sched2.get_legend_handles_labels()
    ax_phase.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right", handlelength=1.4)

    ax_hw.plot(x_llc_full, y_llc_full, color="#0f766e", lw=1.1, label="full LLC miss")
    ax_hw.plot(x_llc_path, y_llc_path, color="#6b7280", lw=0.95, label="control LLC miss")
    ax_hw_dtlb = ax_hw.twinx()
    ax_hw_dtlb.plot(x_dtlb_full, y_dtlb_full, color="#f97316", lw=0.9, ls="--", alpha=0.9, label="full DTLB walk")
    ax_hw_dtlb.plot(x_dtlb_path, y_dtlb_path, color="#9ca3af", lw=0.8, ls="--", alpha=0.75, label="control DTLB walk")
    ax_hw.set_title("(c) Hardware pressure", loc="left", pad=1)
    ax_hw.set_ylabel("LLC miss\n(M/s)")
    ax_hw_dtlb.set_ylabel("DTLB walk\n(M/s)")
    ax_hw.set_xlabel("Elapsed time (s)")
    annotate_window(ax_hw, windows, labels=False)
    hw_lines, hw_labels = ax_hw.get_legend_handles_labels()
    dtlb_lines, dtlb_labels = ax_hw_dtlb.get_legend_handles_labels()
    ax_hw.legend(hw_lines + dtlb_lines, hw_labels + dtlb_labels, frameon=False, loc="upper right", ncol=2, handlelength=1.4)

    for ax in (ax_lat, ax_phase, ax_hw):
        ax.grid(axis="y", color="#e5e7eb", lw=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_sched2.spines["top"].set_visible(False)
    ax_hw_dtlb.spines["top"].set_visible(False)
    ax_lat.tick_params(labelbottom=False)
    ax_phase.tick_params(labelbottom=False)
    ax_lat.set_xlim(0, horizon)
    ax_phase.set_xlim(0, horizon)
    ax_hw.set_xlim(0, horizon)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"[m3-odin-style] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
