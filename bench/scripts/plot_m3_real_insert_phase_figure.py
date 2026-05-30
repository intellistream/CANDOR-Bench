#!/usr/bin/env python3
"""Plot realistic full-insert phase evidence without ablation modes."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


LABEL = "sift1m_b200_round_robin_burst_w40000_r5000_nolock"
CASE_TEMPLATE = "high_writer_concurrency_rep{rep:02d}/" + LABEL
REPS = (1, 2, 3)


plt.rcParams.update(
    {
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.4,
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


def load_search_batches(path: Path, batch_size: int) -> dict[int, dict[str, float]]:
    batches: dict[int, dict[str, float]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            idx = int(f(row.get("sample_idx"))) // batch_size
            batch = batches.setdefault(idx, {"op_ms": 0.0, "e2e_ms": 0.0, "finish_ms": 0.0, "count": 0.0})
            batch["op_ms"] += f(row.get("op_ms"))
            batch["e2e_ms"] += f(row.get("e2e_ms"))
            batch["finish_ms"] = max(batch["finish_ms"], f(row.get("finish_ms")))
            batch["count"] += 1.0
    for batch in batches.values():
        if batch["count"] > 0:
            batch["op_ms"] /= batch["count"]
            batch["e2e_ms"] /= batch["count"]
    return batches


def collect_overheads(full_root: Path, readonly_root: Path, batch_size: int) -> list[dict[str, float]]:
    rows = []
    for rep in REPS:
        case = CASE_TEMPLATE.format(rep=rep)
        full = load_search_batches(full_root / "full_insert" / case / "raw_latency/search_wide.csv", batch_size)
        readonly = load_search_batches(readonly_root / "readonly_noop" / case / "raw_latency/search_wide.csv", batch_size)
        for idx in sorted(set(full) & set(readonly)):
            rows.append(
                {
                    "rep": float(rep),
                    "batch_idx": float(idx),
                    "elapsed_s": full[idx]["finish_ms"] / 1000.0,
                    "exec_overhead_ms": (full[idx]["op_ms"] - readonly[idx]["op_ms"]) * batch_size,
                    "e2e_overhead_ms": (full[idx]["e2e_ms"] - readonly[idx]["e2e_ms"]) * batch_size,
                }
            )
    return rows


def collect_periods(full_root: Path) -> list[dict[str, float]]:
    rows = []
    for rep in REPS:
        case = CASE_TEMPLATE.format(rep=rep)
        path = full_root / "full_insert" / case / "period_groups.csv"
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                rows.append(
                    {
                        "rep": float(rep),
                        "elapsed_s": (f(row.get("bin_start_ms")) + f(row.get("bin_end_ms"))) / 2000.0,
                        "repair_workers": f(row.get("existing_neighbor_update_active_workers")),
                        "path_workers": f(row.get("insert_path_search_active_workers")),
                        "llc_miss_rate": f(row.get("perf_llc_load_misses_per_s")),
                        "dtlb_walk_rate": f(row.get("perf_dtlb_load_misses_walk_completed_per_s")),
                    }
                )
    return rows


def bin_values(rows: list[dict[str, float]], key: str, horizon: float, bin_s: float, op: str = "median") -> tuple[list[float], list[float]]:
    nbins = int(horizon / bin_s) + 1
    buckets: list[list[float]] = [[] for _ in range(nbins)]
    for row in rows:
        elapsed = row["elapsed_s"]
        if elapsed < 0 or elapsed > horizon:
            continue
        idx = min(nbins - 1, int(elapsed / bin_s))
        value = row.get(key, 0.0)
        if math.isfinite(value):
            buckets[idx].append(value)
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    if op == "mean":
        ys = [statistics.mean(bucket) if bucket else float("nan") for bucket in buckets]
    else:
        ys = [statistics.median(bucket) if bucket else float("nan") for bucket in buckets]
    return xs, ys


def window_label(elapsed: float, windows: list[tuple[float, float]], drain_s: float = 0.0) -> str:
    for start, end in windows:
        if start <= elapsed <= end:
            return "release"
        if end < elapsed <= end + drain_s:
            return "drain"
    return "quiet"


def summarize(overheads: list[dict[str, float]], windows: list[tuple[float, float]]) -> list[dict[str, float | str]]:
    rows = []
    for label in ("release", "drain", "quiet"):
        vals = [row["exec_overhead_ms"] for row in overheads if window_label(row["elapsed_s"], windows, 1.0) == label]
        rows.append(
            {
                "window": label,
                "batches": len(vals),
                "exec_mean_ms": statistics.mean(vals) if vals else 0.0,
                "exec_p50_ms": statistics.median(vals) if vals else 0.0,
                "exec_p95_ms": sorted(vals)[int((len(vals) - 1) * 0.95)] if vals else 0.0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def annotate_windows(ax: plt.Axes, windows: list[tuple[float, float]]) -> None:
    for idx, (start, end) in enumerate(windows):
        ax.axvspan(start, end, color="#e5e7eb", alpha=0.55, lw=0)
        ax.axvline(start, color="#6b7280", lw=0.45)
        ax.axvline(end, color="#6b7280", lw=0.45)
        if idx == 0:
            ymax = ax.get_ylim()[1]
            ax.annotate("begin\ninsert", xy=(start, ymax * 0.55), xytext=(start - 0.8, ymax * 0.78), arrowprops={"arrowstyle": "->", "lw": 0.45}, ha="right", fontsize=6.3)
            ax.annotate("end\ninsert", xy=(end, ymax * 0.34), xytext=(end + 0.55, ymax * 0.61), arrowprops={"arrowstyle": "->", "lw": 0.45}, ha="left", fontsize=6.3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--readonly-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--bin-s", type=float, default=0.20)
    args = parser.parse_args()

    windows, horizon, insert_rate = load_config(args.config)
    overheads = collect_overheads(args.full_root, args.readonly_root, args.batch_size)
    periods = collect_periods(args.full_root)
    summary = summarize(overheads, windows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "m3_real_full_insert_phase_overheads.csv", overheads)
    write_csv(args.out_dir / "m3_real_full_insert_phase_summary.csv", summary)

    x_lat, y_lat = bin_values(overheads, "exec_overhead_ms", horizon, args.bin_s, "median")
    x_repair, y_repair = bin_values(periods, "repair_workers", horizon, args.bin_s, "mean")
    x_path, y_path = bin_values(periods, "path_workers", horizon, args.bin_s, "mean")
    x_sched = [(idx + 0.5) * args.bin_s for idx in range(int(horizon / args.bin_s) + 1)]
    y_sched = [insert_rate if any(start <= x <= end for start, end in windows) else 0.0 for x in x_sched]

    fig = plt.figure(figsize=(6.9, 2.78), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.72, 1.0], height_ratios=[1.0, 1.0], wspace=0.18, hspace=0.10)
    ax_lat = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[1, 0], sharex=ax_lat)
    ax_sum = fig.add_subplot(gs[:, 1])
    fig.patch.set_facecolor("white")

    ax_lat.plot(x_lat, y_lat, color="#4f46e5", lw=1.2)
    ax_lat.set_title("(a) Search P50 execution overhead", loc="left", pad=1)
    ax_lat.set_ylabel("Latency\n(ms)")
    ax_lat.set_ylim(0, max(55.0, max(v for v in y_lat if v == v) * 1.12))
    annotate_windows(ax_lat, windows)

    ax_phase.plot(x_repair, y_repair, color="#dc2626", lw=1.05, label="existing-node repair")
    ax_phase.plot(x_path, y_path, color="#4b5563", lw=0.9, label="insert path search")
    ax_sched = ax_phase.twinx()
    ax_sched.plot(x_sched, y_sched, color="#111827", lw=0.85, ls="--", label="insert release")
    ax_phase.set_title("(b) Real full-insert periods", loc="left", pad=1)
    ax_phase.set_ylabel("Phase\nworkers")
    ax_sched.set_ylabel("Release\n(k pts/s)")
    ax_phase.set_xlabel("Elapsed time (s)")
    ax_phase.set_ylim(0, max(1.0, max(y_path + y_repair) * 1.22))
    ax_sched.set_ylim(0, max(1.0, insert_rate * 1.35))
    annotate_windows(ax_phase, windows)
    lines, labels = ax_phase.get_legend_handles_labels()
    lines2, labels2 = ax_sched.get_legend_handles_labels()
    ax_phase.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right", handlelength=1.3)

    labels = ["release", "drain", "quiet"]
    means = [float(next(row for row in summary if row["window"] == label)["exec_mean_ms"]) for label in labels]
    p95s = [float(next(row for row in summary if row["window"] == label)["exec_p95_ms"]) for label in labels]
    xs = list(range(len(labels)))
    ax_sum.plot(xs, means, marker="s", color="#4f46e5", lw=1.1, label="mean")
    ax_sum.plot(xs, p95s, marker="^", color="#111827", lw=1.0, label="p95")
    ax_sum.set_title("(c) Full-insert windows", loc="left", pad=1)
    ax_sum.set_ylabel("Execution overhead (ms)")
    ax_sum.set_xticks(xs)
    ax_sum.set_xticklabels(labels)
    ax_sum.set_ylim(0, max(means + p95s) * 1.18)
    ax_sum.legend(frameon=False, loc="upper right", handlelength=1.4)

    for ax in (ax_lat, ax_phase, ax_sum):
        ax.grid(axis="y", color="#e5e7eb", lw=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_sched.spines["top"].set_visible(False)
    ax_lat.tick_params(labelbottom=False)
    ax_lat.set_xlim(0, horizon)
    ax_phase.set_xlim(0, horizon)

    out = args.out_dir / "m3_real_full_insert_odin_style_figure2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[m3-real-full-insert] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
