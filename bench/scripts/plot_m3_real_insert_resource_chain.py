#!/usr/bin/env python3
"""Plot full-insert latency, writer phase, and resource-pressure alignment."""

from __future__ import annotations

import argparse
import bisect
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
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.75,
    }
)


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * pct)]


def corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return 0.0
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    mx = statistics.mean(xvals)
    my = statistics.mean(yvals)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xvals))
    sy = math.sqrt(sum((y - my) ** 2 for y in yvals))
    if sx <= 0 or sy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy)


def normalize(values: list[float]) -> list[float]:
    scale = percentile([v for v in values if v > 0 and math.isfinite(v)], 0.95)
    if scale <= 0:
        scale = max([v for v in values if math.isfinite(v)] or [1.0])
    return [min(v / scale, 1.25) if scale > 0 and math.isfinite(v) else float("nan") for v in values]


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
    return windows, f(workload.get("schedule_horizon_ms")) / 1000.0, f(workload.get("insert_event_rate")) / 1000.0


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


def load_period_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            rows.append(
                {
                    "elapsed_s": (f(row.get("bin_start_ms")) + f(row.get("bin_end_ms"))) / 2000.0,
                    "path_workers": f(row.get("insert_path_search_active_workers")),
                    "existing_node_repair_workers": f(row.get("existing_neighbor_update_active_workers")),
                    "l2_miss_rate": f(row.get("perf_l2_rqsts_miss_per_s")),
                    "llc_miss_rate": f(row.get("perf_llc_load_misses_per_s")),
                    "dtlb_walk_rate": f(row.get("perf_dtlb_load_misses_walk_completed_per_s")),
                }
            )
    return rows


def window_class(elapsed: float, windows: list[tuple[float, float]], drain_s: float = 1.0) -> str:
    for start, end in windows:
        if start <= elapsed <= end:
            return "release"
        if end < elapsed <= end + drain_s:
            return "drain"
    return "quiet"


def build_rows(full_root: Path, readonly_root: Path, windows: list[tuple[float, float]], batch_size: int) -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for rep in REPS:
        case = CASE_TEMPLATE.format(rep=rep)
        run_dir = full_root / "full_insert" / case
        full = load_search_batches(run_dir / "raw_latency/search_wide.csv", batch_size)
        readonly = load_search_batches(readonly_root / "readonly_noop" / case / "raw_latency/search_wide.csv", batch_size)
        period_rows = load_period_rows(run_dir / "period_groups.csv")
        period_times = [row["elapsed_s"] for row in period_rows]
        for idx in sorted(set(full) & set(readonly)):
            elapsed = full[idx]["finish_ms"] / 1000.0
            period_idx = bisect.bisect_right(period_times, elapsed) - 1
            if period_idx < 0:
                continue
            period = period_rows[min(period_idx, len(period_rows) - 1)]
            out.append(
                {
                    "rep": rep,
                    "batch_idx": idx,
                    "elapsed_s": elapsed,
                    "window": window_class(elapsed, windows),
                    "execution_overhead_ms": (full[idx]["op_ms"] - readonly[idx]["op_ms"]) * batch_size,
                    "e2e_overhead_ms": (full[idx]["e2e_ms"] - readonly[idx]["e2e_ms"]) * batch_size,
                    **{k: v for k, v in period.items() if k != "elapsed_s"},
                }
            )
    return out


def bin_rows(rows: list[dict[str, float | str]], horizon: float, bin_s: float, key: str, op: str = "median") -> tuple[list[float], list[float]]:
    nbins = int(horizon / bin_s) + 1
    buckets: list[list[float]] = [[] for _ in range(nbins)]
    for row in rows:
        elapsed = float(row["elapsed_s"])
        if elapsed < 0 or elapsed > horizon:
            continue
        idx = min(nbins - 1, int(elapsed / bin_s))
        value = float(row[key])
        if math.isfinite(value):
            buckets[idx].append(value)
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    if op == "mean":
        ys = [statistics.mean(bucket) if bucket else float("nan") for bucket in buckets]
    else:
        ys = [statistics.median(bucket) if bucket else float("nan") for bucket in buckets]
    return xs, ys


def summarize(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    out = []
    metrics = [
        ("execution_overhead_ms", "search execution overhead ms"),
        ("path_workers", "writer graph traversal workers"),
        ("existing_node_repair_workers", "existing-node repair workers"),
        ("llc_miss_rate", "LLC load misses per s"),
        ("dtlb_walk_rate", "DTLB walks per s"),
        ("l2_miss_rate", "L2 misses per s"),
    ]
    for window in ("release", "drain", "quiet"):
        subset = [row for row in rows if row["window"] == window]
        row_out: dict[str, float | str] = {"window": window, "batches": len(subset)}
        for key, label in metrics:
            values = [float(row[key]) for row in subset]
            row_out[label + " mean"] = statistics.mean(values) if values else 0.0
            row_out[label + " p95"] = percentile(values, 0.95)
        out.append(row_out)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, summary_rows: list[dict[str, float | str]], rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        file.write("# M3 Full-Insert Resource Chain\n\n")
        file.write("Scope: real full-insert burst run, matched read-only replay, 3 repeats. The purpose is to explain what inside the insert burst coincides with search execution latency, beyond the fact that writes are present.\n\n")
        file.write("| window | batches | exec mean ms | traversal workers mean | LLC M/s mean | DTLB M/s mean |\n")
        file.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            file.write(
                f"| {row['window']} | {int(row['batches'])} | "
                f"{float(row['search execution overhead ms mean']):.2f} | "
                f"{float(row['writer graph traversal workers mean']):.2f} | "
                f"{float(row['LLC load misses per s mean']) / 1e6:.1f} | "
                f"{float(row['DTLB walks per s mean']) / 1e6:.1f} |\n"
            )
        file.write("\n")
        exec_vals = [float(row["execution_overhead_ms"]) for row in rows]
        for key, label in [
            ("path_workers", "writer graph traversal workers"),
            ("llc_miss_rate", "LLC load miss rate"),
            ("dtlb_walk_rate", "DTLB walk rate"),
            ("l2_miss_rate", "L2 miss rate"),
        ]:
            file.write(f"- correlation with execution overhead, {label}: {corr([float(row[key]) for row in rows], exec_vals):.3f}\n")
        file.write("\nInterpretation: insert bursts matter because they put writer HNSW graph traversal on the same memory-intensive path as readers. The timeline should show whether search overhead aligns with writer traversal occupancy and cache/TLB pressure, not merely with the scheduled write window.\n")


def annotate_windows(ax: plt.Axes, windows: list[tuple[float, float]]) -> None:
    for idx, (start, end) in enumerate(windows):
        ax.axvspan(start, end, color="#e5e7eb", alpha=0.48, lw=0)
        ax.axvline(start, color="#6b7280", lw=0.42)
        ax.axvline(end, color="#6b7280", lw=0.42)
        if idx == 0:
            ymax = ax.get_ylim()[1]
            ax.annotate("begin\ninsert", xy=(start, ymax * 0.55), xytext=(start - 0.85, ymax * 0.78), arrowprops={"arrowstyle": "->", "lw": 0.45}, ha="right", fontsize=6.2)
            ax.annotate("end\ninsert", xy=(end, ymax * 0.35), xytext=(end + 0.58, ymax * 0.62), arrowprops={"arrowstyle": "->", "lw": 0.45}, ha="left", fontsize=6.2)


def plot(out: Path, rows: list[dict[str, float | str]], windows: list[tuple[float, float]], horizon: float, bin_s: float) -> None:
    x_exec, y_exec = bin_rows(rows, horizon, bin_s, "execution_overhead_ms", "median")
    x_path, y_path = bin_rows(rows, horizon, bin_s, "path_workers", "mean")
    x_repair, y_repair = bin_rows(rows, horizon, bin_s, "existing_node_repair_workers", "mean")
    x_llc, y_llc = bin_rows(rows, horizon, bin_s, "llc_miss_rate", "mean")
    _, y_dtlb = bin_rows(rows, horizon, bin_s, "dtlb_walk_rate", "mean")
    _, y_l2 = bin_rows(rows, horizon, bin_s, "l2_miss_rate", "mean")
    y_llc_n = normalize(y_llc)
    y_dtlb_n = normalize(y_dtlb)
    y_l2_n = normalize(y_l2)

    fig = plt.figure(figsize=(5.25, 3.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.10)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    fig.patch.set_facecolor("white")

    ax0.plot(x_exec, y_exec, color="#4f46e5", lw=1.15)
    ax0.set_title("(a) Search P50 execution overhead", loc="left", pad=1)
    ax0.set_ylabel("Latency\n(ms)")
    ax0.set_ylim(0, max(55.0, max(v for v in y_exec if v == v) * 1.12))
    annotate_windows(ax0, windows)

    ax1.plot(x_path, y_path, color="#374151", lw=1.05, label="writer graph traversal")
    ax1.plot(x_repair, y_repair, color="#dc2626", lw=0.95, label="existing-node repair")
    ax1.set_title("(b) Writer phases", loc="left", pad=1)
    ax1.set_ylabel("Phase\nworkers")
    ax1.set_ylim(0, max(1.0, max([v for v in y_path + y_repair if v == v]) * 1.18))
    annotate_windows(ax1, windows)
    ax1.legend(frameon=False, loc="upper right", handlelength=1.3)

    ax2.plot(x_llc, y_l2_n, color="#059669", lw=0.95, label="L2 miss rate")
    ax2.plot(x_llc, y_llc_n, color="#0891b2", lw=0.95, label="LLC load miss rate")
    ax2.plot(x_llc, y_dtlb_n, color="#9333ea", lw=0.95, label="DTLB walk rate")
    ax2.set_title("(c) Memory-hierarchy pressure", loc="left", pad=1)
    ax2.set_ylabel("Normalized\nrate")
    ax2.set_xlabel("Elapsed time (s)")
    ax2.set_ylim(0, 1.3)
    annotate_windows(ax2, windows)
    ax2.legend(frameon=False, loc="upper right", ncol=3, handlelength=1.1, columnspacing=0.8)

    for ax in (ax0, ax1, ax2):
        ax.grid(axis="y", color="#e5e7eb", lw=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, horizon)
    ax0.tick_params(labelbottom=False)
    ax1.tick_params(labelbottom=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--readonly-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--bin-s", type=float, default=0.20)
    args = parser.parse_args()

    windows, horizon, _ = load_config(args.config)
    rows = build_rows(args.full_root, args.readonly_root, windows, args.batch_size)
    summary_rows = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "m3_full_insert_resource_chain_rows.csv", rows)
    write_csv(args.out_dir / "m3_full_insert_resource_chain_summary.csv", summary_rows)
    write_md(args.out_dir / "m3_full_insert_resource_chain.md", summary_rows, rows)
    plot(args.out_dir / "m3_full_insert_resource_chain.png", rows, windows, horizon, args.bin_s)
    print(f"[m3-resource-chain] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
