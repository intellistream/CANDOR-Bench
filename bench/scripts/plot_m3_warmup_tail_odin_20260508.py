#!/usr/bin/env python3
"""Plot the corrected M3 tail-latency attribution."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


INSERT_WINDOWS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]
EARLY_WINDOW = (500.0, 2500.0)
STABLE_OUTSIDE = [(3500.0, 6000.0), (7000.0, 9500.0), (10500.0, 12000.0)]


def percentile(values: list[float], q: float) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def in_ranges(ms: float, ranges: list[tuple[float, float]]) -> bool:
    return any(start <= ms < end for start, end in ranges)


def load_samples(root: Path, dataset: str, algorithm: str) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    for run_dir in sorted(root.glob(f"{dataset}_{algorithm}_rep*")):
        path = run_dir / "raw_latency" / "search_compact.csv"
        if not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("measured") != "1":
                    continue
                try:
                    samples.append((float(row["finish_ms"]), float(row["op_ms"])))
                except (KeyError, ValueError):
                    continue
    return samples


def binned_p99(samples: list[tuple[float, float]], bin_ms: float) -> tuple[list[float], list[float]]:
    bins: dict[int, list[float]] = defaultdict(list)
    for finish_ms, op_ms in samples:
        if finish_ms < 0:
            continue
        idx = int(finish_ms // bin_ms)
        bins[idx].append(op_ms)
    xs: list[float] = []
    ys: list[float] = []
    for idx in sorted(bins):
        if len(bins[idx]) < 20:
            continue
        xs.append((idx + 0.5) * bin_ms / 1000.0)
        ys.append(percentile(bins[idx], 0.99))
    return xs, ys


def segment_stats(samples: list[tuple[float, float]]) -> dict[str, float]:
    early = [op for ms, op in samples if EARLY_WINDOW[0] <= ms < EARLY_WINDOW[1]]
    insert = [op for ms, op in samples if in_ranges(ms, INSERT_WINDOWS)]
    stable = [op for ms, op in samples if in_ranges(ms, STABLE_OUTSIDE)]
    all_ops = [op for _, op in samples]
    threshold = percentile(all_ops, 0.99)
    top = [(ms, op) for ms, op in samples if op >= threshold]
    return {
        "early_p99": percentile(early, 0.99),
        "insert_p99": percentile(insert, 0.99),
        "stable_p99": percentile(stable, 0.99),
        "top_insert_pct": 100.0
        * sum(1 for ms, _ in top if in_ranges(ms, INSERT_WINDOWS))
        / max(1, len(top)),
        "insert_sample_pct": 100.0
        * sum(1 for ms, _ in samples if in_ranges(ms, INSERT_WINDOWS))
        / max(1, len(samples)),
    }


def work_counter_summary(path: Path) -> dict[str, float]:
    batches: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("measured") != "1":
                continue
            key = (row["start_raw_ns"], row["finish_raw_ns"])
            item = batches.setdefault(
                key,
                {
                    "op_ms": float(row["op_ms"]),
                    "sum_base": 0.0,
                    "sum_dist": 0.0,
                    "sum_edges": 0.0,
                    "sum_lock": 0.0,
                },
            )
            item["sum_base"] += float(row.get("work_base_search_ms", 0) or 0)
            item["sum_dist"] += float(row.get("work_distance_computations", 0) or 0)
            item["sum_edges"] += float(row.get("work_level0_edges_scanned", 0) or 0)
            item["sum_lock"] += float(row.get("work_level0_lock_wait_ms", 0) or 0)
    rows = list(batches.values())
    threshold = percentile([r["op_ms"] for r in rows], 0.99)
    top = [r for r in rows if r["op_ms"] >= threshold]
    rest = [r for r in rows if r["op_ms"] < threshold]

    def mean(items: list[dict[str, float]], key: str) -> float:
        vals = [r[key] for r in items if math.isfinite(r[key])]
        return sum(vals) / len(vals) if vals else float("nan")

    def ratio(key: str) -> float:
        denom = mean(rest, key)
        return mean(top, key) / denom if denom > 0 else 0.0

    return {
        "base_time_ratio": ratio("sum_base"),
        "distance_ratio": ratio("sum_dist"),
        "edges_ratio": ratio("sum_edges"),
        "lock_wait_ratio": ratio("sum_lock"),
    }


def write_summary_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def shade_periods(ax: plt.Axes) -> None:
    ax.axvspan(EARLY_WINDOW[0] / 1000.0, EARLY_WINDOW[1] / 1000.0, color="#d8dee9", alpha=0.65, lw=0)
    for start, end in INSERT_WINDOWS:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#f4c7c3", alpha=0.62, lw=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("result/m3_prune_only_tail_matrix_20260508/clean"))
    parser.add_argument("--workcounter", type=Path, default=Path("result/m3_prune_only_tail_matrix_20260508_workcounter1/clean/sift200k_annchor-m3_rep1/raw_latency/search_wide.csv"))
    parser.add_argument("--out", type=Path, default=Path("../paper-progress/m3/search-tail-rootcause/figures/15_odin_m3_warmup_tail.png"))
    parser.add_argument("--pdf", type=Path, default=Path("../paper-progress/m3/search-tail-rootcause/figures/15_odin_m3_warmup_tail.pdf"))
    parser.add_argument("--csv", type=Path, default=Path("../paper-progress/m3/search-tail-rootcause/figures/15_odin_m3_warmup_tail.csv"))
    args = parser.parse_args()

    datasets = ["sift200k", "sift10m200k", "sift1m", "gist200k", "dbpedia50k"]
    display = {
        "sift200k": "SIFT\n200K",
        "sift10m200k": "SIFT10M\n200K",
        "sift1m": "SIFT\n1M",
        "gist200k": "GIST\n200K",
        "dbpedia50k": "DBpedia\n50K",
    }

    all_stats: list[dict[str, str | float]] = []
    samples_by_dataset: dict[str, list[tuple[float, float]]] = {}
    for dataset in datasets:
        samples = load_samples(args.root, dataset, "annchor-m3")
        samples_by_dataset[dataset] = samples
        stats = segment_stats(samples)
        all_stats.append({"dataset": dataset, **stats})

    work = work_counter_summary(args.workcounter)

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.8,
        }
    )

    blue = "#2f6fdd"
    navy = "#263445"
    coral = "#d95f59"
    slate = "#6b7280"
    green = "#2f9b7a"
    grid = "#e5e7eb"

    fig = plt.figure(figsize=(7.15, 4.25), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.03], width_ratios=[1.28, 1.0], hspace=0.36, wspace=0.29)
    ax_t = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_w = fig.add_subplot(gs[1, 1])

    # Panel A: time line.
    shade_periods(ax_t)
    x, y = binned_p99(samples_by_dataset["sift200k"], 250.0)
    ax_t.plot(x, y, color=navy, lw=1.8, solid_capstyle="round", label="Measured read P99")
    ax_t.set_xlim(0.0, 12.0)
    ax_t.set_ylim(0.0, max(3.0, np.nanmax(y) * 1.15))
    ax_t.set_ylabel("Latency (ms)")
    ax_t.set_xlabel("Elapsed time (s)")
    ax_t.grid(True, axis="y", color=grid, lw=0.7)
    ax_t.spines["top"].set_visible(False)
    ax_t.spines["right"].set_visible(False)
    ax_t.legend(
        handles=[
            plt.Line2D([0], [0], color=navy, lw=1.8, label="Measured read P99"),
            Patch(facecolor="#d8dee9", edgecolor="none", alpha=0.65, label="Early warmup"),
            Patch(facecolor="#f4c7c3", edgecolor="none", alpha=0.62, label="Insert window"),
        ],
        loc="upper right",
        ncols=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )
    ax_t.text(0.012, 0.93, "a", transform=ax_t.transAxes, ha="left", va="top", fontsize=11.0)

    # Panel B: dataset replication.
    xpos = np.arange(len(datasets))
    width = 0.24
    early = [float(r["early_p99"]) for r in all_stats]
    insert = [float(r["insert_p99"]) for r in all_stats]
    stable = [float(r["stable_p99"]) for r in all_stats]
    ax_b.bar(xpos - width, early, width=width, color=slate, label="Early warmup")
    ax_b.bar(xpos, insert, width=width, color=coral, label="Insert window")
    ax_b.bar(xpos + width, stable, width=width, color=blue, label="Later outside")
    ax_b.set_ylabel("P99 latency (ms)")
    ax_b.set_xticks(xpos)
    ax_b.set_xticklabels([display[d] for d in datasets])
    ax_b.grid(True, axis="y", color=grid, lw=0.7)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.legend(frameon=False, ncols=1, loc="upper right", handlelength=1.3)
    ax_b.text(0.012, 0.94, "b", transform=ax_b.transAxes, ha="left", va="top", fontsize=11.0)

    # Panel C: sanity check from search work counters.
    labels = ["Base\nTime", "Distance\nWork", "Edges\nScanned", "Lock\nWait"]
    ratios = [
        work["base_time_ratio"],
        work["distance_ratio"],
        work["edges_ratio"],
        work["lock_wait_ratio"],
    ]
    colors = [green, slate, slate, slate]
    y_pos = np.arange(len(labels))
    ax_w.barh(y_pos, ratios, color=colors, height=0.55)
    ax_w.axvline(1.0, color="#222222", lw=0.85, linestyle=(0, (2, 2)))
    ax_w.set_yticks(y_pos)
    ax_w.set_yticklabels(labels)
    ax_w.invert_yaxis()
    ax_w.set_xlabel("Top-tail / rest")
    ax_w.set_xlim(0.0, max(2.5, max(ratios) * 1.15))
    ax_w.grid(True, axis="x", color=grid, lw=0.7)
    ax_w.spines["top"].set_visible(False)
    ax_w.spines["right"].set_visible(False)
    for idx, value in enumerate(ratios):
        ax_w.text(value + 0.04, idx, f"{value:.2f}x", ha="left", va="center", fontsize=9.0, color="#1f2937")
    ax_w.text(0.012, 0.94, "c", transform=ax_w.transAxes, ha="left", va="top", fontsize=11.0)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.12)

    for path in [args.out, args.pdf, args.csv]:
        path.parent.mkdir(parents=True, exist_ok=True)
    write_summary_csv(args.csv, all_stats + [{"dataset": "work_counter", **work}])
    fig.savefig(args.out, dpi=300)
    fig.savefig(args.pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
