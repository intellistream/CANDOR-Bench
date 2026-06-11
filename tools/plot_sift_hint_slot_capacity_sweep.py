#!/usr/bin/env python3
"""Plot SIFT hint table slot/capacity sweep QPS-recall points."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


SYSTEMS = [
    ("streamseed_hybrid", "HNSW-BriskSeed", "ef-120"),
    ("streamseed_hybrid_symphonyqg", "SymphonyQG-BriskSeed", "ef-120"),
    ("streamseed_hybrid_freshdiskann", "FreshDiskANN-BriskSeed", "L-120_R-48"),
]
DIR_RE = re.compile(r"^test(?P<trial>\d+)-slot(?P<slots>\d+)-capacity(?P<capacity>\d+)$")
CONFIG_ORDER = [
    (6000, 10),
    (6000, 20),
    (8000, 30),
    (6000, 50),
    (6000, 70),
]
SELECTED_CONFIGS = set(CONFIG_ORDER)
CONFIG_RANK = {config: idx for idx, config in enumerate(CONFIG_ORDER)}

OVERLAY_STYLE = {
    "streamseed_hybrid": {"color": "#D55E00", "marker": "o", "label": "HNSW-BriskSeed"},
    "streamseed_hybrid_symphonyqg": {"color": "#009E73", "marker": "s", "label": "SymphonyQG-BriskSeed"},
    "streamseed_hybrid_freshdiskann": {"color": "#E69F00", "marker": "^", "label": "FreshDiskANN-BriskSeed"},
}


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def format_k_ticks(x, _pos=None) -> str:
    if abs(x) < 1000:
        return f"{x:g}"
    value = x / 1000.0
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}k"
    return f"{value:g}k"


def read_final_results(csv_path: Path) -> dict[int, tuple[float, float]]:
    values: dict[int, tuple[float, float]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if row.get("recall") in (None, "") or row.get("query_qps") in (None, ""):
                continue
            batch_idx = int(row.get("batch_idx", row_idx))
            values[batch_idx] = (float(row["recall"]), float(row["query_qps"]))
    if not values:
        raise ValueError(f"No recall/query_qps rows found in {csv_path}")
    return values


def final_results_path(test_dir: Path) -> Path | None:
    expected = test_dir / f"{test_dir.name}_final_results.csv"
    if expected.exists():
        return expected
    matches = sorted(test_dir.glob("*_final_results.csv"))
    return matches[0] if matches else None


def collect_rows(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for system, label, _run_group in SYSTEMS:
        system_dir = results_root / system
        if not system_dir.exists():
            raise FileNotFoundError(f"Missing system result directory: {system_dir}")

        grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
        for test_dir in sorted(system_dir.iterdir()):
            if not test_dir.is_dir():
                continue
            match = DIR_RE.match(test_dir.name)
            if not match:
                continue

            csv_path = final_results_path(test_dir)
            if csv_path is None:
                raise FileNotFoundError(f"No final_results.csv found under {test_dir}")

            batch_values = read_final_results(csv_path)
            slots = int(match.group("slots"))
            capacity = int(match.group("capacity"))
            if (slots, capacity) not in SELECTED_CONFIGS:
                continue
            trial = int(match.group("trial"))
            grouped[(slots, capacity)].append(
                {
                    "trial": trial,
                    "trial_dir": test_dir.name,
                    "batch_values": batch_values,
                    "rows": len(batch_values),
                    "source": str(csv_path),
                }
            )

        for (slots, capacity), trials in sorted(grouped.items(), reverse=True):
            trials.sort(key=lambda row: int(row["trial"]))
            common_batches = sorted(
                set.intersection(*(set(row["batch_values"]) for row in trials))
            )
            if not common_batches:
                raise ValueError(f"No common batch_idx values for {system} slot{slots}-capacity{capacity}")

            median_recalls: list[float] = []
            median_qps_values: list[float] = []
            for batch_idx in common_batches:
                recall_values = [
                    float(row["batch_values"][batch_idx][0]) for row in trials
                ]
                qps_values = [
                    float(row["batch_values"][batch_idx][1]) for row in trials
                ]
                median_recalls.append(float(median(recall_values)))
                median_qps_values.append(float(median(qps_values)))

            rows.append(
                {
                    "system": system,
                    "label": label,
                    "hint_table_slots": slots,
                    "hint_slot_capacity": capacity,
                    "config": f"slot{slots}-capacity{capacity}",
                    "trials": len(trials),
                    "trial_dirs": ";".join(str(row["trial_dir"]) for row in trials),
                    "rows": len(common_batches),
                    "raw_rows": sum(int(row["rows"]) for row in trials),
                    "mean_recall": mean(median_recalls),
                    "mean_query_qps": mean(median_qps_values),
                    "aggregation": "per_batch_median_over_trials_then_mean",
                    "sources": ";".join(str(row["source"]) for row in trials),
                }
            )

    return rows


def write_summary(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "label",
        "hint_table_slots",
        "hint_slot_capacity",
        "config",
        "trials",
        "trial_dirs",
        "rows",
        "raw_rows",
        "mean_recall",
        "mean_query_qps",
        "aggregation",
        "sources",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def config_key(row: dict[str, object]) -> int:
    config = (int(row["hint_table_slots"]), int(row["hint_slot_capacity"]))
    return CONFIG_RANK[config]


def config_label(row: dict[str, object]) -> str:
    return f"{row['hint_table_slots']}/{row['hint_slot_capacity']}"


def style_qps_recall_axis(ax) -> None:
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linestyle="-", linewidth=1.2, alpha=1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(format_k_ticks))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("#333333")


def apply_cache_miss_like_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.75,
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def set_padded_limits(ax, x_values: list[float], y_values: list[float]) -> None:
    if not x_values or not y_values:
        return
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max((x_max - x_min) * 0.14, 0.02)
    y_pad = max((y_max - y_min) * 0.14, 500.0)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def annotate_configs(ax, points: list[dict[str, object]], x_values: list[float], y_values: list[float]) -> None:
    for row, x, y in zip(points, x_values, y_values):
        ax.annotate(
            config_label(row),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9.0,
            color="#333333",
            zorder=4,
        )


def plot_subplots(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    apply_cache_miss_like_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5), dpi=400)

    for ax, (system, _label, _run_group) in zip(axes, SYSTEMS):
        points = [row for row in rows if row["system"] == system]
        points.sort(key=config_key)
        recalls = [float(row["mean_recall"]) * 100.0 for row in points]
        qps_values = [float(row["mean_query_qps"]) for row in points]
        style = OVERLAY_STYLE[system]

        ax.plot(
            recalls,
            qps_values,
            color=style["color"],
            linestyle="-",
            linewidth=1.8,
            marker=style["marker"],
            markersize=8.2,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=1.05,
            alpha=0.96,
        )
        set_padded_limits(ax, recalls, qps_values)
        ax.set_title(style["label"], fontsize=18, fontweight="semibold")
        ax.set_xlabel("Recall (%)", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        style_qps_recall_axis(ax)

    axes[0].set_ylabel("Query QPS", fontsize=16)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0.08)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def overlay_marker_size(marker: str) -> int:
    return 72 if marker != "^" else 78


def plot_overlay(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    apply_cache_miss_like_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=400)
    all_recalls: list[float] = []
    all_qps: list[float] = []

    for system, _label, _run_group in SYSTEMS:
        points = [row for row in rows if row["system"] == system]
        points.sort(key=config_key)
        recalls = [float(row["mean_recall"]) * 100.0 for row in points]
        qps_values = [float(row["mean_query_qps"]) for row in points]
        all_recalls.extend(recalls)
        all_qps.extend(qps_values)
        style = OVERLAY_STYLE[system]

        ax.plot(
            recalls,
            qps_values,
            color=style["color"],
            linestyle="-",
            linewidth=2.1,
            alpha=0.95,
            zorder=2,
        )
        for row, x, y in zip(points, recalls, qps_values):
            marker = style["marker"]
            ax.scatter(
                x,
                y,
                color=style["color"],
                marker=marker,
                s=overlay_marker_size(marker),
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )

    ax.set_xlabel("Recall (%)", fontsize=16)
    ax.set_ylabel("Query QPS", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    style_qps_recall_axis(ax)
    set_padded_limits(ax, all_recalls, all_qps)

    handles = [
        Line2D(
            [0],
            [0],
            color=OVERLAY_STYLE[system]["color"],
            marker=OVERLAY_STYLE[system]["marker"],
            linestyle="-",
            linewidth=2.1,
            markersize=7.8,
            markerfacecolor=OVERLAY_STYLE[system]["color"],
            markeredgecolor="white",
            label=OVERLAY_STYLE[system]["label"],
        )
        for system, _label, _run_group in SYSTEMS
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=True,
        facecolor="#f0f0f0",
        edgecolor="#bdbdbd",
        framealpha=1.0,
        fontsize=12.8,
        handlelength=2.2,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.12, top=0.80)
    fig.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0.05)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def print_ranges(rows: list[dict[str, object]]) -> None:
    print("QPS/recall ranges by system:")
    for system, _label, _run_group in SYSTEMS:
        points = [row for row in rows if row["system"] == system]
        if not points:
            continue

        max_qps = max(float(row["mean_query_qps"]) for row in points)
        min_qps = min(float(row["mean_query_qps"]) for row in points)
        max_recall = max(float(row["mean_recall"]) for row in points)
        min_recall = min(float(row["mean_recall"]) for row in points)
        label = OVERLAY_STYLE[system]["label"]
        print(
            f"  {label}: "
            f"max_qps={max_qps:.3f}, min_qps={min_qps:.3f}, "
            f"max_recall={max_recall:.6f} ({max_recall * 100.0:.3f}%), "
            f"min_recall={min_recall:.6f} ({min_recall * 100.0:.3f}%)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average SIFT slot/capacity sweep final_results.csv files and plot QPS-recall points."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/sift"),
        help="Root containing streamseed_hybrid result directories.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("tools/sift_hint_slot_capacity_qps_recall_summary.csv"),
        help="Output CSV with averaged points.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/sift_hint_slot_capacity_qps_recall_subplots.png"),
        help="Output path for the three-subplot plot.",
    )
    parser.add_argument(
        "--overlay-output",
        type=Path,
        default=Path("tools/sift_hint_slot_capacity_qps_recall_overlay.png"),
        help="Output path for the single-axis overlay plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.results_root)
    if not rows:
        raise SystemExit(f"No matching test*-slot*-capacity* final_results.csv files under {args.results_root}")
    write_summary(rows, args.summary)
    plot_subplots(rows, args.output)
    plot_overlay(rows, args.overlay_output)
    print_ranges(rows)
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.output}")
    if args.output.suffix.lower() != ".pdf":
        print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.overlay_output}")
    if args.overlay_output.suffix.lower() != ".pdf":
        print(f"Wrote {args.overlay_output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
