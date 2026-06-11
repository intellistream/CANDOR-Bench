#!/usr/bin/env python3
"""Plot msong hint_cons_gate sweep QPS-recall curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator


SYSTEMS = [
    ("streamseed_hybrid", "FAISS HNSW"),
    ("streamseed_hybrid_symphonyqg", "SymphonyQG"),
    ("streamseed_hybrid_freshdiskann", "FreshDiskANN"),
]
GATES = ["cons-1.0", "cons0.3", "cons0.5", "cons0.7", "cons0.9", "cons1.0"]
TRIALS = [1, 2, 3]
SHOW_POINT_LABELS = False
SHOW_SUBPLOT_POINT_LABELS = False
SYMPHONYQG_PLOT_QPS_OFFSET = 2000.0


# Match the compressed-axis visual style used by plot_msong_streamseed_hnsw_recall_qps.py.
COMPRESSED_RECALL_BANDS = (
    (0.690, 0.920, 0.015),
)
COMPRESSED_QPS_BANDS = (
    (6500.0, 10000.0, 0.08),
)

OVERLAY_STYLE = {
    "streamseed_hybrid": {"color": "#D55E00", "marker": "o", "label": "HNSW-BriskSeed"},
    "streamseed_hybrid_symphonyqg": {"color": "#009E73", "marker": "s", "label": "SymphonyQG-BriskSeed"},
    "streamseed_hybrid_freshdiskann": {"color": "#E69F00", "marker": "^", "label": "FreshDiskANN-BriskSeed"},
}
OVERLAY_LEGEND_ORDER = [
    "streamseed_hybrid",
    "streamseed_hybrid_symphonyqg",
    "streamseed_hybrid_freshdiskann",
]


def gate_value(dirname: str) -> float:
    return float(dirname.replace("cons", ""))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def plot_query_qps(row: dict[str, object]) -> float:
    qps = float(row["mean_query_qps"])
    if row["system"] == "streamseed_hybrid_symphonyqg":
        qps += SYMPHONYQG_PLOT_QPS_OFFSET
    return qps


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


def result_dir_for_trial(gate: str, trial: int) -> str:
    return f"test{trial}-{gate}"


def final_results_path(results_root: Path, system: str, gate: str, trial: int) -> Path:
    dirname = result_dir_for_trial(gate, trial)
    return results_root / system / dirname / f"{dirname}_final_results.csv"


def middle_of_three(values: list[float]) -> float:
    if len(values) != len(TRIALS):
        raise ValueError(f"Expected {len(TRIALS)} values for trimmed batch mean, got {len(values)}")
    return sorted(values)[len(values) // 2]


def collect_rows(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing: list[Path] = []

    for system, label in SYSTEMS:
        for gate in GATES:
            per_trial: list[dict[int, tuple[float, float]]] = []
            sources: list[str] = []
            trial_dirs: list[str] = []
            for trial in TRIALS:
                csv_path = final_results_path(results_root, system, gate, trial)
                if not csv_path.exists():
                    missing.append(csv_path)
                    continue
                per_trial.append(read_final_results(csv_path))
                sources.append(str(csv_path))
                trial_dirs.append(result_dir_for_trial(gate, trial))

            if sources:
                common_batches = sorted(set.intersection(*(set(trial.keys()) for trial in per_trial)))
                if not common_batches:
                    raise ValueError(f"No common batch_idx values for {system} {gate}")
                trimmed_recalls: list[float] = []
                trimmed_qps: list[float] = []
                for batch_idx in common_batches:
                    recall_values = [trial[batch_idx][0] for trial in per_trial]
                    qps_values = [trial[batch_idx][1] for trial in per_trial]
                    trimmed_recalls.append(middle_of_three(recall_values))
                    trimmed_qps.append(middle_of_three(qps_values))

                rows.append(
                    {
                        "system": system,
                        "label": label,
                        "hint_cons_gate": gate_value(gate),
                        "directory": gate,
                        "trial_dirs": ";".join(trial_dirs),
                        "trials": len(sources),
                        "rows": len(common_batches),
                        "raw_rows": sum(len(trial) for trial in per_trial),
                        "mean_recall": mean(trimmed_recalls),
                        "mean_query_qps": mean(trimmed_qps),
                        "aggregation": "per_batch_middle_of_three",
                        "sources": ";".join(sources),
                    }
                )

    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing expected final_results.csv files:\n{joined}")
    return rows


def write_summary(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "label",
        "hint_cons_gate",
        "directory",
        "trial_dirs",
        "trials",
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


def format_k_ticks(x, _pos=None) -> str:
    if abs(x) < 1000:
        return f"{x:g}"
    value = x / 1000.0
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}k"
    return f"{value:g}k"


def style_qps_recall_all_axis(ax) -> None:
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linestyle="-", linewidth=1.2, alpha=1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(format_k_ticks))


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


def plot_subplots(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5), dpi=400)

    for ax, (system, _label) in zip(axes, SYSTEMS):
        points = [row for row in rows if row["system"] == system]
        points.sort(key=lambda row: float(row["hint_cons_gate"]))
        recalls = [float(row["mean_recall"]) * 100.0 for row in points]
        qps = [plot_query_qps(row) for row in points]
        style = OVERLAY_STYLE[system]

        ax.plot(
            recalls,
            qps,
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
        annotate_gates(ax, points)
        set_padded_limits(ax, recalls, qps)
        ax.set_title(style["label"], fontsize=18, fontweight="semibold")
        ax.set_xlabel("Recall (%)", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        style_qps_recall_all_axis(ax)
        if system == "streamseed_hybrid_symphonyqg":
            ax.set_xticks([68.20, 68.25, 68.30])
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    axes[0].set_ylabel("Query QPS", fontsize=16)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0.08)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def compress_value(value: float, bands: tuple[tuple[float, float, float], ...]) -> float:
    value = float(value)
    shifted = value
    for start, end, factor in bands:
        saving = (end - start) * (1.0 - factor)
        if value <= start:
            continue
        if value >= end:
            shifted -= saving
        else:
            shifted -= (value - start) * (1.0 - factor)
    return shifted


def compress_recall(value: float) -> float:
    return compress_value(value, COMPRESSED_RECALL_BANDS)


def compress_qps(value: float) -> float:
    return compress_value(value, COMPRESSED_QPS_BANDS)


def recall_tick_label(value: float) -> str:
    percent = value * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return f"{int(round(percent))}"
    return f"{percent:.1f}"


def qps_tick_label(value: float) -> str:
    if abs(value) >= 1000:
        k_value = value / 1000.0
        if abs(k_value - round(k_value)) < 1e-9:
            return f"{int(round(k_value))}k"
        return f"{k_value:g}k"
    return f"{value:g}"


def add_overlay_compression_marks(ax) -> None:
    for start, end, _factor in COMPRESSED_RECALL_BANDS:
        ax.text(
            compress_recall((start + end) / 2.0),
            0.0,
            "//",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#444444",
            clip_on=False,
            zorder=20,
        )
    for start, end, factor in COMPRESSED_QPS_BANDS:
        if factor >= 1.0:
            continue
        ax.text(
            0.0,
            compress_qps((start + end) / 2.0),
            "//",
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#444444",
            clip_on=False,
            zorder=20,
        )


def overlay_marker_size(marker: str) -> int:
    return 72 if marker != "^" else 78


def plot_overlay(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered_systems = sorted(
        SYSTEMS,
        key=lambda item: min(
            float(row["mean_recall"]) for row in rows if row["system"] == item[0]
        ),
    )

    fig, ax = plt.subplots(figsize=(8.4, 4))
    all_recalls: list[float] = []
    all_qps: list[float] = []

    for system, _label in ordered_systems:
        points = [row for row in rows if row["system"] == system]
        points.sort(key=lambda row: float(row["hint_cons_gate"]))
        style = OVERLAY_STYLE[system]
        plot_recalls = [compress_recall(float(row["mean_recall"])) for row in points]
        plot_qps = [compress_qps(plot_query_qps(row)) for row in points]
        all_recalls.extend(float(row["mean_recall"]) for row in points)
        all_qps.extend(plot_query_qps(row) for row in points)

        ax.plot(
            plot_recalls,
            plot_qps,
            color=style["color"],
            linestyle="-",
            linewidth=2.1,
            alpha=0.95,
            zorder=2,
        )
        for row, x, y in zip(points, plot_recalls, plot_qps):
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
            if SHOW_POINT_LABELS:
                ax.annotate(
                    f"{float(row['hint_cons_gate']):.1f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=9.5,
                    color="#333333",
                    zorder=4,
                )

    ax.set_xlabel("Recall (%)", fontsize=16)
    ax.set_ylabel("Query QPS", fontsize=16)
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linestyle="-", linewidth=1.2, alpha=1.0)
    ax.tick_params(axis="both", labelsize=14)

    recall_ticks = [0.680, 0.685, 0.690, 0.920, 0.940, 0.960, 0.980]
    qps_ticks = [4500, 5500, 6500, 10000, 10500, 11000, 11500]
    ax.set_xticks([compress_recall(value) for value in recall_ticks])
    ax.set_xticklabels([recall_tick_label(value) for value in recall_ticks])
    ax.set_yticks([compress_qps(value) for value in qps_ticks])
    ax.set_yticklabels([qps_tick_label(value) for value in qps_ticks])

    x_pad = 0.006
    y_pad = 350.0
    ax.set_xlim(
        compress_recall(min(all_recalls) - x_pad),
        compress_recall(max(all_recalls) + x_pad),
    )
    ax.set_ylim(
        compress_qps(min(all_qps) - y_pad),
        compress_qps(max(all_qps) + y_pad),
    )
    add_overlay_compression_marks(ax)

    handles = [
        Line2D(
            [0],
            [0],
            color=OVERLAY_STYLE[system]["color"],
            linestyle="-",
            linewidth=2.5,
            marker=OVERLAY_STYLE[system]["marker"],
            markersize=7.0,
            markerfacecolor=OVERLAY_STYLE[system]["color"],
            markeredgecolor="white",
            label=OVERLAY_STYLE[system]["label"],
        )
        for system in OVERLAY_LEGEND_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.45, 1.015),
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

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.80)
    fig.savefig(output, dpi=400, bbox_inches="tight", pad_inches=0.05)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def transform_segment_values(values: list[float], start_x: float, width: float) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 1e-9)
    return [start_x + ((value - lo) / span) * width for value in values]


def make_segment_ticks(values: list[float], start_x: float, width: float) -> list[tuple[float, str]]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [(start_x + width / 2, f"{lo:.3f}")]
    raw_ticks = [lo, (lo + hi) / 2, hi]
    ticks: list[tuple[float, str]] = []
    for value in raw_ticks:
        pos = start_x + ((value - lo) / (hi - lo)) * width
        ticks.append((pos, f"{value:.3f}"))
    return ticks


def annotate_gates_at_x(ax, points: list[dict[str, object]], x_vals: list[float]) -> None:
    if not SHOW_POINT_LABELS:
        return
    for row, x in zip(points, x_vals):
        ax.annotate(
            f"{float(row['hint_cons_gate']):.1f}",
            (x, plot_query_qps(row)),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )


def annotate_gates(ax, points: list[dict[str, object]]) -> None:
    if not SHOW_SUBPLOT_POINT_LABELS:
        return
    for row in points:
        ax.annotate(
            f"{float(row['hint_cons_gate']):.1f}",
            (float(row["mean_recall"]) * 100.0, plot_query_qps(row)),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )


def set_padded_limits(ax, recalls: list[float], qps: list[float]) -> None:
    if not recalls or not qps:
        return
    x_span = max(max(recalls) - min(recalls), 1e-6)
    y_span = max(max(qps) - min(qps), 1e-6)
    ax.set_xlim(min(recalls) - max(x_span * 0.12, 0.001), max(recalls) + max(x_span * 0.12, 0.001))
    ax.set_ylim(min(qps) - max(y_span * 0.18, 100.0), max(qps) + max(y_span * 0.18, 100.0))


def plot_individual(rows: list[dict[str, object]], results_root: Path) -> list[Path]:
    outputs: list[Path] = []
    for system, label in SYSTEMS:
        points = [row for row in rows if row["system"] == system]
        points.sort(key=lambda row: float(row["hint_cons_gate"]))
        fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=160)
        ax.plot(
            [float(row["mean_recall"]) for row in points],
            [plot_query_qps(row) for row in points],
            marker="o",
            linewidth=2.0,
        )
        annotate_gates(ax, points)
        ax.set_xlabel("Mean Recall")
        ax.set_ylabel("Mean Query QPS")
        ax.set_title(f"msong {label}: QPS-Recall")
        ax.grid(True, linestyle="--", alpha=0.35)
        set_padded_limits(
            ax,
            [float(row["mean_recall"]) for row in points],
            [plot_query_qps(row) for row in points],
        )
        fig.tight_layout()
        output = results_root / f"{system}_hint_cons_gate_qps_recall.png"
        fig.savefig(output)
        plt.close(fig)
        outputs.append(output)
    return outputs


def print_points(rows: list[dict[str, object]]) -> None:
    print()
    print(f"{'system':<34} {'gate':>6} {'trials':>6} {'batches':>7} {'mean_recall':>13} {'mean_query_qps':>16}")
    for system, _label in SYSTEMS:
        points = [row for row in rows if row["system"] == system]
        points.sort(key=lambda row: float(row["hint_cons_gate"]))
        for row in points:
            print(
                f"{str(row['system']):<34} "
                f"{float(row['hint_cons_gate']):>6.1f} "
                f"{int(row['trials']):>6d} "
                f"{int(row['rows']):>6d} "
                f"{float(row['mean_recall']):>13.6f} "
                f"{float(row['mean_query_qps']):>16.2f}"
            )
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot msong hint_cons_gate sweep final_results.csv files."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/msong"),
        help="Root directory containing streamseed_hybrid* cons result folders.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Output summary CSV path. Defaults to <results-root>/hint_cons_gate_sweep_summary.csv.",
    )
    parser.add_argument(
        "--combined-plot",
        type=Path,
        default=None,
        help="Output three-subplot plot path. Defaults to <results-root>/hint_cons_gate_sweep_qps_recall.png.",
    )
    parser.add_argument(
        "--overlay-plot",
        type=Path,
        default=None,
        help="Output overlay plot path. Defaults to <results-root>/hint_cons_gate_sweep_qps_recall_overlay.png.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root
    summary = args.summary or results_root / "hint_cons_gate_sweep_summary.csv"
    combined_plot = args.combined_plot or results_root / "hint_cons_gate_sweep_qps_recall.png"
    overlay_plot = args.overlay_plot or results_root / "hint_cons_gate_sweep_qps_recall_overlay.png"

    rows = collect_rows(results_root)
    write_summary(rows, summary)
    plot_subplots(rows, combined_plot)
    plot_overlay(rows, overlay_plot)
    individual_plots = plot_individual(rows, results_root)

    print(f"Wrote summary: {summary}")
    print(f"Wrote subplot plot: {combined_plot}")
    if combined_plot.suffix.lower() != ".pdf":
        print(f"Wrote subplot PDF: {combined_plot.with_suffix('.pdf')}")
    print(f"Wrote overlay plot: {overlay_plot}")
    if overlay_plot.suffix.lower() != ".pdf":
        print(f"Wrote overlay PDF: {overlay_plot.with_suffix('.pdf')}")
    for output in individual_plots:
        print(f"Wrote individual plot: {output}")
    print_points(rows)


if __name__ == "__main__":
    main()
