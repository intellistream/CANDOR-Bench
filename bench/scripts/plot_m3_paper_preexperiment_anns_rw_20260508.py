#!/usr/bin/env python3
"""Plot M3 paper pre-experiment: ANN hardness and update pressure."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POST_START_MS = 2500.0

GROUPS = [
    ("fast100", "Fast"),
    ("mid100", "Mid"),
    ("slow100", "Slow"),
]

CASES = [
    ("searchonly", "No Inserts"),
    ("real", "With Inserts"),
]

ROOTS = {
    ("searchonly", "fast100"): Path("result/m3_targeted_fast100_static180k_searchonly_20260508"),
    ("searchonly", "mid100"): Path("result/m3_targeted_mid100_static180k_searchonly_20260508"),
    ("searchonly", "slow100"): Path("result/m3_targeted_slow100_static180k_searchonly_20260508"),
    ("real", "fast100"): Path("result/m3_targeted_fast100_static180k_real_20260508"),
    ("real", "mid100"): Path("result/m3_targeted_mid100_static180k_real_20260508"),
    ("real", "slow100"): Path("result/m3_targeted_slow100_static180k_real_20260508"),
}

WORK_COLS = [
    ("work_distance_computations", "Distance comps"),
    ("work_level0_edges_scanned", "L0 edges"),
    ("work_visited_nodes", "Visited nodes"),
]

PERF_ROOTS = {
    ("low", "loadmiss"): Path("result/m3_perf_fast100_searchonly_loadmiss_20260508"),
    ("tail", "loadmiss"): Path("result/m3_perf_slow100_searchonly_loadmiss_20260508"),
    ("low", "loadwait"): Path("result/m3_perf_fast100_searchonly_loadwait_20260508"),
    ("tail", "loadwait"): Path("result/m3_perf_slow100_searchonly_loadwait_20260508"),
}


def to_float(value: str | float | int | None) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return math.nan


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else math.nan


def minmax_err(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return (math.nan, math.nan)
    m = mean(vals)
    return (m - min(vals), max(vals) - m)


def read_mixed_summary(root: Path) -> list[dict[str, str]]:
    path = root / "mixed_path_locality_summary.csv"
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_search_wides(root: Path) -> list[Path]:
    paths = sorted(root.glob("**/raw_latency/search_wide.csv"))
    if not paths:
        raise FileNotFoundError(f"missing search_wide.csv under {root}")
    return paths


def read_work_means(root: Path) -> list[dict[str, float]]:
    reps: list[dict[str, float]] = []
    for path in find_search_wides(root):
        sums = {col: 0.0 for col, _ in WORK_COLS}
        rows = 0.0
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("measured") != "1":
                    continue
                if to_float(row.get("finish_ms")) < POST_START_MS:
                    continue
                rows += 1.0
                for col, _ in WORK_COLS:
                    sums[col] += to_float(row.get(col))
        if rows > 0:
            reps.append({col: sums[col] / rows for col, _ in WORK_COLS})
    return reps


def collect(bench_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    p99_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []

    for case_key, case_label in CASES:
        for group_key, group_label in GROUPS:
            root = bench_root / ROOTS[(case_key, group_key)]
            summaries = read_mixed_summary(root)
            p99s = [to_float(row["post_p99_ms"]) for row in summaries]
            insert_p99s = [to_float(row["insert_p99_ms"]) for row in summaries]
            outside_p99s = [to_float(row["outside_p99_ms"]) for row in summaries]
            ratios = [
                ins / out
                for ins, out in zip(insert_p99s, outside_p99s)
                if math.isfinite(ins) and math.isfinite(out) and out > 0
            ]

            low, high = minmax_err(p99s)
            p99_rows.append(
                {
                    "panel": "p99",
                    "case": case_key,
                    "case_label": case_label,
                    "group": group_key,
                    "group_label": group_label,
                    "mean": mean(p99s),
                    "err_low": low,
                    "err_high": high,
                    "n": len(p99s),
                }
            )

            low, high = minmax_err(ratios)
            window_rows.append(
                {
                    "panel": "insert_outside_ratio",
                    "case": case_key,
                    "case_label": case_label,
                    "group": group_key,
                    "group_label": group_label,
                    "mean": mean(ratios),
                    "err_low": low,
                    "err_high": high,
                    "n": len(ratios),
                }
            )

    return p99_rows, window_rows


def read_summary_row(root: Path) -> dict[str, str]:
    with (root / "summary.csv").open(newline="") as f:
        return next(csv.DictReader(f))


def rate(row: dict[str, str], num: str, den: str) -> float:
    denominator = to_float(row.get(den))
    return to_float(row.get(num)) / denominator if denominator > 0 else math.nan


def collect_hardware_rows(bench_root: Path) -> list[dict[str, object]]:
    low_miss = read_summary_row(bench_root / PERF_ROOTS[("low", "loadmiss")])
    tail_miss = read_summary_row(bench_root / PERF_ROOTS[("tail", "loadmiss")])
    low_wait = read_summary_row(bench_root / PERF_ROOTS[("low", "loadwait")])
    tail_wait = read_summary_row(bench_root / PERF_ROOTS[("tail", "loadwait")])

    low_ipc = to_float(low_miss["perf_post_ipc"])
    tail_ipc = to_float(tail_miss["perf_post_ipc"])
    low_l1d = rate(low_wait, "perf_cycle_activity_stalls_l1d_miss_post_rate", "perf_cycles_post_rate")
    tail_l1d = rate(tail_wait, "perf_cycle_activity_stalls_l1d_miss_post_rate", "perf_cycles_post_rate")
    low_l3 = rate(low_wait, "perf_cycle_activity_stalls_l3_miss_post_rate", "perf_cycles_post_rate")
    tail_l3 = rate(tail_wait, "perf_cycle_activity_stalls_l3_miss_post_rate", "perf_cycles_post_rate")

    return [
        {
            "metric": "P99",
            "value": to_float(tail_miss["post_p99_ms"]) / to_float(low_miss["post_p99_ms"]),
            "kind": "latency",
        },
        {
            "metric": "CPI",
            "value": low_ipc / tail_ipc,
            "kind": "cpu",
        },
        {
            "metric": "LLC\nMiss",
            "value": to_float(tail_miss["perf_post_llc_load_miss_rate"])
            / to_float(low_miss["perf_post_llc_load_miss_rate"]),
            "kind": "memory",
        },
        {
            "metric": "L1D\nStall",
            "value": tail_l1d / low_l1d,
            "kind": "memory",
        },
        {
            "metric": "L3\nStall",
            "value": tail_l3 / low_l3,
            "kind": "memory",
        },
    ]


def row_value(rows: list[dict[str, object]], **filters: str) -> float:
    for row in rows:
        if all(str(row.get(k)) == v for k, v in filters.items()):
            return float(row["mean"])
    return math.nan


def row_err(rows: list[dict[str, object]], **filters: str) -> tuple[float, float]:
    for row in rows:
        if all(str(row.get(k)) == v for k, v in filters.items()):
            return (float(row["err_low"]), float(row["err_high"]))
    return (math.nan, math.nan)


def write_csv(path: Path, groups: list[list[dict[str, object]]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    rows = [row for group in groups for row in group]
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 6.2,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.65,
            "axes.linewidth": 0.62,
            "mathtext.fontset": "dejavusans",
        }
    )


def style_axis(ax: plt.Axes, grid: str) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=grid, linewidth=0.58)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#111827")
    ax.spines["bottom"].set_color("#111827")
    ax.tick_params(axis="both", colors="#111827", width=0.62, length=2.6, pad=1.8)


def draw_hardware_panel(ax: plt.Axes, hardware_rows: list[dict[str, object]]) -> None:
    labels = [str(row["metric"]) for row in hardware_rows]
    vals = [float(row["value"]) for row in hardware_rows]
    x = np.arange(len(labels))
    grid = "#e8edf3"
    colors = ["#4c78a8", "#7f7f7f", "#e15759", "#e15759", "#e15759"]
    dark = "#1f2937"

    ax.bar(x, vals, 0.56, color=colors, edgecolor=dark, linewidth=0.38)
    ax.axhline(1.0, color=dark, linewidth=0.65)
    for xi, val in zip(x, vals):
        ax.text(xi, val + 0.16, f"{val:.1f}x", ha="center", va="bottom", fontsize=5.5, color=dark)
    ax.set_ylabel("Tail / Low Lat.\n(x)")
    ax.set_xticks(x, labels)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5.45)
    style_axis(ax, grid)


def draw_write_lift_panel(
    hardware_rows: list[dict[str, object]],
    window_rows: list[dict[str, object]],
    p99_rows: list[dict[str, object]],
    ax: plt.Axes,
) -> None:
    grid = "#e8edf3"
    blue = "#4c78a8"
    red = "#e15759"
    dark = "#1f2937"

    search_gap = next(float(row["value"]) for row in hardware_rows if row["metric"] == "P99")
    write_lift = row_value(window_rows, case="real", group="slow100")
    write_err_low, write_err_high = row_err(window_rows, case="real", group="slow100")
    labels = ["ANNS\nPath", "Insert\nWindow"]
    vals = [search_gap, write_lift]
    x = np.arange(len(labels))

    ax.bar(x, vals, 0.48, color=[blue, red], edgecolor=dark, linewidth=0.38)
    ax.errorbar(
        [x[1]],
        [vals[1]],
        yerr=np.array([[write_err_low], [write_err_high]]),
        fmt="none",
        ecolor=dark,
        elinewidth=0.65,
        capsize=2,
    )
    ax.axhline(1.0, color=dark, linewidth=0.65)
    for xi, val in zip(x, vals):
        ax.text(xi, val + 0.16, f"{val:.2f}x" if val < 2 else f"{val:.1f}x", ha="center", va="bottom", fontsize=5.6, color=dark)
    ax.set_ylabel("P99 Multiplier\n(x)")
    ax.set_xticks(x, labels)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 5.45)
    style_axis(ax, grid)


def save_figure(fig: plt.Figure, out_png: Path, out_pdf: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot(
    p99_rows: list[dict[str, object]],
    hardware_rows: list[dict[str, object]],
    window_rows: list[dict[str, object]],
    out_png: Path,
    out_pdf: Path,
    panel_a_png: Path,
    panel_a_pdf: Path,
    panel_b_png: Path,
    panel_b_pdf: Path,
) -> None:
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.24), constrained_layout=True)
    draw_hardware_panel(axes[0], hardware_rows)
    draw_write_lift_panel(hardware_rows, window_rows, p99_rows, axes[1])
    save_figure(fig, out_png, out_pdf)

    fig, ax = plt.subplots(1, 1, figsize=(1.62, 1.12), constrained_layout=True)
    draw_hardware_panel(ax, hardware_rows)
    save_figure(fig, panel_a_png, panel_a_pdf)

    fig, ax = plt.subplots(1, 1, figsize=(1.62, 1.12), constrained_layout=True)
    draw_write_lift_panel(hardware_rows, window_rows, p99_rows, ax)
    save_figure(fig, panel_b_png, panel_b_pdf)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, default=Path("bench"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper-progress/m3/search-tail-rootcause/figures/18_m3_preexperiment_anns_rw.png"),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("paper-progress/m3/search-tail-rootcause/figures/18_m3_preexperiment_anns_rw.pdf"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("paper-progress/m3/search-tail-rootcause/figures/18_m3_preexperiment_anns_rw.csv"),
    )
    parser.add_argument(
        "--panel-a-out",
        type=Path,
        default=Path("paper-progress/m3/search-tail-rootcause/figures/18_m3_preexperiment_anns_rw_a.png"),
    )
    parser.add_argument(
        "--panel-a-pdf",
        type=Path,
        default=Path("paper-progress/m3/search-tail-rootcause/figures/18_m3_preexperiment_anns_rw_a.pdf"),
    )
    parser.add_argument(
        "--panel-b-out",
        type=Path,
        default=Path("paper-progress/m3/search-tail-rootcause/figures/18_m3_preexperiment_anns_rw_b.png"),
    )
    parser.add_argument(
        "--panel-b-pdf",
        type=Path,
        default=Path("paper-progress/m3/search-tail-rootcause/figures/18_m3_preexperiment_anns_rw_b.pdf"),
    )
    args = parser.parse_args()

    p99_rows, window_rows = collect(args.bench_root)
    hardware_rows = collect_hardware_rows(args.bench_root)
    write_csv(args.csv, [p99_rows, window_rows, hardware_rows])
    plot(
        p99_rows,
        hardware_rows,
        window_rows,
        args.out,
        args.pdf,
        args.panel_a_out,
        args.panel_a_pdf,
        args.panel_b_out,
        args.panel_b_pdf,
    )
    print(args.out)
    print(args.pdf)
    print(args.panel_a_out)
    print(args.panel_a_pdf)
    print(args.panel_b_out)
    print(args.panel_b_pdf)
    print(args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
