#!/usr/bin/env python3
"""Plot M3 occupancy-matched search-vs-insert tail comparison."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def label_for_case(case: str) -> str:
    labels = {
        "insert_efc200": "insert efc=200",
        "insert_efc35": "insert efc=35",
        "insert_efc35_skip_update": "insert efc=35\nskip update",
    }
    return labels.get(case, case)


def case_color(case: str) -> str:
    if case == "insert_efc200":
        return "#ea580c"
    if case == "insert_efc35":
        return "#dc2626"
    if case == "insert_efc35_skip_update":
        return "#16a34a"
    return "#334155"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-bg-lowq-rows", type=float, default=1000.0)
    args = parser.parse_args()

    rows = load_csv(args.summary)
    pairs = load_csv(args.matched)
    bg = sorted(
        [
            row for row in rows
            if row.get("kind") == "bg_search"
            and f(row.get("lowq_rows")) >= args.min_bg_lowq_rows
        ],
        key=lambda row: f(row.get("background_search_workers")),
    )
    inserts = [
        row for row in rows
        if str(row.get("kind", "")).startswith("insert")
    ]

    plt.rcParams.update({
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.2,
    })

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 6.4),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.25, 1.0, 0.9]},
    )
    fig.patch.set_facecolor("white")

    ax = axes[0]
    if bg:
        xs = [f(row.get("background_search_workers")) for row in bg]
        p95 = [f(row.get("op_p95_lowq_ms")) for row in bg]
        p99 = [f(row.get("op_p99_lowq_ms")) for row in bg]
        ax.plot(xs, p95, color="#94a3b8", lw=1.0, marker="o", ms=3.0, label="bg search p95")
        ax.plot(xs, p99, color="#475569", lw=1.25, marker="o", ms=3.2, label="bg search p99")
    for row in inserts:
        x = f(row.get("insert_total_highlevel_workers"))
        y95 = f(row.get("op_p95_lowq_ms"))
        y99 = f(row.get("op_p99_lowq_ms"))
        color = case_color(str(row.get("case")))
        ax.scatter([x], [y95], marker="^", s=42, color=color, edgecolor="white", linewidth=0.5)
        ax.scatter([x], [y99], marker="s", s=46, color=color, edgecolor="white", linewidth=0.5)
        if row.get("case") == "insert_efc200":
            offset = (8, -10)
        elif row.get("case") == "insert_efc35":
            offset = (8, 8)
        else:
            offset = (8, -2)
        ax.annotate(label_for_case(str(row.get("case"))), (x, y99),
                    textcoords="offset points", xytext=offset, fontsize=6.9, color=color)
    ax.set_title("(a) Measured-search latency at matched competing occupancy", loc="left", pad=2)
    ax.set_ylabel("low-queue op latency (ms)")
    ax.set_xlabel("competing active workers")
    ax.grid(axis="both", color="#e5e7eb", lw=0.8)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    yvals = []
    if bg:
        yvals.extend([f(row.get("op_p95_lowq_ms")) for row in bg])
        yvals.extend([f(row.get("op_p99_lowq_ms")) for row in bg])
    yvals.extend([f(row.get("op_p95_lowq_ms")) for row in inserts])
    yvals.extend([f(row.get("op_p99_lowq_ms")) for row in inserts])
    finite_y = [value for value in yvals if math.isfinite(value)]
    if finite_y:
        ax.set_ylim(0.0, max(finite_y) * 1.18)

    ax = axes[1]
    pair_rows = [row for row in pairs if row.get("insert_case")]
    xlabels = [label_for_case(row["insert_case"]) for row in pair_rows]
    xpos = list(range(len(pair_rows)))
    width = 0.34
    p95_delta = [f(row.get("op_p95_delta_vs_bg_ms")) for row in pair_rows]
    p99_delta = [f(row.get("op_p99_delta_vs_bg_ms")) for row in pair_rows]
    ax.axhline(0, color="#94a3b8", lw=0.8)
    ax.bar([x - width / 2 for x in xpos], p95_delta, width=width, color="#f97316", label="p95 delta vs matched bg")
    ax.bar([x + width / 2 for x in xpos], p99_delta, width=width, color="#dc2626", label="p99 delta vs matched bg")
    ax.set_xticks(xpos, xlabels)
    ax.set_title("(b) Insert residual after occupancy-matched bg-search control", loc="left", pad=2)
    ax.set_ylabel("delta ms")
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.legend(frameon=False, loc="upper left")
    finite_delta = [value for value in p95_delta + p99_delta if math.isfinite(value)]
    if finite_delta:
        ax.set_ylim(min(-0.25, min(finite_delta) * 1.15), max(0.5, max(finite_delta) * 1.25))

    ax = axes[2]
    insert_labels = [label_for_case(str(row.get("case"))) for row in inserts]
    ix = list(range(len(inserts)))
    path_workers = [f(row.get("insert_path_workers")) for row in inserts]
    update_workers = [f(row.get("existing_update_workers")) for row in inserts]
    task_workers = [f(row.get("insert_task_workers")) for row in inserts]
    has_phase = any(value > 0 for value in path_workers + update_workers)
    if has_phase:
        ax.bar(ix, path_workers, color="#2563eb", label="insert path traversal")
        ax.bar(ix, update_workers, bottom=path_workers, color="#9333ea", label="existing-neighbor update")
    else:
        ax.bar(ix, task_workers, color="#64748b", label="insert task occupancy fallback")
    ax.set_xticks(ix, insert_labels)
    ax.set_title("(c) Insert occupancy used for matching", loc="left", pad=2)
    ax.set_ylabel("active workers")
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
