#!/usr/bin/env python3
"""Plot which writer-side component drives reader latency under concurrency."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKLOADS = [
    ("high", "chasing"),
    ("high", "round_robin"),
    ("low", "chasing"),
    ("low", "round_robin"),
]

COMPONENT_MODES = [
    ("cpu_dummy", "Writer CPU-only"),
    ("ann_search_dummy", "Writer graph/vector search"),
    ("full_insert", "Real insert"),
]

COMPONENT_COLORS = {
    "cpu_dummy": "#7b8794",
    "ann_search_dummy": "#c44949",
    "full_insert": "#2f6db2",
}

PHASES = [
    ("top_delta_e2e_mean_wait_insert_path_search_active_workers", "Graph search"),
    ("top_delta_e2e_mean_wait_existing_neighbor_update_active_workers", "Neighbor repair"),
    ("top_delta_e2e_mean_wait_existing_neighbor_prune_active_workers", "Prune scan"),
    ("top_delta_e2e_mean_wait_existing_neighbor_rewrite_active_workers", "Rewrite"),
]

PHASE_COLORS = {
    "Graph search": "#2f6db2",
    "Neighbor repair": "#d97706",
    "Prune scan": "#8b5cf6",
    "Rewrite": "#64748b",
}


def to_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_label(label: str) -> tuple[str, str] | None:
    pattern = (
        r"insert_i"
        r"w(?P<workers>\d+)_rep(?P<rep>\d+)/sift_b200_"
        r"(?P<query>chasing|round_robin)_w40000_r40000_nolock"
    )
    match = re.match(pattern, label)
    if not match:
        return None
    concurrency = "high" if int(match.group("workers")) >= 16 else "low"
    return concurrency, match.group("query")


def load_rows(path: Path) -> dict[tuple[str, str, str], list[dict[str, float]]]:
    groups: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed = parse_label(row["label"])
            if not parsed:
                continue
            concurrency, query = parsed
            groups[(concurrency, query, row["mode"])].append(
                {k: to_float(v) for k, v in row.items() if k not in {"mode", "label"}}
            )
    return groups


def mean_metric(rows: list[dict[str, float]], metric: str) -> float:
    return statistics.mean(r.get(metric, 0.0) for r in rows) if rows else 0.0


def sd_metric(rows: list[dict[str, float]], metric: str) -> float:
    vals = [r.get(metric, 0.0) for r in rows]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def workload_label(concurrency: str, query: str) -> str:
    left = "High writers" if concurrency == "high" else "Low writers"
    right = "chasing" if query == "chasing" else "round-robin"
    return f"{left}\n{right}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    groups = load_rows(args.summary_csv)
    x = list(range(len(WORKLOADS)))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    width = 0.22
    offsets = [-width, 0.0, width]
    for offset, (mode, label) in zip(offsets, COMPONENT_MODES):
        ys = [
            mean_metric(groups.get((concurrency, query, mode), []), "p99_delta_e2e_ms")
            for concurrency, query in WORKLOADS
        ]
        yerr = [
            sd_metric(groups.get((concurrency, query, mode), []), "p99_delta_e2e_ms")
            for concurrency, query in WORKLOADS
        ]
        axes[0].bar(
            [v + offset for v in x],
            ys,
            width=width,
            yerr=yerr,
            capsize=2,
            color=COMPONENT_COLORS[mode],
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )

    axes[0].set_title("Which writer work hurts reader latency?")
    axes[0].set_ylabel("reader p99 e2e delta over read-only (ms)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([workload_label(*w) for w in WORKLOADS])
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    phase_width = 0.18
    phase_offsets = [
        (i - (len(PHASES) - 1) / 2.0) * phase_width for i in range(len(PHASES))
    ]
    for offset, (metric, label) in zip(phase_offsets, PHASES):
        ys = [
            mean_metric(groups.get((concurrency, query, "full_insert"), []), metric)
            for concurrency, query in WORKLOADS
        ]
        axes[1].bar(
            [v + offset for v in x],
            ys,
            width=phase_width,
            color=PHASE_COLORS[label],
            edgecolor="black",
            linewidth=0.4,
            label=label,
        )

    axes[1].set_title("In real insert, high-latency reader windows overlap graph search")
    axes[1].set_ylabel("active writer workers during reader high-latency windows")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([workload_label(*w) for w in WORKLOADS])
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"[m3-writer-components] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
