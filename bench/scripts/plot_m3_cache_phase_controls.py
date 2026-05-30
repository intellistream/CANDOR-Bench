#!/usr/bin/env python3
"""Plot the M3 cache/phase control matrix summary."""

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


MODES = [
    "cpu_dummy",
    "ann_search_dummy",
    "path_only_insert_control",
    "full_insert",
    "hot_filter",
]

MODE_LABELS = {
    "cpu_dummy": "CPU dummy",
    "ann_search_dummy": "ANN search dummy",
    "path_only_insert_control": "Path-only control",
    "full_insert": "Full insert",
    "hot_filter": "Hot filter",
}

COLORS = {
    "cpu_dummy": "#6f7f8f",
    "ann_search_dummy": "#c44949",
    "path_only_insert_control": "#8a8a8a",
    "full_insert": "#2f6db2",
    "hot_filter": "#2f9c6b",
}

WORKLOAD_ORDER = [
    (47, "chasing"),
    (47, "round_robin"),
    (8, "chasing"),
    (8, "round_robin"),
]


def to_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_label(label: str) -> tuple[int, str, int]:
    match = re.match(
        r"insert_i" r"w(?P<iw>\d+)_rep(?P<rep>\d+)/sift_b200_(?P<query>chasing|round_robin)_w40000_r40000_nolock",
        label,
    )
    if not match:
        raise ValueError(f"cannot parse label: {label}")
    return int(match.group("iw")), match.group("query"), int(match.group("rep"))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def load_groups(path: Path) -> dict[tuple[int, str, str], list[dict[str, float]]]:
    groups: dict[tuple[int, str, str], list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            iw, query, rep = parse_label(row["label"])
            parsed = {k: to_float(v) for k, v in row.items() if k not in {"mode", "label"}}
            parsed["rep"] = float(rep)
            groups[(iw, query, row["mode"])].append(parsed)
    return groups


def aggregate(groups: dict[tuple[int, str, str], list[dict[str, float]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metrics = [
        "p99_delta_e2e_ms",
        "p99_delta_start_delay_ms",
        "p99_delta_op_ms",
        "mean_delta_perf_l2_rqsts_miss_per_s",
        "top_delta_e2e_mean_perf_l2_rqsts_miss_per_s",
        "top_delta_e2e_mean_insert_path_search_active_workers",
    ]
    for iw, query in WORKLOAD_ORDER:
        for mode in MODES:
            vals = groups.get((iw, query, mode), [])
            if not vals:
                continue
            concurrency = "high writer concurrency" if iw >= 47 else "low writer concurrency"
            row: dict[str, object] = {
                "_iw": iw,
                "writer_concurrency": concurrency,
                "query": query,
                "mode": mode,
                "reps": len(vals),
            }
            for metric in metrics:
                series = [float(v.get(metric, 0.0)) for v in vals]
                row[f"{metric}_mean"] = mean(series)
                row[f"{metric}_sd"] = stdev(series)
            rows.append(row)
    return rows


def write_aggregate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key.startswith("_"):
                continue
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in fields})


def workload_label(iw: int, query: str) -> str:
    concurrency = "High writer concurrency" if iw >= 47 else "Low writer concurrency"
    query_label = "chasing queries" if query == "chasing" else "round-robin queries"
    return f"{concurrency}\n{query_label}"


def grouped_bar(
    rows: list[dict[str, object]],
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    scale: float = 1.0,
    err_metric: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = list(range(len(WORKLOAD_ORDER)))
    width = 0.15
    offsets = [(i - (len(MODES) - 1) / 2) * width for i in range(len(MODES))]
    lookup = {(int(r["_iw"]), str(r["query"]), str(r["mode"])): r for r in rows}

    for idx, mode in enumerate(MODES):
        ys: list[float] = []
        yerr: list[float] = []
        for iw, query in WORKLOAD_ORDER:
            row = lookup.get((iw, query, mode), {})
            ys.append(float(row.get(metric, 0.0)) / scale)
            if err_metric:
                yerr.append(float(row.get(err_metric, 0.0)) / scale)
        ax.bar(
            [v + offsets[idx] for v in x],
            ys,
            width=width,
            label=MODE_LABELS[mode],
            color=COLORS[mode],
            edgecolor="black",
            linewidth=0.4,
            yerr=yerr if err_metric else None,
            capsize=2 if err_metric else 0,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([workload_label(iw, query) for iw, query in WORKLOAD_ORDER])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def start_op_figure(rows: list[dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    specs = [
        ("p99_delta_start_delay_ms_mean", "p99 start-delay delta (ms)", "Start-delay dominates e2e jitter"),
        ("p99_delta_op_ms_mean", "p99 op delta (ms)", "Search-op delta is much smaller"),
    ]
    x = list(range(len(WORKLOAD_ORDER)))
    width = 0.15
    offsets = [(i - (len(MODES) - 1) / 2) * width for i in range(len(MODES))]
    lookup = {(int(r["_iw"]), str(r["query"]), str(r["mode"])): r for r in rows}

    for ax, (metric, ylabel, title) in zip(axes, specs):
        for idx, mode in enumerate(MODES):
            ys = [
                float(lookup.get((iw, query, mode), {}).get(metric, 0.0))
                for iw, query in WORKLOAD_ORDER
            ]
            ax.bar(
                [v + offsets[idx] for v in x],
                ys,
                width=width,
                label=MODE_LABELS[mode],
                color=COLORS[mode],
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([workload_label(iw, query) for iw, query in WORKLOAD_ORDER])
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, fontsize=8, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def l2_path_figure(rows: list[dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    specs = [
        (
            "top_delta_e2e_mean_perf_l2_rqsts_miss_per_s_mean",
            1.0e9,
            "top-window L2 misses (1e9/s)",
            "ANN graph/vector dummy has the largest L2 pressure",
        ),
        (
            "top_delta_e2e_mean_insert_path_search_active_workers_mean",
            1.0,
            "active insert path-search workers",
            "Real insert pressure window is insert_path_search",
        ),
    ]
    x = list(range(len(WORKLOAD_ORDER)))
    width = 0.15
    offsets = [(i - (len(MODES) - 1) / 2) * width for i in range(len(MODES))]
    lookup = {(int(r["_iw"]), str(r["query"]), str(r["mode"])): r for r in rows}

    for ax, (metric, scale, ylabel, title) in zip(axes, specs):
        for idx, mode in enumerate(MODES):
            ys = [
                float(lookup.get((iw, query, mode), {}).get(metric, 0.0)) / scale
                for iw, query in WORKLOAD_ORDER
            ]
            ax.bar(
                [v + offsets[idx] for v in x],
                ys,
                width=width,
                label=MODE_LABELS[mode],
                color=COLORS[mode],
                edgecolor="black",
                linewidth=0.4,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([workload_label(iw, query) for iw, query in WORKLOAD_ORDER])
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, fontsize=8, frameon=False, loc="lower center")
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.out_dir or args.summary_csv.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = aggregate(load_groups(args.summary_csv))
    write_aggregate_csv(out_dir / "cache_phase_aggregate.csv", rows)

    grouped_bar(
        rows,
        "p99_delta_e2e_ms_mean",
        "p99 e2e delta over read-only (ms)",
        "M3 cache-phase controls: e2e p99 delta",
        out_dir / "m3_cache_phase_e2e_delta.png",
        err_metric="p99_delta_e2e_ms_sd",
    )
    start_op_figure(rows, out_dir / "m3_cache_phase_start_vs_op_delta.png")
    l2_path_figure(rows, out_dir / "m3_cache_phase_l2_and_path.png")

    print(f"[m3-cache-phase-plot] wrote {out_dir / 'cache_phase_aggregate.csv'}")
    print(f"[m3-cache-phase-plot] wrote {out_dir / 'm3_cache_phase_e2e_delta.png'}")
    print(f"[m3-cache-phase-plot] wrote {out_dir / 'm3_cache_phase_start_vs_op_delta.png'}")
    print(f"[m3-cache-phase-plot] wrote {out_dir / 'm3_cache_phase_l2_and_path.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
