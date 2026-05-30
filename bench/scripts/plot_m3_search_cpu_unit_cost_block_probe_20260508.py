#!/usr/bin/env python3
"""Plot M3 search-CPU unit-cost block probe."""

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

BLOCKS = [
    {
        "key": "low",
        "label": "low\n080-099",
        "root": Path("result/m3_perf_cpu_block_low_nondist_loadwait_080_099_20260508"),
        "graph_root": Path("result/m3_graph_timing_cpu_block_low_nondist_080_099_20260508"),
    },
    {
        "key": "high_nondist",
        "label": "high non-dist\n340-359",
        "root": Path("result/m3_perf_cpu_block_high_nondist_loadwait_340_359_20260508"),
        "graph_root": Path("result/m3_graph_timing_cpu_block_high_nondist_340_359_20260508"),
    },
    {
        "key": "high_work",
        "label": "high work\n300-319",
        "root": Path("result/m3_perf_cpu_block_high_work_loadwait_300_319_20260508"),
        "graph_root": Path("result/m3_graph_timing_cpu_block_high_work_300_319_20260508"),
    },
]


def read_summary(root: Path) -> dict[str, float]:
    with (root / "summary.csv").open(newline="") as f:
        row = next(csv.DictReader(f))
    return {k: to_float(v) for k, v in row.items()}


def to_float(value: str | float | int | None) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return math.nan


def find_search_wide(root: Path) -> Path:
    matches = sorted(root.glob("**/raw_latency/search_wide.csv"))
    if not matches:
        raise FileNotFoundError(f"missing search_wide.csv under {root}")
    return matches[0]


def load_batch_means(root: Path) -> dict[str, float]:
    path = find_search_wide(root)
    groups: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("measured") != "1":
                continue
            if to_float(row.get("finish_ms")) < POST_START_MS:
                continue
            key = (row.get("start_raw_ns", ""), row.get("finish_raw_ns", ""))
            item = groups.setdefault(key, {"rows": 0.0, "op_ms": to_float(row.get("op_ms"))})
            item["rows"] += 1.0
            for col in [
                "work_searchknn_thread_cpu_ms",
                "work_base_search_ms",
                "work_distance_computations",
                "work_level0_edges_scanned",
                "work_visited_nodes",
                "work_candidate_pops",
                "work_candidate_pushes",
                "work_level0_queue_pop_ms",
                "work_level0_adj_fetch_ms",
                "work_level0_locality_capture_ms",
                "work_level0_candidate_loop_ms",
                "work_level0_visited_check_ms",
                "work_level0_visibility_check_ms",
                "work_level0_candidate_accept_ms",
            ]:
                item[col] = item.get(col, 0.0) + to_float(row.get(col))

    batches = [item for item in groups.values() if item["rows"] == 20.0 and item.get("work_base_search_ms", 0.0) > 0.0]
    if not batches:
        raise ValueError(f"no post-window measured batches in {path}")

    out: dict[str, float] = {}
    for col in batches[0]:
        if col == "rows":
            continue
        out[col] = sum(item.get(col, 0.0) for item in batches) / len(batches)
    out["batches"] = float(len(batches))
    out["p99_ms"] = percentile([item["op_ms"] for item in batches], 99.0)
    return out


def percentile(values: list[float], pct: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.nan
    pos = (len(vals) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def build_rows(bench_root: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for block in BLOCKS:
        perf_root = bench_root / block["root"]
        graph_root = bench_root / block["graph_root"]
        summary = read_summary(perf_root)
        raw = load_batch_means(perf_root)
        graph = load_batch_means(graph_root)

        cycles = summary["perf_cycles_post_rate"]
        row: dict[str, float | str] = {
            "key": block["key"],
            "label": block["label"],
            "p99_ms": summary["post_p99_ms"],
            "thread_cpu_ms": raw["work_searchknn_thread_cpu_ms"],
            "base_ms": raw["work_base_search_ms"],
            "dist_comps": raw["work_distance_computations"],
            "edges": raw["work_level0_edges_scanned"],
            "visited": raw["work_visited_nodes"],
            "ipc": summary["perf_post_ipc"],
            "l1d_per_cycle": summary["perf_cycle_activity_stalls_l1d_miss_post_rate"] / cycles,
            "l2_per_cycle": summary["perf_cycle_activity_stalls_l2_miss_post_rate"] / cycles,
            "l3_per_cycle": summary["perf_cycle_activity_stalls_l3_miss_post_rate"] / cycles,
            "mem_any_per_cycle": summary["perf_cycle_activity_cycles_mem_any_post_rate"] / cycles,
            "scoreboard_per_cycle": summary["perf_resource_stalls_scoreboard_post_rate"] / cycles,
            "offcore_data_per_cycle": summary["perf_offcore_requests_outstanding_cycles_with_demand_data_rd_post_rate"] / cycles,
            "graph_thread_cpu_ms": graph["work_searchknn_thread_cpu_ms"],
            "graph_dist_comps": graph["work_distance_computations"],
            "graph_edges": graph["work_level0_edges_scanned"],
            "graph_loop_per_edge_ns": graph["work_level0_candidate_loop_ms"] * 1e6 / graph["work_level0_edges_scanned"],
            "graph_accept_per_push_ns": graph["work_level0_candidate_accept_ms"] * 1e6 / graph["work_candidate_pushes"],
            "graph_adj_fetch_per_exp_ns": graph["work_level0_adj_fetch_ms"] * 1e6 / graph["work_candidate_pops"],
            "graph_queue_per_pop_ns": graph["work_level0_queue_pop_ms"] * 1e6 / graph["work_candidate_pops"],
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
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


def ratio(values: list[float], baseline: float) -> list[float]:
    return [v / baseline if baseline else math.nan for v in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-root", type=Path, default=Path("bench"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper-progress/m3/search-tail-rootcause/figures/17_m3_search_cpu_unit_cost_block_probe.png"),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("paper-progress/m3/search-tail-rootcause/figures/17_m3_search_cpu_unit_cost_block_probe.pdf"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("paper-progress/m3/search-tail-rootcause/figures/17_m3_search_cpu_unit_cost_block_probe.csv"),
    )
    args = parser.parse_args()

    rows = build_rows(args.bench_root)
    write_csv(args.csv, rows)

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )

    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(rows))
    colors = ["#6b7280", "#2f6fdd", "#d95f59"]
    navy = "#263445"
    green = "#2f9b7a"
    amber = "#c47f17"
    grid = "#e5e7eb"

    fig = plt.figure(figsize=(7.25, 3.35), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.20, 1.35], wspace=0.44)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    width = 0.34
    p99 = [float(row["p99_ms"]) for row in rows]
    thread = [float(row["thread_cpu_ms"]) for row in rows]
    ax0.bar(x - width / 2, p99, width, color=colors, alpha=0.88, label="P99 wall")
    ax0b = ax0.twinx()
    ax0b.plot(x + width / 2, thread, marker="o", color=navy, lw=1.7, label="search CPU")
    for xi, val in zip(x + width / 2, thread):
        ax0b.text(xi, val + 0.25, f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, color=navy)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("P99 wall (ms)")
    ax0b.set_ylabel("Search CPU / batch (ms)")
    ax0.set_title("A. Replay blocks")
    ax0.grid(True, axis="y", color=grid, lw=0.7)
    ax0.spines["top"].set_visible(False)
    ax0b.spines["top"].set_visible(False)
    ax0.set_ylim(0, max(p99) * 1.28)
    ax0b.set_ylim(0, max(thread) * 1.22)

    low = rows[0]
    counter_names = ["IPC", "L1D stall", "mem-any", "L3 stall", "offcore data"]
    high_nondist = [
        float(rows[1]["ipc"]) / float(low["ipc"]),
        float(rows[1]["l1d_per_cycle"]) / float(low["l1d_per_cycle"]),
        float(rows[1]["mem_any_per_cycle"]) / float(low["mem_any_per_cycle"]),
        float(rows[1]["l3_per_cycle"]) / float(low["l3_per_cycle"]),
        float(rows[1]["offcore_data_per_cycle"]) / float(low["offcore_data_per_cycle"]),
    ]
    high_work = [
        float(rows[2]["ipc"]) / float(low["ipc"]),
        float(rows[2]["l1d_per_cycle"]) / float(low["l1d_per_cycle"]),
        float(rows[2]["mem_any_per_cycle"]) / float(low["mem_any_per_cycle"]),
        float(rows[2]["l3_per_cycle"]) / float(low["l3_per_cycle"]),
        float(rows[2]["offcore_data_per_cycle"]) / float(low["offcore_data_per_cycle"]),
    ]
    y = np.arange(len(counter_names))
    h = 0.34
    ax1.axvline(1.0, color="#9ca3af", lw=0.9)
    ax1.barh(y - h / 2, high_nondist, h, color="#2f6fdd", label="340-359 / low")
    ax1.barh(y + h / 2, high_work, h, color="#d95f59", label="300-319 / low")
    ax1.set_yticks(y)
    ax1.set_yticklabels(counter_names)
    ax1.invert_yaxis()
    ax1.set_xlabel("Ratio vs low block")
    ax1.set_title("B. Counter direction")
    ax1.grid(True, axis="x", color=grid, lw=0.7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, loc="lower right", handlelength=1.0)
    ax1.set_xlim(0.45, 1.32)

    path_names = ["dist comps", "L0 edges", "cand. loop\nper edge", "accept\nper push", "adj. fetch\nper exp."]
    graph_vals = {
        "340-359 / low": [
            float(rows[1]["graph_dist_comps"]) / float(low["graph_dist_comps"]),
            float(rows[1]["graph_edges"]) / float(low["graph_edges"]),
            float(rows[1]["graph_loop_per_edge_ns"]) / float(low["graph_loop_per_edge_ns"]),
            float(rows[1]["graph_accept_per_push_ns"]) / float(low["graph_accept_per_push_ns"]),
            float(rows[1]["graph_adj_fetch_per_exp_ns"]) / float(low["graph_adj_fetch_per_exp_ns"]),
        ],
        "300-319 / low": [
            float(rows[2]["graph_dist_comps"]) / float(low["graph_dist_comps"]),
            float(rows[2]["graph_edges"]) / float(low["graph_edges"]),
            float(rows[2]["graph_loop_per_edge_ns"]) / float(low["graph_loop_per_edge_ns"]),
            float(rows[2]["graph_accept_per_push_ns"]) / float(low["graph_accept_per_push_ns"]),
            float(rows[2]["graph_adj_fetch_per_exp_ns"]) / float(low["graph_adj_fetch_per_exp_ns"]),
        ],
    }
    xx = np.arange(len(path_names))
    ax2.axhline(1.0, color="#9ca3af", lw=0.9)
    ax2.plot(xx, graph_vals["340-359 / low"], marker="o", lw=1.8, color="#2f6fdd", label="340-359 / low")
    ax2.plot(xx, graph_vals["300-319 / low"], marker="s", lw=1.8, color="#d95f59", label="300-319 / low")
    ax2.set_xticks(xx)
    ax2.set_xticklabels(path_names)
    ax2.set_ylabel("Ratio vs low block")
    ax2.set_title("C. Path localization")
    ax2.grid(True, axis="y", color=grid, lw=0.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, loc="upper left", handlelength=1.4)
    ax2.set_ylim(0.92, 1.34)

    fig.suptitle(
        "M3 search-CPU residual: L1D/load wait in non-distance level-0 graph traversal",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=11.2,
        fontweight="bold",
        color=navy,
    )
    fig.text(
        0.02,
        0.025,
        "Static180K, search-only, post-warmup. Low/high blocks are original SIFT query tags. Graph timing is path-localization only.",
        ha="left",
        va="bottom",
        fontsize=7.8,
        color="#4b5563",
    )
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.92])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    fig.savefig(args.pdf)
    print(f"wrote {args.out}")
    print(f"wrote {args.pdf}")
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
