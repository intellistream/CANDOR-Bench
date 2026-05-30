#!/usr/bin/env python3
"""Create a compact main evidence figure for the no-lock HNSW spike."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def normalized_case(row: dict[str, str]) -> str:
    case = row.get("case", "")
    if case.startswith("normal_iw") and not case.endswith("8"):
        return "normal_insert40k"
    if case.startswith("skip_") and case.endswith("insert40k"):
        return "skip_existing_node_repair_insert40k"
    return case


def load_period_rows(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    rows = list(csv.DictReader(path.open()))
    baseline_case = "normal_insert40k"
    ablation_case = "skip_existing_node_repair_insert40k"
    selected = [
        r
        for r in rows
        if r.get("query_mode") == "chasing"
        and r.get("write_rate") == "40000"
        and r.get("read_rate") == "40000"
        and normalized_case(r) in {baseline_case, ablation_case}
    ]
    by_case = {normalized_case(r): r for r in selected}
    return by_case[baseline_case], by_case[ablation_case]


def load_perf_ratios(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open()))
    metrics = {
        "L1D miss": "L1-dcache-load-misses",
        "L2 miss": "l2_rqsts.miss",
        "LLC miss": "LLC-load-misses",
        "Cache miss": "cache-misses",
        "DTLB walk": "dtlb_load_misses.walk_completed",
    }
    ratios: dict[str, list[float]] = {label: [] for label in metrics}
    for mode in {"chasing", "round_robin"}:
        normal = [r for r in rows if r.get("query_mode") == mode and r.get("case", "").startswith("normal")]
        skip = [r for r in rows if r.get("query_mode") == mode and r.get("case", "").startswith("skip_")]
        for label, key in metrics.items():
            n = next((f(r, key) for r in normal if f(r, key) > 0), 0.0)
            s = next((f(r, key) for r in skip if f(r, key) > 0), 0.0)
            if n > 0 and s > 0:
                ratios[label].append(n / s)
    return {label: sum(vals) / len(vals) for label, vals in ratios.items() if vals}


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: plot_m3_rewrite_core_evidence.py <period_summary.csv> <perf_summary.csv> <out.png>",
            file=sys.stderr,
        )
        return 2

    period_summary = Path(sys.argv[1])
    perf_summary = Path(sys.argv[2])
    out = Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)

    normal, skip = load_period_rows(period_summary)
    ratios = load_perf_ratios(perf_summary)

    normal_p99 = f(normal, "p99_e2e_ms")
    skip_p99 = f(skip, "p99_e2e_ms")
    period_prob = f(normal, "period_query_hit_prob_mean")
    skip_period = f(skip, "period_query_hit_prob_mean")
    active_prob_pct = f(normal, "active_node_query_hit_prob_mean") * 100.0
    recent_prob = f(normal, "recent_node_query_hit_prob_mean")
    latency_ratio = normal_p99 / skip_p99 if skip_p99 > 0 else 0.0

    fig = plt.figure(figsize=(15.5, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.18, 0.82], width_ratios=[1.1, 1.25, 1.35])
    fig.patch.set_facecolor("white")

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.72,
        "No-lock HNSW spike: existing-node rewrite creates a cache-pressure period",
        fontsize=20,
        fontweight="bold",
        va="center",
    )
    title_ax.text(
        0.0,
        0.18,
        "SIFT, batch=200, writer-heavy steady insert, write/read=40000/40000, query_mode=chasing",
        fontsize=11,
        color="#475569",
        va="center",
    )

    ax0 = fig.add_subplot(gs[1, 0])
    labels = ["normal\nold-rewrite", "skip\nold-rewrite"]
    vals = [normal_p99, skip_p99]
    colors = ["#2563eb", "#059669"]
    ax0.bar(labels, vals, color=colors, alpha=0.92, width=0.58)
    ax0.set_ylabel("search batch e2e p99 (ms)", fontsize=11)
    ax0.set_title("1. Tail latency collapses", loc="left", fontsize=13, fontweight="bold")
    ax0.text(0.5, max(vals) * 0.78, f"{latency_ratio:.0f}x\nlower", ha="center", fontsize=18, fontweight="bold")
    for i, v in enumerate(vals):
        ax0.text(i, v + max(vals) * 0.035, f"{v:.1f} ms", ha="center", fontsize=11)
    ax0.grid(True, axis="y", color="#e5e7eb")
    ax0.set_axisbelow(True)

    ax1 = fig.add_subplot(gs[1, 1])
    y = [2.0, 1.0, 0.0]
    names = ["rewrite period\n(normal)", "rewrite period\n(skip)", "exact active node\n(normal)"]
    values = [period_prob, skip_period, active_prob_pct / 100.0]
    bar_colors = ["#64748b", "#cbd5e1", "#7c3aed"]
    ax1.barh(y, values, color=bar_colors, alpha=0.92, height=0.46)
    ax1.set_xlim(0, 1.05)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.set_xlabel("query hit probability")
    ax1.set_title("2. Search hits the period, not the exact node", loc="left", fontsize=13, fontweight="bold")
    ax1.text(period_prob + 0.02, 2.0, f"{period_prob:.2f}", va="center", fontsize=11)
    ax1.text(skip_period + 0.02, 1.0, f"{skip_period:.2f}", va="center", fontsize=11)
    ax1.text(0.04, 0.0, f"{active_prob_pct:.5f}%", va="center", fontsize=11, color="#581c87")
    ax1.text(
        0.02,
        0.08,
        f"recent-node hit is common ({recent_prob:.2f}), but not discriminative",
        transform=ax1.transAxes,
        fontsize=9,
        color="#475569",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    ax1.grid(True, axis="x", color="#e5e7eb")
    ax1.set_axisbelow(True)

    ax2 = fig.add_subplot(gs[1, 2])
    metric_names = list(ratios)
    metric_vals = [ratios[k] for k in metric_names]
    metric_colors = ["#1d4ed8", "#0f766e", "#ea580c", "#9333ea", "#be123c"]
    ax2.bar(metric_names, metric_vals, color=metric_colors, alpha=0.9)
    ax2.set_ylabel("normal / skip-old-rewrite ratio")
    ax2.set_title("3. Cache/TLB pressure explodes", loc="left", fontsize=13, fontweight="bold")
    for i, v in enumerate(metric_vals):
        ax2.text(i, v + max(metric_vals) * 0.025, f"{v:.1f}x", ha="center", fontsize=10)
    ax2.tick_params(axis="x", rotation=18)
    ax2.grid(True, axis="y", color="#e5e7eb")
    ax2.set_axisbelow(True)

    fig.text(
        0.02,
        0.02,
        "Interpretation: no-lock search does not wait on a lock. It slows because batches overlap the existing-node rewrite/prune period, where graph-memory rewrites amplify cache/TLB misses.",
        fontsize=11,
        color="#334155",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
