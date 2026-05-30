#!/usr/bin/env python3
"""Plot M3 insert-side mechanism attribution controls."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


MODE_ORDER = [
    ("cpu_dummy", "CPU task"),
    ("ann_search_dummy", "ANN traversal"),
    ("path_only_insert_control", "Path-only insert"),
    ("full_insert", "Full insert"),
]

FEATURES = [
    ("graph", "graph traversal"),
    ("new_links", "new-node links"),
    ("old_repair", "existing-node repair"),
]

FEATURE_ON = {
    "cpu_dummy": set(),
    "ann_search_dummy": {"graph"},
    "path_only_insert_control": {"graph", "new_links"},
    "full_insert": {"graph", "new_links", "old_repair"},
}

RELEASE_RATE = 40000.0


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as file:
        return {row["mode"]: row for row in csv.DictReader(file)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    modes = [mode for mode, _ in MODE_ORDER]
    labels = [label for _, label in MODE_ORDER]

    e2e = [f(rows[mode]["e2e_overhead_mean_ms"]) for mode in modes]
    wait = [max(0.0, f(rows[mode]["wait_overhead_mean_ms"])) for mode in modes]
    op = [f(rows[mode]["op_overhead_mean_ms"]) for mode in modes]
    llc = [f(rows[mode]["llc_load_miss_rate_mean"]) / 1e6 for mode in modes]
    dtlb = [f(rows[mode]["dtlb_walk_rate_mean"]) / 1e6 for mode in modes]
    insert_qps = [f(rows[mode]["insert_qps"]) for mode in modes]
    backlog_proxy = [max(0.0, RELEASE_RATE - qps) / 1000.0 for qps in insert_qps]

    fig = plt.figure(figsize=(12.8, 7.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.0], wspace=0.32, hspace=0.42)
    ax_matrix = fig.add_subplot(grid[0, 0])
    ax_latency = fig.add_subplot(grid[0, 1])
    ax_cache = fig.add_subplot(grid[1, 0])
    ax_backlog = fig.add_subplot(grid[1, 1])
    fig.patch.set_facecolor("white")

    # Panel A: explicit insert-side condition toggles.
    for y, (_, feature_label) in enumerate(FEATURES):
        for x, mode in enumerate(modes):
            on = FEATURES[y][0] in FEATURE_ON[mode]
            color = "#1f2937" if on else "#f3f4f6"
            edge = "#1f2937" if on else "#d1d5db"
            ax_matrix.add_patch(Rectangle((x - 0.32, y - 0.32), 0.64, 0.64, facecolor=color, edgecolor=edge, linewidth=1.0))
    ax_matrix.set_xlim(-0.65, len(modes) - 0.35)
    ax_matrix.set_ylim(len(FEATURES) - 0.45, -0.55)
    ax_matrix.set_xticks(range(len(modes)))
    ax_matrix.set_xticklabels(labels, rotation=14, ha="right")
    ax_matrix.set_yticks(range(len(FEATURES)))
    ax_matrix.set_yticklabels([label for _, label in FEATURES])
    ax_matrix.set_title("(a) Insert-side condition toggles", loc="left", fontsize=11)
    ax_matrix.tick_params(axis="both", length=0)
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)

    # Panel B: search e2e overhead decomposition.
    x = list(range(len(modes)))
    ax_latency.bar(x, wait, color="#b91c1c", label="wait / dispatch")
    ax_latency.bar(x, op, bottom=wait, color="#9ca3af", label="search op")
    ax_latency.set_xticks(x)
    ax_latency.set_xticklabels(labels, rotation=14, ha="right")
    ax_latency.set_ylabel("matched search overhead (ms)")
    ax_latency.set_title("(b) Full insert overhead is mostly wait", loc="left", fontsize=11)
    ax_latency.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax_latency.set_axisbelow(True)
    ax_latency.legend(frameon=False, loc="upper left")
    for i, value in enumerate(e2e):
        offset = 0.22 if value < 1.0 else 0.45
        ax_latency.text(i, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax_latency.set_ylim(0, max(e2e) * 1.22)

    # Panel C: cache/TLB counters reject a cache-only explanation.
    width = 0.34
    ax_cache.bar([i - width / 2 for i in x], llc, width=width, color="#0369a1", label="LLC load misses")
    ax_cache.bar([i + width / 2 for i in x], dtlb, width=width, color="#6d28d9", label="DTLB walks")
    ax_cache.set_xticks(x)
    ax_cache.set_xticklabels(labels, rotation=14, ha="right")
    ax_cache.set_ylabel("event rate (M/s, log scale)")
    ax_cache.set_yscale("log")
    ax_cache.set_title("(c) Cache/TLB pressure alone is not sufficient", loc="left", fontsize=11)
    ax_cache.grid(axis="y", color="#e5e7eb", linewidth=0.8, which="both")
    ax_cache.set_axisbelow(True)
    ax_cache.legend(frameon=False, loc="upper left")
    ax_cache.annotate(
        "highest cache/TLB,\nlow search overhead",
        xy=(1, max(llc[1], dtlb[1])),
        xytext=(1.55, max(llc[1], dtlb[1]) * 0.55),
        arrowprops={"arrowstyle": "->", "color": "#374151", "linewidth": 0.9},
        fontsize=8,
        ha="left",
    )

    # Panel D: backlog proxy rejects a backlog-only explanation.
    ax_backlog.bar(x, backlog_proxy, color="#64748b", label="release minus insert QPS")
    ax_backlog.set_xticks(x)
    ax_backlog.set_xticklabels(labels, rotation=14, ha="right")
    ax_backlog.set_ylabel("unserved insert release (k pts/s)")
    ax_backlog.set_title("(d) Insert backlog alone is not sufficient", loc="left", fontsize=11)
    ax_backlog.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax_backlog.set_axisbelow(True)
    ax_e2e = ax_backlog.twinx()
    ax_e2e.plot(x, e2e, color="#b91c1c", marker="o", linewidth=1.8, label="search e2e overhead")
    ax_e2e.set_ylabel("search e2e overhead (ms)")
    ax_e2e.set_ylim(0, max(e2e) * 1.25)
    ax_backlog.annotate(
        "larger backlog proxy,\nlow latency",
        xy=(1, backlog_proxy[1]),
        xytext=(1.55, max(backlog_proxy) * 0.58),
        arrowprops={"arrowstyle": "->", "color": "#374151", "linewidth": 0.9},
        fontsize=8,
        ha="left",
    )
    handles1, labels1 = ax_backlog.get_legend_handles_labels()
    handles2, labels2 = ax_e2e.get_legend_handles_labels()
    ax_backlog.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left")

    for ax in [ax_latency, ax_cache, ax_backlog]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_e2e.spines["top"].set_visible(False)

    fig.suptitle(
        "Insert-side mechanism attribution",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=13,
    )
    fig.text(
        0.02,
        0.018,
        "SIFT1M steady read/write, 40k insert pts/s and 5k read pts/s, matched against read-only replay. "
        "Full insert turns write pressure into reader wait; this does not yet isolate a single repair micro-step.",
        fontsize=9,
        color="#374151",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(f"[m3-insert-mechanism-figure] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
