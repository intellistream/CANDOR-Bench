#!/usr/bin/env python3
"""Plots for the 2026-04-17 OCC experiment sweep.

Figure 1: Search QPS — MVCC+OCC vs RWLock.
  Bars per (threads, rate_family) variant. SIFT only (rwlock compare data
  lives for SIFT only). Two bars per variant: MVCC+OCC and RWLock.

Figure 2: OCC merge cost as % of search op latency.
  Four subplots: (SIFT b=20), (SIFT b=200), (GIST b=20), (GIST b=200).
  Bars across 9 variants (3 threads × 3 rate families) per subplot.
"""
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RESULT_DIR = "/home/junyao/code/ANN-CC-Bench/results"
OUT_DIR = os.path.join(RESULT_DIR, "plots_occ_20260417")

# Column names for OCC benchmark CSVs
OCC_CSVS = {
    "sift": [
        f"{RESULT_DIR}/sift_occ_t16/benchmark_results.csv",
        f"{RESULT_DIR}/sift_occ_t32/benchmark_results.csv",
        f"{RESULT_DIR}/sift_occ_t64/benchmark_results.csv",
    ],
    "gist": [
        f"{RESULT_DIR}/gist_occ_t16/benchmark_results.csv",
        f"{RESULT_DIR}/gist_occ_t32/benchmark_results.csv",
        f"{RESULT_DIR}/gist_occ_t64/benchmark_results.csv",
    ],
}

RWLOCK_SUMMARY = (
    f"{RESULT_DIR}/sift_compare_realtime_sat_rates_summary_20260417.csv"
)


THREAD_ORDER = [16, 32, 64]
FAMILY_ORDER = ["BL", "HR", "HW"]
FAMILY_LABEL = {"BL": "BL", "HR": "HR", "HW": "HW"}

# Map (insert_rate, search_rate) -> family identifier used in the rwlock CSV.
# BL rates differ by thread count (160k:160k at 16T, 320k:320k at 32T/64T).
FAMILY_RATES = {
    16: {"BL": (160000, 160000), "HR": (80000, 720000), "HW": (720000, 80000)},
    32: {"BL": (320000, 320000), "HR": (80000, 720000), "HW": (720000, 80000)},
    64: {"BL": (320000, 320000), "HR": (80000, 720000), "HW": (720000, 80000)},
}


# Colors chosen for good contrast and print readability.
COLOR_OCC = "#4C78A8"   # steel blue
COLOR_RW = "#E45756"    # warm red
COLOR_PCT = "#F58518"   # amber
COLOR_GRID = "#555555"
EDGE = "#222222"


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def family_of(row):
    t = int(float(row["threads"]))
    ir = round(float(row["insert_input_rate"]))
    sr = round(float(row["search_input_rate"]))
    table = FAMILY_RATES.get(t, {})
    for fam, (tir, tsr) in table.items():
        if ir == tir and sr == tsr:
            return fam
    return None


def collect_occ(dataset):
    """Return dict[(threads, batch, family)] -> row."""
    out = {}
    for path in OCC_CSVS[dataset]:
        for row in load_csv(path):
            if row.get("query_mode") != "round_robin":
                continue
            fam = family_of(row)
            if fam is None:
                continue
            t = int(float(row["threads"]))
            bs = int(float(row["write_batch_size"]))
            out[(t, bs, fam)] = row
    return out


def collect_rwlock():
    """Return dict[(threads, family)] -> row from the sift rwlock summary."""
    out = {}
    for row in load_csv(RWLOCK_SUMMARY):
        t = int(float(row["threads"]))
        fam = row["family"]
        out[(t, fam)] = row
    return out


def style_axes(ax, *, ygrid=True):
    if ygrid:
        ax.grid(axis="y", color=COLOR_GRID, alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors="black", labelsize=10)
    ax.tick_params(axis="y", colors="black", labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.9)


def variant_labels():
    return [f"{t}T·{FAMILY_LABEL[f]}" for t in THREAD_ORDER for f in FAMILY_ORDER]


def make_qps_plot(sift_occ, rwlock):
    """Fig 1: SIFT batch=20 search QPS, OCC vs RWLock."""
    labels = variant_labels()
    occ_vals = []
    rw_nolock_vals = []
    rw_vals = []
    speedup_vals = []
    for t in THREAD_ORDER:
        for fam in FAMILY_ORDER:
            occ_row = sift_occ.get((t, 20, fam))
            rw_row = rwlock.get((t, fam))
            occ_qps = float(occ_row["search_qps (per-point)"]) if occ_row else 0.0
            rw_nolock = float(rw_row["nolock_search_qps"]) if rw_row else 0.0
            rw_qps = float(rw_row["rwlock_search_qps"]) if rw_row else 0.0
            occ_vals.append(occ_qps)
            rw_nolock_vals.append(rw_nolock)
            rw_vals.append(rw_qps)
            if rw_qps > 0:
                speedup_vals.append(occ_qps / rw_qps)
            else:
                speedup_vals.append(0.0)

    n = len(labels)
    xs = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5.6))
    bars_occ = ax.bar(xs - width / 2, occ_vals, width,
                      color=COLOR_OCC, edgecolor=EDGE, linewidth=0.8,
                      label="MVCC + OCC (no lock)")
    bars_rw = ax.bar(xs + width / 2, rw_vals, width,
                     color=COLOR_RW, edgecolor=EDGE, linewidth=0.8,
                     label="RWLock")
    ax.set_ylabel("Search QPS  (higher is better)", fontsize=11)
    ax.set_xticks(xs, labels, rotation=0, fontsize=10)
    style_axes(ax)

    ymax = max(occ_vals + rw_vals) if (occ_vals + rw_vals) else 1.0
    ax.set_ylim(0, ymax * 1.22)

    # Annotate: absolute QPS on each bar
    for bar, v in zip(bars_occ, occ_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.012,
                f"{v/1000:.1f}k", ha="center", va="bottom", fontsize=8.5, color=COLOR_OCC,
                fontweight="bold")
    for bar, v in zip(bars_rw, rw_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.012,
                f"{v/1000:.1f}k", ha="center", va="bottom", fontsize=8.5, color=COLOR_RW,
                fontweight="bold")
    # Speedup annotation above each pair
    for x, sp in zip(xs, speedup_vals):
        ax.text(x, ymax * 1.14, f"{sp:.2f}x",
                ha="center", va="center", fontsize=9.5, color="black",
                bbox=dict(boxstyle="round,pad=0.22", fc="#f2f2f2", ec="#888", lw=0.6))

    # Vertical separators between thread groups
    for i in range(1, len(THREAD_ORDER)):
        ax.axvline(i * len(FAMILY_ORDER) - 0.5, color="#bbbbbb", linewidth=0.6, linestyle="--")

    ax.set_title("SIFT search QPS — MVCC+OCC vs RWLock  (batch=20, round-robin)",
                 fontsize=12, pad=14)
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig1_sift_qps_occ_vs_rwlock.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def make_occ_pct_plot(sift_occ, gist_occ):
    """Fig 2: OCC merge cost as % of search op latency. 2×2 subplots."""
    datasets = [
        ("sift", 20, sift_occ),
        ("sift", 200, sift_occ),
        ("gist", 20, gist_occ),
        ("gist", 200, gist_occ),
    ]
    labels = variant_labels()

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharey=True)
    axes = axes.flatten()

    for idx, (ds, bs, data) in enumerate(datasets):
        ax = axes[idx]
        pct_vals = []
        for t in THREAD_ORDER:
            for fam in FAMILY_ORDER:
                row = data.get((t, bs, fam))
                pct_vals.append(float(row["inflight_occ_pct_search_op"]) if row else 0.0)
        xs = np.arange(len(labels))
        bars = ax.bar(xs, pct_vals, width=0.62,
                      color=COLOR_PCT, edgecolor=EDGE, linewidth=0.8)
        ax.set_xticks(xs, labels, rotation=0, fontsize=9)
        ax.set_title(f"{ds.upper()}  batch={bs}", fontsize=11, pad=6)
        ax.set_ylim(0, 100)
        if idx % 2 == 0:
            ax.set_ylabel("OCC cost  (% of search op)", fontsize=10.5)
        style_axes(ax)
        for bar, v in zip(bars, pct_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.6,
                    f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=8.5, color="black",
                    fontweight="bold")
        # Vertical separators between thread groups
        for i in range(1, len(THREAD_ORDER)):
            ax.axvline(i * len(FAMILY_ORDER) - 0.5, color="#bbbbbb",
                       linewidth=0.6, linestyle="--")
        # Horizontal 50% guideline
        ax.axhline(50, color="#888888", linewidth=0.6, linestyle=":")

    fig.suptitle(
        "OCC merge overhead — share of per-query search latency",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = os.path.join(OUT_DIR, "fig2_occ_pct_share.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sift_occ = collect_occ("sift")
    gist_occ = collect_occ("gist")
    rwlock = collect_rwlock()

    missing_sift = [k for k in [(t, 20, f) for t in THREAD_ORDER for f in FAMILY_ORDER] if k not in sift_occ]
    missing_gist = [k for k in [(t, 20, f) for t in THREAD_ORDER for f in FAMILY_ORDER] if k not in gist_occ]
    if missing_sift:
        print(f"WARN: missing SIFT OCC rows: {missing_sift}", file=sys.stderr)
    if missing_gist:
        print(f"WARN: missing GIST OCC rows: {missing_gist}", file=sys.stderr)

    p1 = make_qps_plot(sift_occ, rwlock)
    p2 = make_occ_pct_plot(sift_occ, gist_occ)
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
