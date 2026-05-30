#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "paper-progress" / "m3" / "search-tail-rootcause"
TABLE_DIR = PAPER / "tables"
FIG_DIR = PAPER / "figures"

TIMELINE = TABLE_DIR / "23_m2_node_lock_dataset_bins.csv"
MATRIX = TABLE_DIR / "22_m2_node_lock_dataset_matrix.csv"
HW_SUMMARY = TABLE_DIR / "25_m2_node_lock_hardware_summary.csv"
OUT = FIG_DIR / "13_odin_m2_node_lock_preexperiment.png"
OUT_PDF = OUT.with_suffix(".pdf")
OUT_SUMMARY = OUT.with_suffix(".summary.csv")

WINDOWS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]


def shade_insert(ax, alpha=0.52):
    for start, end in WINDOWS:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#f1d5d2", alpha=alpha, lw=0, zorder=0)


def draw_lanes(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.axis("off")
    read_dark = "#253044"
    bg_read = "#dfe6ef"
    insert = "#c94b55"
    text = "#27313f"
    y0, y1, h = 0.20, 0.60, 0.24
    ax.add_patch(Rectangle((0, y0), 12, h, facecolor=read_dark, edgecolor="none"))
    ax.text(-0.02, y0 + h / 2, "Measured\nReads", transform=ax.get_yaxis_transform(),
            color=text, va="center", ha="right", fontsize=6.8, linespacing=0.92)
    ax.text(0.20, y0 + h / 2, "8", color="white", va="center", ha="left", fontsize=7.0)
    cursor = 0.0
    for start, end in WINDOWS:
        start_s, end_s = start / 1000.0, end / 1000.0
        if start_s > cursor:
            ax.add_patch(Rectangle((cursor, y1), start_s - cursor, h, facecolor=bg_read, edgecolor="none"))
        ax.add_patch(Rectangle((start_s, y1), end_s - start_s, h, facecolor=insert, edgecolor="none"))
        ax.text((start_s + end_s) / 2, y1 + h / 2, "Insert", color="white", va="center", ha="center", fontsize=7.2)
        cursor = end_s
    if cursor < 12:
        ax.add_patch(Rectangle((cursor, y1), 12 - cursor, h, facecolor=bg_read, edgecolor="none"))
    ax.text(-0.02, y1 + h / 2, "Competing\nThreads", transform=ax.get_yaxis_transform(),
            color=text, va="center", ha="right", fontsize=6.8, linespacing=0.92)
    ax.text(0.20, y1 + h / 2, "40", color=text, va="center", ha="left", fontsize=7.0)


def plot_timeline(ax_lane, ax_tail, ax_wait, timeline):
    draw_lanes(ax_lane)
    sift = timeline[(timeline["dataset"] == "sift200k") & (timeline["node_lock"].isin(["on", "off"]))].copy()
    colors = {"on": "#c94b55", "off": "#2f69b9"}
    for lock in ["off", "on"]:
        frame = sift[sift["node_lock"] == lock].sort_values("bin_ms")
        label = "Node Lock Off" if lock == "off" else "Node Lock On"
        ax_tail.plot(frame["bin_ms"] / 1000.0, frame["op_p99_ms"], color=colors[lock], lw=1.55, label=label)
    on = sift[sift["node_lock"] == "on"].sort_values("bin_ms")
    ax_wait.fill_between(on["bin_ms"] / 1000.0, 0, on["rdlock_wait_max_ms"], color="#c94b55", alpha=0.18, lw=0)
    ax_wait.plot(on["bin_ms"] / 1000.0, on["rdlock_wait_max_ms"], color="#c94b55", lw=1.25)

    for ax in (ax_tail, ax_wait):
        shade_insert(ax)
        ax.set_xlim(0, 12)
        ax.grid(True, axis="y", color="#e6e9ef", lw=0.62)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=2.4, width=0.62)

    ax_tail.set_ylabel("Read P99\n(ms)")
    ax_tail.set_ylim(0, 7.9)
    ax_tail.legend(loc="upper right", frameon=False, handlelength=1.8, labelspacing=0.25)
    ax_tail.tick_params(axis="x", labelbottom=False)
    ax_wait.set_ylabel("Read-Lock\nWait (ms)")
    ax_wait.set_ylim(0, 150)
    ax_wait.set_yticks([0, 75, 150])
    ax_wait.set_xlabel("Time (s)")


def plot_ablation(ax, matrix):
    datasets = [("sift200k", "SIFT\n128d"), ("gist200k", "GIST\n960d")]
    x = np.arange(len(datasets))
    width = 0.34
    on_vals, off_vals = [], []
    for dataset, _ in datasets:
        on = matrix[(matrix["dataset"] == dataset) & (matrix["node_lock"] == "on")].iloc[0]
        off = matrix[(matrix["dataset"] == dataset) & (matrix["node_lock"] == "off")].iloc[0]
        on_vals.append(float(on["insert_op_p99_max_ms"]))
        off_vals.append(float(off["insert_op_p99_max_ms"]))
    ax.bar(x - width / 2, on_vals, width=width, color="#c94b55", label="On")
    ax.bar(x + width / 2, off_vals, width=width, color="#2f69b9", label="Off")
    for i, (a, b) in enumerate(zip(on_vals, off_vals)):
        change = (a - b) / a * 100.0 if a else 0
        text = f"-{change:.0f}%" if change > 0 else "No drop"
        ax.text(i, max(a, b) + 0.55, text, ha="center", va="bottom", color="#27313f", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in datasets])
    ax.set_ylabel("Insert-Window\nP99 Max (ms)")
    ax.set_ylim(0, max(max(on_vals), max(off_vals)) * 1.32)
    ax.grid(True, axis="y", color="#e6e9ef", lw=0.62)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.2, columnspacing=0.9)


def plot_correlation(ax, summary):
    rows = []
    for dataset, label in [("sift200k", "SIFT 128d"), ("gist200k", "GIST 960d")]:
        row = summary[(summary["dataset"] == dataset) & (summary["node_lock"] == "on")].iloc[0]
        rows.append(
            {
                "label": label,
                "Lock": float(row["corr_insert_p99_vs_level0_lock_wait_p99_ms"]),
                "RFO": float(row["corr_insert_p99_vs_l2_rqsts.rfo_miss"]),
                "Long Load": float(row["corr_insert_p99_vs_longest_lat_cache.miss"]),
            }
        )
    colors = {"Lock": "#c94b55", "RFO": "#c9972f", "Long Load": "#596879"}
    y_positions = []
    y_labels = []
    values = []
    bar_colors = []
    group_centers = []
    y = 0
    for row in rows:
        start = y
        for j, (metric, label) in enumerate([("Lock", "Lock"), ("RFO", "RFO"), ("Long Load", "Long\nLoad")]):
            y_positions.append(y)
            if j == 0:
                y_labels.append(f"{row['label']}\n{label}")
            else:
                y_labels.append(label)
            values.append(row[metric])
            bar_colors.append(colors[metric])
            y += 1
        group_centers.append((row["label"], (start + y - 1) / 2.0))
        y += 0.7
    ax.barh(y_positions, values, height=0.56, color=bar_colors)
    for x, y_pos in zip(values, y_positions):
        if x >= 0:
            ax.text(x + 0.025, y_pos, f"{x:.2f}", va="center", ha="left", fontsize=6.8, color="#27313f")
        else:
            ax.text(x + 0.035, y_pos, f"{x:.2f}", va="center", ha="left", fontsize=6.8, color="white")
    ax.axvline(0, color="#a9b1bc", lw=0.85)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlim(-0.62, 0.90)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_xlabel("Correlation With P99")
    ax.grid(True, axis="x", color="#e6e9ef", lw=0.62)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()


def main():
    timeline = pd.read_csv(TIMELINE)
    matrix = pd.read_csv(MATRIX)
    summary = pd.read_csv(HW_SUMMARY)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.70,
        }
    )

    fig = plt.figure(figsize=(8.2, 3.35), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.78, 0.78, 1.18],
        height_ratios=[0.43, 1.0, 0.82],
        left=0.078,
        right=0.975,
        top=0.92,
        bottom=0.16,
        wspace=0.44,
        hspace=0.17,
    )
    ax_lane = fig.add_subplot(gs[0, 0])
    ax_tail = fig.add_subplot(gs[1, 0], sharex=ax_lane)
    ax_wait = fig.add_subplot(gs[2, 0], sharex=ax_lane)
    ax_ablation = fig.add_subplot(gs[:, 1])
    ax_corr = fig.add_subplot(gs[:, 2])

    plot_timeline(ax_lane, ax_tail, ax_wait, timeline)
    plot_ablation(ax_ablation, matrix)
    plot_correlation(ax_corr, summary)

    ax_tail.set_title("Timeline Evidence", loc="left", fontweight="normal")
    ax_ablation.set_title("Node-Lock Ablation", loc="left", fontweight="normal")
    ax_corr.set_title("Metric Alignment", loc="left", fontweight="normal")

    fig.savefig(OUT, dpi=280)
    fig.savefig(OUT_PDF)

    rows = []
    for dataset in ["sift200k", "gist200k"]:
        on = matrix[(matrix["dataset"] == dataset) & (matrix["node_lock"] == "on")].iloc[0]
        off = matrix[(matrix["dataset"] == dataset) & (matrix["node_lock"] == "off")].iloc[0]
        hw = summary[(summary["dataset"] == dataset) & (summary["node_lock"] == "on")].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "insert_p99_on_ms": on["insert_op_p99_max_ms"],
                "insert_p99_off_ms": off["insert_op_p99_max_ms"],
                "lock_corr": hw["corr_insert_p99_vs_level0_lock_wait_p99_ms"],
                "rfo_corr": hw["corr_insert_p99_vs_l2_rqsts.rfo_miss"],
                "long_load_corr": hw["corr_insert_p99_vs_longest_lat_cache.miss"],
            }
        )
    pd.DataFrame(rows).to_csv(OUT_SUMMARY, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
