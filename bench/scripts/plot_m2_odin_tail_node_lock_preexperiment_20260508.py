#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "paper-progress" / "m3" / "search-tail-rootcause"
TABLE_DIR = PAPER / "tables"
FIG_DIR = PAPER / "figures"

CONCLUSION = TABLE_DIR / "32_m2_tail_causality_matched_conclusion.csv"
TAIL_LOCK = TABLE_DIR / "26_tail_batch_lock_wait_summary.csv"

OUT = FIG_DIR / "14_odin_m2_tail_node_lock_preexperiment.png"
OUT_PDF = OUT.with_suffix(".pdf")
OUT_CSV = OUT.with_suffix(".csv")

WINDOWS = [(2.5, 3.5), (6.0, 7.0), (9.5, 10.5)]

INK = "#263140"
MUTED = "#657184"
GRID = "#e8ebf0"
RED = "#c94b55"
RED_LIGHT = "#efd5d2"
BLUE = "#2f69b9"
BLUE_LIGHT = "#dbe5f2"
GOLD = "#c99a32"
GRAY = "#9aa3ad"
PALE = "#f6f7f9"


def style_axes(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c7cdd5")
    ax.spines["bottom"].set_color("#c7cdd5")
    ax.tick_params(axis="both", colors=INK, length=2.5, width=0.7)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.7, zorder=0)


def draw_lanes(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    rows = [
        (2.22, "Measured search", "8", "#263140", "white"),
        (1.30, "Background search", "20", BLUE_LIGHT, INK),
        (0.38, "Insert", "20", "#f4eeee", INK),
    ]
    for y, label, count, color, text_color in rows:
        ax.add_patch(Rectangle((0, y), 12, 0.48, facecolor=color, edgecolor="none", zorder=1))
        ax.text(-0.08, y + 0.24, label, ha="right", va="center", color=INK, fontsize=7.9)
        ax.text(0.18, y + 0.24, count, ha="left", va="center", color=text_color, fontsize=8.0)

    for start, end in WINDOWS:
        ax.add_patch(Rectangle((start, 0.38), end - start, 0.48, facecolor=RED, edgecolor="none", zorder=2))
        ax.text((start + end) / 2, 0.62, "Insert", ha="center", va="center", color="white", fontsize=7.5)
        ax.axvspan(start, end, ymin=0.02, ymax=0.96, color=RED_LIGHT, alpha=0.36, lw=0, zorder=0)

    for x in [0, 3, 6, 9, 12]:
        ax.text(x, 0.02, f"{x}", ha="center", va="bottom", color=MUTED, fontsize=7.0)
    ax.text(12.22, 0.02, "s", ha="left", va="bottom", color=MUTED, fontsize=7.0)


def plot_clean_effect(ax, conclusion):
    rows = conclusion[conclusion["dataset"].isin(["sift200k", "sift10m200k"])].copy()
    rows["label"] = rows["dataset"].map({"sift200k": "SIFT\n200k", "sift10m200k": "SIFT10M\n200k"})
    x = np.arange(len(rows))
    width = 0.32
    off = rows["clean_insert_p99_off_ms"].to_numpy()
    on = rows["clean_insert_p99_on_ms"].to_numpy()
    lift = rows["clean_insert_p99_lift_pct"].to_numpy()
    top01 = rows["clean_top01_tail_insert_on"].to_numpy()

    ax.bar(x - width / 2, off, width, color=BLUE, label="Node Lock Off", zorder=3)
    ax.bar(x + width / 2, on, width, color=RED, label="Node Lock On", zorder=3)
    for i, (a, pct, frac) in enumerate(zip(on, lift, top01)):
        ax.text(i, a + 2.2, f"+{pct:.1f}%", ha="center", va="bottom", color=RED, fontsize=8.2)
        ax.text(i, a + 6.4, f"Top 0.1%: {frac:.0%}",
                ha="center", va="bottom", color=INK, fontsize=7.2)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["label"])
    ax.set_ylabel("Insert-Window\nBatch P99 (ms)")
    ax.set_ylim(0, max(on.max(), off.max()) * 1.42)
    ax.legend(loc="upper left", frameon=False, handlelength=1.2, labelspacing=0.18)
    style_axes(ax)


def plot_tail_wait(ax, tail_lock):
    pick = tail_lock[
        (tail_lock["case"] == "sift200k_node_lock_on")
        & (tail_lock["tail"] == "top0.1%")
        & (tail_lock["scope"].isin(["all", "insert"]))
    ].copy()
    order = {"insert": 0, "all": 1}
    pick["order"] = pick["scope"].map(order)
    pick = pick.sort_values("order")
    labels = ["Insert-window\ntail", "Overall\ntail"]
    batch = pick["batch_op_median_ms"].to_numpy()
    lock = pick["lock_max_median_ms"].to_numpy()
    rest = np.maximum(batch - lock, 0)
    y = np.arange(len(pick))
    ax.barh(y, lock, color=RED, height=0.46, label="Node-lock wait", zorder=3)
    ax.barh(y, rest, left=lock, color="#d7dce3", height=0.46, label="Other time", zorder=3)
    for yi, l, b in zip(y, lock, batch):
        frac = l / b if b else 0
        ax.text(min(l * 0.5, l - 2) if l > 10 else l + 2, yi, f"{l:.1f} ms",
                ha="center" if l > 10 else "left", va="center",
                color="white" if l > 10 else INK, fontsize=8.0)
        ax.text(b + 5, yi, f"{frac:.0%}", ha="left", va="center", color=RED, fontsize=8.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Median Batch Latency (ms)")
    ax.set_xlim(0, max(batch) * 1.34)
    ax.legend(loc="lower right", frameon=False, handlelength=1.3, labelspacing=0.18)
    style_axes(ax, grid_axis="x")
    ax.invert_yaxis()


def plot_probe(ax, conclusion):
    rows = conclusion[conclusion["dataset"].isin(["sift200k", "sift10m200k", "gist200k", "dbpedia50k"])].copy()
    rows["label"] = rows["dataset"].map(
        {
            "sift200k": "SIFT\n200k",
            "sift10m200k": "SIFT10M\n200k",
            "gist200k": "GIST\n200k",
            "dbpedia50k": "DBpedia\n50k",
        }
    )
    x = np.arange(len(rows))
    max_wait = rows["probe_rdlock_max_on_ms"].to_numpy()
    count = rows["probe_rdlock_count_on"].to_numpy()
    off_count = rows["probe_rdlock_count_off"].to_numpy()
    corr = rows["probe_corr_bin_p99_rdlock_max_on"].to_numpy()
    colors = [RED if lab.startswith("SIFT") else GRAY for lab in rows["label"]]

    ax.vlines(x, 0, max_wait, color=colors, lw=2.0, zorder=2)
    ax.scatter(x, max_wait, s=42, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(0, color="#c7cdd5", lw=0.8)
    for xi, wait, c, off, r in zip(x, max_wait, count, off_count, corr):
        ax.text(xi, wait + 8, f"{wait:.0f} ms", ha="center", va="bottom", color=INK, fontsize=7.4)
        ax.text(xi, 7, f"n={int(c)}\noff={int(off)}\nr={r:.2f}", ha="center", va="bottom",
                color=MUTED, fontsize=6.6, linespacing=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["label"])
    ax.set_ylabel("Slow rdlock\nMax (ms)")
    ax.set_ylim(0, max_wait.max() * 1.28)
    style_axes(ax)


def main():
    conclusion = pd.read_csv(CONCLUSION)
    tail_lock = pd.read_csv(TAIL_LOCK)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.72,
            "axes.titleweight": "normal",
        }
    )

    fig = plt.figure(figsize=(8.5, 4.15), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.42, 1.05, 1.18],
        height_ratios=[0.72, 1.05],
        left=0.12,
        right=0.985,
        bottom=0.15,
        top=0.925,
        wspace=0.42,
        hspace=0.48,
    )
    ax_lanes = fig.add_subplot(gs[0, 0])
    ax_clean = fig.add_subplot(gs[1, 0])
    ax_wait = fig.add_subplot(gs[:, 1])
    ax_probe = fig.add_subplot(gs[:, 2])

    draw_lanes(ax_lanes)
    plot_clean_effect(ax_clean, conclusion)
    plot_tail_wait(ax_wait, tail_lock)
    plot_probe(ax_probe, conclusion)

    ax_lanes.set_title("Matched Read-Write Window", loc="left", pad=2)
    ax_clean.set_title("Clean Latency Effect", loc="left", pad=4)
    ax_wait.set_title("How Much Tail Is Lock Wait", loc="left", pad=4)
    ax_probe.set_title("External Mechanism Check", loc="left", pad=4)

    summary = []
    for _, row in conclusion.iterrows():
        summary.append(
            {
                "dataset": row["dataset"],
                "insert_p99_off_ms": row["clean_insert_p99_off_ms"],
                "insert_p99_on_ms": row["clean_insert_p99_on_ms"],
                "insert_p99_lift_pct": row["clean_insert_p99_lift_pct"],
                "top01_tail_insert_on": row["clean_top01_tail_insert_on"],
                "probe_rdlock_count_on": row["probe_rdlock_count_on"],
                "probe_rdlock_max_on_ms": row["probe_rdlock_max_on_ms"],
                "probe_rdlock_count_off": row["probe_rdlock_count_off"],
            }
        )
    for _, row in tail_lock.iterrows():
        if row["case"] == "sift200k_node_lock_on" and row["tail"] == "top0.1%" and row["scope"] in {"all", "insert"}:
            summary.append(
                {
                    "dataset": f"sift200k_tail_{row['scope']}",
                    "insert_p99_off_ms": "",
                    "insert_p99_on_ms": row["batch_op_median_ms"],
                    "insert_p99_lift_pct": "",
                    "top01_tail_insert_on": "",
                    "probe_rdlock_count_on": "",
                    "probe_rdlock_max_on_ms": row["lock_max_median_ms"],
                    "probe_rdlock_count_off": "",
                }
            )
    pd.DataFrame(summary).to_csv(OUT_CSV, index=False)

    fig.savefig(OUT, dpi=320)
    fig.savefig(OUT_PDF)
    print(OUT)
    print(OUT_PDF)
    print(OUT_CSV)


if __name__ == "__main__":
    raise SystemExit(main())
