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

OUT = FIG_DIR / "14_odin_m2_tail_node_lock_preexperiment_v2.png"
OUT_PDF = OUT.with_suffix(".pdf")
OUT_CSV = OUT.with_suffix(".csv")

WINDOWS = [(2.5, 3.5), (6.0, 7.0), (9.5, 10.5)]

INK = "#273241"
MUTED = "#6b7280"
GRID = "#e6e9ef"
RED = "#c94b55"
RED_SOFT = "#efd8d6"
BLUE = "#2f69b9"
BLUE_SOFT = "#dfe7f3"
GRAY = "#d5dbe3"
PALE = "#f7f8fa"


def setup_style():
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "axes.titleweight": "normal",
        }
    )


def clean_axis(ax, grid="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c8ced7")
    ax.spines["bottom"].set_color("#c8ced7")
    ax.tick_params(axis="both", colors=INK, length=2.4, width=0.65)
    ax.grid(True, axis=grid, color=GRID, lw=0.62, zorder=0)


def draw_window(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_measured = 0.56
    y_bg = 0.29
    y_insert = 0.09
    h = 0.11

    ax.text(-0.015, y_measured + h / 2, "Measured search", transform=ax.transAxes,
            ha="right", va="center", fontsize=7.0, color=INK)
    ax.text(-0.015, y_bg + h / 2, "Background search", transform=ax.transAxes,
            ha="right", va="center", fontsize=7.0, color=INK)
    ax.text(-0.015, y_insert + h / 2, "Insert", transform=ax.transAxes,
            ha="right", va="center", fontsize=7.0, color=INK)

    ax.add_patch(Rectangle((0, y_measured), 12, h, color=INK, lw=0))
    ax.add_patch(Rectangle((0, y_bg), 12, h, color=BLUE_SOFT, lw=0))
    ax.add_patch(Rectangle((0, y_insert), 12, h, color="#f0f1f3", lw=0))

    ax.text(0.18, y_measured + h / 2, "8", ha="left", va="center", color="white", fontsize=7.2)
    ax.text(0.18, y_bg + h / 2, "20", ha="left", va="center", color=INK, fontsize=7.2)
    ax.text(0.18, y_insert + h / 2, "20", ha="left", va="center", color=INK, fontsize=7.2)

    for start, end in WINDOWS:
        ax.axvspan(start, end, ymin=0.0, ymax=0.93, color=RED_SOFT, alpha=0.42, lw=0, zorder=0)
        ax.add_patch(Rectangle((start, y_insert), end - start, h, color=RED, lw=0, zorder=2))
    for x in [0, 3, 6, 9, 12]:
        ax.text(x, 0.0, str(x), ha="center", va="bottom", color=MUTED, fontsize=6.8)
    ax.text(12.16, 0.0, "s", ha="left", va="bottom", color=MUTED, fontsize=6.8)


def plot_clean(ax, c):
    rows = c[c["dataset"].isin(["sift200k", "sift10m200k"])].copy()
    rows["label"] = rows["dataset"].map({"sift200k": "SIFT 200k", "sift10m200k": "SIFT10M 200k"})
    x = np.arange(len(rows))
    off = rows["clean_insert_p99_off_ms"].to_numpy()
    on = rows["clean_insert_p99_on_ms"].to_numpy()
    lift = rows["clean_insert_p99_lift_pct"].to_numpy()
    top01 = rows["clean_top01_tail_insert_on"].to_numpy()

    for xi, a, b, pct, frac in zip(x, off, on, lift, top01):
        ax.plot([xi, xi], [a, b], color="#aab2bd", lw=1.2, zorder=2)
        ax.scatter([xi], [a], s=48, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter([xi], [b], s=56, color=RED, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(xi + 0.08, b + 1.2, f"+{pct:.1f}%", ha="left", va="bottom", color=RED, fontsize=8.2)
        ax.text(xi, 4.0, f"Top 0.1%: {frac:.0%}", ha="center", va="center", color=MUTED, fontsize=7.0)

    ax.set_xlim(-0.45, len(rows) - 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["label"])
    ax.set_ylabel("Insert-window\nbatch P99 (ms)")
    ax.set_ylim(0, 45)
    clean_axis(ax)
    ax.scatter([], [], s=48, color=BLUE, label="Node lock off")
    ax.scatter([], [], s=56, color=RED, label="Node lock on")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.35, borderpad=0.1)


def plot_wait(ax, t):
    rows = t[
        (t["case"] == "sift200k_node_lock_on")
        & (t["tail"] == "top0.1%")
        & (t["scope"].isin(["insert", "all"]))
    ].copy()
    rows["label"] = rows["scope"].map({"insert": "Insert-window tail", "all": "Overall tail"})
    rows["order"] = rows["scope"].map({"insert": 0, "all": 1})
    rows = rows.sort_values("order")

    y = np.arange(len(rows))
    batch = rows["batch_op_median_ms"].to_numpy()
    lock = rows["lock_max_median_ms"].to_numpy()
    rest = np.maximum(batch - lock, 0)
    frac = lock / batch

    ax.barh(y, lock, color=RED, height=0.36, label="Node-lock wait", zorder=3)
    ax.barh(y, rest, left=lock, color=GRAY, height=0.36, label="Other", zorder=3)
    for yi, b, l, f in zip(y, batch, lock, frac):
        ax.text(l / 2, yi, f"{l:.1f} ms", ha="center", va="center", color="white", fontsize=8.0)
        ax.text(b + 4, yi, f"{f:.0%}", ha="left", va="center", color=RED, fontsize=8.4)

    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"])
    ax.set_xlabel("Median tail-batch latency (ms)")
    ax.set_xlim(0, 170)
    clean_axis(ax, grid="x")
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False, handlelength=1.2, handletextpad=0.4)


def draw_probe_note(ax, c):
    ax.axis("off")
    rows = c[c["dataset"].isin(["sift200k", "sift10m200k"])].copy()
    x0 = 0.03
    y0 = 0.86
    ax.text(x0, y0, "External mechanism check", ha="left", va="top", color=INK, fontsize=8.4)
    y = y0 - 0.25
    for _, r in rows.iterrows():
        name = "SIFT 200k" if r["dataset"] == "sift200k" else "SIFT10M 200k"
        ax.text(x0, y, name, ha="left", va="center", color=INK, fontsize=7.3)
        ax.text(
            x0 + 0.42,
            y,
            f"rdlock max {r['probe_rdlock_max_on_ms']:.0f} ms",
            ha="left",
            va="center",
            color=RED,
            fontsize=7.3,
        )
        ax.text(x0 + 0.42, y - 0.13, "off: 0", ha="left", va="center", color=MUTED, fontsize=6.9)
        y -= 0.35


def main():
    setup_style()
    c = pd.read_csv(CONCLUSION)
    t = pd.read_csv(TAIL_LOCK)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.85, 2.82), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.18, 1.24, 0.86],
        height_ratios=[0.58, 1.0],
        left=0.135,
        right=0.945,
        bottom=0.18,
        top=0.92,
        wspace=0.50,
        hspace=0.32,
    )
    ax_win = fig.add_subplot(gs[0, 0])
    ax_clean = fig.add_subplot(gs[1, 0])
    ax_wait = fig.add_subplot(gs[:, 1])
    ax_note = fig.add_subplot(gs[:, 2])

    draw_window(ax_win)
    plot_clean(ax_clean, c)
    plot_wait(ax_wait, t)
    draw_probe_note(ax_note, c)

    ax_win.set_title("Matched read-write window", loc="left", pad=1)
    ax_clean.set_title("Clean latency", loc="left", pad=3)
    ax_wait.set_title("Tail batch composition", loc="left", pad=3)

    c.to_csv(OUT_CSV, index=False)
    fig.savefig(OUT, dpi=340)
    fig.savefig(OUT_PDF)
    print(OUT)
    print(OUT_PDF)
    print(OUT_CSV)


if __name__ == "__main__":
    raise SystemExit(main())
