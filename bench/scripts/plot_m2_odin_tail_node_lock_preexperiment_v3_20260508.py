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

OUT = FIG_DIR / "14_odin_m2_tail_node_lock_preexperiment_v3.png"
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
GRAY = "#d7dce3"
PAPER_BG = "#fbfbfc"


def setup_style():
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "axes.linewidth": 0.7,
            "axes.titleweight": "normal",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
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
    ax.set_ylim(0, 3.0)
    ax.set_yticks([])
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xticklabels(["0", "3", "6", "9", "12"])
    ax.tick_params(axis="x", colors=MUTED, length=0, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)

    lanes = [
        ("Measured search (8)", 2.18, INK, "white"),
        ("Background search (20)", 1.25, BLUE_SOFT, INK),
        ("Insert (20)", 0.32, "#f0f1f3", INK),
    ]
    h = 0.48

    for start, end in WINDOWS:
        ax.axvspan(start, end, ymin=0.06, ymax=0.93, color=RED_SOFT, alpha=0.42, lw=0, zorder=0)

    for label, y, fill, text_color in lanes:
        ax.add_patch(Rectangle((0, y), 12, h, facecolor=fill, edgecolor="none", zorder=1))
        ax.text(
            0.22,
            y + h / 2,
            label,
            ha="left",
            va="center",
            color=text_color,
            fontsize=8.2,
            zorder=4,
        )

    for start, end in WINDOWS:
        ax.add_patch(Rectangle((start, 0.32), end - start, h, facecolor=RED, edgecolor="none", zorder=3))

    ax.text(12.18, -0.07, "s", ha="left", va="top", color=MUTED, fontsize=7.2)
    ax.set_title("Matched Window", loc="left", pad=2, color=INK)


def load_sift_rows(conclusion):
    rows = conclusion[conclusion["dataset"].isin(["sift200k", "sift10m200k"])].copy()
    rows["label"] = rows["dataset"].map({"sift200k": "SIFT 200k", "sift10m200k": "SIFT10M 200k"})
    rows["order"] = rows["dataset"].map({"sift200k": 0, "sift10m200k": 1})
    return rows.sort_values("order")


def plot_clean(ax, conclusion):
    rows = load_sift_rows(conclusion)
    x = np.arange(len(rows))
    off = rows["clean_insert_p99_off_ms"].to_numpy()
    on = rows["clean_insert_p99_on_ms"].to_numpy()
    lift = rows["clean_insert_p99_lift_pct"].to_numpy()
    top01 = rows["clean_top01_tail_insert_on"].to_numpy()

    for xi, a, b, pct, frac in zip(x, off, on, lift, top01):
        ax.plot([xi, xi], [a, b], color="#aab2bd", lw=1.3, zorder=2)
        ax.scatter([xi], [a], s=50, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter([xi], [b], s=58, color=RED, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(xi + 0.07, b + 1.25, f"+{pct:.1f}%", ha="left", va="bottom", color=RED, fontsize=8.4)
        ax.text(xi, 4.1, f"Top 0.1% in insert: {frac:.0%}", ha="center", va="center", color=MUTED, fontsize=7.4)

    ax.set_xlim(-0.42, len(rows) - 0.58)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["label"])
    ax.set_ylabel("Insert-window P99 (ms)")
    ax.set_ylim(0, 46)
    ax.set_title("Search Tail Rises During Insert", loc="left", pad=3, color=INK)
    clean_axis(ax)


def plot_wait(ax, tail_lock, conclusion):
    rows = tail_lock[
        (tail_lock["case"] == "sift200k_node_lock_on")
        & (tail_lock["tail"] == "top0.1%")
        & (tail_lock["scope"].isin(["insert", "all"]))
    ].copy()
    rows["label"] = rows["scope"].map({"insert": "Insert tail", "all": "Overall tail"})
    rows["order"] = rows["scope"].map({"insert": 0, "all": 1})
    rows = rows.sort_values("order")

    y = np.arange(len(rows))
    batch = rows["batch_op_median_ms"].to_numpy()
    lock = rows["lock_max_median_ms"].to_numpy()
    rest = np.maximum(batch - lock, 0)
    frac = rows["lock_max_frac_median"].to_numpy()

    ax.barh(y, lock, color=RED, height=0.42, zorder=3)
    ax.barh(y, rest, left=lock, color=GRAY, height=0.42, zorder=3)
    for yi, b, l, f in zip(y, batch, lock, frac):
        ax.text(l / 2, yi, f"{l:.1f} ms", ha="center", va="center", color="white", fontsize=8.2)
        ax.text(b + 5, yi, f"{f:.0%}", ha="left", va="center", color=RED, fontsize=8.2)

    sift = load_sift_rows(conclusion)
    probe = "rdlock probe max: "
    probe += " / ".join(f"{r['probe_rdlock_max_on_ms']:.0f} ms" for _, r in sift.iterrows())
    probe += "; off = 0"
    ax.text(
        0.99,
        0.96,
        probe,
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=MUTED,
        fontsize=7.0,
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="square,pad=0.12", alpha=0.92),
    )

    ax.add_patch(Rectangle((4, 1.33), 10, 0.09, facecolor=RED, edgecolor="none", clip_on=False))
    ax.text(17, 1.38, "Node-lock wait", ha="left", va="center", color=INK, fontsize=7.2)
    ax.add_patch(Rectangle((72, 1.33), 10, 0.09, facecolor=GRAY, edgecolor="none", clip_on=False))
    ax.text(85, 1.38, "Other", ha="left", va="center", color=INK, fontsize=7.2)

    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"])
    ax.set_xlabel("Median tail-batch latency (ms)")
    ax.set_xlim(0, 170)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_ylim(-0.55, 1.52)
    ax.invert_yaxis()
    ax.set_title("Tail Is Mostly Node-Lock Wait", loc="left", pad=3, color=INK)
    clean_axis(ax, grid="x")


def main():
    setup_style()
    conclusion = pd.read_csv(CONCLUSION)
    tail_lock = pd.read_csv(TAIL_LOCK)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.9, 3.25), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.82, 1.36],
        width_ratios=[1.0, 1.2],
        left=0.085,
        right=0.975,
        bottom=0.16,
        top=0.92,
        wspace=0.36,
        hspace=0.44,
    )

    ax_window = fig.add_subplot(gs[0, :])
    ax_clean = fig.add_subplot(gs[1, 0])
    ax_wait = fig.add_subplot(gs[1, 1])

    draw_window(ax_window)
    plot_clean(ax_clean, conclusion)
    plot_wait(ax_wait, tail_lock, conclusion)

    conclusion.to_csv(OUT_CSV, index=False)
    fig.savefig(OUT, dpi=360)
    fig.savefig(OUT_PDF)
    print(OUT)
    print(OUT_PDF)
    print(OUT_CSV)


if __name__ == "__main__":
    raise SystemExit(main())
