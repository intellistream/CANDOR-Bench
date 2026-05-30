#!/usr/bin/env python3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "paper-progress" / "m3" / "search-tail-rootcause"
CLEAN_BINS = PAPER / "tables" / "23_m2_node_lock_dataset_bins.csv"
HW_BINS = PAPER / "tables" / "24_m2_node_lock_hardware_bins.csv"
HW_SUMMARY = PAPER / "tables" / "25_m2_node_lock_hardware_summary.csv"
OUT = PAPER / "figures" / "12_m2_node_lock_multicausal_evidence.png"
OUT_PDF = OUT.with_suffix(".pdf")
SUMMARY = PAPER / "figures" / "12_m2_node_lock_multicausal_evidence_summary.csv"
WINDOWS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]


DATASETS = [
    ("sift200k", "SIFT 128d"),
    ("gist200k", "GIST 960d"),
]


def shade_insert(ax):
    for start, end in WINDOWS:
        ax.axvspan(start / 1000.0, end / 1000.0, color="#f2d7d5", alpha=0.55, lw=0, zorder=0)


def normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    mx = float(values.max())
    if mx <= 0.0:
        return values
    return values / mx


def plot_p99(ax, clean: pd.DataFrame, dataset: str):
    sub = clean[clean["dataset"] == dataset].copy()
    shade_insert(ax)
    colors = {"on": "#c94758", "off": "#2f68b8"}
    labels = {"on": "Node Lock On", "off": "Node Lock Off"}
    for lock in ["on", "off"]:
        frame = sub[sub["node_lock"] == lock].sort_values("bin_ms")
        ax.plot(
            frame["bin_ms"] / 1000.0,
            frame["op_p99_ms"],
            color=colors[lock],
            lw=1.55,
            label=labels[lock],
        )
    ax.set_xlim(0, 12)
    ax.grid(True, axis="y", color="#e6e8ec", lw=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2.6, width=0.65)


def plot_pressure(ax, hw: pd.DataFrame, dataset: str):
    sub = hw[(hw["dataset"] == dataset) & (hw["node_lock"] == "on")].sort_values("bin_ms").copy()
    shade_insert(ax)
    x = sub["bin_ms"] / 1000.0
    lock = normalize(sub["level0_lock_wait_p99_ms"])
    rfo = normalize(sub["l2_rqsts.rfo_miss"])
    long_load = normalize(sub["longest_lat_cache.miss"])
    ax.fill_between(x, 0, long_load, color="#8793a5", alpha=0.20, lw=0, label="Long-Latency Load")
    ax.plot(x, lock, color="#c94758", lw=1.45, label="Node-Lock Wait")
    ax.plot(x, rfo, color="#c9972f", lw=1.15, ls=(0, (3, 2)), label="RFO Miss")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0, 12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(True, axis="y", color="#e6e8ec", lw=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=2.6, width=0.65)


def plot_corr(ax, summary: pd.DataFrame, dataset: str):
    row = summary[(summary["dataset"] == dataset) & (summary["node_lock"] == "on")].iloc[0]
    labels = ["Lock", "RFO", "LLC", "Long\nLoad"]
    cols = [
        "corr_insert_p99_vs_level0_lock_wait_p99_ms",
        "corr_insert_p99_vs_l2_rqsts.rfo_miss",
        "corr_insert_p99_vs_LLC-load-misses",
        "corr_insert_p99_vs_longest_lat_cache.miss",
    ]
    vals = [float(row[c]) if pd.notna(row[c]) and row[c] != "" else 0.0 for c in cols]
    colors = ["#c94758", "#c9972f", "#8f99a8", "#526170"]
    y = np.arange(len(labels))
    ax.axvline(0, color="#a8afb9", lw=0.8)
    ax.barh(y, vals, color=colors, height=0.52)
    for yi, v in zip(y, vals):
        ax.text(0.82, yi, f"{v:.2f}", va="center", ha="right", fontsize=7.4, color="#27313f")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.62, 0.90)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.grid(True, axis="x", color="#e6e8ec", lw=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.6, width=0.65)
    ax.invert_yaxis()


def main() -> int:
    clean = pd.read_csv(CLEAN_BINS)
    hw = pd.read_csv(HW_BINS)
    summary = pd.read_csv(HW_SUMMARY)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.72,
        }
    )

    fig = plt.figure(figsize=(8.05, 4.95), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.70, 1.58, 1.12],
        hspace=0.34,
        wspace=0.40,
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.13,
    )

    axes = {}
    for r, (dataset, name) in enumerate(DATASETS):
        ax_p99 = fig.add_subplot(gs[r, 0])
        ax_pressure = fig.add_subplot(gs[r, 1], sharex=ax_p99)
        ax_corr = fig.add_subplot(gs[r, 2])
        axes[(dataset, "p99")] = ax_p99
        axes[(dataset, "pressure")] = ax_pressure
        axes[(dataset, "corr")] = ax_corr

        plot_p99(ax_p99, clean, dataset)
        plot_pressure(ax_pressure, hw, dataset)
        plot_corr(ax_corr, summary, dataset)
        ax_p99.set_ylabel(f"{name}\nRead P99 (ms)")
        ax_pressure.set_ylabel("Normalized\nPressure")
        if r == 0:
            ax_p99.set_title("Read Tail", loc="left", fontsize=8.8, fontweight="normal")
            ax_pressure.set_title("Wait And Hardware Pressure", loc="left", fontsize=8.8, fontweight="normal")
            ax_corr.set_title("Insert-Bin Correlation", loc="left", fontsize=8.8, fontweight="normal")
            ax_p99.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.8)
            ax_pressure.legend(
                loc="upper center",
                bbox_to_anchor=(0.54, 1.03),
                frameon=False,
                ncol=1,
                handlelength=1.7,
                labelspacing=0.25,
            )
        else:
            ax_p99.set_xlabel("Time (s)")
            ax_pressure.set_xlabel("Time (s)")
            ax_corr.set_xlabel("Pearson r")
        if r == 0:
            ax_p99.tick_params(axis="x", labelbottom=False)
            ax_pressure.tick_params(axis="x", labelbottom=False)

    rows = []
    for dataset, _ in DATASETS:
        s = summary[(summary["dataset"] == dataset) & (summary["node_lock"] == "on")].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "lock_corr": s["corr_insert_p99_vs_level0_lock_wait_p99_ms"],
                "rfo_corr": s["corr_insert_p99_vs_l2_rqsts.rfo_miss"],
                "llc_corr": s["corr_insert_p99_vs_LLC-load-misses"],
                "long_load_corr": s["corr_insert_p99_vs_longest_lat_cache.miss"],
                "insert_p99_max_ms": s["insert_op_p99_max_ms"],
                "outside_p99_max_ms": s["outside_op_p99_max_ms"],
            }
        )
    pd.DataFrame(rows).to_csv(SUMMARY, index=False)
    fig.savefig(OUT, dpi=260)
    fig.savefig(OUT_PDF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
