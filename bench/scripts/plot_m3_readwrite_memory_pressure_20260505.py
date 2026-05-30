#!/usr/bin/env python3
"""Build the M3 read-write memory-pressure evidence figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = [
    ("efc400_r1", "real_insert_pressure_efc400_sift_cw40_20260505"),
    ("efc800_r1", "real_insert_pressure_efc800_sift_cw40_20260505"),
    ("efc800_r2", "real_insert_pressure_efc800_sift_cw40_r2_20260505"),
]

CASE_ORDER = ["insert_cpu", "read_traversal_x8", "real_insert"]
CASE_LABEL = {
    "insert_cpu": "CPU work inside insert period",
    "read_traversal_x8": "ANN graph reads inside insert period",
    "real_insert": "Real insert request",
}
PLOT_LABEL = {
    "insert_cpu": "CPU work\ninside insert period",
    "read_traversal_x8": "ANN graph reads\ninside insert period",
    "real_insert": "Real insert\nrequest",
}
CASE_COLOR = {
    "insert_cpu": "#64748b",
    "read_traversal_x8": "#2563eb",
    "real_insert": "#dc2626",
}


def load_rows(base_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id, rel in RUNS:
        table = base_dir / rel / "tables" / "graph_write_probe_matched_delta_vs_background.csv"
        frame = pd.read_csv(table)
        for rec in frame.itertuples(index=False):
            rows.append(
                {
                    "run": run_id,
                    "case": rec.case_key,
                    "case_label": CASE_LABEL.get(rec.case_key, rec.case_label),
                    "delta_search_index_ms_mean": float(rec.delta_search_index_ms_mean),
                    "delta_search_index_ms_p99": float(rec.delta_search_index_ms_p99),
                    "delta_base_us_per_l0_edge": float(rec.delta_base_us_per_l0_edge),
                    "delta_llc_load_misses_per_l0_edge": float(
                        rec.delta_llc_load_misses_per_l0_edge
                    ),
                    "delta_dtlb_load_misses_per_l0_edge": float(
                        rec.delta_dtlb_load_misses_per_l0_edge
                    ),
                    "delta_remote_hitm_per_l0_edge": float(rec.delta_remote_hitm_per_l0_edge),
                    "delta_rfo_miss_per_l0_edge": float(rec.delta_rfo_miss_per_l0_edge),
                }
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "delta_search_index_ms_mean",
        "delta_search_index_ms_p99",
        "delta_base_us_per_l0_edge",
        "delta_llc_load_misses_per_l0_edge",
        "delta_dtlb_load_misses_per_l0_edge",
        "delta_remote_hitm_per_l0_edge",
        "delta_rfo_miss_per_l0_edge",
    ]
    rows: list[dict[str, object]] = []
    for case in CASE_ORDER:
        part = frame[frame["case"] == case]
        row: dict[str, object] = {
            "case": case,
            "case_label": CASE_LABEL[case],
            "replicates": int(len(part)),
        }
        for metric in metrics:
            vals = part[metric].astype(float)
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def scatter_replicates(ax: plt.Axes, frame: pd.DataFrame, metric: str) -> None:
    rng = np.random.default_rng(7)
    for idx, case in enumerate(CASE_ORDER):
        part = frame[frame["case"] == case]
        jitter = rng.uniform(-0.08, 0.08, size=len(part))
        ax.scatter(
            np.full(len(part), idx) + jitter,
            part[metric],
            s=26,
            color=CASE_COLOR[case],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )


def bar_with_points(
    ax: plt.Axes,
    summary_frame: pd.DataFrame,
    frame: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    means = []
    errs = []
    colors = []
    for case in CASE_ORDER:
        row = summary_frame[summary_frame["case"] == case].iloc[0]
        means.append(float(row[f"{metric}_mean"]))
        errs.append(float(row[f"{metric}_std"]))
        colors.append(CASE_COLOR[case])

    x = np.arange(len(CASE_ORDER))
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.bar(x, means, yerr=errs, color=colors, alpha=0.84, capsize=3, width=0.62)
    scatter_replicates(ax, frame, metric)
    ax.set_xticks(x, [PLOT_LABEL[c] for c in CASE_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_plot(frame: pd.DataFrame, summary_frame: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.6,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "M3 pre-experiment: measured-search latency tracks ANN read pressure more than graph writes",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )

    bar_with_points(
        axes[0, 0],
        summary_frame,
        frame,
        "delta_search_index_ms_mean",
        "delta mean search latency (ms)",
        "(a) Search latency vs matched pure-search background",
    )
    bar_with_points(
        axes[0, 1],
        summary_frame,
        frame,
        "delta_base_us_per_l0_edge",
        "delta base time per L0 edge (us)",
        "(b) Unit traversal cost",
    )

    x = np.arange(len(CASE_ORDER))
    width = 0.34
    llc = [
        float(summary_frame[summary_frame["case"] == c]["delta_llc_load_misses_per_l0_edge_mean"].iloc[0])
        for c in CASE_ORDER
    ]
    dtlb = [
        float(summary_frame[summary_frame["case"] == c]["delta_dtlb_load_misses_per_l0_edge_mean"].iloc[0])
        for c in CASE_ORDER
    ]
    ax = axes[1, 0]
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.bar(x - width / 2, llc, width, color="#0f766e", label="LLC load misses")
    ax.bar(x + width / 2, dtlb, width, color="#7c3aed", label="dTLB load misses")
    ax.set_xticks(x, [PLOT_LABEL[c] for c in CASE_ORDER])
    ax.set_ylabel("delta misses per L0 edge")
    ax.set_title("(c) Cache/TLB miss pressure", loc="left", fontweight="bold")
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    hitm = [
        float(summary_frame[summary_frame["case"] == c]["delta_remote_hitm_per_l0_edge_mean"].iloc[0])
        for c in CASE_ORDER
    ]
    rfo = [
        float(summary_frame[summary_frame["case"] == c]["delta_rfo_miss_per_l0_edge_mean"].iloc[0])
        for c in CASE_ORDER
    ]
    ax = axes[1, 1]
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.bar(x - width / 2, hitm, width, color="#f97316", label="remote HITM")
    ax.bar(x + width / 2, rfo, width, color="#92400e", label="RFO misses")
    ax.set_xticks(x, [PLOT_LABEL[c] for c in CASE_ORDER])
    ax.set_ylabel("delta events per L0 edge")
    ax.set_title("(d) Store/coherence counters stay small", loc="left", fontweight="bold")
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = load_rows(args.base_dir)
    summary_frame = summarize(frame)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "matched_delta_runs.csv", index=False)
    summary_frame.to_csv(args.out_dir / "matched_delta_summary.csv", index=False)
    write_plot(frame, summary_frame, args.out_dir / "readwrite_memory_pressure.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
