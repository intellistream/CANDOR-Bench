#!/usr/bin/env python3
"""Plot M3 closed-loop root-cause ablation summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASES = [
    ("full", "full update"),
    ("skip_undo", "skip undo"),
    ("update_readonly", "read-only update"),
    ("skip_update", "skip update"),
]


def load_case(root: Path, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(root / f"closed_loop_odin_{key}_tail_summary.csv")
    timeline = pd.read_csv(root / f"closed_loop_odin_{key}_tail.csv")
    return summary, timeline


def phase_value(summary: pd.DataFrame, phase: str, col: str) -> float:
    row = summary[summary["phase"] == phase]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][col])


def active_mean(timeline: pd.DataFrame, col: str) -> float:
    part = timeline[timeline["in_insert_active_window"] == True]
    if part.empty or col not in part:
        return 0.0
    return float(part[col].mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    rows = []
    for key, label in CASES:
        summary, timeline = load_case(args.root, key)
        outside = phase_value(summary, "outside", "lowq_op_p99_ms")
        active = phase_value(summary, "insert_active", "lowq_op_p99_ms")
        rows.append(
            {
                "case": key,
                "label": label,
                "outside_p99_ms": outside,
                "insert_active_p99_ms": active,
                "delta_p99_ms": active - outside,
                "queue_p95_ms": phase_value(summary, "insert_active", "lowq_queue_p95_ms"),
                "insert_path_workers": phase_value(summary, "insert_active", "insert_path_workers_mean"),
                "existing_update_workers": phase_value(summary, "insert_active", "existing_update_workers_mean"),
                "llc_load_misses_per_s": phase_value(summary, "insert_active", "llc_load_misses_per_s_mean"),
                "load_scan_workers": active_mean(timeline, "existing_neighbor_load_scan_active_workers"),
                "prune_workers": active_mean(timeline, "existing_neighbor_prune_active_workers"),
                "undo_workers": active_mean(timeline, "existing_neighbor_undo_record_active_workers"),
                "rewrite_workers": active_mean(timeline, "existing_neighbor_rewrite_active_workers"),
            }
        )
    df = pd.DataFrame(rows)
    csv_out = args.csv_out or args.out.with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.4), constrained_layout=True)
    fig.patch.set_facecolor("white")

    x = np.arange(len(df))
    width = 0.34
    ax = axes[0]
    ax.bar(x - width / 2, df["outside_p99_ms"], width, color="#94a3b8", label="outside insert")
    ax.bar(x + width / 2, df["insert_active_p99_ms"], width, color="#dc2626", label="insert active")
    for idx, row in df.iterrows():
        ax.text(idx + width / 2, row["insert_active_p99_ms"] + 0.035, f"+{row['delta_p99_ms']:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("(a) measured-search p99: only full mutation creates large insert-active tail", loc="left")
    ax.set_ylabel("p99 op latency (ms)")
    ax.set_xticks(x, df["label"], rotation=0)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    bottom = np.zeros(len(df))
    parts = [
        ("load_scan_workers", "#60a5fa", "load/scan"),
        ("prune_workers", "#2563eb", "prune compute"),
        ("undo_workers", "#f97316", "undo record"),
        ("rewrite_workers", "#dc2626", "rewrite/writeback"),
    ]
    for col, color, label in parts:
        vals = df[col].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, label=label)
        bottom += vals
    ax.set_title("(b) active insert-update phase occupancy during insert windows", loc="left")
    ax.set_ylabel("worker-equiv occupancy")
    ax.set_xticks(x, df["label"])
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.legend(frameon=False, ncol=4)

    ax = axes[2]
    ax.bar(x, df["llc_load_misses_per_s"] / 1e6, color="#111827", alpha=0.82)
    ax.set_title("(c) LLC does not explain the p99 ordering", loc="left")
    ax.set_ylabel("LLC-load-misses/s (M)")
    ax.set_xticks(x, df["label"])
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)

    for ax in axes:
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(args.out)
    print(csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
