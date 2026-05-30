#!/usr/bin/env python3
"""Plot full-insert-only closed-loop replicate evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    overlap = pd.read_csv(args.overlap)
    summary["rep_idx"] = np.arange(1, len(summary) + 1)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0, 0]
    ax.plot(summary["rep_idx"], summary["outside_p99_ms"], marker="o", color="#64748b", lw=1.8, label="pure-search periods")
    ax.plot(summary["rep_idx"], summary["insert_p99_ms"], marker="o", color="#dc2626", lw=1.8, label="real-insert periods")
    for row in summary.itertuples(index=False):
        ax.plot([row.rep_idx, row.rep_idx], [row.outside_p99_ms, row.insert_p99_ms], color="#ef4444", alpha=0.35, lw=1.2)
    ax.set_title("(a) real full insert raises measured-search p99 in every replicate", loc="left")
    ax.set_xlabel("replicate")
    ax.set_ylabel("period p99 op latency (ms)")
    ax.set_xticks(summary["rep_idx"])
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ratio = pd.DataFrame(
        {
            "rep_idx": summary["rep_idx"],
            "p99 latency": summary["insert_p99_ms"] / summary["outside_p99_ms"],
            "search dist p95": summary["insert_dist_p95"] / summary["outside_dist_p95"],
            "search hops p95": summary["insert_hops_p95"] / summary["outside_hops_p95"],
        }
    )
    x = np.arange(len(summary))
    width = 0.24
    ax.axhline(1.0, color="#6b7280", lw=1.0)
    ax.bar(x - width, ratio["p99 latency"], width, color="#dc2626", label="p99 latency")
    ax.bar(x, ratio["search dist p95"], width, color="#2563eb", label="search dist p95")
    ax.bar(x + width, ratio["search hops p95"], width, color="#60a5fa", label="search hops p95")
    ax.set_title("(b) latency rises while measured-search traversal work drops", loc="left")
    ax.set_ylabel("insert / outside")
    ax.set_xlabel("replicate")
    ax.set_xticks(x, summary["rep_idx"])
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.bar(summary["rep_idx"] - 0.18, summary["insert_path_workers"], 0.36, color="#9ca3af", label="insert traversal")
    ax.bar(summary["rep_idx"] + 0.18, summary["existing_update_workers"], 0.36, color="#dc2626", label="existing-neighbor update")
    ax.set_title("(c) real insert consistently executes existing-neighbor update", loc="left")
    ax.set_xlabel("replicate")
    ax.set_ylabel("worker-equiv occupancy")
    ax.set_xticks(summary["rep_idx"])
    ax.legend(frameon=False)

    ax = axes[1, 1]
    groups = ["outside_all", "insert_no_write_overlap", "insert_write_overlap"]
    labels = ["outside", "insert no write overlap", "insert write overlap"]
    med = overlap[overlap["group"].isin(groups)].groupby("group")["op_p99_ms"].median().reindex(groups)
    ns = overlap[overlap["group"].isin(groups)].groupby("group")["n"].sum().reindex(groups)
    bars = ax.bar(np.arange(len(groups)), med.to_numpy(), color=["#64748b", "#f59e0b", "#dc2626"], alpha=0.88)
    for idx, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"n={int(ns.iloc[idx])}", ha="center", va="bottom", fontsize=8)
    ax.set_title("(d) natural write-overlap samples carry the insert-period tail", loc="left")
    ax.set_ylabel("median per-rep p99 op latency (ms)")
    ax.set_xticks(np.arange(len(groups)), labels, rotation=12, ha="right")

    for ax in axes.ravel():
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
