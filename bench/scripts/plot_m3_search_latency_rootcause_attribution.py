#!/usr/bin/env python3
"""Plot the M3 search-latency root-cause attribution figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def metric_pair(frame: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    row = frame[frame["metric"] == metric]
    if row.empty:
        raise KeyError(metric)
    rec = row.iloc[0]
    return float(rec["outside"]), float(rec["insert"]), float(rec["insert_over_outside"])


def row_for(frame: pd.DataFrame, case: str, scope: str) -> pd.Series:
    rows = frame[(frame["case"] == case) & (frame["scope"] == scope)]
    if rows.empty:
        raise KeyError(f"{case}/{scope}")
    return rows.iloc[0]


def enrich_row(frame: pd.DataFrame, case: str, group: str) -> pd.Series:
    rows = frame[(frame["case"] == case) & (frame["group"] == group)]
    if rows.empty:
        raise KeyError(f"{case}/{group}")
    return rows.iloc[0]


def write_summary(
    out: Path,
    front_summary: pd.DataFrame,
    tail_enrichment: pd.DataFrame,
    matched: pd.DataFrame,
) -> None:
    rows: list[dict[str, object]] = []

    for metric in [
        "amortized_op_p99_ms",
        "queue_p95_ms",
        "dist_p95_per_query",
        "hops_p95_per_query",
    ]:
        outside, insert, ratio = metric_pair(front_summary, metric)
        rows.append(
            {
                "panel": "front_exclusion",
                "metric": metric,
                "case": "full_insert",
                "outside": outside,
                "insert": insert,
                "ratio": ratio,
            }
        )

    for case in ["insert_efc200", "insert_efc35"]:
        rec = matched[matched["insert_case"] == case].iloc[0]
        for metric in ["op_p99", "op_p999"]:
            rows.append(
                {
                    "panel": "matched_search_control",
                    "metric": metric,
                    "case": case,
                    "outside": float(rec[f"bg_{metric}_lowq_ms"]),
                    "insert": float(rec[f"insert_{metric}_lowq_ms"]),
                    "ratio": float(rec[f"insert_{metric}_lowq_ms"]) / float(rec[f"bg_{metric}_lowq_ms"]),
                }
            )

    for group in ["tail_top_5pct", "rest"]:
        rec = enrich_row(tail_enrichment, "efc35_full", group)
        for metric in [
            "op_p95",
            "existing_update_mean",
            "prune_mean",
            "undo_mean",
            "rewrite_mean",
            "dist_mean",
            "hops_mean",
        ]:
            rows.append(
                {
                    "panel": "tail_enrichment",
                    "metric": metric,
                    "case": group,
                    "outside": np.nan,
                    "insert": float(rec[metric]),
                    "ratio": np.nan,
                }
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def annotate_bar(ax: plt.Axes, bars, fmt: str = "{:.2f}", dy: float = 0.03) -> None:
    ymax = ax.get_ylim()[1]
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + ymax * dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    tables = args.artifact_dir / "tables"
    front_summary = pd.read_csv(tables / "odin_preexperiment_full_insert_fair_summary.csv")
    tail_enrichment = pd.read_csv(tables / "anns_controlled_residual_tail_enrichment.csv")
    matched = pd.read_csv(tables / "occupancy_matched_pairs.csv")

    write_summary(args.csv_out, front_summary, tail_enrichment, matched)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "M3 root cause: search tail follows existing-neighbor graph mutation",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )

    ax = axes[0, 0]
    front_metrics = [
        ("p99 latency", "amortized_op_p99_ms", "#dc2626"),
        ("queue p95", "queue_p95_ms", "#64748b"),
        ("dist p95", "dist_p95_per_query", "#2563eb"),
        ("hops p95", "hops_p95_per_query", "#60a5fa"),
    ]
    labels = [label for label, _, _ in front_metrics]
    ratios = [metric_pair(front_summary, metric)[2] for _, metric, _ in front_metrics]
    colors = [color for _, _, color in front_metrics]
    bars = ax.bar(np.arange(len(labels)), ratios, color=colors)
    ax.axhline(1.0, color="#111827", lw=0.8)
    ax.set_title("(a) insert raises latency without queue or more traversal", loc="left")
    ax.set_ylabel("insert / outside")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0, max(ratios) * 1.28)
    annotate_bar(ax, bars, "{:.2f}x", dy=0.015)

    ax = axes[0, 1]
    matched_cases = [
        ("insert_efc200", "default insert"),
        ("insert_efc35", "same-ef insert"),
    ]
    x = np.arange(len(matched_cases))
    width = 0.34
    bg_p99 = []
    insert_p99 = []
    bg_p999 = []
    insert_p999 = []
    worker_labels = []
    traversal_labels = []
    for case, _ in matched_cases:
        rec = matched[matched["insert_case"] == case].iloc[0]
        bg_p99.append(float(rec["bg_op_p99_lowq_ms"]))
        insert_p99.append(float(rec["insert_op_p99_lowq_ms"]))
        bg_p999.append(float(rec["bg_op_p999_lowq_ms"]))
        insert_p999.append(float(rec["insert_op_p999_lowq_ms"]))
        worker_labels.append(f"{rec['target_insert_total_workers']:.1f}w vs {rec['matched_bg_workers']:.1f}w")
        traversal_labels.append(f"dist {rec['insert_dist_p95_lowq']:.0f} vs {rec['bg_dist_p95_lowq']:.0f}")
    ax.bar(x - width / 2, bg_p99, width, color="#64748b")
    bars = ax.bar(x + width / 2, insert_p99, width, color="#dc2626", label="insert p99")
    ax.scatter(x - width / 2, bg_p999, marker="D", s=22, color="#334155")
    ax.scatter(x + width / 2, insert_p999, marker="D", s=22, color="#7f1d1d")
    ax.set_title("(b) search also replaces cache: compare real insert against matched bg-search", loc="left")
    ax.set_ylabel("low-queue op latency (ms)")
    ax.set_xticks(x, [label for _, label in matched_cases])
    ax.set_ylim(0, max(insert_p999 + bg_p999) * 1.28)
    annotate_bar(ax, bars, "{:.2f}", dy=0.014)
    ax.text(
        0.02,
        0.98,
        "bars=p99, diamonds=p99.9; gray=matched bg-search, red=insert",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="#111827",
    )
    for idx, (workers, traversal) in enumerate(zip(worker_labels, traversal_labels, strict=False)):
        ax.text(idx, ax.get_ylim()[1] * 0.73, workers, ha="center", va="top", fontsize=7, color="#111827")
        ax.text(idx, ax.get_ylim()[1] * 0.64, traversal, ha="center", va="top", fontsize=7, color="#334155")

    ax = axes[1, 0]
    groups = ["rest", "tail_top_5pct"]
    group_labels = ["rest", "slowest 5%"]
    bottom = np.zeros(len(groups))
    phase_parts = [
        ("existing_update_mean", "#ef4444", "existing update"),
        ("prune_mean", "#f97316", "prune"),
        ("undo_mean", "#a855f7", "undo"),
        ("rewrite_mean", "#7f1d1d", "rewrite"),
    ]
    for col, color, label in phase_parts:
        vals = np.array([float(enrich_row(tail_enrichment, "efc35_full", group)[col]) for group in groups])
        ax.bar(np.arange(len(groups)), vals, bottom=bottom, color=color, label=label)
        bottom += vals
    tail = enrich_row(tail_enrichment, "efc35_full", "tail_top_5pct")
    rest = enrich_row(tail_enrichment, "efc35_full", "rest")
    ax.set_title(
        f"(c) slow queries overlap mutation; dist {tail['dist_mean']:.1f} vs {rest['dist_mean']:.1f}, hops {tail['hops_mean']:.2f} vs {rest['hops_mean']:.2f}",
        loc="left",
    )
    ax.set_ylabel("mean overlap samples")
    ax.set_xticks(np.arange(len(groups)), group_labels)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    ax = axes[1, 1]
    tail = enrich_row(tail_enrichment, "efc35_full", "tail_top_5pct")
    rest = enrich_row(tail_enrichment, "efc35_full", "rest")
    ratio_metrics = [
        ("op p95", float(tail["op_p95"]) / float(rest["op_p95"]), "#dc2626"),
        ("existing\nupdate", float(tail["existing_update_mean"]) / float(rest["existing_update_mean"]), "#ef4444"),
        ("prune", float(tail["prune_mean"]) / float(rest["prune_mean"]), "#f97316"),
        ("undo", float(tail["undo_mean"]) / float(rest["undo_mean"]), "#a855f7"),
        ("dist", float(tail["dist_mean"]) / float(rest["dist_mean"]), "#2563eb"),
        ("hops", float(tail["hops_mean"]) / float(rest["hops_mean"]), "#60a5fa"),
    ]
    labels = [item[0] for item in ratio_metrics]
    vals = [item[1] for item in ratio_metrics]
    colors = [item[2] for item in ratio_metrics]
    bars = ax.bar(np.arange(len(labels)), vals, color=colors)
    ax.axhline(1.0, color="#111827", lw=0.8)
    ax.set_title("(d) slow-query enrichment is mutation-heavy, not traversal-heavy", loc="left")
    ax.set_ylabel("slowest 5% / rest")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0, max(vals) * 1.25)
    annotate_bar(ax, bars, "{:.1f}x", dy=0.012)

    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e7eb", lw=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=260, bbox_inches="tight")
    print(args.out)
    print(args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
