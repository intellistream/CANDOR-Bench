#!/usr/bin/env python3
"""
Plot MVCC experiment results.
Reads benchmark_results.csv from result/mvcc_exp/{dataset}/{qmode}_{lock}/
Generates comparison plots: rwlock vs nolock vs mvcc.

Usage: cd bench && python3 scripts/plot_mvcc_exp.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_DIR = SCRIPT_DIR.parent
RESULT_BASE = BENCH_DIR / "result" / "mvcc_exp"
PLOT_DIR = os.path.join(RESULT_BASE, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DATASETS = ["sift", "gist", "deep1M", "msong"]
QUERY_MODES = ["chasing", "round_robin"]
LOCK_MODES = ["rwlock", "nolock", "mvcc"]
LOCK_COLORS = {"rwlock": "#e74c3c", "nolock": "#3498db", "mvcc": "#2ecc71"}
LOCK_LABELS = {"rwlock": "RWLock", "nolock": "NoLock", "mvcc": "MVCC"}
# Only plot these rate groups (insert_rate, search_rate)
RATE_FILTER = {(40000, 40000), (10000, 90000), (90000, 10000)}


def load_all():
    """Load all benchmark_results.csv into a single DataFrame with lock_mode column."""
    frames = []
    for ds in DATASETS:
        for qm in QUERY_MODES:
            for lock in LOCK_MODES:
                csv_path = os.path.join(RESULT_BASE, ds, f"{qm}_{lock}", "benchmark_results.csv")
                if not os.path.exists(csv_path):
                    continue
                df = pd.read_csv(csv_path)
                df["lock_mode"] = lock
                df["query_mode"] = qm
                frames.append(df)
    if not frames:
        print(f"No results found under {RESULT_BASE}. Has the experiment finished?")
        return None
    combined = pd.concat(frames, ignore_index=True)
    # Filter to selected rate groups only
    if RATE_FILTER:
        combined["insert_input_rate"] = pd.to_numeric(combined["insert_input_rate"], errors="coerce")
        combined["search_input_rate"] = pd.to_numeric(combined["search_input_rate"], errors="coerce")
        mask = combined.apply(
            lambda r: (int(r["insert_input_rate"]), int(r["search_input_rate"])) in RATE_FILTER, axis=1)
        combined = combined[mask]
    return combined


def _rate_label(row):
    ir = int(row["insert_input_rate"])
    sr = int(row["search_input_rate"])
    return f"w{ir//1000}k/r{sr//1000}k"


def plot_metric_vs_threads(df, dataset, qmode, metric_col, ylabel, title_suffix, filename_suffix):
    """Bar chart: metric vs threads, one subplot per (batch, rate), grouped by lock mode."""
    sub = df[(df["dataset_name"] == dataset) & (df["query_mode"] == qmode)].copy()
    if sub.empty:
        return

    sub["rate_label"] = sub.apply(_rate_label, axis=1)
    batch_sizes = sorted(sub["write_batch_size"].unique())
    rate_labels = sorted(sub["rate_label"].unique())
    threads_list = sorted(sub["threads"].unique())

    ncols = len(rate_labels)
    nrows = len(batch_sizes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ri, bs in enumerate(batch_sizes):
        for ci, rl in enumerate(rate_labels):
            ax = axes[ri][ci]
            sub_br = sub[(sub["write_batch_size"] == bs) & (sub["rate_label"] == rl)]
            width = 0.25
            x_pos = range(len(threads_list))

            for i, lock in enumerate(LOCK_MODES):
                sub_lock = sub_br[sub_br["lock_mode"] == lock]
                vals = []
                for t in threads_list:
                    row = sub_lock[sub_lock["threads"] == t]
                    vals.append(row[metric_col].mean() if not row.empty else 0)
                offset = (i - 1) * width
                ax.bar([x + offset for x in x_pos], vals, width,
                       label=LOCK_LABELS[lock], color=LOCK_COLORS[lock], alpha=0.85)

            ax.set_xlabel("Threads")
            if ci == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f"batch={bs}, {rl}")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(threads_list)
            ax.legend(fontsize=7)

    fig.suptitle(f"{dataset} / {qmode} — {title_suffix}", fontsize=13)
    fig.tight_layout()
    fname = os.path.join(PLOT_DIR, f"{dataset}_{qmode}_{filename_suffix}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_metric_vs_batch(df, dataset, qmode, metric_col, ylabel, title_suffix, filename_suffix):
    """Bar chart: metric vs batch_size, one subplot per (threads, rate), grouped by lock mode."""
    sub = df[(df["dataset_name"] == dataset) & (df["query_mode"] == qmode)].copy()
    if sub.empty:
        return

    sub["rate_label"] = sub.apply(_rate_label, axis=1)
    threads_list = sorted(sub["threads"].unique())
    rate_labels = sorted(sub["rate_label"].unique())
    batch_sizes = sorted(sub["write_batch_size"].unique())

    ncols = len(rate_labels)
    nrows = len(threads_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ri, nt in enumerate(threads_list):
        for ci, rl in enumerate(rate_labels):
            ax = axes[ri][ci]
            sub_tr = sub[(sub["threads"] == nt) & (sub["rate_label"] == rl)]
            width = 0.25
            x_pos = range(len(batch_sizes))

            for i, lock in enumerate(LOCK_MODES):
                sub_lock = sub_tr[sub_tr["lock_mode"] == lock]
                vals = []
                for bs in batch_sizes:
                    row = sub_lock[sub_lock["write_batch_size"] == bs]
                    vals.append(row[metric_col].mean() if not row.empty else 0)
                offset = (i - 1) * width
                ax.bar([x + offset for x in x_pos], vals, width,
                       label=LOCK_LABELS[lock], color=LOCK_COLORS[lock], alpha=0.85)

            ax.set_xlabel("Batch Size")
            if ci == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f"threads={nt}, {rl}")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(batch_sizes)
            ax.legend(fontsize=7)

    fig.suptitle(f"{dataset} / {qmode} — {title_suffix}", fontsize=13)
    fig.tight_layout()
    fname = os.path.join(PLOT_DIR, f"{dataset}_{qmode}_{filename_suffix}_by_batch.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def main():
    df = load_all()
    if df is None:
        return

    # Ensure numeric
    for col in ["threads", "write_batch_size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    metrics = [
        ("search_qps (per-point)",                "Search QPS",           "Search QPS",           "search_qps"),
        ("search_p99_op_latency_ms (per-batch)",   "Search p99 (ms)",      "Search p99 Latency",   "search_p99"),
        ("search_mean_op_latency_ms (per-batch)",  "Search mean (ms)",     "Search Mean Latency",  "search_mean"),
        ("insert_qps (per-point)",                 "Insert QPS",           "Insert QPS",           "insert_qps"),
        ("insert_p99_op_latency_ms (per-batch)",   "Insert p99 (ms)",      "Insert p99 Latency",   "insert_p99"),
        ("recall",                                 "Recall@10",            "Recall",               "recall"),
        ("freshness_total_pts_p99",                "Freshness p99 (pts)",  "Freshness p99",        "freshness_p99"),
    ]

    for ds in DATASETS:
        for qm in QUERY_MODES:
            print(f"\n=== {ds} / {qm} ===")
            for metric_col, ylabel, title_suffix, fname_suffix in metrics:
                if metric_col not in df.columns:
                    continue
                df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
                plot_metric_vs_threads(df, ds, qm, metric_col, ylabel, title_suffix, fname_suffix)
                plot_metric_vs_batch(df, ds, qm, metric_col, ylabel, title_suffix, fname_suffix)

    # ── Summary table ──
    print("\n=== Summary (mean across rate groups) ===")
    summary_cols = ["dataset_name", "query_mode", "lock_mode", "threads", "write_batch_size",
                    "search_qps (per-point)", "search_p99_op_latency_ms (per-batch)",
                    "insert_qps (per-point)", "recall"]
    available = [c for c in summary_cols if c in df.columns]
    summary = df[available].groupby(
        ["dataset_name", "query_mode", "lock_mode", "threads", "write_batch_size"]
    ).mean(numeric_only=True).round(2)
    summary_path = os.path.join(PLOT_DIR, "summary.csv")
    summary.to_csv(summary_path)
    print(f"Summary table: {summary_path}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
