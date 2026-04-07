#!/usr/bin/env python3
"""Plot hit-skewness diagnostics from ANN result HDF5.
Generates:
1) rank_frequency.png/pdf: rank-frequency curve of neighbor hit counts
2) lorenz_curve.png/pdf: Lorenz curve of hit concentration (+ Gini)
3) batch_hotspot_coverage.png/pdf: per-batch coverage by global hotspot IDs
4) skewness_dashboard.png/pdf: combined 3-panel figure
"""
# python tools/plot_hit_skewness.py   /home/ghr/SAGE-DB-NEW/SAGE-DB-Bench/results/sift/faiss_HNSW/test1/ef-120.hdf5   --batch-size 10000   --batch-scope all   --out-dir /home/ghr/SAGE-DB-NEW/SAGE-DB-Bench/results/sift/faiss_HNSW/test1/skewness_plots_last_batch

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


FIGSIZE = (8, 6)
DASHBOARD_FIGSIZE = (18, 5)


def load_neighbors(h5_path: Path, dataset: str = "auto") -> Tuple[np.ndarray, str]:
    with h5py.File(h5_path, "r") as f:
        if dataset != "auto":
            if dataset not in f:
                raise KeyError(f"dataset '{dataset}' not found in {h5_path}")
            data = f[dataset][()]
            return data, dataset

        if "neighbors_continuous" in f:
            data = f["neighbors_continuous"][()]
            return data, "neighbors_continuous"
        if "neighbors" in f:
            data = f["neighbors"][()]
            return data, "neighbors"

    raise RuntimeError("No neighbors dataset found (tried neighbors_continuous, neighbors)")


def normalize_batches(data: np.ndarray, batch_size: int) -> np.ndarray:
    if data.ndim == 3:
        # (num_batches, batch_size, k)
        if data.shape[1] != batch_size:
            raise ValueError(f"batch_size mismatch: data has {data.shape[1]}, expected {batch_size}")
        return data

    if data.ndim == 2:
        # (num_batches * batch_size, k)
        total_queries, k = data.shape
        if total_queries % batch_size != 0:
            raise ValueError(f"rows={total_queries} not divisible by batch_size={batch_size}")
        return data.reshape(total_queries // batch_size, batch_size, k)

    raise ValueError(f"Unexpected neighbors shape: {data.shape}")


def frequency_counts(flat_ids: np.ndarray, drop_negative: bool = True) -> np.ndarray:
    if drop_negative:
        flat_ids = flat_ids[flat_ids >= 0]
    if flat_ids.size == 0:
        return np.array([], dtype=np.int64)
    _, counts = np.unique(flat_ids, return_counts=True)
    return counts


def gini_coefficient(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    x = np.sort(values.astype(np.float64))
    n = x.size
    if np.all(x == 0):
        return 0.0
    # G = (2*sum(i*x_i)/(n*sum(x)))-((n+1)/n), i starts from 1
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(i * x) / (n * np.sum(x))) - (n + 1.0) / n)


def parse_top_fracs(s: str) -> list[float]:
    out = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        v = float(item)
        if v <= 0 or v > 1:
            raise ValueError(f"top fraction must be in (0,1], got {v}")
        out.append(v)
    if not out:
        raise ValueError("no valid top fractions")
    return out


def save_fig(fig: plt.Figure, png_path: Path) -> None:
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    try:
        fig.savefig(png_path.with_suffix(".pdf"))
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot hit-skewness diagnostics from HDF5 neighbors")
    ap.add_argument("hdf5", type=Path, help="Path to result .hdf5")
    ap.add_argument(
        "--dataset",
        choices=["auto", "neighbors_continuous", "neighbors"],
        default="auto",
        help="Dataset key in HDF5",
    )
    ap.add_argument("--batch-size", type=int, default=10000, help="Queries per batch")
    ap.add_argument(
        "--batch-scope",
        choices=["all", "last", "index"],
        default="all",
        help="Use all batches, only the last batch, or a specific batch index",
    )
    ap.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Batch index used when --batch-scope index",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <hdf5_dir>/skewness_plots)",
    )
    ap.add_argument(
        "--top-fracs",
        type=str,
        default="0.01,0.05",
        help="Comma-separated hotspot fractions for batch coverage, e.g. '0.01,0.05'",
    )
    ap.add_argument("--show", action="store_true", help="Show figures interactively")
    args = ap.parse_args()

    if not args.hdf5.exists():
        raise FileNotFoundError(args.hdf5)

    # Match the style used in plot_qps_batch.py
    sns.set_theme(style="whitegrid", context="talk")

    data, dataset_name = load_neighbors(args.hdf5, args.dataset)
    batches = normalize_batches(data, args.batch_size)
    num_batches, _, k = batches.shape

    if args.batch_scope == "all":
        selected_batch_ids = np.arange(num_batches)
    elif args.batch_scope == "last":
        selected_batch_ids = np.array([num_batches - 1])
    else:
        if args.batch_index < 0 or args.batch_index >= num_batches:
            raise IndexError(
                f"batch-index out of range: {args.batch_index}, valid=[0,{num_batches - 1}]"
            )
        selected_batch_ids = np.array([args.batch_index])

    selected_batches = batches[selected_batch_ids]

    out_dir = args.out_dir or args.hdf5.parent / "skewness_plots"
    top_fracs = parse_top_fracs(args.top_fracs)

    # Flatten selected hits for concentration analysis
    flat_all = selected_batches.reshape(-1)
    counts = frequency_counts(flat_all, drop_negative=True)
    if counts.size == 0:
        raise RuntimeError("No valid (non-negative) neighbor IDs found")

    counts_desc = np.sort(counts)[::-1]
    rank = np.arange(1, counts_desc.size + 1)

    # Rank-frequency plot
    fig1 = plt.figure(figsize=FIGSIZE)
    ax1 = fig1.add_subplot(111)
    ax1.plot(rank, counts_desc, color="#D62828", linewidth=1.8)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("hit ID rank (log scale)")
    ax1.set_ylabel("hit frequency (log scale)")
    ax1.set_title("Rank-Frequency of Top-k Hit IDs")
    ax1.grid(True)
    save_fig(fig1, out_dir / "rank_frequency.png")

    # Lorenz curve + Gini
    counts_asc = np.sort(counts.astype(np.float64))
    cum_hits = np.cumsum(counts_asc)
    cum_hits = np.insert(cum_hits, 0, 0.0)
    cum_hits_share = cum_hits / cum_hits[-1]
    cum_ids_share = np.linspace(0.0, 1.0, cum_hits_share.size)
    gini = gini_coefficient(counts)

    fig2 = plt.figure(figsize=FIGSIZE)
    ax2 = fig2.add_subplot(111)
    ax2.plot(cum_ids_share, cum_hits_share, color="#D62828", linewidth=2.0, label=f"Lorenz (Gini={gini:.3f})")
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="Uniform")
    ax2.set_xlabel("cumulative share of hit IDs")
    ax2.set_ylabel("cumulative share of top-k hits")
    ax2.set_title("Lorenz Curve of Hit Concentration")
    ax2.legend(frameon=True)
    ax2.grid(True)
    save_fig(fig2, out_dir / "lorenz_curve.png")

    # Batch hotspot coverage
    unique_ids, unique_counts = np.unique(flat_all[flat_all >= 0], return_counts=True)
    order_desc = np.argsort(unique_counts)[::-1]
    sorted_ids = unique_ids[order_desc]

    fig3 = plt.figure(figsize=FIGSIZE)
    ax3 = fig3.add_subplot(111)

    for frac in top_fracs:
        m = max(1, int(np.ceil(frac * sorted_ids.size)))
        hotspot_ids = sorted_ids[:m]
        coverage = []
        x_vals = []
        for b in selected_batch_ids:
            batch_flat = batches[b].reshape(-1)
            valid = batch_flat[batch_flat >= 0]
            if valid.size == 0:
                coverage.append(0.0)
            else:
                c = np.isin(valid, hotspot_ids).sum() / valid.size * 100.0
                coverage.append(float(c))
            x_vals.append(int(b))
        ax3.plot(
            x_vals,
            coverage,
            linewidth=2.0,
            label=f"Top {frac * 100:.1f}% hotspot IDs",
        )

    ax3.set_xlabel("batch_id")
    ax3.set_ylabel("coverage by hotspot IDs (%)")
    ax3.set_title("Per-Batch Coverage by Global Hotspot IDs")
    ax3.grid(True)
    ax3.legend(frameon=True)
    save_fig(fig3, out_dir / "batch_hotspot_coverage.png")

    # Dashboard figure
    fig4, axes = plt.subplots(1, 3, figsize=DASHBOARD_FIGSIZE)

    axes[0].plot(rank, counts_desc, color="#D62828", linewidth=1.6)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("rank (log)")
    axes[0].set_ylabel("frequency (log)")
    axes[0].set_title("Rank-Frequency")
    axes[0].grid(True)

    axes[1].plot(cum_ids_share, cum_hits_share, color="#D62828", linewidth=1.8)
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("cum. ID share")
    axes[1].set_ylabel("cum. hit share")
    axes[1].set_title(f"Lorenz (Gini={gini:.3f})")
    axes[1].grid(True)

    for frac in top_fracs:
        m = max(1, int(np.ceil(frac * sorted_ids.size)))
        hotspot_ids = sorted_ids[:m]
        coverage = []
        x_vals = []
        for b in selected_batch_ids:
            batch_flat = batches[b].reshape(-1)
            valid = batch_flat[batch_flat >= 0]
            c = (np.isin(valid, hotspot_ids).sum() / valid.size * 100.0) if valid.size else 0.0
            coverage.append(float(c))
            x_vals.append(int(b))
        axes[2].plot(x_vals, coverage, linewidth=1.8, label=f"Top {frac * 100:.1f}%")
    axes[2].set_xlabel("batch_id")
    axes[2].set_ylabel("coverage (%)")
    axes[2].set_title("Hotspot Coverage")
    axes[2].legend(frameon=True)
    axes[2].grid(True)

    fig4.suptitle(
        f"Hit-Skewness Diagnostics ({dataset_name}, k={k}, selected={args.batch_scope}, total_batches={num_batches})",
        y=1.02,
    )
    save_fig(fig4, out_dir / "skewness_dashboard.png")

    print(f"Saved plots to: {out_dir}")
    print("- rank_frequency.png/.pdf")
    print("- lorenz_curve.png/.pdf")
    print("- batch_hotspot_coverage.png/.pdf")
    print("- skewness_dashboard.png/.pdf")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
