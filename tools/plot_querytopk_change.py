#!/usr/bin/env python3
"""Plot per-query top-k overlap ratio between two batches from ANN result HDF5.

Example:
python tools/plot_querytopk_change.py \
    results/sift/faiss_hnsw_streamseed/ef-120/ef-120.hdf5 \
    --batch-size 10000 \
    --batch-a 10 \
    --batch-b 11
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


FIGSIZE = (10, 5)
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 20


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


def per_query_set_overlap_ratio(batch_a: np.ndarray, batch_b: np.ndarray) -> np.ndarray:
    """Compute query-wise set-overlap ratio |A∩B|/k between two (num_queries, k) arrays."""
    if batch_a.shape != batch_b.shape:
        raise ValueError(f"Batch shapes mismatch: {batch_a.shape} vs {batch_b.shape}")
    if batch_a.ndim != 2:
        raise ValueError(f"Expected 2D batch arrays, got {batch_a.ndim}D")

    num_queries, k = batch_a.shape
    if k <= 0:
        raise ValueError("k must be > 0")

    ratios = np.zeros(num_queries, dtype=np.float64)
    for i in range(num_queries):
        a = batch_a[i]
        b = batch_b[i]
        a = a[a >= 0]
        b = b[b >= 0]
        if a.size == 0 and b.size == 0:
            ratios[i] = 1.0
            continue
        inter = np.intersect1d(a, b, assume_unique=False).size
        ratios[i] = inter / float(k)
    return ratios


def per_query_positional_overlap_ratio(batch_a: np.ndarray, batch_b: np.ndarray) -> np.ndarray:
    """Compute query-wise positional-overlap ratio: same-position matches / k."""
    if batch_a.shape != batch_b.shape:
        raise ValueError(f"Batch shapes mismatch: {batch_a.shape} vs {batch_b.shape}")
    if batch_a.ndim != 2:
        raise ValueError(f"Expected 2D batch arrays, got {batch_a.ndim}D")

    k = batch_a.shape[1]
    if k <= 0:
        raise ValueError("k must be > 0")
    return (batch_a == batch_b).sum(axis=1).astype(np.float64) / float(k)


def save_fig(fig: plt.Figure, png_path: Path) -> None:
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    try:
        fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot per-query top-k overlap ratio between two batches from HDF5 neighbors"
    )
    ap.add_argument("hdf5", type=Path, help="Path to result .hdf5")
    ap.add_argument(
        "--dataset",
        choices=["auto", "neighbors_continuous", "neighbors"],
        default="auto",
        help="Dataset key in HDF5",
    )
    ap.add_argument("--batch-size", type=int, default=10000, help="Queries per batch")
    ap.add_argument("--batch-a", type=int, default=10, help="First batch index (default: 10)")
    ap.add_argument("--batch-b", type=int, default=11, help="Second batch index (default: 11)")
    ap.add_argument(
        "--overlap-mode",
        choices=["set", "position"],
        default="set",
        help="Overlap definition: set=|A∩B|/k, position=same-position/k",
    )
    ap.add_argument(
        "--sort",
        choices=["none", "desc", "asc"],
        default="none",
        help="How to order queries for plotting: none=original query id order",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <hdf5_dir>/querytopk_change_plots)",
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

    if args.batch_a < 0 or args.batch_a >= num_batches:
        raise IndexError(f"batch-a out of range: {args.batch_a}, valid=[0,{num_batches - 1}]")
    if args.batch_b < 0 or args.batch_b >= num_batches:
        raise IndexError(f"batch-b out of range: {args.batch_b}, valid=[0,{num_batches - 1}]")

    out_dir = args.out_dir or args.hdf5.parent / "querytopk_change_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_a = batches[args.batch_a]
    batch_b = batches[args.batch_b]
    if args.overlap_mode == "set":
        overlap = per_query_set_overlap_ratio(batch_a, batch_b)
        ylabel = "Overlap Percent (%)"
    else:
        overlap = per_query_positional_overlap_ratio(batch_a, batch_b)
        ylabel = "top-k positional overlap (%)"

    exact_match_ratio = float(np.all(batch_a == batch_b, axis=1).mean())

    if args.sort == "desc":
        order = np.argsort(overlap)[::-1]
        y = overlap[order] * 100.0
        x = np.arange(y.size)
        xlabel = "Sorted Query ID (by overlap, desc)"
    elif args.sort == "asc":
        order = np.argsort(overlap)
        y = overlap[order] * 100.0
        x = np.arange(y.size)
        xlabel = "query id (sorted by overlap, asc)"
    else:
        order = np.arange(overlap.size)
        y = overlap * 100.0
        x = order
        xlabel = "query id"

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    ax.plot(x, y, color="#D62828", linewidth=2.0)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    y_upper = min(110.0, max(5.0, float(np.max(y)) * 1.05))
    ax.set_ylim(0.0, y_upper)
    ax.grid(True)

    png_path = out_dir / (
        f"query_overlap_{args.overlap_mode}_{args.sort}_batch{args.batch_a}_vs_batch{args.batch_b}.png"
    )
    save_fig(fig, png_path)

    np.savetxt(
        out_dir
        / f"query_overlap_{args.overlap_mode}_{args.sort}_batch{args.batch_a}_vs_batch{args.batch_b}.csv",
        np.column_stack([x, y, order]),
        delimiter=",",
        header="x,overlap_percent,original_query_id",
        comments="",
    )

    print(f"Saved plot to: {png_path}")
    print(
        f"Overlap stats: mean={overlap.mean():.4f}, median={np.median(overlap):.4f}, "
        f"min={overlap.min():.4f}, max={overlap.max():.4f}"
    )
    print(f"Exact row match ratio: {exact_match_ratio:.4f}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
