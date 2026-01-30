#!/usr/bin/env python3
"""Inspect neighbors stored in benchmark HDF5 results.

This repo writes results in HDF5 via `run_benchmark.py`:
- `neighbors`: results from the final `search` op (shape: nq x k)
- `neighbors_continuous`: results from periodic queries during `batch_insert`
  (shape: (num_batches*nq) x k)

This script prints the top-k neighbor IDs for a given query index across all
batches, and optionally compares batches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


@dataclass(frozen=True)
class BatchView:
    batch_idx: int
    neighbors: np.ndarray  # (k,)


def _format_neighbors(arr: np.ndarray, max_items: int = 0) -> str:
    vals = [int(x) for x in arr.tolist()]
    if max_items and len(vals) > max_items:
        vals = vals[:max_items]
        return "[" + ", ".join(map(str, vals)) + f", ...] (len={len(arr)})"
    return "[" + ", ".join(map(str, vals)) + "]"


def _overlap_ratio(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) == 0:
        return 0.0
    return len(set(a) & set(b)) / float(len(a))


def _exact_match(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def _load_nq_and_k(f: h5py.File, dataset_name: str) -> Tuple[int, int]:
    if dataset_name not in f:
        raise KeyError(f"dataset '{dataset_name}' not found in HDF5")
    ds = f[dataset_name]
    if ds.ndim != 2:
        raise ValueError(f"expected 2D dataset '{dataset_name}', got shape={ds.shape}")
    return int(ds.shape[0]), int(ds.shape[1])


def _infer_nq(f: h5py.File, dataset_name: str, nq_arg: Optional[int]) -> int:
    if nq_arg is not None:
        return int(nq_arg)

    # Best effort: infer nq from the final search dataset.
    if "neighbors" in f:
        nq, _ = _load_nq_and_k(f, "neighbors")
        return nq

    # Fallback: cannot infer.
    raise ValueError("cannot infer nq (query count). Please pass --nq")


def iter_query_across_batches(
    f: h5py.File,
    dataset_name: str,
    query_idx: int,
    nq: int,
) -> List[BatchView]:
    if dataset_name not in f:
        raise KeyError(f"dataset '{dataset_name}' not found in HDF5")

    ds = f[dataset_name]
    if ds.ndim != 2:
        raise ValueError(f"expected 2D dataset '{dataset_name}', got shape={ds.shape}")

    total_rows, k = int(ds.shape[0]), int(ds.shape[1])

    if query_idx < 0 or query_idx >= nq:
        raise IndexError(f"query_idx out of range: {query_idx} (nq={nq})")

    if total_rows % nq != 0:
        raise ValueError(
            f"dataset rows not divisible by nq: rows={total_rows}, nq={nq}. "
            "Pass correct --nq or check result file."
        )

    num_batches = total_rows // nq

    out: List[BatchView] = []
    for b in range(num_batches):
        row = b * nq + query_idx
        neigh = np.array(ds[row, :], dtype=np.int64)
        out.append(BatchView(batch_idx=b, neighbors=neigh))

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Print top-k neighbors for a query across batches in HDF5 results")
    ap.add_argument(
        "--hdf5",
        required=True,
        help="Path to result .hdf5 (e.g., results/sift/faiss_HNSW/ef-120/ef-120.hdf5)",
    )
    ap.add_argument(
        "--dataset",
        default="neighbors_continuous",
        choices=["neighbors_continuous", "neighbors"],
        help="Which dataset to read from the HDF5",
    )
    ap.add_argument("--query", type=int, default=0, help="Query index (0-based)")
    ap.add_argument(
        "--nq",
        type=int,
        default=None,
        help="Number of queries per batch. If omitted, inferred from 'neighbors' dataset when available.",
    )
    ap.add_argument(
        "--compare",
        choices=["none", "prev", "last", "first"],
        default="last",
        help="Compare each batch against previous/last/first (prints exact + overlap)",
    )
    ap.add_argument(
        "--with-batch0",
        action="store_true",
        help="Also print overlap with batch0 (repeatedwithbatch0=xx.x%)",
    )
    ap.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Print only first N items of topK (0 = print all)",
    )

    args = ap.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        nq = _infer_nq(f, args.dataset, args.nq)

        if args.dataset == "neighbors":
            # For a single search, treat it as 1 "batch".
            ds = f["neighbors"]
            if ds.ndim != 2:
                raise ValueError(f"expected 2D dataset 'neighbors', got shape={ds.shape}")
            if args.query < 0 or args.query >= ds.shape[0]:
                raise IndexError(f"query out of range: {args.query} (nq={ds.shape[0]})")
            neigh = np.array(ds[args.query, :], dtype=np.int64)
            print(f"dataset=neighbors nq={ds.shape[0]} k={ds.shape[1]}")
            print(f"q{args.query}: {_format_neighbors(neigh, args.max_items)}")
            return 0

        views = iter_query_across_batches(f, args.dataset, args.query, nq)
        k = int(views[0].neighbors.shape[0]) if views else 0
        print(f"dataset={args.dataset} nq={nq} k={k} num_batches={len(views)}")

        if not views:
            return 0

        ref_first = views[0].neighbors
        ref_last = views[-1].neighbors

        exact_count = 0
        for i, v in enumerate(views):
            suffix = ""

            # Optional: always show overlap with batch0
            if args.with_batch0:
                overlap0 = _overlap_ratio(v.neighbors.tolist(), ref_first.tolist())
                suffix += f" | repeatedwithbatch0={overlap0*100:.1f}%"

            if args.compare != "none":
                if args.compare == "prev":
                    ref = views[i - 1].neighbors if i > 0 else v.neighbors
                elif args.compare == "first":
                    ref = ref_first
                elif args.compare == "last":
                    ref = ref_last
                else:
                    ref = v.neighbors

                exact = _exact_match(v.neighbors, ref)
                overlap = _overlap_ratio(v.neighbors.tolist(), ref.tolist())
                if exact:
                    exact_count += 1

                if args.compare == "prev" and i == 0:
                    suffix += " | repeated percent=N/A (no previous batch)"
                else:
                    suffix += (
                        f" | exact_vs_{args.compare}={int(exact)}"
                        f" repeated percent={overlap*100:.1f}%"
                    )

            print(f"batch{v.batch_idx:02d} q{args.query}: {_format_neighbors(v.neighbors, args.max_items)}{suffix}")

        if args.compare != "none":
            print(f"\nsummary: exact_matches_vs_{args.compare}={exact_count}/{len(views)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
