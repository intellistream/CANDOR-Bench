#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import h5py
import pandas as pd


def _load_neighbors(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        if "neighbors_continuous" in f:
            data = f["neighbors_continuous"][()]
            dataset_name = "neighbors_continuous"
        elif "neighbors" in f:
            data = f["neighbors"][()]
            dataset_name = "neighbors"
        else:
            raise RuntimeError("No neighbors dataset found in HDF5")
    return data, dataset_name


def _normalize_batches(data: np.ndarray, batch_size: int):
    if data.ndim == 3:
        # (num_batches, batch_size, k)
        num_batches = data.shape[0]
        if data.shape[1] != batch_size:
            raise ValueError(f"batch_size mismatch: data has {data.shape[1]}, expected {batch_size}")
        return data, num_batches, data.shape[2]

    if data.ndim == 2:
        # (num_batches*batch_size, k)
        total_queries, k = data.shape
        if total_queries % batch_size != 0:
            raise ValueError(f"total queries {total_queries} not divisible by batch_size {batch_size}")
        num_batches = total_queries // batch_size
        reshaped = data.reshape(num_batches, batch_size, k)
        return reshaped, num_batches, k

    raise ValueError(f"Unexpected neighbors shape: {data.shape}")


def _compute_per_query_overlap(current_batch: np.ndarray, ref_batch: np.ndarray):
    """Compute overlap by matching the same query index across two batches."""
    if current_batch.shape != ref_batch.shape:
        raise ValueError(
            f"batch shape mismatch: current {current_batch.shape}, ref {ref_batch.shape}"
        )

    batch_size, k = current_batch.shape
    repeated_count = 0
    for qid in range(batch_size):
        repeated_count += np.intersect1d(
            current_batch[qid], ref_batch[qid], assume_unique=False
        ).size

    total_neighbors = batch_size * k
    repeated_percent = (repeated_count / total_neighbors) * 100 if total_neighbors else 0.0
    return repeated_count, total_neighbors, repeated_percent


def _compute_batch_set_overlap(current_batch: np.ndarray, ref_batch: np.ndarray):
    """Compute overlap by treating all neighbors in a batch as one global set."""
    current_set = set(current_batch.reshape(-1).tolist())
    ref_set = set(ref_batch.reshape(-1).tolist())
    repeated_count = len(current_set.intersection(ref_set))
    total_neighbors = current_batch.size
    repeated_percent = (repeated_count / total_neighbors) * 100 if total_neighbors else 0.0
    return repeated_count, total_neighbors, repeated_percent


def main():
    p = argparse.ArgumentParser(description="Compute batch-to-batch neighbor repeat ratio")
    p.add_argument("hdf5", help="Path to .hdf5 file")
    p.add_argument("--batch-size", type=int, default=10000, help="Queries per batch")
    p.add_argument("--output", type=str, default=None, help="Output CSV path")
    p.add_argument("--compare", choices=["prev", "first"], default="prev",
                   help="Compare each batch to previous batch or first batch")
    p.add_argument(
        "--mode",
        choices=["per-query", "batch-set"],
        default="per-query",
        help=(
            "Overlap definition: per-query compares each query's top-k with the same "
            "query index in reference batch; batch-set compares global neighbor sets"
        ),
    )
    args = p.parse_args()

    h5_path = Path(args.hdf5)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    data, dataset_name = _load_neighbors(h5_path)
    batches, num_batches, k = _normalize_batches(data, args.batch_size)

    rows = []
    baseline_batch = None
    prev_batch = None
    for batch_id in range(num_batches):
        batch = batches[batch_id]

        if batch_id == 0:
            repeated_count = 0
            repeated_percent = 0.0
            total_neighbors = batch.size
            baseline_batch = batch
        else:
            if args.compare == "first":
                ref_batch = baseline_batch
            else:
                ref_batch = prev_batch

            if args.mode == "per-query":
                repeated_count, total_neighbors, repeated_percent = _compute_per_query_overlap(
                    batch, ref_batch
                )
            else:
                repeated_count, total_neighbors, repeated_percent = _compute_batch_set_overlap(
                    batch, ref_batch
                )

        rows.append({
            "batch_id": batch_id,
            "repeated_count": repeated_count,
            "total_neighbors": total_neighbors,
            "repeated_percent": repeated_percent,
            "dataset": dataset_name,
            "compare": args.compare,
            "mode": args.mode,
            "k": k,
        })

        prev_batch = batch

    out_path = Path(args.output) if args.output else h5_path.with_name("result.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
