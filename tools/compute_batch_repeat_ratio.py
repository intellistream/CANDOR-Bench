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


def main():
    p = argparse.ArgumentParser(description="Compute batch-to-batch neighbor repeat ratio")
    p.add_argument("hdf5", help="Path to .hdf5 file")
    p.add_argument("--batch-size", type=int, default=10000, help="Queries per batch")
    p.add_argument("--output", type=str, default=None, help="Output CSV path")
    p.add_argument("--compare", choices=["prev", "first"], default="prev",
                   help="Compare each batch to previous batch or first batch")
    args = p.parse_args()

    h5_path = Path(args.hdf5)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    data, dataset_name = _load_neighbors(h5_path)
    batches, num_batches, k = _normalize_batches(data, args.batch_size)

    rows = []
    baseline_set = None
    for batch_id in range(num_batches):
        batch = batches[batch_id]
        flat = batch.reshape(-1)
        total_neighbors = flat.size
        current_set = set(flat.tolist())

        if batch_id == 0:
            repeated_count = 0
            repeated_percent = 0.0
            baseline_set = current_set
        else:
            if args.compare == "first":
                ref_set = baseline_set
            else:
                ref_set = prev_set
            repeated_count = len(current_set.intersection(ref_set))
            repeated_percent = (repeated_count / total_neighbors) * 100 if total_neighbors else 0.0

        rows.append({
            "batch_id": batch_id,
            "repeated_count": repeated_count,
            "total_neighbors": total_neighbors,
            "repeated_percent": repeated_percent,
            "dataset": dataset_name,
            "compare": args.compare
        })

        prev_set = current_set

    out_path = Path(args.output) if args.output else h5_path.with_name("result.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(out_path)


if __name__ == "__main__":
    main()
