#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: Path):
    data = pd.read_csv(path)
    if "batch_idx" not in data.columns or "query_qps" not in data.columns:
        raise ValueError(f"{path} must contain batch_idx and query_qps columns")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Plot QPS over batch_idx for two CSV files"
    )
    parser.add_argument(
        "--csv-a",
        type=Path,
        required=True,
        help="First CSV (e.g., faiss_hnsw_incremental)"
    )
    parser.add_argument(
        "--csv-b",
        type=Path,
        required=True,
        help="Second CSV (e.g., faiss_HNSW)"
    )
    parser.add_argument(
        "--label-a",
        type=str,
        default="faiss_hnsw_incremental",
        help="Legend label for csv-a"
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default="faiss_HNSW",
        help="Legend label for csv-b"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("qps_over_batches.png"),
        help="Output image path"
    )
    args = parser.parse_args()

    data_a = load_csv(args.csv_a)
    data_b = load_csv(args.csv_b)

    plt.figure(figsize=(10, 5))
    plt.plot(data_a["batch_idx"], data_a["query_qps"], marker="o", label=args.label_a)
    plt.plot(data_b["batch_idx"], data_b["query_qps"], marker="o", label=args.label_b)
    plt.xlabel("batch_idx")
    plt.ylabel("query_qps")
    plt.title("QPS over batch_idx")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
