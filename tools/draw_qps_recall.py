#!/usr/bin/env python3
"""Compare recall and query QPS vs batch_id from two results CSVs."""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare recall and query QPS vs batch_id from two results CSVs."
    )
    parser.add_argument("csv_a", help="First CSV with batch_idx, recall, query_qps.")
    parser.add_argument("csv_b", help="Second CSV with batch_idx, recall, query_qps.")
    parser.add_argument("--label-a", default="A", help="Label for first CSV.")
    parser.add_argument("--label-b", default="B", help="Label for second CSV.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: <csv_dir>/qps_recall_compare.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_a = args.csv_a
    csv_b = args.csv_b

    for csv_path in (csv_a, csv_b):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    required = {"batch_idx", "recall", "query_qps"}

    df_a = pd.read_csv(csv_a)
    missing_a = required - set(df_a.columns)
    if missing_a:
        raise ValueError(f"First CSV missing columns: {', '.join(sorted(missing_a))}")
    df_a = df_a.sort_values("batch_idx")

    df_b = pd.read_csv(csv_b)
    missing_b = required - set(df_b.columns)
    if missing_b:
        raise ValueError(f"Second CSV missing columns: {', '.join(sorted(missing_b))}")
    df_b = df_b.sort_values("batch_idx")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(
        df_a["batch_idx"],
        df_a["recall"],
        color="#D62828",
        linewidth=2,
        label=args.label_a,
    )
    axes[0].plot(
        df_b["batch_idx"],
        df_b["recall"],
        color="#0057B7",
        linewidth=2,
        label=args.label_b,
    )
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Recall vs Batch ID")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        df_a["batch_idx"],
        df_a["query_qps"],
        color="#D62828",
        linewidth=2,
        label=args.label_a,
    )
    axes[1].plot(
        df_b["batch_idx"],
        df_b["query_qps"],
        color="#0057B7",
        linewidth=2,
        label=args.label_b,
    )
    axes[1].set_xlabel("Batch ID")
    axes[1].set_ylabel("Query QPS")
    axes[1].set_title("Query QPS vs Batch ID")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if args.out is None:
        out_dir = os.path.dirname(csv_a)
        out_path = os.path.join(out_dir, "qps_recall_compare.png")
    else:
        out_path = args.out

    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
