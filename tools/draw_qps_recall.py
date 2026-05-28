#!/usr/bin/env python3
"""Compare recall and query QPS vs batch_id from two results CSVs."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#python tools/draw_qps_recall.py --label-a streamseed --label-b streamseed_off  results/sift/faiss_hnsw_streamseed/ef-120/ef-120_final_results.csv  results/sift/faiss_hnsw_streamseed_off/ef-120/ef-120_final_results.csv

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

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    axes[0].plot(
        df_a["batch_idx"],
        df_a["query_qps"],
        color="#D62828",
        linewidth=1.6,
        label=args.label_a,
    )
    axes[0].plot(
        df_b["batch_idx"],
        df_b["query_qps"],
        color="#0057B7",
        linewidth=1.6,
        label=args.label_b,
    )
    axes[0].set_xlabel("Batch ID",  fontsize=14)
    axes[0].set_ylabel("Query QPS", fontsize=14)
    axes[0].set_title("Query QPS vs Batch ID", fontsize=16)
    axes[0].tick_params(axis="both", labelsize=13)
    axes[0].tick_params(axis="x", labelbottom=True)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        df_a["batch_idx"],
        df_a["recall"],
        color="#D62828",
        linewidth=1.4,
        label=args.label_a,
    )
    axes[1].plot(
        df_b["batch_idx"],
        df_b["recall"],
        color="#0057B7",
        linewidth=1.4,
        label=args.label_b,
    )
    axes[1].set_xlabel("Batch ID", fontsize=14)
    axes[1].set_ylabel("Recall", fontsize=14)
    axes[1].set_title("Recall vs Batch ID", fontsize=16)
    axes[1].tick_params(axis="both", labelsize=13)
    axes[1].legend(loc="upper right",fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Inset zoom: highlight recall regions where the two curves differ noticeably.
    recall_cmp = pd.merge(
        df_a[["batch_idx", "recall"]],
        df_b[["batch_idx", "recall"]],
        on="batch_idx",
        suffixes=("_a", "_b"),
    ).sort_values("batch_idx")
    recall_cmp["abs_delta"] = (recall_cmp["recall_a"] - recall_cmp["recall_b"]).abs()

    inset_df = pd.DataFrame()
    if not recall_cmp.empty:
        # Focus inset on one local region centered at the maximum delta point.
        max_idx = int(recall_cmp["abs_delta"].values.argmax())
        left = max(0, max_idx - 1)
        right = min(len(recall_cmp), max_idx + 2)
        inset_df = recall_cmp.iloc[left:right]

    if not inset_df.empty:
        x_min = float(inset_df["batch_idx"].min())
        x_max = float(inset_df["batch_idx"].max())
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        else:
            x_pad = max(0.1, 0.02 * (x_max - x_min))
            x_min -= x_pad
            x_max += x_pad

        y_vals = np.concatenate([inset_df["recall_a"].values, inset_df["recall_b"].values])
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))
        if y_min == y_max:
            y_min -= 0.005
            y_max += 0.005
        else:
            y_pad = max(0.0008, 0.08 * (y_max - y_min))
            y_min -= y_pad
            y_max += y_pad

        inset_ax = axes[1].inset_axes([0.72, 0.12, 0.20, 0.20])
        inset_ax.plot(
            df_a["batch_idx"],
            df_a["recall"],
            color="#D62828",
            linewidth=0.6,
        )
        inset_ax.plot(
            df_b["batch_idx"],
            df_b["recall"],
            color="#0057B7",
            linewidth=0.6,
        )
        inset_ax.set_xlim(x_min, x_max)
        inset_ax.set_ylim(y_min, y_max)
        inset_ax.grid(True, alpha=0.25)
        inset_ax.tick_params(axis="both", labelsize=9)
        axes[1].indicate_inset_zoom(inset_ax, edgecolor="0.35", alpha=0.8)

    plt.tight_layout()

    if args.out is None:
        out_dir = os.path.dirname(csv_a)
        out_path = os.path.join(out_dir, "qps_recall_compare.png")
    else:
        out_path = args.out

    fig.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
