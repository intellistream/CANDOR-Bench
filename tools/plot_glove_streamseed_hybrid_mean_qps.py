#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

#python tools/plot_glove_streamseed_hybrid_mean_qps.py   --base-dir results/sift/streamseed_hybrid   --output tools/sift_streamseed_hybrid_mean_qps.png   --trim-k 3

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean query_qps vs batch_idx from test1-10 CSV files."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/glove/streamseed_hybrid"),
        help="Base directory containing test1..test5 subfolders.",
    )
    parser.add_argument(
        "--tests",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=(1, 10),
        help="Inclusive test index range, e.g. 1 10.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="ef-120_batch_query_qps.csv",
        help="CSV filename under each test directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/glove_streamseed_hybrid_mean_qps.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--trim-k",
        type=int,
        default=1,
        help="Drop the lowest and highest K values per batch before averaging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start, end = args.tests
    if start > end:
        raise ValueError("Invalid test range: START must be <= END")
    if args.trim_k < 0:
        raise ValueError("--trim-k must be >= 0")

    frames: list[pd.DataFrame] = []

    for i in range(start, end + 1):
        csv_path = args.base_dir / f"test{i}" / args.csv_name
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        required = {"batch_idx", "query_qps"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        frames.append(df[["batch_idx", "query_qps"]].copy())

    all_data = pd.concat(frames, ignore_index=True)

    def trimmed_mean(series: pd.Series) -> float:
        vals = series.sort_values().to_numpy()
        trim = int(args.trim_k)
        # Drop K values from each side when enough points are available.
        if trim > 0 and vals.size > 2 * trim:
            vals = vals[trim:-trim]
        return float(vals.mean())

    mean_df = (
        all_data.groupby("batch_idx", as_index=False)["query_qps"]
        .apply(trimmed_mean)
        .rename(columns={"query_qps": "trimmed_mean_qps"})
        .sort_values("batch_idx")
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        mean_df["batch_idx"],
        mean_df["trimmed_mean_qps"],
        marker="o",
        linewidth=2,
        markersize=4,
        label=f"Trimmed mean query_qps (trim-k={args.trim_k})",
    )
    plt.title(
        f"Glove streamseed_hybrid: Trimmed Mean Query QPS vs Batch Index (trim-k={args.trim_k})"
    )
    plt.xlabel("Batch Index")
    plt.ylabel("Mean Query QPS")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
