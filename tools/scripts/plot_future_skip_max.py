#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compact_number(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v.is_integer():
        v = int(v)
    if v % 1_000_000 == 0:
        return f"{int(v / 1_000_000)}m"
    if v % 1_000 == 0:
        return f"{int(v / 1_000)}k"
    return str(v)


def rate_label(insert_rate, search_rate):
    return f"w{compact_number(insert_rate)}-r{compact_number(search_rate)}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot future_skip_hop_max by thread count and r/w rate."
    )
    parser.add_argument("--csv", required=True, help="Input CSV path.")
    parser.add_argument("--out", help="Output PNG path.")
    parser.add_argument(
        "--dataset",
        help="Optional dataset filter (matches dataset_name).",
    )
    return parser.parse_args()


def build_plot(df, out_path):
    datasets = (
        sorted(df["dataset_name"].dropna().unique().tolist())
        if "dataset_name" in df.columns
        else ["unknown"]
    )
    if not datasets:
        datasets = ["unknown"]

    n = len(datasets)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12 * ncols, 5 * nrows),
        squeeze=False,
    )

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        if "dataset_name" in df.columns:
            sub = df[df["dataset_name"] == dataset].copy()
        else:
            sub = df.copy()

        if sub.empty:
            ax.set_title(str(dataset))
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.axis("off")
            continue

        sub["insert_rate_val"] = pd.to_numeric(
            sub["insert_input_rate"], errors="coerce"
        ).fillna(0.0)
        sub["search_rate_val"] = pd.to_numeric(
            sub["search_input_rate"], errors="coerce"
        ).fillna(0.0)
        sub["ratio"] = sub.apply(
            lambda r: (
                float(r["search_rate_val"]) / float(r["insert_rate_val"])
                if float(r["insert_rate_val"]) != 0
                else float("inf")
            ),
            axis=1,
        )
        sub["category"] = sub["ratio"].apply(
            lambda v: "write-heavy" if v < 0.8 else "read-heavy" if v > 1.25 else "balanced"
        )
        prefix_map = {"write-heavy": "W", "balanced": "B", "read-heavy": "R"}
        sub["rate_label"] = sub.apply(
            lambda r: f"{prefix_map[r['category']]} {rate_label(r['insert_input_rate'], r['search_input_rate'])}",
            axis=1,
        )
        sub = sub.sort_values(
            ["insert_rate_val", "search_rate_val", "threads"],
            ascending=[False, False, True],
        )

        # Deduplicate repeated runs with the same rate/thread.
        sub_grouped = (
            sub.groupby(
                ["threads", "rate_label", "insert_rate_val", "search_rate_val"],
                as_index=False,
            )["future_skip_hop_max"]
            .max()
        )

        rate_meta = (
            sub_grouped[["rate_label", "insert_rate_val", "search_rate_val"]]
            .drop_duplicates()
            .sort_values(
                ["insert_rate_val", "search_rate_val"],
                ascending=[False, False],
            )
        )
        rate_order = rate_meta["rate_label"].tolist()
        x_map = {label: i for i, label in enumerate(rate_order)}
        sub_grouped["x"] = sub_grouped["rate_label"].map(x_map)

        thread_list = sorted(sub_grouped["threads"].dropna().unique().tolist())
        color_map = plt.get_cmap("tab10")
        for idx_t, threads in enumerate(thread_list):
            grp = sub_grouped[sub_grouped["threads"] == threads].sort_values("x")
            ax.plot(
                grp["x"].to_numpy(),
                grp["future_skip_hop_max"].to_numpy(),
                marker="o",
                linewidth=2,
                color=color_map(idx_t % 10),
                label=f"t{int(threads)}",
            )

        ax.set_title(str(dataset))
        ax.set_xticks(range(len(rate_order)))
        ax.set_xticklabels(rate_order, rotation=0, ha="center")
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylabel("Max skipped future nodes per query")
        ax.set_xlabel("r/w rate (ordered by r/w ratio, W/B/R)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title="threads", fontsize=9)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"missing csv: {csv_path}")

    df = pd.read_csv(csv_path)
    if "future_skip_hop_max" not in df.columns:
        raise SystemExit("missing future_skip_hop_max column")

    if args.dataset:
        df = df[df.get("dataset_name", "") == args.dataset].copy()
        if df.empty:
            raise SystemExit(f"no rows for dataset {args.dataset}")

    out_path = args.out
    if not out_path:
        out_path = csv_path.with_name(csv_path.stem + "_future_skip_max.png")
    build_plot(df, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
