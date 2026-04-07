#!/usr/bin/env python3
"""Plot repeated_percent vs batch_id (mean with 95% CI) across test runs."""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FAISS_HNSW_COLOR = "#D62828"
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16


def find_csv_files(base_dir: str, pattern: str):
    path = Path(base_dir)
    if not path.exists():
        raise FileNotFoundError(f"base dir not found: {base_dir}")
    patterns = [p.strip() for p in pattern.split(",") if p.strip()]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(base_dir, p)))
    return sorted(set(files))


def read_repeat_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    required_cols = {"batch_id", "repeated_percent"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)} in {csv_path}")
    out = df[["batch_id", "repeated_percent"]].copy()
    out = out.dropna(subset=["batch_id", "repeated_percent"])
    out["batch_id"] = pd.to_numeric(out["batch_id"], errors="coerce")
    out["repeated_percent"] = pd.to_numeric(out["repeated_percent"], errors="coerce")
    out = out.dropna(subset=["batch_id", "repeated_percent"])
    return out


def load_all_rows(files):
    all_rows = []
    found = 0
    for f in files:
        try:
            df = read_repeat_csv(f)
        except Exception as e:
            print(f"skip {f}: {e}", file=sys.stderr)
            continue
        if df.empty:
            print(f"skip {f}: empty data", file=sys.stderr)
            continue
        df = df.sort_values("batch_id")
        df["test_name"] = os.path.basename(os.path.dirname(f))
        all_rows.append(df)
        found += 1
    return found, all_rows


def plot_repeated_percent(files, out_path: str, show: bool, title: str):
    found, rows = load_all_rows(files)
    if found == 0:
        raise RuntimeError("no valid result.csv files found")

    all_df = pd.concat(rows, ignore_index=True)

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(
        data=all_df,
        x="batch_id",
        y="repeated_percent",
        estimator="mean",
        errorbar=("ci", 95),
        color=FAISS_HNSW_COLOR,
        err_kws={"alpha": 0.12},
        linewidth=2.2,
    )
    plt.xlabel("batch_id (1w->10w expansion, batchsize=2500)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("repeated_percent (%)", fontsize=LABEL_FONTSIZE)
    plt.title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    plt.grid(True)
    plt.tight_layout()

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    try:
        plt.savefig(output.with_suffix(".pdf"))
    except Exception:
        pass
    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot repeated_percent vs batch_id (mean with 95% CI)"
    )
    parser.add_argument(
        "--base-dir",
        default="/home/ghr/SAGE-DB-NEW/SAGE-DB-Bench/results/sift/faiss_HNSW",
        help="Algorithm result directory containing test folders",
    )
    parser.add_argument(
        "--pattern",
        default="test[1-9]/result.csv,test10/result.csv",
        help="Glob pattern(s) relative to base-dir, comma-separated",
    )
    parser.add_argument(
        "--out",
        default="results/sift/repeat_topk_compare.png",
        help="Output image path",
    )
    parser.add_argument(
        "--title",
        default="repeated_percent vs batch_id (mean with 95% CI)",
        help="Figure title",
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    files = find_csv_files(args.base_dir, args.pattern)
    if not files:
        print(
            f"no files found under {args.base_dir} with pattern {args.pattern}",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        plot_repeated_percent(files, args.out, args.show, args.title)
    except Exception as e:
        print(f"plot failed: {e}", file=sys.stderr)
        sys.exit(3)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
