#!/usr/bin/env python3
"""
Plot cache_misses vs query_idx with mean and 95% CI.
Reads ef-120_query_cache_miss.csv from multiple experiment folders.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def find_csv_files(base_dir, pattern):
    path = Path(base_dir)
    if not path.exists():
        raise FileNotFoundError(f"base dir not found: {base_dir}")
    patterns = [p.strip() for p in pattern.split(',') if p.strip()]
    files = []
    for p in patterns:
        glob_pattern = os.path.join(base_dir, p)
        files.extend(glob.glob(glob_pattern))
    return sorted(set(files))


def read_cache_csv(csv_path):
    df = pd.read_csv(csv_path)
    col_lower = {c.lower(): c for c in df.columns}
    if 'query_idx' in col_lower:
        query_col = col_lower['query_idx']
    elif 'query' in col_lower:
        query_col = col_lower['query']
    else:
        query_col = df.columns[0]
    if 'cache_misses' in col_lower:
        miss_col = col_lower['cache_misses']
    elif 'cache_miss' in col_lower:
        miss_col = col_lower['cache_miss']
    else:
        miss_col = df.columns[-1]
    return df[[query_col, miss_col]].rename(columns={query_col: 'query_idx', miss_col: 'cache_misses'})


def load_all_rows(files, algo_label):
    found = 0
    all_rows = []
    for f in files:
        try:
            df = read_cache_csv(f)
        except Exception as e:
            print(f"skip {f}: read failed: {e}", file=sys.stderr)
            continue
        if df.empty:
            print(f"skip {f}: empty", file=sys.stderr)
            continue
        df = df.dropna(subset=['query_idx', 'cache_misses'])
        try:
            df['query_idx'] = pd.to_numeric(df['query_idx'], errors='coerce')
            df['cache_misses'] = pd.to_numeric(df['cache_misses'], errors='coerce')
        except Exception:
            pass
        df = df.dropna(subset=['query_idx', 'cache_misses'])
        if df.empty:
            print(f"skip {f}: no numeric data", file=sys.stderr)
            continue
        df = df.sort_values('query_idx')
        df['algo'] = algo_label
        df['test_name'] = os.path.basename(os.path.dirname(f))
        all_rows.append(df)
        found += 1
    return found, all_rows


def plot_files(algo_to_files, out_path, show):
    sns.set_theme(style='whitegrid', context='talk')
    plt.figure(figsize=(10, 6))
    found_total = 0
    all_rows = []
    for algo_label, files in algo_to_files.items():
        found, rows = load_all_rows(files, algo_label)
        found_total += found
        all_rows.extend(rows)
    if found_total == 0:
        raise RuntimeError('no CSV files found')
    all_df = pd.concat(all_rows, ignore_index=True)
    algo_labels = list(algo_to_files.keys())
    palette = None
    if len(algo_labels) == 2:
        palette = {
            algo_labels[0]: '#D62828',
            algo_labels[1]: '#0057B7'
        }
    sns.lineplot(
        data=all_df,
        x='query_idx',
        y='cache_misses',
        hue='algo',
        estimator='mean',
        errorbar=('ci', 95),
        palette=palette,
        err_kws={'alpha': 0.12},
        linewidth=2.2
    )
    plt.xlabel('query_idx')
    plt.ylabel('cache_misses')
    plt.title('cache_misses vs query_idx (mean with 95% CI)')
    plt.grid(True)
    plt.legend(title='algo')
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    try:
        plt.savefig(out_path.with_suffix('.pdf'))
    except Exception:
        pass
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot cache_misses vs query_idx')
    parser.add_argument(
        '--base-dirs',
        default='/home/ghr/candor-bench/SAGE-DB-Bench/results/sift/faiss_HNSW, /home/ghr/candor-bench/SAGE-DB-Bench/results/sift/faiss_hnsw_incremental',
        help='comma-separated algorithm directories containing test*_result folders'
    )
    parser.add_argument(
        '--labels',
        default='',
        help='comma-separated labels for each algorithm directory (optional)'
    )
    parser.add_argument(
        '--pattern',
        default='ef-120/ef-120_query_cache_miss.csv',
        help='glob pattern(s) relative to base-dir, comma-separated'
    )
    parser.add_argument(
        '--out',
        default='results/sift/cache_miss_compare.png',
        help='output image path'
    )
    parser.add_argument('--show', action='store_true', help='show plot interactively')
    args = parser.parse_args()

    base_dirs = [d.strip() for d in args.base_dirs.split(',') if d.strip()]
    labels = [l.strip() for l in args.labels.split(',') if l.strip()]
    if labels and len(labels) != len(base_dirs):
        print('labels count must match base-dirs', file=sys.stderr)
        sys.exit(2)
    if not labels:
        labels = [os.path.basename(d.rstrip('/')) for d in base_dirs]

    algo_to_files = {}
    for base_dir, label in zip(base_dirs, labels):
        files = find_csv_files(base_dir, args.pattern)
        if not files:
            print(f'no files found: {os.path.join(base_dir, args.pattern)}', file=sys.stderr)
            sys.exit(2)
        algo_to_files[label] = files
    try:
        plot_files(algo_to_files, args.out, args.show)
    except Exception as e:
        print(f'plot failed: {e}', file=sys.stderr)
        sys.exit(3)
    print(f'saved: {args.out}')


if __name__ == '__main__':
    main()
