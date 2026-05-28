#!/usr/bin/env python3
"""
使用 seaborn 绘制 qps 随 batch_id 变化的折线图（按 batch_id 求均值并显示置信区间）。
默认对比两个算法目录，并在同一张图中绘制两条均值曲线。
搜索目录示例: results/sift/faiss_HNSW/test1_result/.../ef-120_batch_query_qps.csv
输出: qps_batch_compare.png (默认)
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def find_csv_files(base_dir, pattern):
    path = Path(base_dir)
    if not path.exists():
        raise FileNotFoundError(f"base dir not found: {base_dir}")
    # Support comma-separated patterns
    patterns = [p.strip() for p in pattern.split(',') if p.strip()]
    files = []
    for p in patterns:
        glob_pattern = os.path.join(base_dir, p)
        files.extend(glob.glob(glob_pattern))
    files = sorted(set(files))
    return files


def read_qps_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Try to find batch id and qps columns
    col_lower = {c.lower(): c for c in df.columns}
    # find batch id
    if 'batch_id' in col_lower:
        batch_col = col_lower['batch_id']
    elif 'batch' in col_lower:
        batch_col = col_lower['batch']
    else:
        # fallback to first column
        batch_col = df.columns[0]
    # find qps
    if 'qps' in col_lower:
        qps_col = col_lower['qps']
    elif 'throughput' in col_lower:
        qps_col = col_lower['throughput']
    else:
        # try last column
        qps_col = df.columns[-1]
    return df[[batch_col, qps_col]].rename(columns={batch_col: 'batch_id', qps_col: 'qps'})


def load_all_rows(files, algo_label):
    found = 0
    all_rows = []
    for f in files:
        try:
            df = read_qps_csv(f)
        except Exception as e:
            print(f"跳过文件 {f}: 读取失败：{e}", file=sys.stderr)
            continue
        if df.empty:
            print(f"跳过文件 {f}: 内容为空", file=sys.stderr)
            continue
        # ensure numeric
        df = df.dropna(subset=['batch_id', 'qps'])
        try:
            df['batch_id'] = pd.to_numeric(df['batch_id'], errors='coerce')
            df['qps'] = pd.to_numeric(df['qps'], errors='coerce')
        except Exception:
            pass
        df = df.dropna(subset=['batch_id', 'qps'])
        if df.empty:
            print(f"跳过文件 {f}: 无可用数值列", file=sys.stderr)
            continue
        df = df.sort_values('batch_id')
        df['algo'] = algo_label
        df['test_name'] = os.path.basename(os.path.dirname(f))
        all_rows.append(df)
        found += 1
    return found, all_rows


def plot_files(
    algo_to_files,
    out_path,
    show,
    no_trim_labels=None,
    trim_overrides=None,
    smooth_window=0):
    sns.set_theme(style='whitegrid', context='notebook', font_scale=0.9)
    plt.figure(figsize=(10, 6))
    found_total = 0
    all_rows = []
    no_trim_labels = set(no_trim_labels or [])
    trim_overrides = dict(trim_overrides or {})
    for algo_label, files in algo_to_files.items():
        found, rows = load_all_rows(files, algo_label)
        found_total += found
        all_rows.extend(rows)
    if found_total == 0:
        raise RuntimeError('未找到可绘图的 CSV 文件')
    all_df = pd.concat(all_rows, ignore_index=True)
    algo_labels = list(algo_to_files.keys())
    palette = None
    if len(algo_labels) == 2:
        # Darker line colors; CI band will be light via alpha
        palette = {
            algo_labels[0]: '#D62828',
            algo_labels[1]: '#0057B7'
        }
    elif len(algo_labels) == 3:
        palette = {
            algo_labels[0]: '#D62828',
            algo_labels[1]: '#0057B7',
            algo_labels[2]: "#1D6E1D"
        }
    def trimmed_stats(group):
        values = group['qps'].dropna().to_numpy()
        if values.size > 0:
            algo_label = str(group.name[0])
            if algo_label not in no_trim_labels:
                if algo_label in trim_overrides:
                    drop_low, drop_high = trim_overrides[algo_label]
                else:
                    algo_label_lower = algo_label.lower()
                    if 'incremental' in algo_label_lower:
                        drop_low = 2
                        drop_high = 1
                    else:
                        drop_low = 1
                        drop_high = 1
                if values.size > (drop_low + drop_high):
                    values = np.sort(values)[drop_low:values.size - drop_high]
        if values.size == 0:
            return pd.Series({'mean': np.nan, 'ci': np.nan, 'n': 0})
        mean = float(np.mean(values))
        if values.size >= 2:
            std = float(np.std(values, ddof=1))
            ci = 1.96 * (std / np.sqrt(values.size))
        else:
            ci = np.nan
        return pd.Series({'mean': mean, 'ci': ci, 'n': int(values.size)})

    summary = (
        all_df.groupby(['algo', 'batch_id'], as_index=False)
        .apply(trimmed_stats)
        .reset_index(drop=True)
        .sort_values(['algo', 'batch_id'])
    )

    for algo_label in algo_labels:
        algo_df = summary[summary['algo'] == algo_label]
        if algo_df.empty:
            continue
        color = palette.get(algo_label) if palette else None
        plot_df = algo_df.sort_values('batch_id')
        if smooth_window and smooth_window > 1:
            plot_df = plot_df.copy()
            plot_df['mean'] = (
                plot_df['mean']
                .rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
            )
            plot_df['ci'] = (
                plot_df['ci']
                .rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
            )
        plt.plot(plot_df['batch_id'], plot_df['mean'], label=algo_label, color=color, linewidth=2.2)
        if algo_df['ci'].notna().any():
            lower = plot_df['mean'] - plot_df['ci']
            upper = plot_df['mean'] + plot_df['ci']
            plt.fill_between(plot_df['batch_id'], lower, upper, color=color, alpha=0.12)
    plt.xlabel('batch_id (1w->50w expansion, batchsize=2500)', labelpad=10)
    plt.ylabel('qps', labelpad=10)
    plt.title('qps vs batch_id (trimmed mean with 95% CI)')
    plt.grid(True)
    plt.legend(title='algo')
    plt.tight_layout(pad=1.2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    # also save pdf
    try:
        plt.savefig(out_path.with_suffix('.pdf'))
    except Exception:
        pass
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot qps vs batch_id for multiple tests')
    parser.add_argument(
        '--base-dirs',
        default='results/sift/faiss_HNSW, results/sift/faiss_HNSW_Frequent_reoptimization',
        help='comma-separated algorithm directories containing test*_result folders'
    )
    parser.add_argument(
        '--labels',
        default='',
        help='comma-separated labels for each algorithm directory (optional)'
    )
    parser.add_argument('--pattern', default='test[0-9]/ef-120_batch_query_qps.csv, test10/ef-120_batch_query_qps.csv', help='glob pattern(s) relative to base-dir, comma-separated')
    parser.add_argument('--out', default='results/sift/qps_batch_compare.png', help='output image path')
    parser.add_argument(
        '--smooth-window',
        type=int,
        default=3,
        help='rolling mean window size for smoothing (0 or 1 to disable)'
    )
    parser.add_argument('--show', action='store_true', help='show plot interactively')
    parser.add_argument(
        '--extra-dir',
        default='results/sift/faiss_hnsw_incremental',
        help='extra algorithm directory to plot (optional)'
    )
    parser.add_argument(
        '--extra-pattern',
        default='test1[5-9]/ef-120_batch_query_qps.csv, test20/ef-120_batch_query_qps.csv',
        help='glob pattern(s) for extra-dir, comma-separated'
    )
    parser.add_argument(
        '--extra-label',
        default='faiss_hnsw_incremental_avg',
        help='label for extra-dir series'
    )
    args = parser.parse_args()

    base_dirs = [d.strip() for d in args.base_dirs.split(',') if d.strip()]
    labels = [l.strip() for l in args.labels.split(',') if l.strip()]
    if labels and len(labels) != len(base_dirs):
        print('labels 数量需要与 base-dirs 数量一致', file=sys.stderr)
        sys.exit(2)
    if not labels:
        labels = [os.path.basename(d.rstrip('/')) for d in base_dirs]

    algo_to_files = {}
    for base_dir, label in zip(base_dirs, labels):
        files = find_csv_files(base_dir, args.pattern)
        if not files:
            print(f'未找到文件，搜索: {os.path.join(base_dir, args.pattern)}', file=sys.stderr)
            sys.exit(2)
        algo_to_files[label] = files
    no_trim_labels = set()
    trim_overrides = {}
    extra_dir = args.extra_dir.strip()
    if extra_dir:
        extra_files = find_csv_files(extra_dir, args.extra_pattern)
        if not extra_files:
            print(f'未找到文件，搜索: {os.path.join(extra_dir, args.extra_pattern)}', file=sys.stderr)
            sys.exit(2)
        algo_to_files[args.extra_label] = extra_files
        trim_overrides[args.extra_label] = (2, 0)
    try:
        plot_files(
            algo_to_files,
            args.out,
            args.show,
            no_trim_labels=no_trim_labels,
            trim_overrides=trim_overrides,
            smooth_window=args.smooth_window,
        )
    except Exception as e:
        print(f'绘图失败: {e}', file=sys.stderr)
        sys.exit(3)
    print(f'已保存: {args.out}')


if __name__ == '__main__':
    main()
