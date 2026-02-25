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
    # Mean across tests with 95% CI band
    sns.lineplot(
        data=all_df,
        x='batch_id',
        y='qps',
        hue='algo',
        estimator='mean',
        errorbar=('ci', 95),
        palette=palette,
        err_kws={'alpha': 0.12},
        linewidth=2.2
    )
    plt.xlabel('batch_id (1w->50w expansion, batchsize=2500)')
    plt.ylabel('qps')
    plt.title('qps vs batch_id (mean with 95% CI)')
    plt.grid(True)
    plt.legend(title='algo')
    plt.tight_layout()
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
        default='results/sift/faiss_HNSW1, results/sift/faiss_hnsw_incremental1',
        help='comma-separated algorithm directories containing test*_result folders'
    )
    parser.add_argument(
        '--labels',
        default='',
        help='comma-separated labels for each algorithm directory (optional)'
    )
    parser.add_argument('--pattern', default='test[0-9]/ef-120_batch_query_qps.csv, test10/ef-120_batch_query_qps.csv', help='glob pattern(s) relative to base-dir, comma-separated')
    parser.add_argument('--out', default='results/sift/qps_batch_compare.png', help='output image path')
    parser.add_argument('--show', action='store_true', help='show plot interactively')
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
    try:
        plot_files(algo_to_files, args.out, args.show)
    except Exception as e:
        print(f'绘图失败: {e}', file=sys.stderr)
        sys.exit(3)
    print(f'已保存: {args.out}')


if __name__ == '__main__':
    main()
