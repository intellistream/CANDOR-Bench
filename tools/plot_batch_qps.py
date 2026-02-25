#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def gaussian_smooth(y, sigma: float):
    if sigma <= 0:
        return y
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return np.convolve(y, kernel, mode='same')


def main():
    p = argparse.ArgumentParser(description='Plot batch qps with optional smoothing')
    p.add_argument('csv', nargs='?', default='results/sift/faiss_HNSW/ef-120/ef-120_batch_query_qps.csv')
    p.add_argument('--method', choices=['raw', 'rolling', 'ewm', 'gaussian'], default='gaussian')
    p.add_argument('--window', type=int, default=5, help='window size for rolling/ewm')
    p.add_argument('--sigma', type=float, default=2.0, help='sigma for gaussian smoothing')
    p.add_argument('--plot-raw', action='store_true', help='also plot raw series in background')
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f'CSV not found: {csv_path}', file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(csv_path)
    x = df['batch_idx'].to_numpy()
    y = df['query_qps'].to_numpy()

    if args.method == 'raw':
        y_smooth = y
    elif args.method == 'rolling':
        y_smooth = pd.Series(y).rolling(window=args.window, center=True, min_periods=1).mean().to_numpy()
    elif args.method == 'ewm':
        # use span equivalent to window
        span = max(1, args.window)
        y_smooth = pd.Series(y).ewm(span=span, adjust=False).mean().to_numpy()
    else:  # gaussian
        y_smooth = gaussian_smooth(y, sigma=args.sigma)

    plt.figure(figsize=(12, 5))
    if args.plot_raw:
        plt.plot(x, y, color='0.7', linewidth=1, label='raw', alpha=0.6)
    else:
        plt.plot(x, y, color='0.9', linewidth=0.5, alpha=0.6)

    plt.plot(x, y_smooth, color='tab:blue', linewidth=2, label=f'smoothed ({args.method})')
    plt.scatter(x, y_smooth, color='tab:blue', s=12)
    plt.xlabel('batch_idx')
    plt.ylabel('query_qps')
    plt.title(csv_path.name)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_path = csv_path.with_name(csv_path.stem + f'_{args.method}.png')
    plt.savefig(out_path, dpi=200)
    print(out_path)


if __name__ == '__main__':
    main()
