#!/usr/bin/env python3
"""
绘制单个 CSV 文件中 batch_idx -> query_qps 的折线图。

用法示例:
  python tools/draw_onecsv_qps.py results/sift/faiss_HNSW/test16_result/ef-120_batch_query_qps.csv

支持选项: --out 指定输出图片文件, --show 在运行后显示图像
"""
import argparse
import csv
import os
import sys

def read_csv_simple(path):
    x = []
    y = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return x, y
        # try to detect header names
        lower = [h.lower() for h in headers]
        if 'batch_idx' in lower and 'query_qps' in lower:
            bi = lower.index('batch_idx')
            qi = lower.index('query_qps')
            for row in reader:
                if len(row) <= max(bi, qi):
                    continue
                try:
                    x.append(int(float(row[bi])))
                    y.append(float(row[qi]))
                except Exception:
                    continue
        else:
            # fallback: first two columns
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    x.append(int(float(row[0])))
                    y.append(float(row[1]))
                except Exception:
                    continue
    return x, y

def read_csv_dict(path):
    x = []
    y = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return x, y
        lower = [h.lower() for h in reader.fieldnames]
        if 'batch_idx' in lower and 'query_qps' in lower:
            # use DictReader rows
            for row in reader:
                try:
                    x.append(int(float(row[next(h for h in reader.fieldnames if h.lower()=='batch_idx')])) )
                    y.append(float(row[next(h for h in reader.fieldnames if h.lower()=='query_qps')]))
                except Exception:
                    continue
        else:
            # fallback to positional via reader
            f.seek(0)
            return read_csv_simple(path)
    return x, y

def load(path):
    # Prefer DictReader for robustness, but fall back
    try:
        x, y = read_csv_dict(path)
        if x and y:
            return x, y
    except Exception:
        pass
    return read_csv_simple(path)

def plot_xy(x, y, out_path, title=None, show=False):
    import matplotlib
    # Use Agg backend so script works headless
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not x or not y:
        print('No data to plot', file=sys.stderr)
        return 1

    # sort by x
    pairs = sorted(zip(x, y), key=lambda t: t[0])
    xs, ys = zip(*pairs)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(xs, ys, marker='o', linestyle='-', linewidth=1)
    ax.set_xlabel('batch_idx')
    ax.set_ylabel('query_qps')
    ax.grid(True, linestyle='--', alpha=0.5)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f'Saved plot to {out_path}')
    if show:
        # display using a GUI backend if possible
        try:
            import matplotlib.pyplot as plt_show
            matplotlib.use(None)
            plt_show.show()
        except Exception:
            print('Cannot show plot (no GUI); saved to file instead')
    return 0

def main():
    p = argparse.ArgumentParser(description='Draw QPS vs batch_idx from one CSV file')
    p.add_argument('csvfile', help='CSV file path')
    p.add_argument('--out', '-o', help='Output image file (png). Defaults to <csvname>_qps.png')
    p.add_argument('--title', '-t', help='Plot title', default=None)
    p.add_argument('--show', action='store_true', help='Show the plot interactively (may not work on headless)')
    args = p.parse_args()

    csvfile = args.csvfile
    if not os.path.exists(csvfile):
        print('CSV file not found:', csvfile, file=sys.stderr)
        return 2

    out = args.out
    if not out:
        base = os.path.splitext(os.path.basename(csvfile))[0]
        out = base + '_qps.png'

    x, y = load(csvfile)
    return plot_xy(x, y, out, title=args.title, show=args.show)

if __name__ == '__main__':
    sys.exit(main())
