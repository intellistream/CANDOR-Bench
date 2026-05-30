#!/usr/bin/env python3
import csv
import os
import sys

import matplotlib.pyplot as plt


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fval(row, key):
    return float(row[key])


def build_rate_map(rows):
    out = {}
    for row in rows:
        key = (int(fval(row, "search_input_rate")), int(fval(row, "insert_input_rate")))
        out[key] = row
    return out


def main():
    if len(sys.argv) != 4:
        print("usage: plot_sift_sat_compare.py NOLOCK_CSV RWLOCK_CSV OUT_DIR", file=sys.stderr)
        return 2

    nolock_csv, rwlock_csv, out_dir = sys.argv[1:]
    os.makedirs(out_dir, exist_ok=True)

    nolock = build_rate_map(load_rows(nolock_csv))
    rwlock = build_rate_map(load_rows(rwlock_csv))
    keys = sorted(set(nolock) & set(rwlock))

    out_table = os.path.join(out_dir, "sift_sat_16t_batch20_compare.csv")
    with open(out_table, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "search_rate",
                "write_rate",
                "nolock_search_qps",
                "rwlock_search_qps",
                "nolock_insert_qps",
                "rwlock_insert_qps",
                "search_speedup",
            ]
        )
        for key in keys:
            nr = nolock[key]
            rr = rwlock[key]
            ns = fval(nr, "search_qps (per-point)")
            rs = fval(rr, "search_qps (per-point)")
            ni = fval(nr, "insert_qps (per-point)")
            ri = fval(rr, "insert_qps (per-point)")
            speedup = ns / rs if rs else 0.0
            w.writerow([key[0], key[1], f"{ns:.4f}", f"{rs:.4f}", f"{ni:.4f}", f"{ri:.4f}", f"{speedup:.3f}"])

    rates = [k[0] for k in keys]
    nolock_search = [fval(nolock[k], "search_qps (per-point)") for k in keys]
    rwlock_search = [fval(rwlock[k], "search_qps (per-point)") for k in keys]
    nolock_insert = [fval(nolock[k], "insert_qps (per-point)") for k in keys]
    rwlock_insert = [fval(rwlock[k], "insert_qps (per-point)") for k in keys]

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    for ax in axes:
        ax.grid(axis="y", color="black", alpha=0.28, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.9)

    axes[0].plot(rates, nolock_search, marker="o", color="#4C78A8", linewidth=2, label="NoLock")
    axes[0].plot(rates, rwlock_search, marker="o", color="#E45756", linewidth=2, label="RWLock")
    axes[0].set_xlabel("Offered search/write rate")
    axes[0].set_ylabel("Search QPS")

    axes[1].plot(rates, nolock_insert, marker="o", color="#4C78A8", linewidth=2, label="NoLock")
    axes[1].plot(rates, rwlock_insert, marker="o", color="#E45756", linewidth=2, label="RWLock")
    axes[1].set_xlabel("Offered search/write rate")
    axes[1].set_ylabel("Insert QPS")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_png = os.path.join(out_dir, "sift_sat_16t_batch20_compare.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
