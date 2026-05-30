#!/usr/bin/env python3
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_int(row, key):
    return int(float(row[key]))


def to_float(row, key):
    return float(row[key])


def rate_bucket(row):
    insert_rate = to_float(row, "insert_input_rate")
    search_rate = to_float(row, "search_input_rate")
    if math.isclose(insert_rate, search_rate):
        return "BAL"
    return "RH" if search_rate > insert_rate else "WH"


def key_of(row):
    return (
        to_int(row, "write_batch_size"),
        to_int(row, "threads"),
        rate_bucket(row),
        row["query_mode"],
    )


def label_of(row):
    return f"{to_int(row, 'threads')}T {rate_bucket(row)}"


def sort_key(row):
    bucket_order = {"RH": 0, "BAL": 1, "WH": 2}
    return (to_int(row, "threads"), bucket_order[rate_bucket(row)])


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def style_axes(ax):
    ax.grid(axis="y", color="black", alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors="black", labelsize=10, rotation=35)
    ax.tick_params(axis="y", colors="black", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)


def save_occ_plots(occ_rows, out_dir):
    rows = [r for r in occ_rows if r["query_mode"] == "round_robin"]
    batches = sorted({to_int(r, "write_batch_size") for r in rows})

    dist_color = "#4C78A8"
    pct_color = "#F58518"

    manifest = []
    for batch in batches:
        batch_rows = sorted(
            [r for r in rows if to_int(r, "write_batch_size") == batch],
            key=sort_key,
        )
        labels = [label_of(r) for r in batch_rows]
        extra_dist = [to_float(r, "inflight_occ_l2_dists_mean") for r in batch_rows]
        occ_pct = [to_float(r, "inflight_occ_pct_search_op") for r in batch_rows]

        fig, ax = plt.subplots(figsize=(11, 5.8))
        xs = range(len(batch_rows))
        bars = ax.bar(xs, extra_dist, width=0.72, color=dist_color, edgecolor="black", linewidth=0.8)
        ax.set_xticks(list(xs), labels)
        ax.set_ylabel("OCC extra dist / query")
        style_axes(ax)
        ymax = max(extra_dist) if extra_dist else 1.0
        ax.set_ylim(0, ymax * 1.18 + 1)
        for bar, v in zip(bars, extra_dist):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )
        fig.tight_layout()
        dist_path = os.path.join(out_dir, f"sift_roundrobin_batch{batch}_distance.png")
        fig.savefig(dist_path, dpi=180)
        plt.close(fig)
        manifest.append(("distance", batch, dist_path))

        fig, ax = plt.subplots(figsize=(11, 5.8))
        xs = range(len(batch_rows))
        bars = ax.bar(xs, occ_pct, width=0.72, color=pct_color, edgecolor="black", linewidth=0.8)
        ax.set_xticks(list(xs), labels)
        ax.set_ylabel("OCC cost (% of search op)")
        style_axes(ax)
        ymax = max(occ_pct) if occ_pct else 1.0
        ax.set_ylim(0, max(100.0, ymax * 1.18))
        for bar, v in zip(bars, occ_pct):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1.2, ymax * 0.02),
                f"{v:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color=pct_color,
                fontweight="bold",
            )
        fig.tight_layout()
        pct_path = os.path.join(out_dir, f"sift_roundrobin_batch{batch}_latency.png")
        fig.savefig(pct_path, dpi=180)
        plt.close(fig)
        manifest.append(("latency", batch, pct_path))

    return manifest


def save_rwlock_compare(occ_rows, rw_rows, out_dir):
    occ_rows = [r for r in occ_rows if r["query_mode"] == "round_robin"]
    rw_rows = [r for r in rw_rows if r["query_mode"] == "round_robin"]

    occ_map = {key_of(r): r for r in occ_rows}
    rw_map = {key_of(r): r for r in rw_rows}
    shared_keys = sorted(set(occ_map) & set(rw_map))

    out_csv = os.path.join(out_dir, "sift_roundrobin_rwlock_vs_occ_qps.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "batch",
                "threads",
                "rate",
                "query_mode",
                "occ_search_qps",
                "rwlock_search_qps",
                "rwlock_delta_pct",
            ]
        )
        for key in shared_keys:
            occ = occ_map[key]
            rw = rw_map[key]
            occ_qps = to_float(occ, "search_qps (per-point)")
            rw_qps = to_float(rw, "search_qps (per-point)")
            delta = (rw_qps - occ_qps) / occ_qps * 100.0 if occ_qps else 0.0
            writer.writerow(
                [
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    f"{occ_qps:.4f}",
                    f"{rw_qps:.4f}",
                    f"{delta:.2f}",
                ]
            )

    batches = sorted({k[0] for k in shared_keys})
    bucket_order = {"RH": 0, "BAL": 1, "WH": 2}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.flatten()

    occ_color = "#4C78A8"
    rw_color = "#E45756"

    for ax, batch in zip(axes, batches):
        batch_keys = sorted(
            [k for k in shared_keys if k[0] == batch],
            key=lambda k: (k[1], bucket_order[k[2]]),
        )
        labels = [f"{k[1]}T {k[2]}" for k in batch_keys]
        occ_vals = [to_float(occ_map[k], "search_qps (per-point)") for k in batch_keys]
        rw_vals = [to_float(rw_map[k], "search_qps (per-point)") for k in batch_keys]
        xs = list(range(len(batch_keys)))
        width = 0.38
        ax.bar([x - width / 2 for x in xs], occ_vals, width=width, color=occ_color, edgecolor="black", linewidth=0.8, label="MVCC+OCC")
        ax.bar([x + width / 2 for x in xs], rw_vals, width=width, color=rw_color, edgecolor="black", linewidth=0.8, label="RWLock")
        ax.set_xticks(xs, labels)
        ax.set_ylabel("Search QPS")
        ax.text(0.01, 0.96, f"batch={batch}", transform=ax.transAxes, ha="left", va="top", fontsize=11, color="black")
        style_axes(ax)
        ymax = max(occ_vals + rw_vals) if (occ_vals or rw_vals) else 1.0
        ax.set_ylim(0, ymax * 1.15 + 1)

    for ax in axes[len(batches):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = os.path.join(out_dir, "sift_roundrobin_rwlock_vs_occ_qps.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return out_csv, out_png


def main():
    if len(sys.argv) != 4:
        print("usage: plot_sift_occ_and_rwlock.py OCC_CSV RWLOCK_CSV OUT_DIR", file=sys.stderr)
        return 2

    occ_csv, rw_csv, out_dir = sys.argv[1:]
    ensure_dir(out_dir)

    occ_rows = load_csv(occ_csv)
    rw_rows = load_csv(rw_csv)

    manifest = save_occ_plots(occ_rows, out_dir)
    out_csv, out_png = save_rwlock_compare(occ_rows, rw_rows, out_dir)

    readme = os.path.join(out_dir, "README.txt")
    with open(readme, "w") as f:
        f.write("Source OCC CSV:\n")
        f.write(f"{occ_csv}\n\n")
        f.write("Source RWLock CSV:\n")
        f.write(f"{rw_csv}\n\n")
        f.write("Generated files:\n")
        for kind, batch, path in manifest:
            f.write(f"- batch={batch} {kind}: {path}\n")
        f.write(f"- qps csv: {out_csv}\n")
        f.write(f"- qps png: {out_png}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
