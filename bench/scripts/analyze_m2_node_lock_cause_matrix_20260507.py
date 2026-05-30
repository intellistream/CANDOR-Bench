#!/usr/bin/env python3
import csv
import math
import os
import re
import statistics
from collections import defaultdict


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_ROOT = os.path.join(ROOT, "result", "m2_node_lock_cause_matrix_20260507")
PAPER_ROOT = os.path.abspath(os.path.join(ROOT, "..", "paper-progress", "m3", "search-tail-rootcause"))
TABLE_DIR = os.path.join(PAPER_ROOT, "tables")
FIG_DIR = os.path.join(PAPER_ROOT, "figures")
BIN_MS = 50.0
INSERT_WINDOWS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]


CASES = [
    ("sift200k", "on", "sift200k_node_lock_on"),
    ("sift200k", "off", "sift200k_node_lock_off"),
    ("sift10m200k", "on", "sift10m200k_node_lock_on"),
    ("sift10m200k", "off", "sift10m200k_node_lock_off"),
    ("sift1m", "on", "sift1m_node_lock_on"),
    ("sift1m", "off", "sift1m_node_lock_off"),
    ("gist200k", "on", "gist200k_node_lock_on"),
    ("gist200k", "off", "gist200k_node_lock_off"),
    ("dbpedia50k", "on", "dbpedia50k_node_lock_on"),
    ("dbpedia50k", "off", "dbpedia50k_node_lock_off"),
]


def percentile(values, pct):
    if not values:
        return math.nan
    values = sorted(values)
    idx = int(math.ceil(pct / 100.0 * len(values))) - 1
    idx = max(0, min(idx, len(values) - 1))
    return values[idx]


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return math.nan
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def in_insert_window(ms):
    return any(start <= ms < end for start, end in INSERT_WINDOWS)


def bin_start_ms(ms):
    return math.floor(ms / BIN_MS) * BIN_MS


def read_measured_bins(case_dir):
    path = os.path.join(case_dir, "raw_latency", "search_compact.csv")
    bins = defaultdict(list)
    origin_samples = []
    measured_count = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("measured") != "1":
                continue
            measured_count += 1
            finish_ms = float(row["finish_ms"])
            finish_raw_ns = int(row["finish_raw_ns"])
            op_ms = float(row["op_ms"])
            origin_samples.append(finish_raw_ns - finish_ms * 1_000_000.0)
            bins[bin_start_ms(finish_ms)].append(op_ms)
    if not origin_samples:
        raise RuntimeError(f"no measured search samples in {path}")
    origin_ns = statistics.median(origin_samples)
    bin_rows = []
    for b, values in sorted(bins.items()):
        bin_rows.append(
            {
                "bin_ms": b,
                "search_count": len(values),
                "op_p50_ms": percentile(values, 50),
                "op_p95_ms": percentile(values, 95),
                "op_p99_ms": percentile(values, 99),
                "insert_active": in_insert_window(b),
            }
        )
    return origin_ns, measured_count, bin_rows


def parse_probe_line(line):
    parts = line.strip().split(",")
    if not parts or not parts[0]:
        return None
    try:
        ts_ns = int(parts[0])
    except ValueError:
        return None
    values = {"ts_ns": ts_ns}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key] = value
    try:
        values["wait_us"] = int(values.get("wait_us", "0"))
    except ValueError:
        values["wait_us"] = 0
    return values


def read_probe_bins(case_dir, origin_ns):
    path = os.path.join(case_dir, "rwlock_probe.csv")
    bins = defaultdict(lambda: {"rd_count": 0, "rd_sum_ms": 0.0, "rd_max_ms": 0.0, "wr_count": 0, "wr_sum_ms": 0.0, "wr_max_ms": 0.0})
    if not os.path.exists(path):
        return bins
    with open(path, errors="replace") as f:
        for line in f:
            row = parse_probe_line(line)
            if not row:
                continue
            sym = row.get("sym", "")
            op = row.get("op", "")
            if "searchBaseLayer" not in sym:
                continue
            elapsed_ms = (row["ts_ns"] - origin_ns) / 1_000_000.0
            if elapsed_ms < 0 or elapsed_ms > 13000:
                continue
            b = bin_start_ms(elapsed_ms)
            wait_ms = row["wait_us"] / 1000.0
            if op == "rdlock":
                bins[b]["rd_count"] += 1
                bins[b]["rd_sum_ms"] += wait_ms
                bins[b]["rd_max_ms"] = max(bins[b]["rd_max_ms"], wait_ms)
            elif op == "wrlock":
                bins[b]["wr_count"] += 1
                bins[b]["wr_sum_ms"] += wait_ms
                bins[b]["wr_max_ms"] = max(bins[b]["wr_max_ms"], wait_ms)
    return bins


def summarize_case(dataset, node_lock, case_name):
    case_dir = os.path.join(RESULT_ROOT, case_name)
    origin_ns, measured_count, search_bins = read_measured_bins(case_dir)
    lock_bins = read_probe_bins(case_dir, origin_ns)
    joined = []
    for row in search_bins:
        lock = lock_bins.get(row["bin_ms"], {})
        joined.append(
            {
                **row,
                "rdlock_wait_count": lock.get("rd_count", 0),
                "rdlock_wait_sum_ms": lock.get("rd_sum_ms", 0.0),
                "rdlock_wait_max_ms": lock.get("rd_max_ms", 0.0),
                "wrlock_wait_count": lock.get("wr_count", 0),
                "wrlock_wait_sum_ms": lock.get("wr_sum_ms", 0.0),
                "wrlock_wait_max_ms": lock.get("wr_max_ms", 0.0),
            }
        )

    insert = [r for r in joined if r["insert_active"]]
    outside = [r for r in joined if not r["insert_active"] and r["bin_ms"] >= 500.0]
    top10 = sorted(joined, key=lambda r: r["op_p99_ms"], reverse=True)[:10]
    insert_top10 = sum(1 for r in top10 if r["insert_active"])
    corr_rows = insert if len(insert) >= 3 else joined
    corr_max = pearson([r["op_p99_ms"] for r in corr_rows], [r["rdlock_wait_max_ms"] for r in corr_rows])
    corr_sum = pearson([r["op_p99_ms"] for r in corr_rows], [r["rdlock_wait_sum_ms"] for r in corr_rows])

    return joined, {
        "dataset": dataset,
        "node_lock": node_lock,
        "case": case_name,
        "measured_search_samples": measured_count,
        "insert_bins": len(insert),
        "outside_bins": len(outside),
        "insert_op_p99_max_ms": max((r["op_p99_ms"] for r in insert), default=math.nan),
        "outside_op_p99_max_ms": max((r["op_p99_ms"] for r in outside), default=math.nan),
        "insert_op_p99_median_ms": statistics.median([r["op_p99_ms"] for r in insert]) if insert else math.nan,
        "outside_op_p99_median_ms": statistics.median([r["op_p99_ms"] for r in outside]) if outside else math.nan,
        "top10_p99_bins_in_insert": insert_top10,
        "rdlock_wait_count_insert": sum(r["rdlock_wait_count"] for r in insert),
        "rdlock_wait_sum_insert_ms": sum(r["rdlock_wait_sum_ms"] for r in insert),
        "rdlock_wait_max_insert_ms": max((r["rdlock_wait_max_ms"] for r in insert), default=0.0),
        "corr_insert_op_p99_vs_rdlock_max": corr_max,
        "corr_insert_op_p99_vs_rdlock_sum": corr_sum,
    }


def fmt(value):
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in fields})


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    summary_rows = []
    joined_rows = []
    for dataset, node_lock, case_name in CASES:
        joined, summary = summarize_case(dataset, node_lock, case_name)
        summary_rows.append(summary)
        for row in joined:
            joined_rows.append({"dataset": dataset, "node_lock": node_lock, "case": case_name, **row})

    summary_fields = [
        "dataset",
        "node_lock",
        "case",
        "measured_search_samples",
        "insert_bins",
        "outside_bins",
        "insert_op_p99_max_ms",
        "outside_op_p99_max_ms",
        "insert_op_p99_median_ms",
        "outside_op_p99_median_ms",
        "top10_p99_bins_in_insert",
        "rdlock_wait_count_insert",
        "rdlock_wait_sum_insert_ms",
        "rdlock_wait_max_insert_ms",
        "corr_insert_op_p99_vs_rdlock_max",
        "corr_insert_op_p99_vs_rdlock_sum",
    ]
    joined_fields = [
        "dataset",
        "node_lock",
        "case",
        "bin_ms",
        "insert_active",
        "search_count",
        "op_p50_ms",
        "op_p95_ms",
        "op_p99_ms",
        "rdlock_wait_count",
        "rdlock_wait_sum_ms",
        "rdlock_wait_max_ms",
        "wrlock_wait_count",
        "wrlock_wait_sum_ms",
        "wrlock_wait_max_ms",
    ]
    write_csv(os.path.join(TABLE_DIR, "22_m2_node_lock_dataset_matrix.csv"), summary_rows, summary_fields)
    write_csv(os.path.join(TABLE_DIR, "23_m2_node_lock_dataset_bins.csv"), joined_rows, joined_fields)

    try:
        import matplotlib.pyplot as plt
        colors = {"on": "#cc4e5c", "off": "#356bb5"}
        datasets = ["sift200k", "sift10m200k", "sift1m", "gist200k", "dbpedia50k"]
        fig, axes = plt.subplots(len(datasets), 1, figsize=(8.6, 10.2), sharex=True)
        for ax, dataset in zip(axes, datasets):
            rows = [r for r in joined_rows if r["dataset"] == dataset]
            for start, end in INSERT_WINDOWS:
                ax.axvspan(start / 1000.0, end / 1000.0, color="#f2c4c1", alpha=0.42, lw=0)
            for node_lock in ["on", "off"]:
                sub = [r for r in rows if r["node_lock"] == node_lock]
                xs = [r["bin_ms"] / 1000.0 for r in sub]
                ys = [r["op_p99_ms"] for r in sub]
                ax.plot(xs, ys, color=colors[node_lock], lw=1.9, label=f"Node Lock {node_lock.capitalize()}")
            ax.set_ylabel("P99 ms")
            ax.set_title(dataset, loc="left", fontsize=11, fontweight="normal")
            ax.grid(True, axis="y", color="#d9d9d9", lw=0.6, alpha=0.65)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].legend(loc="upper right", frameon=False, ncol=2)
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "11_m2_node_lock_dataset_matrix.png"), dpi=220)
        fig.savefig(os.path.join(FIG_DIR, "11_m2_node_lock_dataset_matrix.pdf"))
        plt.close(fig)
    except Exception as exc:
        print(f"figure generation skipped: {exc}")

    for row in summary_rows:
        print(
            row["dataset"],
            row["node_lock"],
            "insert_p99_max_ms=",
            fmt(row["insert_op_p99_max_ms"]),
            "outside_p99_max_ms=",
            fmt(row["outside_op_p99_max_ms"]),
            "rdlock_max_insert_ms=",
            fmt(row["rdlock_wait_max_insert_ms"]),
            "rdlock_sum_insert_ms=",
            fmt(row["rdlock_wait_sum_insert_ms"]),
        )


if __name__ == "__main__":
    main()
