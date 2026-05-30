#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict

import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_ROOT = os.path.join(ROOT, "result", "m2_node_lock_hardware_matrix_20260507")
PAPER_ROOT = os.path.abspath(os.path.join(ROOT, "..", "paper-progress", "m3", "search-tail-rootcause"))
TABLE_DIR = os.path.join(PAPER_ROOT, "tables")
BIN_MS = 50.0
WINDOWS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]
CASES = [
    ("sift200k", "on", "sift200k_node_lock_on"),
    ("sift200k", "off", "sift200k_node_lock_off"),
    ("gist200k", "on", "gist200k_node_lock_on"),
    ("gist200k", "off", "gist200k_node_lock_off"),
]


def in_insert(ms):
    return any(start <= ms < end for start, end in WINDOWS)


def pctl(values, pct):
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not values:
        return math.nan
    idx = max(0, min(len(values) - 1, int(math.ceil(len(values) * pct / 100.0)) - 1))
    return values[idx]


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return math.nan
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return math.nan
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def read_latency_bins(case_dir):
    start_raw = int(open(os.path.join(case_dir, "perf_start_raw_ns.txt")).read().strip())
    path = os.path.join(case_dir, "raw_latency", "search_wide.csv")
    cols = [
        "measured",
        "finish_ms",
        "finish_raw_ns",
        "op_ms",
        "work_searchknn_thread_cpu_ms",
        "work_level0_lock_wait_ms",
        "work_distance_computations",
        "work_level0_expansions",
        "work_level0_edges_scanned",
    ]
    tracked = [c for c in cols if c != "measured"] + ["perf_elapsed_ms"]
    bins = defaultdict(lambda: {c: [] for c in tracked})
    for chunk in pd.read_csv(path, usecols=cols, chunksize=200000):
        chunk = chunk[(chunk["measured"] == 1) & (chunk["finish_ms"] >= 0.0) & (chunk["finish_ms"] < 12000.0)]
        if chunk.empty:
            continue
        chunk["bin_ms"] = (chunk["finish_ms"] // BIN_MS) * BIN_MS
        chunk["perf_elapsed_ms"] = (chunk["finish_raw_ns"] - start_raw) / 1_000_000.0
        for b, frame in chunk.groupby("bin_ms"):
            rec = bins[float(b)]
            for col in ["finish_ms", "perf_elapsed_ms", "op_ms", "work_searchknn_thread_cpu_ms", "work_level0_lock_wait_ms", "work_distance_computations", "work_level0_expansions", "work_level0_edges_scanned"]:
                rec[col].extend(frame[col].astype(float).tolist())
    rows = []
    for b, rec in sorted(bins.items()):
        rows.append(
            {
                "bin_ms": b,
                "perf_elapsed_ms": pctl(rec["perf_elapsed_ms"], 50),
                "insert_active": in_insert(b),
                "search_count": len(rec["op_ms"]),
                "op_p99_ms": pctl(rec["op_ms"], 99),
                "cpu_p99_ms": pctl(rec["work_searchknn_thread_cpu_ms"], 99),
                "level0_lock_wait_p99_ms": pctl(rec["work_level0_lock_wait_ms"], 99),
                "level0_lock_wait_max_ms": max(rec["work_level0_lock_wait_ms"]) if rec["work_level0_lock_wait_ms"] else math.nan,
                "distance_p50": pctl(rec["work_distance_computations"], 50),
                "distance_p99": pctl(rec["work_distance_computations"], 99),
                "level0_expansions_p50": pctl(rec["work_level0_expansions"], 50),
                "level0_edges_p50": pctl(rec["work_level0_edges_scanned"], 50),
            }
        )
    return rows


def read_perf(case_dir):
    path = os.path.join(case_dir, "perf.csv")
    rows = []
    with open(path, errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split(",")
            if len(parts) < 4:
                continue
            try:
                t_ms = float(parts[0]) * 1000.0
            except ValueError:
                continue
            value = parts[1]
            if value in ("<not counted>", "<not supported>"):
                continue
            try:
                count = float(value)
            except ValueError:
                continue
            event = parts[3]
            rows.append((t_ms, event, count))
    by_time = defaultdict(dict)
    for t_ms, event, count in rows:
        by_time[t_ms][event] = count
    return [{"perf_time_ms": t, **events} for t, events in sorted(by_time.items())]


def join_perf(lat_rows, perf_rows):
    if not perf_rows:
        return lat_rows
    times = [r["perf_time_ms"] for r in perf_rows]
    for row in lat_rows:
        t = row["perf_elapsed_ms"]
        idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
        if abs(times[idx] - t) <= 75.0:
            for k, v in perf_rows[idx].items():
                if k != "perf_time_ms":
                    row[k] = v
    return lat_rows


def summarize(rows):
    out = {}
    for prefix, frame in [("insert", [r for r in rows if r["insert_active"]]), ("outside", [r for r in rows if not r["insert_active"] and r["bin_ms"] >= 500.0])]:
        out[f"{prefix}_op_p99_max_ms"] = max((r["op_p99_ms"] for r in frame), default=math.nan)
        out[f"{prefix}_op_p99_median_ms"] = pctl([r["op_p99_ms"] for r in frame], 50)
        out[f"{prefix}_lock_wait_p99_max_ms"] = max((r["level0_lock_wait_p99_ms"] for r in frame), default=math.nan)
        out[f"{prefix}_lock_wait_max_ms"] = max((r["level0_lock_wait_max_ms"] for r in frame), default=math.nan)
        out[f"{prefix}_cpu_p99_median_ms"] = pctl([r["cpu_p99_ms"] for r in frame], 50)
        out[f"{prefix}_dist_p99_median"] = pctl([r["distance_p99"] for r in frame], 50)
        for ev in ["l2_rqsts.all_rfo", "l2_rqsts.rfo_miss", "LLC-load-misses", "LLC-store-misses", "longest_lat_cache.miss", "cycles", "instructions"]:
            vals = [r[ev] for r in frame if ev in r]
            out[f"{prefix}_{ev}_median"] = pctl(vals, 50)
    insert = [r for r in rows if r["insert_active"]]
    for ev in ["level0_lock_wait_p99_ms", "l2_rqsts.all_rfo", "l2_rqsts.rfo_miss", "LLC-load-misses", "LLC-store-misses", "longest_lat_cache.miss", "cycles", "instructions", "distance_p99"]:
        out[f"corr_insert_p99_vs_{ev}"] = pearson([r["op_p99_ms"] for r in insert], [r.get(ev, math.nan) for r in insert])
    return out


def fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.6f}"
    return str(v)


def main():
    os.makedirs(TABLE_DIR, exist_ok=True)
    joined_all = []
    summary = []
    for dataset, node_lock, case in CASES:
        case_dir = os.path.join(RESULT_ROOT, case)
        rows = join_perf(read_latency_bins(case_dir), read_perf(case_dir))
        for r in rows:
            r.update({"dataset": dataset, "node_lock": node_lock, "case": case})
        joined_all.extend(rows)
        summary.append({"dataset": dataset, "node_lock": node_lock, "case": case, **summarize(rows)})

    joined_fields = []
    for r in joined_all:
        for k in r:
            if k not in joined_fields:
                joined_fields.append(k)
    summary_fields = []
    for r in summary:
        for k in r:
            if k not in summary_fields:
                summary_fields.append(k)

    for path, rows, fields in [
        (os.path.join(TABLE_DIR, "24_m2_node_lock_hardware_bins.csv"), joined_all, joined_fields),
        (os.path.join(TABLE_DIR, "25_m2_node_lock_hardware_summary.csv"), summary, summary_fields),
    ]:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: fmt(row.get(k, "")) for k in fields})

    for r in summary:
        def ratio(ev):
            a = r.get(f"insert_{ev}_median", math.nan)
            b = r.get(f"outside_{ev}_median", math.nan)
            if not b or math.isnan(a) or math.isnan(b):
                return math.nan
            return a / b
        print(
            r["dataset"], r["node_lock"],
            "insert_p99_max", fmt(r["insert_op_p99_max_ms"]),
            "outside_p99_max", fmt(r["outside_op_p99_max_ms"]),
            "lock_wait_max", fmt(r["insert_lock_wait_max_ms"]),
            "rfo_miss_lift", fmt(ratio("l2_rqsts.rfo_miss")),
            "llc_load_miss_lift", fmt(ratio("LLC-load-misses")),
            "corr_lock", fmt(r["corr_insert_p99_vs_level0_lock_wait_p99_ms"]),
            "corr_rfo_miss", fmt(r["corr_insert_p99_vs_l2_rqsts.rfo_miss"]),
        )


if __name__ == "__main__":
    main()
