#!/usr/bin/env python3
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = ROOT / "result" / "m2_tail_causality_matrix_20260508"
RESULT_ROOT = Path(__import__("os").environ.get("RESULT_ROOT", str(DEFAULT_RESULT_ROOT)))
PAPER_ROOT = ROOT.parent / "paper-progress" / "m3" / "search-tail-rootcause"
TABLE_DIR = PAPER_ROOT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

INSERT_WINDOWS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]
BIN_MS = 50.0


def percentile(values, q):
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def interval_overlaps_insert(start_ms, finish_ms):
    return any(start_ms < end and finish_ms >= start for start, end in INSERT_WINDOWS)


def point_in_insert(ms):
    return any(start <= ms < end for start, end in INSERT_WINDOWS)


def bin_ms(ms):
    return math.floor(ms / BIN_MS) * BIN_MS


def read_meta(case_dir):
    out = {}
    path = case_dir / "experiment_meta.csv"
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for key, value in csv.reader(f):
            out[key] = value
    return out


def read_batches(case_dir):
    path = case_dir / "raw_latency" / "search_compact.csv"
    if not path.exists():
        return [], float("nan")
    by_key = {}
    origins = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("measured") != "1":
                continue
            start_raw = int(row["start_raw_ns"])
            finish_raw = int(row["finish_raw_ns"])
            finish_ms = float(row["finish_ms"])
            origins.append(finish_raw - finish_ms * 1_000_000.0)
            key = (start_raw, finish_raw)
            item = by_key.setdefault(
                key,
                {
                    "start_raw_ns": start_raw,
                    "finish_raw_ns": finish_raw,
                    "finish_ms": finish_ms,
                    "queries": 0,
                },
            )
            item["queries"] += 1
    if not origins:
        return [], float("nan")
    origin_ns = statistics.median(origins)
    batches = []
    for item in by_key.values():
        start_ms = (item["start_raw_ns"] - origin_ns) / 1_000_000.0
        finish_ms = item["finish_ms"]
        batch_ms = (item["finish_raw_ns"] - item["start_raw_ns"]) / 1_000_000.0
        batches.append(
            {
                "start_ms": start_ms,
                "finish_ms": finish_ms,
                "batch_ms": batch_ms,
                "queries": item["queries"],
                "overlap_insert": interval_overlaps_insert(start_ms, finish_ms),
                "finish_in_insert": point_in_insert(finish_ms),
                "bin_ms": bin_ms(finish_ms),
            }
        )
    batches.sort(key=lambda x: x["finish_ms"])
    return batches, origin_ns


def parse_probe_line(line):
    parts = line.strip().split(",")
    if not parts:
        return None
    try:
        ts_ns = int(parts[0])
    except ValueError:
        return None
    row = {"ts_ns": ts_ns}
    for part in parts[1:]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        row[k] = v
    try:
        row["wait_ms"] = int(row.get("wait_us", "0")) / 1000.0
    except ValueError:
        row["wait_ms"] = 0.0
    return row


def read_probe(case_dir, origin_ns):
    path = case_dir / "rwlock_probe.csv"
    rows = []
    if not path.exists() or not finite(origin_ns):
        return rows
    with path.open(errors="replace") as f:
        for line in f:
            row = parse_probe_line(line)
            if not row:
                continue
            if row.get("op") != "rdlock":
                continue
            if "searchBaseLayer" not in row.get("sym", ""):
                continue
            ms = (row["ts_ns"] - origin_ns) / 1_000_000.0
            if ms < -100.0 or ms > 13000.0:
                continue
            row["elapsed_ms"] = ms
            row["in_insert"] = point_in_insert(ms)
            row["bin_ms"] = bin_ms(ms)
            rows.append(row)
    return rows


def summarize_case(case_dir):
    meta = read_meta(case_dir)
    batches, origin_ns = read_batches(case_dir)
    probe_rows = read_probe(case_dir, origin_ns)

    batch_ms = [b["batch_ms"] for b in batches]
    insert_batches = [b for b in batches if b["overlap_insert"]]
    outside_batches = [b for b in batches if not b["overlap_insert"] and b["finish_ms"] >= 500.0]

    top1_count = max(1, math.ceil(len(batches) * 0.01)) if batches else 0
    top01_count = max(1, math.ceil(len(batches) * 0.001)) if batches else 0
    top1 = sorted(batches, key=lambda b: b["batch_ms"], reverse=True)[:top1_count]
    top01 = sorted(batches, key=lambda b: b["batch_ms"], reverse=True)[:top01_count]

    probe_insert = [r for r in probe_rows if r["in_insert"]]
    probe_outside = [r for r in probe_rows if not r["in_insert"] and r["elapsed_ms"] >= 500.0]

    batch_bins = defaultdict(list)
    for b in batches:
        batch_bins[b["bin_ms"]].append(b["batch_ms"])
    lock_bins = defaultdict(list)
    for r in probe_rows:
        lock_bins[r["bin_ms"]].append(r["wait_ms"])
    common_bins = sorted(set(batch_bins) | set(lock_bins))
    p99_by_bin = [percentile(batch_bins[b], 0.99) for b in common_bins]
    lock_max_by_bin = [max(lock_bins[b]) if lock_bins[b] else 0.0 for b in common_bins]
    lock_sum_by_bin = [sum(lock_bins[b]) for b in common_bins]

    insert_p99 = percentile([b["batch_ms"] for b in insert_batches], 0.99)
    outside_p99 = percentile([b["batch_ms"] for b in outside_batches], 0.99)

    return {
        "result_set": case_dir.parent.name,
        "case": case_dir.name,
        "mode": meta.get("mode", case_dir.parent.name),
        "dataset": meta.get("dataset", ""),
        "node_lock": meta.get("node_lock", ""),
        "competing_workers": int(meta.get("competing_workers", 0) or 0),
        "competing_mode": meta.get("competing_mode", ""),
        "insert_workers": int(meta.get("insert_workers", 0) or 0),
        "measured_workers": int(meta.get("measured_workers", 0) or 0),
        "rep": int(meta.get("rep", 0) or 0),
        "ef_search": int(meta.get("ef_search", 0) or 0),
        "batch_count": len(batches),
        "insert_batch_count": len(insert_batches),
        "outside_batch_count": len(outside_batches),
        "batch_p50_ms": percentile(batch_ms, 0.50),
        "batch_p95_ms": percentile(batch_ms, 0.95),
        "batch_p99_ms": percentile(batch_ms, 0.99),
        "batch_max_ms": max(batch_ms) if batch_ms else float("nan"),
        "insert_batch_p99_ms": insert_p99,
        "outside_batch_p99_ms": outside_p99,
        "insert_over_outside_p99_ratio": insert_p99 / outside_p99 if outside_p99 and finite(outside_p99) else float("nan"),
        "insert_minus_outside_p99_ms": insert_p99 - outside_p99 if finite(insert_p99) and finite(outside_p99) else float("nan"),
        "top1pct_tail_insert_fraction": (sum(1 for b in top1 if b["overlap_insert"]) / len(top1)) if top1 else float("nan"),
        "top01pct_tail_insert_fraction": (sum(1 for b in top01 if b["overlap_insert"]) / len(top01)) if top01 else float("nan"),
        "probe_rdlock_count": len(probe_rows),
        "probe_rdlock_insert_count": len(probe_insert),
        "probe_rdlock_outside_count": len(probe_outside),
        "probe_rdlock_insert_fraction": len(probe_insert) / len(probe_rows) if probe_rows else float("nan"),
        "probe_rdlock_max_ms": max((r["wait_ms"] for r in probe_rows), default=0.0),
        "probe_rdlock_insert_max_ms": max((r["wait_ms"] for r in probe_insert), default=0.0),
        "probe_rdlock_outside_max_ms": max((r["wait_ms"] for r in probe_outside), default=0.0),
        "corr_bin_p99_vs_rdlock_max": pearson(p99_by_bin, lock_max_by_bin),
        "corr_bin_p99_vs_rdlock_sum": pearson(p99_by_bin, lock_sum_by_bin),
    }


def fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.6f}"
    return str(v)


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k, "")) for k in fields})


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["mode"],
            row["dataset"],
            row["node_lock"],
            row["competing_mode"],
            row["competing_workers"],
            row["insert_workers"],
        )
        groups[key].append(row)
    out = []
    for (mode, dataset, node_lock, competing_mode, cw, insert_workers), items in sorted(groups.items()):
        def vals(k):
            return [r[k] for r in items if finite(r.get(k, float("nan")))]

        out.append(
            {
                "mode": mode,
                "dataset": dataset,
                "node_lock": node_lock,
                "competing_mode": competing_mode,
                "competing_workers": cw,
                "insert_workers": insert_workers,
                "reps": len(items),
                "batch_p99_ms_median": percentile(vals("batch_p99_ms"), 0.50),
                "insert_batch_p99_ms_median": percentile(vals("insert_batch_p99_ms"), 0.50),
                "outside_batch_p99_ms_median": percentile(vals("outside_batch_p99_ms"), 0.50),
                "insert_over_outside_p99_ratio_median": percentile(vals("insert_over_outside_p99_ratio"), 0.50),
                "insert_minus_outside_p99_ms_median": percentile(vals("insert_minus_outside_p99_ms"), 0.50),
                "top1pct_tail_insert_fraction_median": percentile(vals("top1pct_tail_insert_fraction"), 0.50),
                "top01pct_tail_insert_fraction_median": percentile(vals("top01pct_tail_insert_fraction"), 0.50),
                "probe_rdlock_count_median": percentile(vals("probe_rdlock_count"), 0.50),
                "probe_rdlock_insert_fraction_median": percentile(vals("probe_rdlock_insert_fraction"), 0.50),
                "probe_rdlock_max_ms_median": percentile(vals("probe_rdlock_max_ms"), 0.50),
                "probe_rdlock_insert_max_ms_median": percentile(vals("probe_rdlock_insert_max_ms"), 0.50),
                "corr_bin_p99_vs_rdlock_max_median": percentile(vals("corr_bin_p99_vs_rdlock_max"), 0.50),
            }
        )
    return out


def main():
    rows = []
    for mode_dir in sorted(RESULT_ROOT.glob("*")):
        if not mode_dir.is_dir():
            continue
        for case_dir in sorted(mode_dir.glob("*")):
            if not case_dir.is_dir():
                continue
            if not (case_dir / "raw_latency" / "search_compact.csv").exists():
                continue
            rows.append(summarize_case(case_dir))

    case_fields = [
        "result_set",
        "case",
        "mode",
        "dataset",
        "node_lock",
        "competing_workers",
        "competing_mode",
        "insert_workers",
        "measured_workers",
        "rep",
        "ef_search",
        "batch_count",
        "insert_batch_count",
        "outside_batch_count",
        "batch_p50_ms",
        "batch_p95_ms",
        "batch_p99_ms",
        "batch_max_ms",
        "insert_batch_p99_ms",
        "outside_batch_p99_ms",
        "insert_over_outside_p99_ratio",
        "insert_minus_outside_p99_ms",
        "top1pct_tail_insert_fraction",
        "top01pct_tail_insert_fraction",
        "probe_rdlock_count",
        "probe_rdlock_insert_count",
        "probe_rdlock_outside_count",
        "probe_rdlock_insert_fraction",
        "probe_rdlock_max_ms",
        "probe_rdlock_insert_max_ms",
        "probe_rdlock_outside_max_ms",
        "corr_bin_p99_vs_rdlock_max",
        "corr_bin_p99_vs_rdlock_sum",
    ]
    agg_fields = [
        "mode",
        "dataset",
        "node_lock",
        "competing_workers",
        "competing_mode",
        "insert_workers",
        "reps",
        "batch_p99_ms_median",
        "insert_batch_p99_ms_median",
        "outside_batch_p99_ms_median",
        "insert_over_outside_p99_ratio_median",
        "insert_minus_outside_p99_ms_median",
        "top1pct_tail_insert_fraction_median",
        "top01pct_tail_insert_fraction_median",
        "probe_rdlock_count_median",
        "probe_rdlock_insert_fraction_median",
        "probe_rdlock_max_ms_median",
        "probe_rdlock_insert_max_ms_median",
        "corr_bin_p99_vs_rdlock_max_median",
    ]
    case_path = TABLE_DIR / "28_m2_tail_causality_matrix_cases.csv"
    agg_path = TABLE_DIR / "29_m2_tail_causality_matrix_aggregate.csv"
    write_csv(case_path, rows, case_fields)
    agg_rows = aggregate(rows)
    write_csv(agg_path, agg_rows, agg_fields)
    print(case_path)
    print(agg_path)
    for row in agg_rows:
        print(
            f"{row['mode']} {row['dataset']} cw={row['competing_workers']} "
            f"iw={row['insert_workers']} {row['competing_mode']} "
            f"lock={row['node_lock']} reps={row['reps']} "
            f"tail_insert={fmt(row['top1pct_tail_insert_fraction_median'])} "
            f"insert/outside={fmt(row['insert_over_outside_p99_ratio_median'])} "
            f"rdlock_max={fmt(row['probe_rdlock_max_ms_median'])}"
        )


if __name__ == "__main__":
    main()
