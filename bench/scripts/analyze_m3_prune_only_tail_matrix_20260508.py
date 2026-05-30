#!/usr/bin/env python3
import csv
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_ROOT = ROOT / "result" / "m3_prune_only_tail_matrix_20260508"
RESULT_ROOT = Path(os.environ.get("RESULT_ROOT", str(DEFAULT_RESULT_ROOT)))
PAPER_ROOT = ROOT.parent / "paper-progress" / "m3" / "search-tail-rootcause"
TABLE_DIR = PAPER_ROOT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

INSERT_WINDOWS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]
BIN_MS = 50.0


def percentile(values, q):
    values = [v for v in values if math.isfinite(v)]
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


def parse_stats(case_dir):
    csv_path = case_dir / "benchmark_results.csv"
    if not csv_path.exists():
        parents = list(case_dir.glob("benchmark_results.csv"))
        csv_path = parents[0] if parents else csv_path
    # Runner writes benchmark_results.csv at the per-algorithm output root.
    candidates = [case_dir / "benchmark_results.csv", case_dir.parent / "benchmark_results.csv"]
    row = None
    for path in candidates:
        if path.exists():
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                row = rows[-1]
                break
    stats = row.get("stats", "") if row else ""
    out = {}
    for key in [
        "graph_unique_lock_wait_ns",
        "graph_existing_neighbor_appends",
        "graph_existing_neighbor_prunes",
        "l0_lockfree_appends",
        "l0_lockfree_append_fallbacks",
        "hot_region_search_marks",
        "l0_adj_overlay_publishes",
        "l0_edge_delta_publishes",
    ]:
        m = re.search(rf"(?:^|, )({re.escape(key)}):([0-9.]+)", stats)
        out[key] = float(m.group(2)) if m else 0.0
    if row:
        for key in [
            "insert_p99_op_latency_ms (per-batch)",
            "search_p99_op_latency_ms (per-batch)",
            "recall",
        ]:
            try:
                out[key] = float(row.get(key, "nan"))
            except ValueError:
                out[key] = float("nan")
    return out


def summarize_case(case_dir):
    meta = read_meta(case_dir)
    batches, origin_ns = read_batches(case_dir)
    probe_rows = read_probe(case_dir, origin_ns)
    stats = parse_stats(case_dir)

    batch_ms = [b["batch_ms"] for b in batches]
    insert_batches = [b for b in batches if b["overlap_insert"]]
    outside_batches = [b for b in batches if not b["overlap_insert"] and b["finish_ms"] >= 500.0]

    top1_count = max(1, math.ceil(len(batches) * 0.01)) if batches else 0
    top01_count = max(1, math.ceil(len(batches) * 0.001)) if batches else 0
    top1 = sorted(batches, key=lambda b: b["batch_ms"], reverse=True)[:top1_count]
    top01 = sorted(batches, key=lambda b: b["batch_ms"], reverse=True)[:top01_count]

    batch_bins = defaultdict(list)
    for b in batches:
        batch_bins[b["bin_ms"]].append(b["batch_ms"])
    lock_bins = defaultdict(list)
    for r in probe_rows:
        lock_bins[r["bin_ms"]].append(r["wait_ms"])
    common_bins = sorted(set(batch_bins) | set(lock_bins))
    p99_by_bin = [percentile(batch_bins[b], 0.99) for b in common_bins]
    lock_max_by_bin = [max(lock_bins[b]) if lock_bins[b] else 0.0 for b in common_bins]

    insert_p99 = percentile([b["batch_ms"] for b in insert_batches], 0.99)
    outside_p99 = percentile([b["batch_ms"] for b in outside_batches], 0.99)
    probe_insert = [r for r in probe_rows if r["in_insert"]]

    out = {
        "mode": meta.get("mode", case_dir.parent.name),
        "algorithm": meta.get("algorithm", ""),
        "dataset": meta.get("dataset", ""),
        "rep": int(meta.get("rep", 0) or 0),
        "batch_count": len(batches),
        "insert_batch_count": len(insert_batches),
        "outside_batch_count": len(outside_batches),
        "batch_p50_ms": percentile(batch_ms, 0.50),
        "batch_p95_ms": percentile(batch_ms, 0.95),
        "batch_p99_ms": percentile(batch_ms, 0.99),
        "batch_max_ms": max(batch_ms) if batch_ms else float("nan"),
        "insert_batch_p99_ms": insert_p99,
        "outside_batch_p99_ms": outside_p99,
        "insert_minus_outside_p99_ms": insert_p99 - outside_p99
        if finite(insert_p99) and finite(outside_p99)
        else float("nan"),
        "top1pct_tail_insert_fraction": (sum(1 for b in top1 if b["overlap_insert"]) / len(top1))
        if top1
        else float("nan"),
        "top01pct_tail_insert_fraction": (sum(1 for b in top01 if b["overlap_insert"]) / len(top01))
        if top01
        else float("nan"),
        "probe_rdlock_count": len(probe_rows),
        "probe_rdlock_insert_count": len(probe_insert),
        "probe_rdlock_max_ms": max((r["wait_ms"] for r in probe_rows), default=0.0),
        "probe_rdlock_insert_max_ms": max((r["wait_ms"] for r in probe_insert), default=0.0),
        "corr_bin_p99_vs_rdlock_max": pearson(p99_by_bin, lock_max_by_bin),
    }
    out.update(stats)
    return out


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
        groups[(row["mode"], row["dataset"], row["algorithm"])].append(row)
    out = []
    numeric = [
        k for k, v in rows[0].items()
        if isinstance(v, (int, float)) and k not in {"rep"}
    ] if rows else []
    for (mode, dataset, algorithm), items in sorted(groups.items()):
        row = {
            "mode": mode,
            "dataset": dataset,
            "algorithm": algorithm,
            "reps": len(items),
        }
        for key in numeric:
            vals = [r[key] for r in items if finite(r.get(key, float("nan")))]
            row[key] = statistics.fmean(vals) if vals else float("nan")
        out.append(row)
    return out


def compare(agg_rows):
    by_key = {(r["mode"], r["dataset"], r["algorithm"]): r for r in agg_rows}
    out = []
    for mode, dataset in sorted({(mode, dataset) for mode, dataset, _ in by_key}):
        off = by_key.get((mode, dataset, "annchor-m3-off"))
        on = by_key.get((mode, dataset, "annchor-m3"))
        if not off or not on:
            continue
        row = {"mode": mode, "dataset": dataset}
        for key in [
            "insert_batch_p99_ms",
            "batch_p99_ms",
            "batch_max_ms",
            "probe_rdlock_max_ms",
            "graph_unique_lock_wait_ns",
            "search_p99_op_latency_ms (per-batch)",
            "insert_p99_op_latency_ms (per-batch)",
            "recall",
        ]:
            a = off.get(key, float("nan"))
            b = on.get(key, float("nan"))
            row[f"{key}_off"] = a
            row[f"{key}_on"] = b
            row[f"{key}_delta"] = b - a if finite(a) and finite(b) else float("nan")
            row[f"{key}_reduction_pct"] = (a - b) / a * 100.0 if finite(a) and a else float("nan")
        row["l0_lockfree_appends_on"] = on.get("l0_lockfree_appends", 0.0)
        row["old_m3_hot_region_on"] = on.get("hot_region_search_marks", 0.0)
        row["old_m3_overlay_on"] = on.get("l0_adj_overlay_publishes", 0.0)
        row["old_m3_delta_on"] = on.get("l0_edge_delta_publishes", 0.0)
        out.append(row)
    return out


def main():
    rows = []
    for mode_dir in sorted(RESULT_ROOT.glob("*")):
        if not mode_dir.is_dir():
            continue
        for case_dir in sorted(mode_dir.glob("*")):
            if (case_dir / "experiment_meta.csv").exists():
                rows.append(summarize_case(case_dir))

    fields = [
        "mode", "algorithm", "dataset", "rep", "batch_count", "insert_batch_count",
        "outside_batch_count", "batch_p50_ms", "batch_p95_ms", "batch_p99_ms",
        "batch_max_ms", "insert_batch_p99_ms", "outside_batch_p99_ms",
        "insert_minus_outside_p99_ms", "top1pct_tail_insert_fraction",
        "top01pct_tail_insert_fraction", "probe_rdlock_count",
        "probe_rdlock_insert_count", "probe_rdlock_max_ms",
        "probe_rdlock_insert_max_ms", "corr_bin_p99_vs_rdlock_max",
        "graph_unique_lock_wait_ns", "graph_existing_neighbor_appends",
        "graph_existing_neighbor_prunes", "l0_lockfree_appends",
        "l0_lockfree_append_fallbacks", "hot_region_search_marks",
        "l0_adj_overlay_publishes", "l0_edge_delta_publishes",
        "insert_p99_op_latency_ms (per-batch)",
        "search_p99_op_latency_ms (per-batch)", "recall",
    ]
    agg_fields = ["mode", "dataset", "algorithm", "reps"] + [
        f for f in fields if f not in {"mode", "dataset", "algorithm", "rep"}
    ]
    cmp_fields = None

    write_csv(TABLE_DIR / "33_m3_prune_only_tail_cases.csv", rows, fields)
    agg_rows = aggregate(rows)
    write_csv(TABLE_DIR / "34_m3_prune_only_tail_aggregate.csv", agg_rows, agg_fields)
    cmp_rows = compare(agg_rows)
    if cmp_rows:
        cmp_fields = list(cmp_rows[0].keys())
        write_csv(TABLE_DIR / "35_m3_prune_only_tail_comparison.csv", cmp_rows, cmp_fields)

    print(f"cases={len(rows)}")
    print(TABLE_DIR / "33_m3_prune_only_tail_cases.csv")
    print(TABLE_DIR / "34_m3_prune_only_tail_aggregate.csv")
    if cmp_fields:
        print(TABLE_DIR / "35_m3_prune_only_tail_comparison.csv")


if __name__ == "__main__":
    raise SystemExit(main())
