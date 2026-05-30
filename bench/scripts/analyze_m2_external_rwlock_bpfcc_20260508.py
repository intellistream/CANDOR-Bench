#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_GLOB = "m2_external_rwlock_bpfcc_20260508*"
OUT_DIR = ROOT.parent / "paper-progress" / "m3" / "search-tail-rootcause" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "27_m2_external_rwlock_bpfcc_summary.csv"

INSERT_WINDOWS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]


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


def in_insert_window(ms):
    return any(start <= ms < end for start, end in INSERT_WINDOWS)


def read_search_batches(case_dir):
    path = case_dir / "raw_latency" / "search_compact.csv"
    batches = {}
    if not path.exists():
        return []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("measured") != "1":
                continue
            key = (row["start_raw_ns"], row["finish_raw_ns"])
            item = batches.setdefault(
                key,
                {
                    "n": 0,
                    "finish_ms": float(row["finish_ms"]),
                    "op_ms": float(row["op_ms"]),
                },
            )
            item["n"] += 1
    out = []
    for item in batches.values():
        out.append(
            {
                "finish_ms": item["finish_ms"],
                "batch_ms": item["op_ms"] * item["n"],
                "in_insert": in_insert_window(item["finish_ms"]),
            }
        )
    return out


FUNC_RE = re.compile(
    r"^\S+\s+\d+\s+(?P<lat_us>[0-9.]+)\s+\S+\s+"
    r"(?P<func>\S*pthread_rwlock_rdlock)\s+(?P<lock>0x[0-9a-fA-F]+)"
)
THRESHOLD_RE = re.compile(r"Tracing function calls slower than (?P<min_us>[0-9]+) us")


def read_funcslower(case_dir):
    path = case_dir / "funcslower_rdlock.txt"
    waits = []
    locks = set()
    min_us = None
    if not path.exists():
        return waits, locks, min_us
    for line in path.read_text(errors="replace").splitlines():
        tm = THRESHOLD_RE.search(line)
        if tm:
            min_us = int(tm.group("min_us"))
        m = FUNC_RE.search(line)
        if not m:
            continue
        waits.append(float(m.group("lat_us")) / 1000.0)
        locks.add(m.group("lock"))
    return waits, locks, min_us


def summarize_case(case_dir):
    waits, locks, min_us = read_funcslower(case_dir)
    batches = read_search_batches(case_dir)
    batch_ms = [b["batch_ms"] for b in batches]
    insert_batch_ms = [b["batch_ms"] for b in batches if b["in_insert"]]
    outside_batch_ms = [b["batch_ms"] for b in batches if not b["in_insert"]]
    if batches:
        top_count = max(1, math.ceil(len(batches) * 0.01))
        top = sorted(batches, key=lambda b: b["batch_ms"], reverse=True)[:top_count]
        top_insert_frac = sum(1 for b in top if b["in_insert"]) / len(top)
    else:
        top_insert_frac = float("nan")

    return {
        "case": case_dir.name,
        "result_set": case_dir.parent.name,
        "min_us": min_us if min_us is not None else "",
        "slow_rdlock_count_ge_threshold": len(waits),
        "slow_rdlock_unique_locks": len(locks),
        "slow_rdlock_p50_ms": percentile(waits, 0.50),
        "slow_rdlock_p95_ms": percentile(waits, 0.95),
        "slow_rdlock_max_ms": max(waits) if waits else 0.0,
        "measured_search_batches": len(batches),
        "batch_p50_ms": percentile(batch_ms, 0.50),
        "batch_p99_ms": percentile(batch_ms, 0.99),
        "batch_max_ms": max(batch_ms) if batch_ms else float("nan"),
        "insert_batch_p99_ms": percentile(insert_batch_ms, 0.99),
        "outside_batch_p99_ms": percentile(outside_batch_ms, 0.99),
        "top1pct_batch_insert_fraction": top_insert_frac,
    }


def main():
    rows = []
    for result_root in sorted((ROOT / "result").glob(RESULT_GLOB)):
        for case_dir in sorted(result_root.glob("*")):
            if not case_dir.is_dir() or not (case_dir / "raw_latency").exists():
                continue
            rows.append(summarize_case(case_dir))

    fieldnames = [
        "result_set",
        "case",
        "min_us",
        "slow_rdlock_count_ge_threshold",
        "slow_rdlock_unique_locks",
        "slow_rdlock_p50_ms",
        "slow_rdlock_p95_ms",
        "slow_rdlock_max_ms",
        "measured_search_batches",
        "batch_p50_ms",
        "batch_p99_ms",
        "batch_max_ms",
        "insert_batch_p99_ms",
        "outside_batch_p99_ms",
        "top1pct_batch_insert_fraction",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(OUT_CSV)
    for row in rows:
        print(
            f"{row['result_set']}/{row['case']} min_us={row['min_us']}: "
            f"slow_rdlock={row['slow_rdlock_count_ge_threshold']} "
            f"max={row['slow_rdlock_max_ms']:.3f}ms "
            f"batch_p99={row['batch_p99_ms']:.3f}ms "
            f"insert_p99={row['insert_batch_p99_ms']:.3f}ms "
            f"outside_p99={row['outside_batch_p99_ms']:.3f}ms "
            f"top1pct_insert={row['top1pct_batch_insert_fraction']:.2f}"
        )


if __name__ == "__main__":
    main()
