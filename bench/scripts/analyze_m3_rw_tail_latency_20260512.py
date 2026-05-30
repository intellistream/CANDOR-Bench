#!/usr/bin/env python3
"""Aggregate read-write tail-latency cells into summary.

For each (variant, N, rep), extract reader-side P50/P95/P99/P999/mean
and writer iQPS, then aggregate across reps.
"""
from __future__ import annotations
import argparse
import csv
import math
import statistics
import sys
from pathlib import Path


def load_meta(p: Path) -> dict:
    out = {}
    if not p.exists(): return out
    for tok in p.read_text().split():
        if "=" in tok:
            k, v = tok.split("=", 1); out[k] = v
    return out


def percentile(xs, q):
    if not xs: return float("nan")
    s = sorted(xs); k = (len(s)-1)*q
    f = math.floor(k); c = math.ceil(k)
    if f == c: return s[int(k)]
    return s[f] + (s[c]-s[f])*(k-f)


def cell_metrics(cell: Path) -> dict:
    meta = load_meta(cell / "cell.meta")
    if not meta: return {}
    # search_op_ms.csv has per-batch latencies
    op_csv = cell / "raw_latency" / "search_op_ms.csv"
    op_ms = []
    if op_csv.exists():
        with open(op_csv) as f:
            rdr = csv.reader(f)
            next(rdr, None)
            for r in rdr:
                if r:
                    try: op_ms.append(float(r[0]))
                    except: pass

    # insert iQPS from insert_wide
    iw = cell / "raw_latency" / "insert_wide.csv"
    n_ins = 0; span_ms = 0.0
    if iw.exists():
        with open(iw) as f:
            fins = [float(r["finish_ms"]) for r in csv.DictReader(f)]
        if fins:
            n_ins = len(fins); span_ms = max(fins) - min(fins) + 1

    iqps = (n_ins * 20 * 1000 / span_ms) if span_ms > 0 else 0.0

    return {
        "cell": cell.name,
        "variant": meta.get("variant", ""),
        "n": int(meta.get("n_writers", "0")),
        "rep": int(meta.get("rep", "0")),
        "n_search_samples": len(op_ms),
        "search_mean_ms": statistics.mean(op_ms) if op_ms else float("nan"),
        "search_p50_ms": percentile(op_ms, 0.50),
        "search_p95_ms": percentile(op_ms, 0.95),
        "search_p99_ms": percentile(op_ms, 0.99),
        "search_p999_ms": percentile(op_ms, 0.999),
        "search_max_ms": max(op_ms) if op_ms else float("nan"),
        "iqps_vec_per_s": iqps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    rows = []
    for cell in sorted(args.root.iterdir()):
        if cell.is_dir() and (cell / "cell.meta").exists():
            m = cell_metrics(cell)
            if m: rows.append(m)
    if not rows:
        print("no cells", file=sys.stderr); return 1
    keys = list(rows[0].keys())
    with open(args.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {args.out}")

    # Aggregate by (variant, n)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["variant"], r["n"])].append(r)

    print(f"\n{'variant':<8} {'N':>3} {'reps':>4} {'rdP50':>7} {'rdP99':>7} {'rdP999':>7} {'rdMAX':>7} {'iQPS':>10}")
    print("-" * 78)
    for (v, n) in sorted(agg.keys()):
        rs = agg[(v, n)]
        p50 = statistics.mean([r["search_p50_ms"] for r in rs])
        p99 = statistics.mean([r["search_p99_ms"] for r in rs])
        p999 = statistics.mean([r["search_p999_ms"] for r in rs])
        pmax = statistics.mean([r["search_max_ms"] for r in rs])
        iq = statistics.mean([r["iqps_vec_per_s"] for r in rs])
        print(f"{v:<8} {n:>3d} {len(rs):>4d} {p50:>7.2f} {p99:>7.2f} {p999:>7.2f} {pmax:>7.1f} {iq:>10.0f}")


if __name__ == "__main__":
    raise SystemExit(main())
