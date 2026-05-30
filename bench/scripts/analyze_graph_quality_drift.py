#!/usr/bin/env python3
"""Analyze graph quality drift: batch-built vs incremental-insert graph."""

import csv
import re
from pathlib import Path

RESULT_ROOT = Path("result/graph_quality_drift")


def parse_stats(stats_str):
    d = {}
    for item in stats_str.split(", "):
        if ":" in item:
            k, v = item.split(":", 1)
            try:
                d[k.strip()] = float(v.strip())
            except ValueError:
                d[k.strip()] = v.strip()
    return d


def sf(r, k, default=0.0):
    try:
        return float(r[k])
    except (ValueError, TypeError, KeyError):
        return default


def main():
    if not RESULT_ROOT.exists():
        print(f"No results found at {RESULT_ROOT}")
        return

    results = {}
    for csv_path in sorted(RESULT_ROOT.rglob("benchmark_results.csv")):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        r = rows[0]
        # Extract dataset and label from path
        parts = csv_path.relative_to(RESULT_ROOT).parts
        if len(parts) >= 2:
            ds = parts[0]
            label = parts[1]
            results[(ds, label)] = r

    if not results:
        print("No results found.")
        return

    print("=" * 80)
    print("  GRAPH QUALITY DRIFT: Batch vs Incremental Build")
    print("=" * 80)

    header = (f"{'dataset':<10} {'config':<15} {'begin%':>7} "
              f"{'recall':>8} {'dist_comp':>12} {'hops':>10} "
              f"{'ins_qps':>10} {'srch_p99':>9}")
    print(f"\n{header}")
    print("-" * 90)

    by_ds = {}
    for (ds, label), r in sorted(results.items()):
        stats = parse_stats(r.get("stats", ""))
        recall = sf(r, "recall")
        dist = stats.get("dist_computations", 0)
        hops = stats.get("hops", 0)
        ins_qps = sf(r, "insert_qps (per-point)")
        s_p99 = sf(r, "search_p99_op_latency_ms (per-batch)")

        # Extract begin% from label
        m = re.search(r"(\d+)pct", label)
        pct = m.group(1) if m else "?"

        print(f"{ds:<10} {label:<15} {pct:>6}% "
              f"{recall:>8.4f} {dist:>12.0f} {hops:>10.0f} "
              f"{ins_qps:>10.1f} {s_p99:>9.2f}")

        if ds not in by_ds:
            by_ds[ds] = {}
        by_ds[ds][label] = {
            "recall": recall, "dist": dist, "hops": hops,
            "pct": pct, "s_p99": s_p99,
        }

    # Print drift analysis
    print("\n" + "=" * 80)
    print("  DRIFT ANALYSIS (relative to batch_99pct baseline)")
    print("=" * 80)
    for ds in sorted(by_ds):
        baseline = by_ds[ds].get("batch_99pct")
        if not baseline:
            continue
        print(f"\n  {ds}:")
        print(f"  {'config':<15} {'recall_delta':>12} {'dist_ratio':>10} {'hops_ratio':>10}")
        print(f"  {'-'*50}")
        for label in ["batch_99pct", "incr_50pct", "incr_10pct", "incr_1pct"]:
            if label not in by_ds[ds]:
                continue
            d = by_ds[ds][label]
            recall_delta = d["recall"] - baseline["recall"]
            dist_ratio = d["dist"] / baseline["dist"] if baseline["dist"] > 0 else 0
            hops_ratio = d["hops"] / baseline["hops"] if baseline["hops"] > 0 else 0
            print(f"  {label:<15} {recall_delta:>+12.4f} {dist_ratio:>10.2f}x {hops_ratio:>10.2f}x")

    print("""
INTERPRETATION:
  - If incr_1pct has lower recall than batch_99pct → incremental insert
    produces a worse graph than batch build. This is the "graph quality drift"
    that segment systems avoid via periodic rebuild.
  - If dist_computations or hops are higher → incremental graph requires more
    work per search to achieve similar results.
  - The gap quantifies the potential value of "online graph repair" as M3.
""")


if __name__ == "__main__":
    main()
