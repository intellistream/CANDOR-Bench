#!/usr/bin/env python3
"""Analyze yield policy sweep: alpha parameter vs search/insert performance."""

import csv
import re
from pathlib import Path

RESULT_ROOT = Path("result/yield_policy_sweep")


def parse_stats(s):
    d = {}
    for item in s.split(", "):
        if ":" in item:
            k, v = item.split(":", 1)
            try: d[k.strip()] = float(v.strip())
            except: pass
    return d


def sf(r, k, d=0.0):
    try: return float(r[k])
    except: return d


def main():
    if not RESULT_ROOT.exists():
        print(f"No results at {RESULT_ROOT}")
        return

    results = {}
    for csv_path in sorted(RESULT_ROOT.rglob("benchmark_results.csv")):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        r = rows[0]
        parts = csv_path.relative_to(RESULT_ROOT).parts
        if len(parts) >= 2:
            label = parts[0]
            alpha_dir = parts[1]
            m = re.search(r"alpha_(\d+p\d+)", alpha_dir)
            if m:
                alpha = m.group(1).replace("p", ".")
                results[(label, alpha)] = r

    if not results:
        print("No results found.")
        return

    # Group by scenario
    scenarios = sorted(set(k[0] for k in results))

    print("=" * 90)
    print("  YIELD POLICY SWEEP: Alpha vs Performance")
    print("=" * 90)

    for scenario in scenarios:
        print(f"\n  === {scenario} ===")
        print(f"  {'alpha':>6} {'srch_mean':>10} {'srch_p99':>9} {'ins_qps':>9} {'ins_p99':>9} "
              f"{'recall':>7} {'yield_cnt':>10} {'starv_ref':>10}")
        print(f"  {'-'*75}")

        for alpha in ["0.05", "0.10", "0.15", "0.20", "0.30", "0.40"]:
            key = (scenario, alpha)
            if key not in results:
                continue
            r = results[key]
            stats = parse_stats(r.get("stats", ""))
            s_mean = sf(r, "search_mean_op_latency_ms (per-batch)")
            s_p99 = sf(r, "search_p99_op_latency_ms (per-batch)")
            i_qps = sf(r, "insert_qps (per-point)")
            i_p99 = sf(r, "insert_p99_op_latency_ms (per-batch)")
            recall = sf(r, "recall")
            yields = stats.get("fiber_yield_count", 0)
            starv = stats.get("fiber_starvation_refuses", 0)

            print(f"  {alpha:>6} {s_mean:>10.2f} {s_p99:>9.2f} {i_qps:>9.1f} {i_p99:>9.2f} "
                  f"{recall:>7.3f} {yields:>10.0f} {starv:>10.0f}")

    # Print comparison summary
    print("\n" + "=" * 90)
    print("  BEST ALPHA PER SCENARIO (minimize srch_p99)")
    print("=" * 90)
    for scenario in scenarios:
        best_alpha = None
        best_p99 = float('inf')
        for alpha in ["0.05", "0.10", "0.15", "0.20", "0.30", "0.40"]:
            key = (scenario, alpha)
            if key not in results:
                continue
            p99 = sf(results[key], "search_p99_op_latency_ms (per-batch)")
            if p99 < best_p99:
                best_p99 = p99
                best_alpha = alpha

        if best_alpha:
            r = results[(scenario, best_alpha)]
            i_qps = sf(r, "insert_qps (per-point)")
            print(f"  {scenario:<35} best_alpha={best_alpha:>5}  srch_p99={best_p99:>8.2f}  ins_qps={i_qps:>8.1f}")

    print("""
INTERPRETATION:
  - If best alpha varies across scenarios → static alpha is fundamentally wrong,
    need adaptive/dynamic yield policy.
  - If lower alpha helps write-heavy → confirms priority inversion hypothesis.
  - If higher alpha helps read-heavy → confirms that aggressive yield helps when reads dominate.
""")


if __name__ == "__main__":
    main()
