#!/usr/bin/env python3
"""Analyze M2 optimization sweep results."""
import csv
from collections import defaultdict
from pathlib import Path

LOG = Path("result/m2_optimize/experiment_log.csv")

def main():
    if not LOG.exists():
        print("No results yet.")
        return

    rows = []
    with open(LOG) as f:
        rows = list(csv.DictReader(f))

    # Group by strategy
    by_strategy = defaultdict(list)
    for r in rows:
        by_strategy[r["strategy"]].append(r)

    # Find strategies where ALL scenarios pass
    print("=" * 90)
    print("  M2 OPTIMIZATION: Strategy × Scenario Matrix")
    print("=" * 90)

    scenarios = sorted(set(r["scenario"] for r in rows))

    # Header
    print(f"\n{'strategy':<20}", end="")
    for s in scenarios:
        print(f" {s:<14}", end="")
    print(f" {'ALL_PASS':>8}")
    print("-" * (20 + 15 * len(scenarios) + 10))

    for strategy in sorted(by_strategy):
        results = {r["scenario"]: r for r in by_strategy[strategy]}
        all_pass = True
        print(f"{strategy:<20}", end="")
        for s in scenarios:
            if s in results:
                r = results[s]
                ps = r.get("pass_search", "?")
                pi = r.get("pass_insert", "?")
                if ps == "PASS" and pi == "PASS":
                    mark = "✓"
                elif ps == "FAIL" and pi == "FAIL":
                    mark = "✗✗"
                    all_pass = False
                else:
                    mark = f"{'S' if ps=='FAIL' else ''}{'I' if pi=='FAIL' else ''}"
                    all_pass = False
                sp99 = r.get("srch_p99", "?")
                try:
                    print(f" {mark}{float(sp99):>10.1f}ms", end="")
                except:
                    print(f" {mark:>14}", end="")
            else:
                print(f" {'—':>14}", end="")
                all_pass = False
        print(f" {'✓ ALL' if all_pass else '':>8}")

    # Detailed view of best strategies
    print("\n" + "=" * 90)
    print("  DETAILED: Best strategy per scenario (minimize srch_p99, pass both criteria)")
    print("=" * 90)
    for s in scenarios:
        best = None
        best_p99 = float('inf')
        for strategy in by_strategy:
            for r in by_strategy[strategy]:
                if r["scenario"] != s: continue
                if r.get("pass_search") != "PASS" or r.get("pass_insert") != "PASS": continue
                try:
                    p99 = float(r["srch_p99"])
                    if p99 < best_p99:
                        best_p99 = p99
                        best = r
                except: pass
        if best:
            print(f"  {s:<20} → {best['strategy']:<15} sp99={best['srch_p99']:>8} iqps={best['ins_qps']:>8} fm={best['fresh_mean']:>8}")
        else:
            print(f"  {s:<20} → NO PASSING STRATEGY")

    # Check if any strategy passes ALL
    print("\n" + "=" * 90)
    print("  STRATEGIES PASSING ALL SCENARIOS")
    print("=" * 90)
    found = False
    for strategy in sorted(by_strategy):
        results = {r["scenario"]: r for r in by_strategy[strategy]}
        if len(results) < len(scenarios): continue
        if all(r.get("pass_search") == "PASS" and r.get("pass_insert") == "PASS"
               for r in results.values()):
            found = True
            print(f"  ✓ {strategy}")
            for s in scenarios:
                r = results[s]
                print(f"    {s:<20} sp99={r['srch_p99']:>8} iqps={r['ins_qps']:>8}")
    if not found:
        print("  NONE - need more strategies")


if __name__ == "__main__":
    main()
