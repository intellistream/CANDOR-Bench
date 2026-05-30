#!/usr/bin/env python3
"""Analyze budget sweep and produce recommendation table."""
import csv
import os

RESULT_BASE = "result/budget_sweep"

BASELINES = {
    'deep1M_rh':  {'rwl_sp':6.34, 'mvcc_sp':22.82,'mvcc_iq':12578,'mvcc_fm':59.1, 'fib_sp':1.55,'fib_iq':15584,'dim':96,'wr':'read-heavy'},
    'deep1M_bal': {'rwl_sp':6.94, 'mvcc_sp':22.51,'mvcc_iq':9406, 'mvcc_fm':61.5, 'fib_sp':2.14,'fib_iq':11306,'dim':96,'wr':'balanced'},
    'deep1M_wh':  {'rwl_sp':6.69, 'mvcc_sp':23.33,'mvcc_iq':8712, 'mvcc_fm':54.1, 'fib_sp':12.22,'fib_iq':6289,'dim':96,'wr':'write-heavy'},
    'gist_rh':    {'rwl_sp':21.12,'mvcc_sp':54.38,'mvcc_iq':4832, 'mvcc_fm':60.1, 'fib_sp':4.70,'fib_iq':5106,'dim':960,'wr':'read-heavy'},
    'gist_bal':   {'rwl_sp':15.77,'mvcc_sp':52.48,'mvcc_iq':4326, 'mvcc_fm':60.8, 'fib_sp':51.20,'fib_iq':2466,'dim':960,'wr':'balanced'},
    'gist_wh':    {'rwl_sp':15.85,'mvcc_sp':53.06,'mvcc_iq':4297, 'mvcc_fm':61.2, 'fib_sp':50.37,'fib_iq':2539,'dim':960,'wr':'write-heavy'},
}

BUDGETS = ['1', '2', '4', '8', 'inf']
SCENARIOS = ['deep1M_rh','deep1M_bal','deep1M_wh','gist_rh','gist_bal','gist_wh']

# Criteria:
# search p99 < mvcc (much better) — at least 30% improvement
# insert qps >= 85% of mvcc
# freshness <= 120% of mvcc
def check_criteria(sp99, iq, fm, bl):
    ok_search = sp99 < 0.7 * bl['mvcc_sp']  # at least 30% better than mvcc
    ok_insert = iq >= 0.85 * bl['mvcc_iq']   # no more than 15% drop
    ok_fresh = fm <= 1.2 * bl['mvcc_fm']      # no more than 20% worse
    return ok_search, ok_insert, ok_fresh

def main():
    # Load all results
    data = {}  # (budget, scenario) -> {sp99, iq, fm}
    for b in BUDGETS:
        for sc in SCENARIOS:
            f = f"{RESULT_BASE}/b{b}/{sc}/benchmark_results.csv"
            if not os.path.exists(f):
                continue
            with open(f) as fh:
                rows = list(csv.DictReader(fh))
            if not rows:
                continue
            r = rows[-1]
            try:
                data[(b, sc)] = {
                    'sp99': float(r['search_p99_op_latency_ms (per-batch)']),
                    'iq': float(r['insert_qps (per-point)']),
                    'fm': float(r['freshness_total_pts_mean']),
                }
            except:
                pass

    if not data:
        print("No results yet.")
        return

    # ── Matrix: search p99 ──
    print("=" * 100)
    print("  SEARCH P99 MATRIX (ms)")
    print("=" * 100)
    print(f"{'scenario':<12} {'rwl':>7} {'mvcc':>7} {'fiber':>7}", end='')
    for b in BUDGETS:
        print(f" {'b='+b:>7}", end='')
    print()
    print("-" * 85)
    for sc in SCENARIOS:
        bl = BASELINES[sc]
        print(f"{sc:<12} {bl['rwl_sp']:>7.1f} {bl['mvcc_sp']:>7.1f} {bl['fib_sp']:>7.1f}", end='')
        for b in BUDGETS:
            if (b, sc) in data:
                v = data[(b, sc)]['sp99']
                ok, _, _ = check_criteria(v, data[(b,sc)]['iq'], data[(b,sc)]['fm'], bl)
                mark = '✓' if ok else ' '
                print(f" {mark}{v:>6.1f}", end='')
            else:
                print(f" {'—':>7}", end='')
        print()

    # ── Matrix: insert QPS (% vs mvcc) ──
    print("\n" + "=" * 100)
    print("  INSERT QPS (% change vs mvcc)")
    print("=" * 100)
    print(f"{'scenario':<12} {'mvcc':>7} {'fiber':>7}", end='')
    for b in BUDGETS:
        print(f" {'b='+b:>7}", end='')
    print()
    print("-" * 75)
    for sc in SCENARIOS:
        bl = BASELINES[sc]
        fib_pct = (bl['fib_iq']/bl['mvcc_iq']-1)*100
        print(f"{sc:<12} {bl['mvcc_iq']:>7} {fib_pct:>+6.0f}%", end='')
        for b in BUDGETS:
            if (b, sc) in data:
                v = data[(b, sc)]['iq']
                pct = (v / bl['mvcc_iq'] - 1) * 100
                _, ok, _ = check_criteria(data[(b,sc)]['sp99'], v, data[(b,sc)]['fm'], bl)
                mark = '✓' if ok else '✗'
                print(f" {mark}{pct:>+5.0f}%", end='')
            else:
                print(f" {'—':>7}", end='')
        print()

    # ── Matrix: freshness ──
    print("\n" + "=" * 100)
    print("  FRESHNESS MEAN (pts)")
    print("=" * 100)
    print(f"{'scenario':<12} {'mvcc':>7} {'fiber':>7}", end='')
    for b in BUDGETS:
        print(f" {'b='+b:>7}", end='')
    print()
    print("-" * 75)
    for sc in SCENARIOS:
        bl = BASELINES[sc]
        print(f"{sc:<12} {bl['mvcc_fm']:>7.1f} {'—':>7}", end='')
        for b in BUDGETS:
            if (b, sc) in data:
                v = data[(b, sc)]['fm']
                _, _, ok = check_criteria(data[(b,sc)]['sp99'], data[(b,sc)]['iq'], v, bl)
                mark = '✓' if ok else '✗'
                print(f" {mark}{v:>6.1f}", end='')
            else:
                print(f" {'—':>7}", end='')
        print()

    # ── Pass/Fail summary ──
    print("\n" + "=" * 100)
    print("  PASS/FAIL SUMMARY (✓ = all 3 criteria met)")
    print("  Criteria: sp99 < 70% mvcc, iqps >= 85% mvcc, fresh <= 120% mvcc")
    print("=" * 100)
    print(f"{'scenario':<12} {'dim':>4} {'workload':<12}", end='')
    for b in BUDGETS:
        print(f" {'b='+b:>5}", end='')
    print()
    print("-" * 60)
    for sc in SCENARIOS:
        bl = BASELINES[sc]
        print(f"{sc:<12} {bl['dim']:>4} {bl['wr']:<12}", end='')
        for b in BUDGETS:
            if (b, sc) in data:
                d = data[(b, sc)]
                s, i, f = check_criteria(d['sp99'], d['iq'], d['fm'], bl)
                if s and i and f:
                    print(f"   ✓ ", end='')
                else:
                    fails = ('S' if not s else '') + ('I' if not i else '') + ('F' if not f else '')
                    print(f"  {fails:<3}", end='')
            else:
                print(f"   — ", end='')
        print()

    # ── Recommendation table ──
    print("\n" + "=" * 100)
    print("  BUDGET RECOMMENDATION TABLE")
    print("=" * 100)
    for sc in SCENARIOS:
        bl = BASELINES[sc]
        best_budget = None
        best_sp99 = float('inf')
        for b in BUDGETS:
            if (b, sc) not in data:
                continue
            d = data[(b, sc)]
            s, i, f = check_criteria(d['sp99'], d['iq'], d['fm'], bl)
            if s and i and f and d['sp99'] < best_sp99:
                best_sp99 = d['sp99']
                best_budget = b
        if best_budget:
            d = data[(best_budget, sc)]
            print(f"  {sc:<12} dim={bl['dim']:>4} {bl['wr']:<12} → budget={best_budget:>3}  "
                  f"sp99={d['sp99']:>6.1f} ({d['sp99']/bl['mvcc_sp']*100:.0f}% of mvcc)  "
                  f"iq={d['iq']:>7.0f} ({d['iq']/bl['mvcc_iq']*100:.0f}% of mvcc)")
        else:
            print(f"  {sc:<12} dim={bl['dim']:>4} {bl['wr']:<12} → NO BUDGET MEETS ALL CRITERIA")

    # ── General recommendation ──
    print("\n" + "-" * 60)
    print("  GENERAL GUIDELINE:")
    # Analyze pattern
    reco = {}
    for sc in SCENARIOS:
        bl = BASELINES[sc]
        for b in BUDGETS:
            if (b, sc) not in data: continue
            d = data[(b, sc)]
            s, i, f = check_criteria(d['sp99'], d['iq'], d['fm'], bl)
            if s and i and f:
                key = (bl['dim'], bl['wr'])
                if key not in reco or d['sp99'] < reco[key][1]:
                    reco[key] = (b, d['sp99'])
    for (dim, wr), (b, sp) in sorted(reco.items()):
        print(f"    dim≈{dim}, {wr}: budget={b}")
    print()


if __name__ == "__main__":
    main()
