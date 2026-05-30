#!/usr/bin/env python3
"""Analyze Diagnostic #1 (search interference tax) and #2 (watermark HOL blocking)."""

import csv
import glob
import os
import sys

def read_results(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row['_path'] = f
                rows.append(row)
    return rows

def safe_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (ValueError, TypeError):
        return default

def main():
    base = os.path.join(os.path.dirname(__file__), '..')
    os.chdir(base)

    # ════════════════════════════════════════════
    # Diagnostic #1: Search Interference Tax
    # ════════════════════════════════════════════
    print("=" * 80)
    print("DIAGNOSTIC #1: Search Interference Tax")
    print("  Compare: search-only baseline vs mixed workload")
    print("=" * 80)

    # Load search-only baselines
    d1_rows = read_results("result/diagnostic_1_2/*/d1_searchonly_*/benchmark_results.csv")
    # Load mixed workload results from fiber_exp (b20, balanced 40k/40k)
    mixed_rows = read_results("result/fiber_exp/*/[!.]*_b20_w40000_r40000/benchmark_results.csv")

    if not d1_rows:
        print("\n  [!] No search-only baseline results found. Run the experiment first.")
    else:
        # Build lookup: (dataset, mode) -> search p99
        d1_map = {}
        for r in d1_rows:
            path = r['_path']
            parts = path.split('/')
            dataset = parts[2]
            config = parts[3]  # d1_searchonly_<mode>_b20
            mode = config.replace('d1_searchonly_', '').replace('_b20', '')
            d1_map[(dataset, mode)] = {
                'p99_op': safe_float(r, 'search_p99_op_latency_ms (per-batch)'),
                'mean_op': safe_float(r, 'search_mean_op_latency_ms (per-batch)'),
                'p99_e2e': safe_float(r, 'search_p99_e2e_latency_ms (per-batch)'),
            }

        mixed_map = {}
        for r in mixed_rows:
            path = r['_path']
            parts = path.split('/')
            dataset = parts[2]
            config = parts[3]
            mode = config.split('_b20')[0]
            key = (dataset, mode)
            if key not in mixed_map:
                mixed_map[key] = []
            mixed_map[key].append({
                'p99_op': safe_float(r, 'search_p99_op_latency_ms (per-batch)'),
                'mean_op': safe_float(r, 'search_mean_op_latency_ms (per-batch)'),
                'p99_e2e': safe_float(r, 'search_p99_e2e_latency_ms (per-batch)'),
            })

        print(f"\n{'dataset':<10} {'mode':<8} {'baseline_p99':>13} {'mixed_p99':>10} {'tax_p99':>10} {'tax_%':>8} │ {'base_mean':>10} {'mix_mean':>10}")
        print("-" * 95)

        for (dataset, mode) in sorted(d1_map.keys()):
            bl = d1_map[(dataset, mode)]
            mk = (dataset, mode)
            if mk in mixed_map:
                # Average across repetitions
                mix_p99 = sum(m['p99_op'] for m in mixed_map[mk]) / len(mixed_map[mk])
                mix_mean = sum(m['mean_op'] for m in mixed_map[mk]) / len(mixed_map[mk])
                tax_p99 = mix_p99 - bl['p99_op']
                tax_pct = (tax_p99 / bl['p99_op'] * 100) if bl['p99_op'] > 0 else float('inf')
                print(f"{dataset:<10} {mode:<8} {bl['p99_op']:>13.2f} {mix_p99:>10.2f} {tax_p99:>+10.2f} {tax_pct:>+7.1f}% │ {bl['mean_op']:>10.2f} {mix_mean:>10.2f}")
            else:
                print(f"{dataset:<10} {mode:<8} {bl['p99_op']:>13.2f} {'N/A':>10} {'N/A':>10} {'N/A':>8} │ {bl['mean_op']:>10.2f} {'N/A':>10}")

    # ════════════════════════════════════════════
    # Diagnostic #2: Watermark HOL Blocking
    # ════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("DIAGNOSTIC #2: Watermark HOL Blocking")
    print("  MVCC mode, batch size sweep: is contiguity lag superlinear in batch size?")
    print("=" * 80)

    d2_rows = read_results("result/diagnostic_1_2/*/d2_hol_*/benchmark_results.csv")
    # Also include existing fiber_exp mvcc results for comparison
    existing_mvcc = read_results("result/fiber_exp/*/mvcc_b*_w40000_r40000/benchmark_results.csv")

    all_d2 = []
    for r in d2_rows + existing_mvcc:
        path = r['_path']
        parts = path.split('/')
        dataset = parts[2]
        config = parts[3]
        bs = int(config.split('_b')[1].split('_')[0])
        all_d2.append({
            'dataset': dataset,
            'batch_size': bs,
            'source': 'diag2' if 'diagnostic' in path else 'fiber_exp',
            'contig_mean': safe_float(r, 'freshness_contiguity_pts_mean'),
            'contig_p99': safe_float(r, 'freshness_contiguity_pts_p99'),
            'total_mean': safe_float(r, 'freshness_total_pts_mean'),
            'total_p99': safe_float(r, 'freshness_total_pts_p99'),
            'insert_qps': safe_float(r, 'insert_qps (per-point)'),
            'search_p99': safe_float(r, 'search_p99_op_latency_ms (per-batch)'),
        })

    if not all_d2:
        print("\n  [!] No D2 results found. Run the experiment first.")
    else:
        for ds in ['sift', 'deep1M']:
            ds_rows = sorted([r for r in all_d2 if r['dataset'] == ds], key=lambda x: x['batch_size'])
            if not ds_rows:
                continue

            print(f"\n── {ds} (MVCC, balanced 40k/40k) ──")
            print(f"{'batch':>6} {'contig_mean':>12} {'contig_p99':>11} {'total_mean':>11} {'total_p99':>10} {'ratio_mean':>11} {'ratio_p99':>10} {'ins_qps':>9} {'srch_p99':>9}")
            print("-" * 105)

            base_contig = None
            base_total = None
            for r in ds_rows:
                if base_contig is None:
                    base_contig = r['contig_mean'] if r['contig_mean'] > 0 else 1
                    base_total = r['total_mean'] if r['total_mean'] > 0 else 1
                ratio_mean = r['contig_mean'] / base_contig if base_contig > 0 else 0
                ratio_p99 = r['contig_p99'] / (ds_rows[0]['contig_p99'] if ds_rows[0]['contig_p99'] > 0 else 1)
                batch_ratio = r['batch_size'] / ds_rows[0]['batch_size']
                # If HOL blocking matters: ratio_mean >> batch_ratio (superlinear)
                marker = " ← SUPERLINEAR" if ratio_mean > batch_ratio * 1.2 else ""
                print(f"{r['batch_size']:>6} {r['contig_mean']:>12.1f} {r['contig_p99']:>11.0f} {r['total_mean']:>11.1f} {r['total_p99']:>10.0f} {ratio_mean:>11.1f}x {ratio_p99:>9.1f}x {r['insert_qps']:>9.0f} {r['search_p99']:>9.2f}{marker}")

            print(f"\n  Key: ratio_mean = contig_mean / contig_mean(smallest_batch)")
            print(f"  If HOL blocking is real: ratio_mean >> batch_size_ratio (superlinear growth)")
            print(f"  If HOL blocking is negligible: ratio_mean ≈ batch_size_ratio (linear growth)")

    # ════════════════════════════════════════════
    # Verdict
    # ════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print("""
  Diagnostic #1 (Interference Tax):
    If tax_% is large (>50%): structural read isolation (#1) is worth pursuing
    If tax_% is small (<20%): M2 fiber already handles it, skip direction #1

  Diagnostic #2 (HOL Blocking):
    If contiguity lag grows superlinearly with batch size: non-prefix visibility (#2) is worth pursuing
    If growth is linear: HOL blocking is NOT the dominant freshness bottleneck, skip direction #2
""")

if __name__ == "__main__":
    main()
