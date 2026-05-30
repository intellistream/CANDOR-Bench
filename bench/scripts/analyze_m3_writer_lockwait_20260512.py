#!/usr/bin/env python3
"""Analyze writer lock-wait audit results.

Output:
- writer_lockwait_summary.csv with per-N columns:
    n_writers, insert_count,
    op_ms_mean, op_ms_p50, op_ms_p99,
    unique_lock_wait_ms_mean, unique_lock_wait_ms_p99,
    unique_lock_wait_fraction       = sum(unique_lock_wait_ms) / sum(op_ms)
    update_loop_fraction            = sum(update_loop_ms)      / sum(op_ms)
    link_critical_fraction          = sum(link_critical_ms)    / sum(op_ms)
    insert_path_ratio_to_op         = sum(insert_path_ms)      / sum(op_ms)   (CPU/wall ratio)
    accounted_fraction              = (sum of known phase wall times) / sum(op_ms)
    cpu_util_competing              = scaled_cycles / (n_writers * window_s * cpu_GHz * 1e9)
    on_cpu_fraction                 = cpu_util_competing
    off_cpu_fraction                = 1 - on_cpu_fraction
    iqps_inserts_per_sec
    reader_p99_ms (from benchmark_results.csv)
"""
from __future__ import annotations
import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean


CPU_FREQ_GHZ = 4.0  # nominal on Xeon 6418H with performance governor


def parse_perf_csv(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(",")
        if len(parts) < 3:
            continue
        cnt_str, unit, event = parts[0], parts[1], parts[2]
        if cnt_str in {"<not counted>", "<not supported>"}:
            out[event] = float("nan"); continue
        try:
            cnt = float(cnt_str)
        except ValueError:
            continue
        out[event] = cnt
        if len(parts) > 4 and parts[4]:
            try:
                out[f"{event}__run_pct"] = float(parts[4])
            except ValueError:
                pass
    return out


def perf_scaled(perf: dict, event: str) -> float:
    cnt = perf.get(event, float("nan"))
    pct = perf.get(f"{event}__run_pct", 100.0)
    if pct <= 0 or math.isnan(cnt): return float("nan")
    return cnt * 100.0 / pct


def load_meta(p: Path) -> dict:
    out = {}
    if not p.exists(): return out
    for tok in p.read_text().split():
        if "=" in tok:
            k, v = tok.split("=", 1); out[k] = v
    return out


def percentile(xs, p):
    if not xs: return float("nan")
    s = sorted(xs)
    k = (len(s)-1) * p
    f = math.floor(k); c = math.ceil(k)
    if f == c: return s[int(k)]
    return s[f] + (s[c]-s[f]) * (k - f)


def analyze_cell(cell: Path) -> dict:
    meta = load_meta(cell / "cell.meta")
    n_writers = int(meta.get("n_writers", "0"))
    sample_ms = int(meta.get("sample_ms", "0"))
    window_s = sample_ms / 1000.0 if sample_ms else float("nan")

    # bench-side insert_wide.csv
    iw = cell / "raw_latency" / "insert_wide.csv"
    if not iw.exists():
        return {"cell": cell.name, "n_writers": n_writers, "error": "no insert_wide.csv"}

    cols = {
        "op_ms": [], "e2e_ms": [],
        "graph_insert_path_ms": [],
        "graph_existing_neighbor_update_loop_ms": [],
        "graph_existing_neighbor_prune_ms": [],
        "graph_unique_lock_wait_ms": [],
        "graph_link_critical_ms": [],
        "graph_connect_ms": [],
        "graph_inserted_node_link_ms": [],
        "graph_select_new_neighbors_ms": [],
        "graph_existing_neighbor_load_scan_ms": [],
        "graph_existing_neighbor_append_ms": [],
        "graph_existing_neighbor_rewrite_ms": [],
        "graph_existing_neighbor_undo_record_ms": [],
        "graph_search_unique_lock_wait_ms": [],
        "graph_search_unique_lock_acqs": [],
        "graph_search_critical_ms": [],
        "finish_ms": [],
    }
    with open(iw) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            for k in cols.keys():
                if k in row:
                    try:
                        cols[k].append(float(row[k]))
                    except ValueError:
                        cols[k].append(0.0)
    n_inserts = len(cols["op_ms"])
    if n_inserts == 0:
        return {"cell": cell.name, "n_writers": n_writers, "error": "0 inserts"}

    # Filter to inserts that completed during the perf window (skip_pre to skip_pre+sample)
    skip_pre_ms = int(meta.get("skip_pre_ms", "0"))
    win_lo = skip_pre_ms
    win_hi = skip_pre_ms + sample_ms
    in_window = [
        i for i, ms in enumerate(cols["finish_ms"]) if win_lo <= ms <= win_hi
    ]
    if not in_window:
        in_window = list(range(n_inserts))  # fall back to all

    def sub(k):
        return [cols[k][i] for i in in_window]

    op = sub("op_ms")
    lw = sub("graph_unique_lock_wait_ms")
    upd = sub("graph_existing_neighbor_update_loop_ms")
    crit = sub("graph_link_critical_ms")
    isp = sub("graph_insert_path_ms")
    conn = sub("graph_connect_ms")
    ino = sub("graph_inserted_node_link_ms")
    sel = sub("graph_select_new_neighbors_ms")
    pru = sub("graph_existing_neighbor_prune_ms")
    slw = sub("graph_search_unique_lock_wait_ms")
    sac = sub("graph_search_unique_lock_acqs")
    scr = sub("graph_search_critical_ms")

    sum_op = sum(op)
    sum_lw = sum(lw)
    sum_upd = sum(upd)
    sum_crit = sum(crit)
    sum_isp = sum(isp)
    sum_conn = sum(conn)
    sum_ino = sum(ino)
    sum_sel = sum(sel)
    sum_slw = sum(slw)
    sum_sac = sum(sac)
    sum_scr = sum(scr)

    # Accounted wall time = update_loop (lock_wait + critical) + select_new + inserted_node_link
    # (graph_insert_path_ms is CPU time across threads, NOT wall; we don't add it here)
    sum_accounted = sum_upd + sum_sel + sum_ino

    # perf side: cycles on competing CPUs (writer pool)
    pc = parse_perf_csv(cell / "perf_competing.txt")
    cycles_scaled = perf_scaled(pc, "cycles")
    # On-CPU time-equivalent = cycles / freq
    on_cpu_seconds = cycles_scaled / (CPU_FREQ_GHZ * 1e9) if cycles_scaled and not math.isnan(cycles_scaled) else float("nan")
    # Total wall-time CPU-seconds available across N writer CPUs
    wall_cpu_seconds = n_writers * window_s if n_writers > 0 and not math.isnan(window_s) else float("nan")
    on_cpu_fraction = on_cpu_seconds / wall_cpu_seconds if wall_cpu_seconds else float("nan")

    # iQPS observed in the window
    iqps = len(in_window) * 20 / window_s if window_s else float("nan")  # batch_size=20

    # bench reader p99 from benchmark_results.csv
    br = cell / "benchmark_results.csv"
    reader_p99 = float("nan")
    insert_p99 = float("nan")
    if br.exists():
        with open(br) as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try: reader_p99 = float(r.get("search_p99_op_latency_ms (per-batch)", "nan"))
                except (TypeError, ValueError): pass
                try: insert_p99 = float(r.get("insert_p99_op_latency_ms (per-batch)", "nan"))
                except (TypeError, ValueError): pass

    return {
        "cell": cell.name,
        "n_writers": n_writers,
        "insert_count_in_window": len(in_window),
        "perf_window_s": window_s,

        "op_ms_mean": mean(op),
        "op_ms_p50": percentile(op, 0.5),
        "op_ms_p99": percentile(op, 0.99),

        "unique_lock_wait_ms_mean": mean(lw),
        "unique_lock_wait_ms_p99": percentile(lw, 0.99),
        "Mstep_unique_lock_wait_fraction": (sum_lw / sum_op) if sum_op else float("nan"),

        "search_unique_lock_wait_ms_mean": mean(slw),
        "search_unique_lock_wait_ms_p99": percentile(slw, 0.99),
        "search_unique_lock_wait_fraction": (sum_slw / sum_op) if sum_op else float("nan"),

        "search_unique_lock_acqs_per_batch": mean(sac),
        "search_unique_lock_avg_wait_us": (sum_slw / sum_sac * 1000) if sum_sac else float("nan"),

        "search_critical_ms_mean": mean(scr),
        "search_critical_fraction": (sum_scr / sum_op) if sum_op else float("nan"),
        "search_critical_avg_us": (sum_scr / sum_sac * 1000) if sum_sac else float("nan"),

        "update_loop_ms_mean": mean(upd),
        "update_loop_fraction": (sum_upd / sum_op) if sum_op else float("nan"),

        "link_critical_ms_mean": mean(crit),
        "link_critical_fraction": (sum_crit / sum_op) if sum_op else float("nan"),

        "total_lock_wait_fraction": ((sum_lw + sum_slw) / sum_op) if sum_op else float("nan"),
        "total_critical_fraction": ((sum_crit + sum_scr) / sum_op) if sum_op else float("nan"),

        "insert_path_ms_mean": mean(isp),
        "insert_path_ratio_to_op": (sum_isp / sum_op) if sum_op else float("nan"),

        "select_new_neighbors_ms_mean": mean(sel),
        "inserted_node_link_ms_mean": mean(ino),
        "connect_ms_mean": mean(conn) if conn else float("nan"),
        "prune_ms_mean": mean(pru),

        "accounted_wall_fraction": (sum_accounted / sum_op) if sum_op else float("nan"),

        "competing_cycles_scaled": cycles_scaled,
        "competing_on_cpu_seconds": on_cpu_seconds,
        "competing_wall_cpu_seconds": wall_cpu_seconds,
        "competing_on_cpu_fraction": on_cpu_fraction,
        "competing_off_cpu_fraction": (1 - on_cpu_fraction) if on_cpu_fraction is not None and not math.isnan(on_cpu_fraction) else float("nan"),

        "iqps_inserts_per_sec": iqps,
        "iqps_commits_per_sec": iqps,
        "reader_p99_ms": reader_p99,
        "insert_p99_ms": insert_p99,
    }


def classify_perf_report(report_path: Path) -> dict:
    """Classify perf_report_bench.txt symbols into buckets.

    Returns: {bucket_name: pct_summed}
    """
    if not report_path.exists():
        return {}
    bucket_kw = {
        "lock_acquire": ["pthread_mutex_lock", "lll_lock_wait", "futex", "lock_wait", "rwlock", "shared_mutex", "unique_lock", "shared_lock", "mutex"],
        "scheduler": ["__schedule", "schedule", "find_busiest_group", "load_balance", "newidle_balance"],
        "distance_compute": ["L2Sqr", "InnerProduct", "fvec_L2sqr", "compute_dist", "DistanceFunc", "distance", "fstdistfunc", "L2SqrSIMD"],
        "hnsw_search": ["searchBaseLayer", "searchKnn", "BaseLayer", "Layer0", "search_base", "search_layer", "buildBfsLayer"],
        "heuristic_neighbors": ["getNeighborsByHeuristic", "MutuallyConnect", "heuristic", "select"],
        "alloc": ["malloc", "free", "memcpy", "memmove", "calloc", "operator new", "operator delete"],
        "go_runtime": ["runtime.", "runtime/"],
        "kernel": ["[k]", "__do_softirq", "irq", "page_fault"],
    }
    counts = {k: 0.0 for k in bucket_kw}
    counts["other"] = 0.0
    total = 0.0
    for line in report_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"): continue
        # Look for percentage prefix: e.g., "  12.34%  bench  ..."
        try:
            pct_str = s.split()[0].rstrip("%")
            pct = float(pct_str)
        except (IndexError, ValueError):
            continue
        symbol = s.lower()
        bucketed = False
        for bucket, kws in bucket_kw.items():
            if any(kw.lower() in symbol for kw in kws):
                counts[bucket] += pct
                bucketed = True; break
        if not bucketed:
            counts["other"] += pct
        total += pct
    counts["__total__"] = total
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cells_dir = args.root / "cells"
    rows = []
    for d in sorted(cells_dir.iterdir()):
        if not d.is_dir(): continue
        if not (d / "cell.meta").exists(): continue
        rows.append(analyze_cell(d))

    if not rows:
        print("no cells", file=sys.stderr); return 1
    keys = list(rows[0].keys())
    for r in rows:
        for k in r: 
            if k not in keys: keys.append(k)
    with open(args.out, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {args.out}")

    print("\n=== Writer lock-wait audit (per-batch aggregate ms, summed across TBB threads) ===")
    print(f"{'N':>4} {'op_ms':>7} {'M-LW%':>6} {'M-CR%':>6} {'S-LW%':>7} {'S-CR%':>7} {'S-acqs':>8} {'S-wait_µs':>10} {'S-crit_µs':>10} {'CPU%':>6} {'iQPS':>6} {'rdP99':>6}")
    print("-" * 110)
    for r in rows:
        if "error" in r: print(f"{r.get('n_writers','?'):>4} {r['error']}"); continue
        print(f"{r['n_writers']:>4d} "
              f"{r['op_ms_mean']:>7.2f} "
              f"{r['Mstep_unique_lock_wait_fraction']*100:>5.1f}% "
              f"{r['link_critical_fraction']*100:>5.1f}% "
              f"{r['search_unique_lock_wait_fraction']*100:>6.1f}% "
              f"{r['search_critical_fraction']*100:>6.1f}% "
              f"{r['search_unique_lock_acqs_per_batch']:>8.0f} "
              f"{r['search_unique_lock_avg_wait_us']:>10.2f} "
              f"{r['search_critical_avg_us']:>10.2f} "
              f"{r['competing_on_cpu_fraction']*100:>5.2f}% "
              f"{r['iqps_inserts_per_sec']:>6.0f} "
              f"{r['reader_p99_ms']:>6.3f}")
    print()
    print("M-LW% = M-step unique_lock wait fraction of op_ms (line 1430 - existing neighbor update)")
    print("S-LW% = SEARCH unique_lock wait fraction of op_ms (line 812 - per-visit unique_lock during insert search)")
    print("Fractions >100% indicate per-batch counters are summed across TBB worker threads doing the batch in parallel.")
    # Classify perf report for the N=16 cell if present
    pr = cells_dir / "insert_n16" / "perf_report_bench.txt"
    if pr.exists():
        print(f"\n=== perf report symbol classification for {pr.parent.name} ===")
        cats = classify_perf_report(pr)
        total = cats.pop("__total__", 0.0)
        for bucket, pct in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"  {bucket:>20s}: {pct:6.2f}%")
        print(f"  {'(total in report)':>20s}: {total:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
