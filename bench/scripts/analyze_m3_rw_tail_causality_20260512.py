#!/usr/bin/env python3
"""Analyze M3 read-write tail causality runs.

This analyzer is intentionally stricter than the older rw-tail script:
- uses per-cell rows, not only mean-of-P99;
- filters samples to the perf window when timestamps exist;
- reports median/IQR across reps;
- emits causal contrasts that subtract generic controls from real inserts;
- flags insert exhaustion and low sample counts.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # New runner writes one key=value per line. Older M3 runners wrote all
        # key=value pairs on a single whitespace-separated line.
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def to_float(value: str | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def percentile(values: list[float], q: float) -> float:
    xs = sorted(x for x in values if not math.isnan(x))
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def median(values: list[float]) -> float:
    xs = [x for x in values if not math.isnan(x)]
    return statistics.median(xs) if xs else float("nan")


def mean(values: list[float]) -> float:
    xs = [x for x in values if not math.isnan(x)]
    return statistics.mean(xs) if xs else float("nan")


def stdev(values: list[float]) -> float:
    xs = [x for x in values if not math.isnan(x)]
    return statistics.stdev(xs) if len(xs) >= 2 else float("nan")


def iqr(values: list[float]) -> tuple[float, float]:
    xs = [x for x in values if not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    return percentile(xs, 0.25), percentile(xs, 0.75)


def parse_perf_csv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(",")
        if len(parts) < 3:
            continue
        event = parts[2]
        try:
            count = float(parts[0])
        except ValueError:
            continue
        out[event] = count
        if len(parts) > 4:
            try:
                out[f"{event}__run_pct"] = float(parts[4])
            except ValueError:
                pass
    return out


def scaled_perf(perf: dict[str, float], event: str) -> float:
    count = perf.get(event, float("nan"))
    pct = perf.get(f"{event}__run_pct", 100.0)
    if math.isnan(count) or pct <= 0:
        return float("nan")
    return count * 100.0 / pct


def load_search_batches(cell: Path, skip_ms: float, sample_ms: float, filter_window: bool) -> list[float]:
    """Return measured search op latencies, deduplicated by batch when possible."""
    if not filter_window:
        op = cell / "raw_latency" / "search_op_ms.csv"
        values = []
        if op.exists():
            with open(op) as f:
                rdr = csv.reader(f)
                next(rdr, None)
                for row in rdr:
                    if row:
                        values.append(to_float(row[0]))
        return [v for v in values if not math.isnan(v)]

    # search_compact has the timing columns needed for window filtering without
    # the wide per-query telemetry payload. Fall back to search_wide if needed.
    wide = cell / "raw_latency" / "search_compact.csv"
    if not wide.exists():
        wide = cell / "raw_latency" / "search_wide.csv"
    window_lo = skip_ms
    window_hi = skip_ms + sample_ms
    if wide.exists():
        seen: set[tuple[str, str, str]] = set()
        values: list[float] = []
        with open(wide) as f:
            for row in csv.DictReader(f):
                if row.get("measured", "1") not in {"1", "true", "True"}:
                    continue
                finish_ms = to_float(row.get("finish_ms"))
                if not math.isnan(finish_ms) and not (window_lo <= finish_ms <= window_hi):
                    continue
                key = (
                    row.get("create_raw_ns", ""),
                    row.get("start_raw_ns", ""),
                    row.get("finish_raw_ns", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                values.append(to_float(row.get("op_ms")))
        return [v for v in values if not math.isnan(v)]

    op = cell / "raw_latency" / "search_op_ms.csv"
    values = []
    if op.exists():
        with open(op) as f:
            rdr = csv.reader(f)
            next(rdr, None)
            for row in rdr:
                if row:
                    values.append(to_float(row[0]))
    return [v for v in values if not math.isnan(v)]


def load_insert_rows(cell: Path, skip_ms: float, sample_ms: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    path = cell / "raw_latency" / "insert_wide.csv"
    all_rows: list[dict[str, str]] = []
    win_rows: list[dict[str, str]] = []
    if not path.exists():
        return all_rows, win_rows
    lo = skip_ms
    hi = skip_ms + sample_ms
    with open(path) as f:
        for row in csv.DictReader(f):
            all_rows.append(row)
            finish_ms = to_float(row.get("finish_ms"))
            if not math.isnan(finish_ms) and lo <= finish_ms <= hi:
                win_rows.append(row)
    return all_rows, win_rows


def sum_col(rows: list[dict[str, str]], col: str) -> float:
    return sum(to_float(r.get(col), 0.0) for r in rows)


def avg_col(rows: list[dict[str, str]], col: str) -> float:
    xs = [to_float(r.get(col)) for r in rows]
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.mean(xs) if xs else float("nan")


def analyze_cell(cell: Path, filter_window: bool) -> dict[str, object]:
    meta = load_meta(cell / "cell.meta")
    skip_ms = to_float(meta.get("skip_pre_ms"), 0.0)
    sample_ms = to_float(meta.get("sample_ms"), 0.0)
    sample_s = sample_ms / 1000.0 if sample_ms > 0 else float("nan")
    batch_size = int(float(meta.get("batch_size", "20")))
    mode = meta.get("mode", cell.name.split("_n", 1)[0])
    if "variant" in meta and mode == cell.name.split("_n", 1)[0]:
        mode = meta["variant"]
    n = int(float(meta.get("n_competing", meta.get("n_writers", "0"))))
    rep = int(float(meta.get("rep", "0")))

    search_ms = load_search_batches(cell, skip_ms, sample_ms, filter_window)
    all_insert, win_insert = load_insert_rows(cell, skip_ms, sample_ms)

    max_insert_finish = max([to_float(r.get("finish_ms")) for r in all_insert], default=float("nan"))
    window_end = skip_ms + sample_ms
    insert_mode = mode.startswith("insert")
    exhausted_early = bool(insert_mode and all_insert and max_insert_finish < window_end - 1000.0)
    zero_insert_in_window = bool(insert_mode and not win_insert)

    sum_insert_op = sum_col(win_insert, "op_ms")
    sum_search_lw = sum_col(win_insert, "graph_search_unique_lock_wait_ms")
    sum_search_acqs = sum_col(win_insert, "graph_search_unique_lock_acqs")
    sum_search_crit = sum_col(win_insert, "graph_search_critical_ms")
    sum_mstep_lw = sum_col(win_insert, "graph_unique_lock_wait_ms")
    sum_link_crit = sum_col(win_insert, "graph_link_critical_ms")
    sum_mstep_loop = sum_col(win_insert, "graph_existing_neighbor_update_loop_ms")
    sum_batch_wall = sum_col(win_insert, "batch_insert_wall_ms")

    reader_perf = parse_perf_csv(cell / "perf_reader.txt")
    competing_perf = parse_perf_csv(cell / "perf_competing.txt")
    uncore_perf = parse_perf_csv(cell / "perf_uncore.txt")
    reader_llc_miss = scaled_perf(reader_perf, "LLC-load-misses")
    competing_llc_miss = scaled_perf(competing_perf, "LLC-load-misses")

    return {
        "cell": cell.name,
        "mode": mode,
        "n": n,
        "rep": rep,
        "dataset_name": meta.get("dataset_name", ""),
        "lock_variant": meta.get("lock_variant", ""),
        "search_batch_samples": len(search_ms),
        "search_p50_ms": percentile(search_ms, 0.50),
        "search_p95_ms": percentile(search_ms, 0.95),
        "search_p99_ms": percentile(search_ms, 0.99),
        "search_p999_ms": percentile(search_ms, 0.999),
        "search_max_ms": max(search_ms) if search_ms else float("nan"),
        "insert_batches_window": len(win_insert),
        "insert_batches_total": len(all_insert),
        "insert_iqps_vec_s": (len(win_insert) * batch_size / sample_s) if sample_s and not math.isnan(sample_s) else float("nan"),
        "insert_max_finish_ms": max_insert_finish,
        "insert_exhausted_early": exhausted_early,
        "insert_zero_in_window": zero_insert_in_window,
        "insert_op_ms_mean": avg_col(win_insert, "op_ms"),
        "batch_insert_wall_ms_mean": avg_col(win_insert, "batch_insert_wall_ms"),
        "search_lock_wait_fraction_of_insert_op": (sum_search_lw / sum_insert_op) if sum_insert_op else float("nan"),
        "search_lock_avg_wait_us": (sum_search_lw / sum_search_acqs * 1000.0) if sum_search_acqs else float("nan"),
        "search_lock_acqs_per_insert_batch": avg_col(win_insert, "graph_search_unique_lock_acqs"),
        "search_critical_fraction_of_insert_op": (sum_search_crit / sum_insert_op) if sum_insert_op else float("nan"),
        "mstep_lock_wait_fraction_of_insert_op": (sum_mstep_lw / sum_insert_op) if sum_insert_op else float("nan"),
        "mstep_link_critical_fraction_of_insert_op": (sum_link_crit / sum_insert_op) if sum_insert_op else float("nan"),
        "mstep_update_loop_fraction_of_insert_op": (sum_mstep_loop / sum_insert_op) if sum_insert_op else float("nan"),
        "batch_insert_wall_fraction_of_insert_op": (sum_batch_wall / sum_insert_op) if sum_insert_op else float("nan"),
        "reader_llc_load_misses": reader_llc_miss,
        "competing_llc_load_misses": competing_llc_miss,
        "uncore_imc_reads": scaled_perf(uncore_perf, "uncore_imc/cas_count_read/"),
        "uncore_imc_writes": scaled_perf(uncore_perf, "uncore_imc/cas_count_write/"),
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("lock_variant", "")), str(row["mode"]), int(row["n"]))].append(row)
    out: list[dict[str, object]] = []
    for (lock_variant, mode, n), rs in sorted(groups.items(), key=lambda x: (x[0][2], x[0][0], x[0][1])):
        p99 = [float(r["search_p99_ms"]) for r in rs]
        p999 = [float(r["search_p999_ms"]) for r in rs]
        p50 = [float(r["search_p50_ms"]) for r in rs]
        q1, q3 = iqr(p99)
        out.append({
            "lock_variant": lock_variant,
            "mode": mode,
            "n": n,
            "reps": len(rs),
            "bad_cells": sum(1 for r in rs if r.get("insert_exhausted_early") or r.get("insert_zero_in_window") or int(r.get("search_batch_samples", 0)) < 100),
            "search_p50_median_ms": median(p50),
            "search_p99_median_ms": median(p99),
            "search_p99_mean_ms": mean(p99),
            "search_p99_stdev_ms": stdev(p99),
            "search_p99_iqr_lo_ms": q1,
            "search_p99_iqr_hi_ms": q3,
            "search_p999_median_ms": median(p999),
            "insert_iqps_median_vec_s": median([float(r["insert_iqps_vec_s"]) for r in rs]),
            "search_lock_wait_fraction_median": median([float(r["search_lock_wait_fraction_of_insert_op"]) for r in rs]),
            "mstep_lock_wait_fraction_median": median([float(r["mstep_lock_wait_fraction_of_insert_op"]) for r in rs]),
            "mstep_link_critical_fraction_median": median([float(r["mstep_link_critical_fraction_of_insert_op"]) for r in rs]),
            "reader_llc_misses_median": median([float(r["reader_llc_load_misses"]) for r in rs]),
            "competing_llc_misses_median": median([float(r["competing_llc_load_misses"]) for r in rs]),
        })
    return out


def build_contrasts(agg_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(str(r.get("lock_variant", "")), str(r["mode"]), int(r["n"])): r for r in agg_rows}
    out: list[dict[str, object]] = []
    variants = sorted({variant for (variant, _mode, _n) in by_key.keys()})
    ns = sorted({n for (_variant, _mode, n) in by_key.keys() if n > 0})
    for variant in variants:
        for n in ns:
            real = by_key.get((variant, "insert_real", n))
            if not real:
                continue
            real_p99 = float(real["search_p99_median_ms"])
            for control in ["spin", "bg_search"]:
                ctrl = by_key.get((variant, control, n))
                if not ctrl:
                    continue
                ctrl_p99 = float(ctrl["search_p99_median_ms"])
                delta = real_p99 - ctrl_p99
                ratio = real_p99 / ctrl_p99 if ctrl_p99 and not math.isnan(ctrl_p99) else float("nan")
                out.append({
                    "lock_variant": variant,
                    "n": n,
                    "claim": f"insert_real_vs_{control}",
                    "insert_real_p99_median_ms": real_p99,
                    "control_p99_median_ms": ctrl_p99,
                    "delta_p99_ms": delta,
                    "ratio": ratio,
                    "interpretation": interpret(control, delta, ratio),
                })
    return out


def interpret(control: str, delta: float, ratio: float) -> str:
    if math.isnan(delta) or math.isnan(ratio):
        return "missing"
    if control == "spin":
        return "above CPU-only load" if delta > 0 else "not above CPU-only load"
    if control == "bg_search":
        return "above read-only graph traversal" if delta > 0 else "not above read-only graph traversal"
    return "positive residual" if delta > 0 else "no positive residual"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--no-window-filter",
        action="store_true",
        help="Use search_op_ms.csv directly. Faster, but does not restrict samples to the perf window.",
    )
    args = parser.parse_args()

    root = args.root
    out_dir = args.out_dir or root
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cell in sorted(root.iterdir()):
        if not cell.is_dir() or not (cell / "cell.meta").exists():
            continue
        rows.append(analyze_cell(cell, filter_window=not args.no_window_filter))
    if not rows:
        raise SystemExit("no cells found")

    agg_rows = aggregate(rows)
    contrasts = build_contrasts(agg_rows)

    write_csv(out_dir / "cell_summary.csv", rows)
    write_csv(out_dir / "aggregate_summary.csv", agg_rows)
    write_csv(out_dir / "causal_contrasts.csv", contrasts)

    print(f"wrote {out_dir / 'cell_summary.csv'}")
    print(f"wrote {out_dir / 'aggregate_summary.csv'}")
    print(f"wrote {out_dir / 'causal_contrasts.csv'}")
    print()
    print("variant mode                 N reps bad   p99_med  p99_iqr        p999_med  iqps_med")
    print("-" * 96)
    for r in agg_rows:
        print(
            f"{str(r.get('lock_variant','')):<7s} {str(r['mode']):<20s} {int(r['n']):>3d} "
            f"{int(r['reps']):>4d} {int(r['bad_cells']):>3d} "
            f"{float(r['search_p99_median_ms']):>9.3f} "
            f"[{float(r['search_p99_iqr_lo_ms']):>7.3f},{float(r['search_p99_iqr_hi_ms']):>7.3f}] "
            f"{float(r['search_p999_median_ms']):>9.3f} "
            f"{float(r['insert_iqps_median_vec_s']):>9.0f}"
        )
    if contrasts:
        print()
        print("Causal contrasts: positive delta means real insert is worse than the control.")
        for r in contrasts:
            print(
                f"N={int(r['n']):>3d} {r['claim']:<34s} "
                f"variant={str(r.get('lock_variant','')):<7s} "
                f"delta={float(r['delta_p99_ms']):>8.3f}ms "
                f"ratio={float(r['ratio']):>6.2f}  {r['interpretation']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
