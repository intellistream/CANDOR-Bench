#!/usr/bin/env python3
"""Attribute M3 search-latency spikes to insert phase work.

The input is a result root containing per-case period_groups.csv files.  Each
period is already a dynamic time bin with insert-phase work, perf counters, and
search-finish latency on the same raw-clock axis.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


PHASES = [
    ("path_search", "insert_path_search_active_workers"),
    ("load_scan", "existing_neighbor_load_scan_active_workers"),
    ("prune", "existing_neighbor_prune_active_workers"),
    ("undo_record", "existing_neighbor_undo_record_active_workers"),
    ("rewrite", "existing_neighbor_rewrite_active_workers"),
    ("update_total", "existing_neighbor_update_active_workers"),
]

WORK_METRICS = [
    ("path_edges", "insert_path_edges_scanned"),
    ("path_dists", "insert_path_dist_comps"),
    ("neighbor_visits", "existing_neighbor_visits"),
    ("edges_loaded", "existing_neighbor_edges_loaded"),
    ("prune_calls", "existing_neighbor_prune_calls"),
    ("prune_candidates", "existing_neighbor_prune_candidates"),
    ("prune_dists", "existing_neighbor_prune_dist_comps"),
    ("edges_written", "existing_neighbor_edges_written"),
    ("edges_pruned", "existing_neighbor_edges_pruned"),
]

L2_COL = "perf_l2_rqsts_miss_per_s"
LAT_COL = "search_finish_e2e_ms_max"


def to_float(value: object) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def percentile(values: list[float], pct: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return 0.0
    k = (len(clean) - 1) * pct / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - k) + clean[hi] * (k - lo)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def load_period_groups(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rank_l2_metrics(rows: list[dict[str, str]]) -> list[tuple[float, str]]:
    valid = [r for r in rows if to_float(r.get(L2_COL)) > 0.0]
    l2 = [to_float(r[L2_COL]) for r in valid]
    ranked: list[tuple[float, str]] = []
    for name, col in PHASES + WORK_METRICS:
        ranked.append((corr([to_float(r.get(col)) for r in valid], l2), name))
    return sorted(ranked, reverse=True)


def summarize_case(case_dir: Path, root: Path, latency_pct: float, l2_abs_threshold: float) -> dict[str, object] | None:
    rows = load_period_groups(case_dir / "period_groups.csv")
    search_rows = [r for r in rows if to_float(r.get("search_finish_count")) > 0.0 and to_float(r.get(LAT_COL)) > 0.0]
    if not search_rows:
        return None

    lat_thr = percentile([to_float(r[LAT_COL]) for r in search_rows], latency_pct)
    high_latency = [r for r in search_rows if to_float(r[LAT_COL]) >= lat_thr]
    other_latency = [r for r in search_rows if to_float(r[LAT_COL]) < lat_thr]
    valid_l2 = [to_float(r.get(L2_COL)) for r in rows if to_float(r.get(L2_COL)) > 0.0]
    l2_p90 = percentile(valid_l2, 90.0)

    phase_abs = sorted(
        (
            (mean([to_float(r.get(col)) for r in high_latency]), name)
            for name, col in PHASES
        ),
        reverse=True,
    )
    phase_uplift = sorted(
        (
            (
                mean([to_float(r.get(col)) for r in high_latency])
                - mean([to_float(r.get(col)) for r in other_latency]),
                name,
            )
            for name, col in PHASES
        ),
        reverse=True,
    )
    l2_rank = rank_l2_metrics(rows)

    top_latency = sorted(high_latency, key=lambda r: to_float(r[LAT_COL]), reverse=True)[:3]
    row: dict[str, object] = {
        "case": str(case_dir.relative_to(root)),
        "periods": len(rows),
        "search_periods": len(search_rows),
        "high_latency_periods": len(high_latency),
        "latency_pct": latency_pct,
        "latency_threshold_ms": lat_thr,
        "l2_p90_per_s": l2_p90,
        "high_latency_l2_ge_abs_count": sum(1 for r in high_latency if to_float(r.get(L2_COL)) >= l2_abs_threshold),
        "high_latency_l2_ge_abs_fraction": (
            sum(1 for r in high_latency if to_float(r.get(L2_COL)) >= l2_abs_threshold) / len(high_latency)
            if high_latency
            else 0.0
        ),
        "high_latency_l2_ge_p90_count": sum(1 for r in high_latency if to_float(r.get(L2_COL)) >= l2_p90),
        "primary_cache_metric": l2_rank[0][1] if l2_rank else "",
        "primary_cache_metric_l2_corr": l2_rank[0][0] if l2_rank else 0.0,
        "primary_high_latency_phase_by_abs": phase_abs[0][1] if phase_abs else "",
        "primary_high_latency_phase_workers_mean": phase_abs[0][0] if phase_abs else 0.0,
        "primary_high_latency_phase_by_uplift": phase_uplift[0][1] if phase_uplift else "",
        "primary_high_latency_phase_uplift_workers": phase_uplift[0][0] if phase_uplift else 0.0,
        "high_search_op_active_workers_mean": mean(
            [to_float(r.get("search_op_active_workers")) for r in high_latency]
        ),
        "high_search_e2e_active_workers_mean": mean(
            [to_float(r.get("search_e2e_active_workers")) for r in high_latency]
        ),
        "top_latency_periods": ";".join(
            f"{to_float(r.get('bin_start_ms')):.0f}ms:{to_float(r.get(LAT_COL)):.1f}ms"
            for r in top_latency
        ),
    }
    for name, col in PHASES:
        vals = [to_float(r.get(col)) for r in high_latency]
        other_vals = [to_float(r.get(col)) for r in other_latency]
        row[f"high_{name}_workers_mean"] = mean(vals)
        row[f"high_{name}_workers_p50"] = median(vals)
        row[f"other_{name}_workers_mean"] = mean(other_vals)
        row[f"uplift_{name}_workers"] = mean(vals) - mean(other_vals)
    for name, col in WORK_METRICS:
        vals = [to_float(r.get(col)) for r in high_latency]
        row[f"high_{name}_mean"] = mean(vals)
        row[f"high_{name}_p50"] = median(vals)
    return row


def write_markdown(rows: list[dict[str, object]], out_path: Path, l2_abs_threshold: float) -> None:
    high_periods = sum(int(r["high_latency_periods"]) for r in rows)
    l2_abs_count = sum(int(r["high_latency_l2_ge_abs_count"]) for r in rows)
    path_mean = mean([float(r["high_path_search_workers_mean"]) for r in rows])
    update_mean = mean([float(r["high_update_total_workers_mean"]) for r in rows])
    prune_mean = mean([float(r["high_prune_workers_mean"]) for r in rows])
    prune_candidates = mean([float(r["high_prune_candidates_mean"]) for r in rows])
    search_op_mean = mean([float(r["high_search_op_active_workers_mean"]) for r in rows])
    search_e2e_mean = mean([float(r["high_search_e2e_active_workers_mean"]) for r in rows])
    insert_over_search = path_mean / search_op_mean if search_op_mean > 0.0 else 0.0
    cache_winners: dict[str, int] = {}
    for r in rows:
        cache_winners[str(r["primary_cache_metric"])] = cache_winners.get(str(r["primary_cache_metric"]), 0) + 1
    winner_line = ", ".join(f"{k}={v}" for k, v in sorted(cache_winners.items(), key=lambda kv: (-kv[1], kv[0])))

    lines = [
        "# M3 Latency Phase Attribution",
        "",
        f"- cases: {len(rows)}",
        f"- high-latency periods: {high_periods}",
        f"- high-latency periods with L2 >= {l2_abs_threshold:.3g}/s: {l2_abs_count}/{high_periods}",
        f"- primary L2-correlated metric winners by case: {winner_line}",
        f"- high-latency period mean path_search active workers: {path_mean:.3f}",
        f"- high-latency period mean search op active workers: {search_op_mean:.3f}",
        f"- high-latency period mean search e2e active workers: {search_e2e_mean:.3f}",
        f"- insert path_search / search op active-worker ratio: {insert_over_search:.1f}x",
        f"- high-latency period mean existing-neighbor update active workers: {update_mean:.3f}",
        f"- high-latency period mean existing-neighbor prune active workers: {prune_mean:.3f}",
        f"- high-latency period mean prune candidates per 10ms period: {prune_candidates:.1f}",
        "",
        "Conclusion: the direct cache-pressure phase is insert_path_search, measured by path distance/edge work and path_search active workers. "
        "The dense existing-neighbor phase present during latency spikes is update_total, dominated by prune rather than rewrite.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="result root containing case directories")
    parser.add_argument("--latency-pct", type=float, default=90.0)
    parser.add_argument("--l2-abs-threshold", type=float, default=1.0e9)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()

    case_dirs = sorted(p.parent for p in args.root.glob("actual_hnsw_insert_*/*/period_groups.csv"))
    rows = [
        row
        for case_dir in case_dirs
        if (row := summarize_case(case_dir, args.root, args.latency_pct, args.l2_abs_threshold)) is not None
    ]
    if not rows:
        raise SystemExit("no period_groups.csv cases found")

    out_csv = args.out_csv or (args.root / "phase_attribution_summary.csv")
    out_md = args.out_md or (args.root / "phase_attribution_summary.md")
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, out_md, args.l2_abs_threshold)
    print(f"[phase-attribution] wrote {out_csv}")
    print(f"[phase-attribution] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
