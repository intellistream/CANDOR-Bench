#!/usr/bin/env python3
"""Analyze mixed read/write M3 search tails with ANNS path locality fields."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean


INSERT_WINDOWS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]
STABLE_OUTSIDE = [(3500.0, 6000.0), (7000.0, 9500.0), (10500.0, 12000.0)]
POST_START_MS = 2500.0


def percentile(values: list[float], q: float) -> float:
    xs = sorted(v for v in values if math.isfinite(v))
    if not xs:
        return math.nan
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return fmean(vals) if vals else math.nan


def ratio(num: float, den: float) -> float:
    if not den or not math.isfinite(num) or not math.isfinite(den):
        return math.nan
    return num / den


def in_ranges(ms: float, ranges: list[tuple[float, float]]) -> bool:
    return any(start <= ms < end for start, end in ranges)


def row_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def find_case_dirs(root: Path) -> list[Path]:
    if (root / "raw_latency" / "search_wide.csv").exists():
        return [root]
    return sorted(p.parents[1] for p in root.glob("**/raw_latency/search_wide.csv"))


def load_batches(case_dir: Path) -> list[dict[str, float]]:
    path = case_dir / "raw_latency" / "search_wide.csv"
    batches: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("measured") != "1":
                continue
            key = (row["start_raw_ns"], row["finish_raw_ns"])
            item = batches.setdefault(
                key,
                {
                    "finish_ms": row_float(row, "finish_ms"),
                    "op_ms": row_float(row, "op_ms"),
                    "rows": 0.0,
                    "thread_cpu_ms": 0.0,
                    "searchknn_ms": 0.0,
                    "base_ms": 0.0,
                    "dist": 0.0,
                    "edges": 0.0,
                    "l0_lock_ms": 0.0,
                    "queue_pts": 0.0,
                    "active_front_pts": 0.0,
                    "path_count": 0.0,
                    "path_span_sum": 0.0,
                    "path_gap_sum": 0.0,
                    "path_unique4k_sum": 0.0,
                    "path_min_age_sum": 0.0,
                    "path_mean_age_sum": 0.0,
                    "path_recent1k": 0.0,
                    "path_recent4k": 0.0,
                    "path_recent16k": 0.0,
                    "work_rewrite_active": 0.0,
                    "work_rewrite_recent": 0.0,
                    "work_rewrite_period": 0.0,
                    "work_rewrite_period_active_sum": 0.0,
                    "work_rewrite_period_active_max": 0.0,
                    "expand_visible": 0.0,
                    "expand_recent1k": 0.0,
                    "expand_recent4k": 0.0,
                    "expand_recent16k": 0.0,
                    "expand_label_gap_sum": 0.0,
                    "expand_label_span_sum": 0.0,
                    "expand_unique_label4k_sum": 0.0,
                    "expand_unique_data4k_sum": 0.0,
                    "expand_unique_data2m_sum": 0.0,
                    "expand_unique_adj4k_sum": 0.0,
                    "expand_unique_adj2m_sum": 0.0,
                    "expand_unique_overflow": 0.0,
                    "phase_link_critical": 0.0,
                    "phase_load_scan": 0.0,
                    "phase_append": 0.0,
                    "phase_prune": 0.0,
                    "phase_rewrite": 0.0,
                },
            )
            count = row_float(row, "work_path_count")
            item["rows"] += 1.0
            item["thread_cpu_ms"] += row_float(row, "work_searchknn_thread_cpu_ms")
            item["searchknn_ms"] += row_float(row, "work_searchknn_ms")
            item["base_ms"] += row_float(row, "work_base_search_ms")
            item["dist"] += row_float(row, "work_distance_computations")
            item["edges"] += row_float(row, "work_level0_edges_scanned")
            item["l0_lock_ms"] += row_float(row, "work_level0_lock_wait_ms")
            item["queue_pts"] += row_float(row, "freshness_queue_pts")
            item["active_front_pts"] += row_float(row, "freshness_active_front_pts")
            item["path_count"] += count
            item["path_span_sum"] += row_float(row, "work_path_label_span")
            item["path_gap_sum"] += row_float(row, "work_path_mean_abs_gap")
            item["path_unique4k_sum"] += row_float(row, "work_path_unique_4k_buckets")
            item["path_min_age_sum"] += row_float(row, "work_path_min_visible_age")
            item["path_mean_age_sum"] += row_float(row, "work_path_mean_visible_age")
            item["path_recent1k"] += row_float(row, "work_path_recent_1k_hits")
            item["path_recent4k"] += row_float(row, "work_path_recent_4k_hits")
            item["path_recent16k"] += row_float(row, "work_path_recent_16k_hits")
            item["work_rewrite_active"] += row_float(row, "work_rewrite_active_expansions")
            item["work_rewrite_recent"] += row_float(row, "work_rewrite_recent_expansions")
            item["work_rewrite_period"] += row_float(row, "work_rewrite_period_expansions")
            item["work_rewrite_period_active_sum"] += row_float(row, "work_rewrite_period_active_sum")
            item["work_rewrite_period_active_max"] = max(
                item["work_rewrite_period_active_max"],
                row_float(row, "work_rewrite_period_active_max"),
            )
            item["expand_visible"] += row_float(row, "work_expand_visible_count")
            item["expand_recent1k"] += row_float(row, "work_expand_recent_1k_hits")
            item["expand_recent4k"] += row_float(row, "work_expand_recent_4k_hits")
            item["expand_recent16k"] += row_float(row, "work_expand_recent_16k_hits")
            item["expand_label_gap_sum"] += row_float(row, "work_expand_label_gap_sum")
            item["expand_label_span_sum"] += row_float(row, "work_expand_label_span")
            item["expand_unique_label4k_sum"] += row_float(row, "work_expand_unique_label_4k_buckets")
            item["expand_unique_data4k_sum"] += row_float(row, "work_expand_unique_data_4k_pages")
            item["expand_unique_data2m_sum"] += row_float(row, "work_expand_unique_data_2m_pages")
            item["expand_unique_adj4k_sum"] += row_float(row, "work_expand_unique_adj_4k_pages")
            item["expand_unique_adj2m_sum"] += row_float(row, "work_expand_unique_adj_2m_pages")
            item["expand_unique_overflow"] += row_float(row, "work_expand_unique_overflow")
            item["phase_link_critical"] += row_float(row, "diag_phase_link_critical_samples")
            item["phase_load_scan"] += row_float(row, "diag_phase_load_scan_samples")
            item["phase_append"] += row_float(row, "diag_phase_append_samples")
            item["phase_prune"] += row_float(row, "diag_phase_prune_samples")
            item["phase_rewrite"] += row_float(row, "diag_phase_rewrite_samples")

    out: list[dict[str, float]] = []
    for item in batches.values():
        rows = max(item["rows"], 1.0)
        path_count = item["path_count"]
        item["path_span_mean"] = item["path_span_sum"] / rows
        item["path_gap_mean"] = item["path_gap_sum"] / rows
        item["path_unique4k_mean"] = item["path_unique4k_sum"] / rows
        item["path_min_age_mean"] = item["path_min_age_sum"] / rows
        item["path_mean_age_mean"] = item["path_mean_age_sum"] / rows
        item["path_recent1k_frac"] = item["path_recent1k"] / path_count if path_count else 0.0
        item["path_recent4k_frac"] = item["path_recent4k"] / path_count if path_count else 0.0
        item["path_recent16k_frac"] = item["path_recent16k"] / path_count if path_count else 0.0
        item["work_rewrite_recent_per_path"] = (
            item["work_rewrite_recent"] / path_count if path_count else 0.0
        )
        item["work_rewrite_period_per_path"] = (
            item["work_rewrite_period"] / path_count if path_count else 0.0
        )
        expand_visible = item["expand_visible"]
        item["expand_recent1k_frac"] = (
            item["expand_recent1k"] / expand_visible if expand_visible else 0.0
        )
        item["expand_recent4k_frac"] = (
            item["expand_recent4k"] / expand_visible if expand_visible else 0.0
        )
        item["expand_recent16k_frac"] = (
            item["expand_recent16k"] / expand_visible if expand_visible else 0.0
        )
        item["expand_label_gap_per_exp"] = (
            item["expand_label_gap_sum"] / expand_visible if expand_visible else 0.0
        )
        item["expand_label_span_mean"] = item["expand_label_span_sum"] / rows
        item["expand_unique_label4k_mean"] = item["expand_unique_label4k_sum"] / rows
        item["expand_unique_data4k_mean"] = item["expand_unique_data4k_sum"] / rows
        item["expand_unique_data2m_mean"] = item["expand_unique_data2m_sum"] / rows
        item["expand_unique_adj4k_mean"] = item["expand_unique_adj4k_sum"] / rows
        item["expand_unique_adj2m_mean"] = item["expand_unique_adj2m_sum"] / rows
        item["expand_unique_data4k_per_exp"] = (
            item["expand_unique_data4k_sum"] / expand_visible if expand_visible else 0.0
        )
        item["expand_unique_data2m_per_exp"] = (
            item["expand_unique_data2m_sum"] / expand_visible if expand_visible else 0.0
        )
        item["expand_unique_adj4k_per_exp"] = (
            item["expand_unique_adj4k_sum"] / expand_visible if expand_visible else 0.0
        )
        item["expand_unique_adj2m_per_exp"] = (
            item["expand_unique_adj2m_sum"] / expand_visible if expand_visible else 0.0
        )
        out.append(item)
    return sorted(out, key=lambda b: b["finish_ms"])


def mean(items: list[dict[str, float]], key: str) -> float:
    return finite_mean([b[key] for b in items])


def summarize(case_dir: Path) -> dict[str, object]:
    batches = load_batches(case_dir)
    post = [b for b in batches if b["finish_ms"] >= POST_START_MS]
    insert = [b for b in post if in_ranges(b["finish_ms"], INSERT_WINDOWS)]
    outside = [b for b in post if in_ranges(b["finish_ms"], STABLE_OUTSIDE)]
    p99 = percentile([b["op_ms"] for b in post], 0.99)
    top = [b for b in post if b["op_ms"] >= p99]
    rest = [b for b in post if b["op_ms"] < p99]

    def top_ratio(key: str) -> float:
        return ratio(mean(top, key), mean(rest, key))

    insert_tail_rate = 100.0 * sum(1 for b in insert if b["op_ms"] >= p99) / max(1, len(insert))
    outside_tail_rate = 100.0 * sum(1 for b in outside if b["op_ms"] >= p99) / max(1, len(outside))
    return {
        "case_dir": str(case_dir),
        "post_batches": len(post),
        "insert_batches": len(insert),
        "outside_batches": len(outside),
        "post_p50_ms": percentile([b["op_ms"] for b in post], 0.50),
        "post_p95_ms": percentile([b["op_ms"] for b in post], 0.95),
        "post_p99_ms": p99,
        "insert_p99_ms": percentile([b["op_ms"] for b in insert], 0.99),
        "outside_p99_ms": percentile([b["op_ms"] for b in outside], 0.99),
        "post_insert_batch_pct": 100.0 * len(insert) / max(1, len(post)),
        "top1_insert_pct": 100.0 * sum(1 for b in top if in_ranges(b["finish_ms"], INSERT_WINDOWS)) / max(1, len(top)),
        "insert_tail_rate_pct": insert_tail_rate,
        "outside_tail_rate_pct": outside_tail_rate,
        "insert_tail_rate_enrichment": ratio(insert_tail_rate, outside_tail_rate),
        "top1_thread_cpu_ratio": top_ratio("thread_cpu_ms"),
        "top1_searchknn_wall_ratio": top_ratio("searchknn_ms"),
        "top1_base_ms_ratio": top_ratio("base_ms"),
        "top1_dist_ratio": top_ratio("dist"),
        "top1_edges_ratio": top_ratio("edges"),
        "top1_l0_lock_ratio": top_ratio("l0_lock_ms"),
        "top1_queue_pts_ratio": top_ratio("queue_pts"),
        "top1_active_front_pts_ratio": top_ratio("active_front_pts"),
        "top1_path_count_ratio": top_ratio("path_count"),
        "top1_path_span_ratio": top_ratio("path_span_mean"),
        "top1_path_gap_ratio": top_ratio("path_gap_mean"),
        "top1_path_unique4k_ratio": top_ratio("path_unique4k_mean"),
        "top1_path_recent1k_frac_ratio": top_ratio("path_recent1k_frac"),
        "top1_path_recent4k_frac_ratio": top_ratio("path_recent4k_frac"),
        "top1_path_recent16k_frac_ratio": top_ratio("path_recent16k_frac"),
        "top1_work_rewrite_active_ratio": top_ratio("work_rewrite_active"),
        "top1_work_rewrite_recent_ratio": top_ratio("work_rewrite_recent"),
        "top1_work_rewrite_period_ratio": top_ratio("work_rewrite_period"),
        "top1_work_rewrite_period_active_sum_ratio": top_ratio("work_rewrite_period_active_sum"),
        "top1_work_rewrite_period_active_max_ratio": top_ratio("work_rewrite_period_active_max"),
        "top1_work_rewrite_recent_per_path_ratio": top_ratio("work_rewrite_recent_per_path"),
        "top1_work_rewrite_period_per_path_ratio": top_ratio("work_rewrite_period_per_path"),
        "top1_expand_visible_ratio": top_ratio("expand_visible"),
        "top1_expand_recent1k_frac_ratio": top_ratio("expand_recent1k_frac"),
        "top1_expand_recent4k_frac_ratio": top_ratio("expand_recent4k_frac"),
        "top1_expand_recent16k_frac_ratio": top_ratio("expand_recent16k_frac"),
        "top1_expand_label_gap_per_exp_ratio": top_ratio("expand_label_gap_per_exp"),
        "top1_expand_label_span_ratio": top_ratio("expand_label_span_mean"),
        "top1_expand_unique_label4k_ratio": top_ratio("expand_unique_label4k_mean"),
        "top1_expand_unique_data4k_ratio": top_ratio("expand_unique_data4k_mean"),
        "top1_expand_unique_data2m_ratio": top_ratio("expand_unique_data2m_mean"),
        "top1_expand_unique_adj4k_ratio": top_ratio("expand_unique_adj4k_mean"),
        "top1_expand_unique_adj2m_ratio": top_ratio("expand_unique_adj2m_mean"),
        "top1_expand_unique_data4k_per_exp_ratio": top_ratio("expand_unique_data4k_per_exp"),
        "top1_expand_unique_data2m_per_exp_ratio": top_ratio("expand_unique_data2m_per_exp"),
        "top1_expand_unique_adj4k_per_exp_ratio": top_ratio("expand_unique_adj4k_per_exp"),
        "top1_expand_unique_adj2m_per_exp_ratio": top_ratio("expand_unique_adj2m_per_exp"),
        "top1_path_min_age_mean": mean(top, "path_min_age_mean"),
        "rest_path_min_age_mean": mean(rest, "path_min_age_mean"),
        "top1_phase_link_critical_ratio": top_ratio("phase_link_critical"),
        "top1_phase_load_scan_ratio": top_ratio("phase_load_scan"),
        "top1_phase_append_ratio": top_ratio("phase_append"),
        "top1_phase_prune_ratio": top_ratio("phase_prune"),
        "top1_phase_rewrite_ratio": top_ratio("phase_rewrite"),
        "insert_thread_cpu_mean_ms": mean(insert, "thread_cpu_ms"),
        "outside_thread_cpu_mean_ms": mean(outside, "thread_cpu_ms"),
        "insert_path_recent16k_frac": mean(insert, "path_recent16k_frac"),
        "outside_path_recent16k_frac": mean(outside, "path_recent16k_frac"),
        "top1_work_rewrite_recent_mean": mean(top, "work_rewrite_recent"),
        "rest_work_rewrite_recent_mean": mean(rest, "work_rewrite_recent"),
        "top1_work_rewrite_period_mean": mean(top, "work_rewrite_period"),
        "rest_work_rewrite_period_mean": mean(rest, "work_rewrite_period"),
        "top1_expand_unique_data4k_mean": mean(top, "expand_unique_data4k_mean"),
        "rest_expand_unique_data4k_mean": mean(rest, "expand_unique_data4k_mean"),
        "top1_expand_unique_adj4k_mean": mean(top, "expand_unique_adj4k_mean"),
        "rest_expand_unique_adj4k_mean": mean(rest, "expand_unique_adj4k_mean"),
        "top1_expand_label_gap_per_exp_mean": mean(top, "expand_label_gap_per_exp"),
        "rest_expand_label_gap_per_exp_mean": mean(rest, "expand_label_gap_per_exp"),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Run root or case dir containing raw_latency/search_wide.csv")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = [summarize(case_dir) for case_dir in find_case_dirs(args.root)]
    if not rows:
        raise SystemExit(f"no search_wide.csv found under {args.root}")

    out = args.out or (args.root / "mixed_path_locality_summary.csv")
    write_csv(out, rows)
    for row in rows:
        print(
            f"{row['case_dir']}: post_p99={row['post_p99_ms']:.3f}ms "
            f"insert_p99={row['insert_p99_ms']:.3f}ms outside_p99={row['outside_p99_ms']:.3f}ms "
            f"top1_insert={row['top1_insert_pct']:.1f}% base/dist/thread="
            f"{row['top1_base_ms_ratio']:.2f}/{row['top1_dist_ratio']:.2f}/"
            f"{row['top1_thread_cpu_ratio']:.2f} path_recent16k_ratio="
            f"{row['top1_path_recent16k_frac_ratio']:.2f} rewrite_recent_ratio="
            f"{row['top1_work_rewrite_recent_ratio']:.2f} rewrite_period_ratio="
            f"{row['top1_work_rewrite_period_ratio']:.2f} data4k_ratio="
            f"{row['top1_expand_unique_data4k_ratio']:.2f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
