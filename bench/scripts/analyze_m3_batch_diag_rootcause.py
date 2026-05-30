#!/usr/bin/env python3
"""Correlate true search batch latency with ANNCHOR-MVCC per-batch work counters."""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path


DIAG_COLUMNS = [
    "diag_dist_comps",
    "diag_hops",
    "diag_future_skip_queries",
    "diag_future_skip_hops",
    "diag_recovery_triggers",
    "diag_recovered_edges",
    "diag_recovered_useful",
    "diag_modified_expansions",
    "diag_recovery_attempts",
    "diag_recovery_candidates",
    "diag_recovery_get_ms",
    "diag_recovery_loop_ms",
    "diag_rewrite_active_expansions",
    "diag_rewrite_active_queries",
    "diag_rewrite_recent_expansions",
    "diag_rewrite_recent_queries",
    "diag_rewrite_period_expansions",
    "diag_rewrite_period_queries",
    "diag_rewrite_period_active_sum",
    "diag_rewrite_period_active_max",
    "diag_invisible_expansions",
    "diag_invisible_expansion_edges",
    "diag_invisible_candidate_dist_comps",
    "diag_invisible_candidate_enqueues",
    "freshness_active_front_pts",
    "freshness_missed_e2e_pts",
]

DERIVED_METRICS = {
    "rewrite_active_query_hit_prob": (
        "diag_rewrite_active_queries",
        "batch_size",
    ),
    "rewrite_active_expansion_hits_per_hop": (
        "diag_rewrite_active_expansions",
        "diag_hops",
    ),
    "rewrite_recent_query_hit_prob": (
        "diag_rewrite_recent_queries",
        "batch_size",
    ),
    "rewrite_recent_expansion_hits_per_hop": (
        "diag_rewrite_recent_expansions",
        "diag_hops",
    ),
    "rewrite_period_query_hit_prob": (
        "diag_rewrite_period_queries",
        "batch_size",
    ),
    "rewrite_period_expansion_hits_per_hop": (
        "diag_rewrite_period_expansions",
        "diag_hops",
    ),
    "rewrite_period_active_writers_per_expansion": (
        "diag_rewrite_period_active_sum",
        "diag_rewrite_period_expansions",
    ),
    "invisible_edges_per_invisible_expansion": (
        "diag_invisible_expansion_edges",
        "diag_invisible_expansions",
    ),
    "invisible_candidate_enqueues_per_hop": (
        "diag_invisible_candidate_enqueues",
        "diag_hops",
    ),
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * p)]


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def batch_size_from_label(label: str) -> int:
    m = re.search(r"_b(\d+)_", label)
    if not m:
        raise ValueError(f"cannot infer batch size from {label}")
    return int(m.group(1))


def parse_label(label: str) -> dict[str, str]:
    out = {"label": label}
    parts = label.split("_")
    for i, part in enumerate(parts):
        if part.startswith("b") and part[1:].isdigit():
            out["batch"] = part[1:]
        elif part == "chasing":
            out["query_mode"] = "chasing"
        elif part == "round" and i + 1 < len(parts) and parts[i + 1] == "robin":
            out["query_mode"] = "round_robin"
        elif part.startswith("w") and part[1:].isdigit():
            out["write_rate"] = part[1:]
        elif part.startswith("r") and part[1:].isdigit():
            out["read_rate"] = part[1:]
        elif part in {"nolock", "nodelock", "rwlock"}:
            out["sync_mode"] = part
    return out


def load_batches(wide: Path, batch_size: int) -> list[dict[str, float]]:
    batches: list[dict[str, float]] = []
    current: list[dict[str, str]] = []
    with wide.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            current.append(row)
            if len(current) == batch_size:
                item: dict[str, float] = {
                    "op_ms": sum(float(r["op_ms"]) for r in current),
                    "e2e_ms": sum(float(r["e2e_ms"]) for r in current),
                }
                for col in DIAG_COLUMNS:
                    item[col] = sum(float(r.get(col, 0.0)) for r in current)
                item["batch_size"] = float(batch_size)
                for name, (num_col, den_col) in DERIVED_METRICS.items():
                    den = item.get(den_col, 0.0)
                    item[name] = item.get(num_col, 0.0) / den if den > 0 else 0.0
                batches.append(item)
                current.clear()
    return batches


def summarize_run(run_dir: Path) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    label = run_dir.name
    batch_size = batch_size_from_label(label)
    batches = load_batches(run_dir / "raw_latency" / "search_wide.csv", batch_size)
    ops = [b["op_ms"] for b in batches]
    e2es = [b["e2e_ms"] for b in batches]
    if not batches:
        raise ValueError(f"no batches in {run_dir}")

    top_n = max(20, min(100, max(1, len(batches) // 100)))
    top_idx = sorted(range(len(batches)), key=lambda i: batches[i]["op_ms"], reverse=True)[:top_n]
    top = [batches[i] for i in top_idx]
    top_rows: list[dict[str, object]] = []
    for rank, idx in enumerate(top_idx, start=1):
        top_rows.append(
            {
                **parse_label(label),
                "rank": rank,
                "batch_idx": idx,
                **batches[idx],
            }
        )

    row: dict[str, object] = {
        **parse_label(label),
        "num_batches": len(batches),
        "mean_op_ms": sum(ops) / len(ops),
        "p99_op_ms": percentile(ops, 0.99),
        "max_op_ms": max(ops),
        "mean_e2e_ms": sum(e2es) / len(e2es),
        "p99_e2e_ms": percentile(e2es, 0.99),
        "top_n": top_n,
        "top_mean_op_ms": sum(b["op_ms"] for b in top) / len(top),
    }

    metric_rows: list[dict[str, object]] = []
    for col in DIAG_COLUMNS + list(DERIVED_METRICS):
        values = [b[col] for b in batches]
        all_mean = sum(values) / len(values)
        top_mean = sum(b[col] for b in top) / len(top)
        enrich = top_mean / all_mean if all_mean > 0 else 0.0
        corr = pearson(values, ops)
        row[f"{col}_mean"] = all_mean
        row[f"{col}_top_mean"] = top_mean
        row[f"{col}_top_enrich"] = enrich
        row[f"{col}_corr_op"] = corr
        metric_rows.append(
            {
                "label": label,
                "metric": col,
                "mean": all_mean,
                "top_mean": top_mean,
                "top_enrich": enrich,
                "corr_op": corr,
            }
        )
    return row, metric_rows, top_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: analyze_m3_batch_diag_rootcause.py <result_dir>", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for wide in sorted(root.glob("*/raw_latency/search_wide.csv")):
        run_dir = wide.parents[1]
        try:
            row, per_metric, per_top = summarize_run(run_dir)
        except Exception as exc:
            print(f"[m3-batch-diag] skip {run_dir}: {exc}", file=sys.stderr)
            continue
        rows.append(row)
        metric_rows.extend(per_metric)
        top_rows.extend(per_top)

    write_csv(root / "m3_batch_diag_summary.csv", rows)
    write_csv(root / "m3_batch_diag_metrics.csv", metric_rows)
    write_csv(root / "m3_batch_diag_top_batches.csv", top_rows)

    report = root / "m3_batch_diag_report.md"
    with report.open("w") as f:
        f.write("# M3 Batch Diagnostic Root Cause\n\n")
        f.write("Runs are ranked by search op p99. `top_enrich` compares the slowest op batches against the run mean.\n\n")
        for row in sorted(rows, key=lambda r: float(r["p99_op_ms"]), reverse=True):
            f.write(
                "## {label}\n\n"
                "- op_ms: mean={mean_op_ms:.2f}, p99={p99_op_ms:.2f}, max={max_op_ms:.2f}, top_mean={top_mean_op_ms:.2f}\n"
                "- recovery_loop_ms top_enrich={diag_recovery_loop_ms_top_enrich:.2f}, corr={diag_recovery_loop_ms_corr_op:.2f}\n"
                "- recovered_edges top_enrich={diag_recovered_edges_top_enrich:.2f}, corr={diag_recovered_edges_corr_op:.2f}\n"
                "- recovery_candidates top_enrich={diag_recovery_candidates_top_enrich:.2f}, corr={diag_recovery_candidates_corr_op:.2f}\n"
                "- future_skip_hops top_enrich={diag_future_skip_hops_top_enrich:.2f}, corr={diag_future_skip_hops_corr_op:.2f}\n"
                "- dist_comps top_enrich={diag_dist_comps_top_enrich:.2f}, corr={diag_dist_comps_corr_op:.2f}\n"
                "- hops top_enrich={diag_hops_top_enrich:.2f}, corr={diag_hops_corr_op:.2f}\n"
                "- rewrite_active_expansions top_enrich={diag_rewrite_active_expansions_top_enrich:.2f}, corr={diag_rewrite_active_expansions_corr_op:.2f}\n"
                "- rewrite_active_queries top_enrich={diag_rewrite_active_queries_top_enrich:.2f}, corr={diag_rewrite_active_queries_corr_op:.2f}\n"
                "- rewrite_recent_expansions top_enrich={diag_rewrite_recent_expansions_top_enrich:.2f}, corr={diag_rewrite_recent_expansions_corr_op:.2f}\n"
                "- rewrite_recent_queries top_enrich={diag_rewrite_recent_queries_top_enrich:.2f}, corr={diag_rewrite_recent_queries_corr_op:.2f}\n"
                "- active query hit prob mean={rewrite_active_query_hit_prob_mean:.6f}, top={rewrite_active_query_hit_prob_top_mean:.6f}, corr={rewrite_active_query_hit_prob_corr_op:.2f}\n"
                "- active expansion hits/hop mean={rewrite_active_expansion_hits_per_hop_mean:.6f}, top={rewrite_active_expansion_hits_per_hop_top_mean:.6f}, corr={rewrite_active_expansion_hits_per_hop_corr_op:.2f}\n"
                "- recent query hit prob mean={rewrite_recent_query_hit_prob_mean:.6f}, top={rewrite_recent_query_hit_prob_top_mean:.6f}, corr={rewrite_recent_query_hit_prob_corr_op:.2f}\n"
                "- recent expansion hits/hop mean={rewrite_recent_expansion_hits_per_hop_mean:.6f}, top={rewrite_recent_expansion_hits_per_hop_top_mean:.6f}, corr={rewrite_recent_expansion_hits_per_hop_corr_op:.2f}\n"
                "- period query hit prob mean={rewrite_period_query_hit_prob_mean:.6f}, top={rewrite_period_query_hit_prob_top_mean:.6f}, corr={rewrite_period_query_hit_prob_corr_op:.2f}\n"
                "- period expansion hits/hop mean={rewrite_period_expansion_hits_per_hop_mean:.6f}, top={rewrite_period_expansion_hits_per_hop_top_mean:.6f}, corr={rewrite_period_expansion_hits_per_hop_corr_op:.2f}\n"
                "- period active writers/exp mean={rewrite_period_active_writers_per_expansion_mean:.2f}, top={rewrite_period_active_writers_per_expansion_top_mean:.2f}, corr={rewrite_period_active_writers_per_expansion_corr_op:.2f}\n"
                "- period active writer max top_enrich={diag_rewrite_period_active_max_top_enrich:.2f}, corr={diag_rewrite_period_active_max_corr_op:.2f}\n\n".format(**row)
            )
    print(f"[m3-batch-diag] wrote {root / 'm3_batch_diag_summary.csv'}")
    print(f"[m3-batch-diag] wrote {root / 'm3_batch_diag_metrics.csv'}")
    print(f"[m3-batch-diag] wrote {root / 'm3_batch_diag_top_batches.csv'}")
    print(f"[m3-batch-diag] wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
