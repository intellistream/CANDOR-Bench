#!/usr/bin/env python3
"""Analyze M3 cache/phase controls against a read-only/noop baseline."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


BASELINE_MODE = "readonly_noop"
CACHE_COLS = [
    "perf_l2_rqsts_miss_per_s",
    "perf_llc_load_misses_per_s",
    "perf_l1_dcache_load_misses_per_s",
    "perf_dtlb_load_misses_walk_completed_per_s",
]
PHASE_COLS = [
    "insert_path_search_active_workers",
    "existing_neighbor_load_scan_active_workers",
    "existing_neighbor_prune_active_workers",
    "existing_neighbor_undo_record_active_workers",
    "existing_neighbor_rewrite_active_workers",
    "existing_neighbor_update_active_workers",
]
WORK_COLS = [
    "insert_path_edges_scanned",
    "insert_path_dist_comps",
    "existing_neighbor_prune_candidates",
    "existing_neighbor_prune_dist_comps",
    "existing_neighbor_edges_written",
    "existing_neighbor_edges_pruned",
]


def to_float(value: object) -> float:
    try:
        return float(value or 0.0)
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


def collapse_batches(search_wide: Path) -> list[dict[str, float]]:
    batches: dict[tuple[int, int, int], dict[str, float]] = {}
    with search_wide.open(newline="") as f:
        for row in csv.DictReader(f):
            create = int(to_float(row.get("create_raw_ns")))
            start = int(to_float(row.get("start_raw_ns")))
            finish = int(to_float(row.get("finish_raw_ns")))
            if finish <= 0:
                continue
            key = (create, start, finish)
            batch = batches.setdefault(
                key,
                {
                    "create_raw_ns": float(create),
                    "start_raw_ns": float(start),
                    "finish_raw_ns": float(finish),
                    "op_ms": max(0.0, (finish - start) / 1_000_000.0),
                    "e2e_ms": max(0.0, (finish - create) / 1_000_000.0),
                    "start_delay_ms": max(0.0, (start - create) / 1_000_000.0),
                    "rows": 0.0,
                },
            )
            batch["rows"] += 1.0
    ordered = sorted(batches.values(), key=lambda r: (r["create_raw_ns"], r["finish_raw_ns"]))
    for i, row in enumerate(ordered):
        row["batch_idx"] = float(i)
    return ordered


def load_periods(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        return [{k: to_float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def weighted_window_metrics(
    periods: list[dict[str, float]],
    window_start: int,
    window_end: int,
) -> dict[str, float]:
    if not periods:
        return {}
    starts = [int(r["bin_start_ns"]) for r in periods]
    if window_start <= 0 or window_end <= window_start:
        return {}
    first = max(0, bisect_right(starts, window_start) - 1)
    out: dict[str, float] = defaultdict(float)
    total = 0.0
    for row in periods[first:]:
        bin_start = int(row["bin_start_ns"])
        bin_end = int(row["bin_end_ns"])
        if bin_start >= window_end:
            break
        overlap = min(window_end, bin_end) - max(window_start, bin_start)
        if overlap <= 0:
            continue
        weight = float(overlap)
        total += weight
        for col in CACHE_COLS + PHASE_COLS + WORK_COLS:
            out[col] += row.get(col, 0.0) * weight
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def weighted_period_metrics(batch: dict[str, float], periods: list[dict[str, float]]) -> dict[str, float]:
    op_start = int(batch.get("start_raw_ns") or 0)
    finish = int(batch.get("finish_raw_ns") or 0)
    if op_start <= 0 or finish <= op_start:
        finish = int(batch["finish_raw_ns"])
        op_start = finish - max(1, int(batch["op_ms"] * 1_000_000.0))
    return weighted_window_metrics(periods, op_start, finish)


def load_case(case_dir: Path) -> list[dict[str, float]]:
    batches = collapse_batches(case_dir / "raw_latency" / "search_wide.csv")
    periods = load_periods(case_dir / "period_groups.csv")
    for batch in batches:
        batch.update(weighted_period_metrics(batch, periods))
        wait_metrics = weighted_window_metrics(
            periods,
            int(batch.get("create_raw_ns") or 0),
            int(batch.get("start_raw_ns") or 0),
        )
        for key, value in wait_metrics.items():
            batch[f"wait_{key}"] = value
    return batches


def mode_label_from_case(path: Path, root: Path) -> tuple[str, str]:
    rel = path.relative_to(root)
    mode = rel.parts[0]
    label = rel.parts[-1]
    case = "/".join(rel.parts[1:-1])
    return mode, f"{case}/{label}"


def display_label(label: str) -> str:
    match = re.match(
        r"insert_i" r"w(?P<workers>\d+)_rep(?P<rep>\d+)/sift_b200_(?P<query>chasing|round_robin)_w40000_r40000_nolock",
        label,
    )
    if not match:
        return label
    workers = int(match.group("workers"))
    concurrency = "high writer concurrency" if workers >= 32 else "low writer concurrency"
    query = "chasing queries" if match.group("query") == "chasing" else "round-robin queries"
    return f"{concurrency}, {query}, repeat {int(match.group('rep'))}"


def summarize_mode(mode: str, label: str, rows: list[dict[str, float]], baseline: list[dict[str, float]]) -> dict[str, object]:
    n = min(len(rows), len(baseline))
    deltas = []
    for i in range(n):
        row = rows[i]
        base = baseline[i]
        item = {
            "delta_op_ms": row.get("op_ms", 0.0) - base.get("op_ms", 0.0),
            "delta_e2e_ms": row.get("e2e_ms", 0.0) - base.get("e2e_ms", 0.0),
            "delta_start_delay_ms": row.get("start_delay_ms", 0.0) - base.get("start_delay_ms", 0.0),
            "op_ms": row.get("op_ms", 0.0),
            "e2e_ms": row.get("e2e_ms", 0.0),
            "start_delay_ms": row.get("start_delay_ms", 0.0),
        }
        for col in CACHE_COLS:
            item[f"delta_{col}"] = row.get(col, 0.0) - base.get(col, 0.0)
            item[col] = row.get(col, 0.0)
            item[f"wait_delta_{col}"] = row.get(f"wait_{col}", 0.0) - base.get(f"wait_{col}", 0.0)
            item[f"wait_{col}"] = row.get(f"wait_{col}", 0.0)
        for col in PHASE_COLS + WORK_COLS:
            item[col] = row.get(col, 0.0)
            item[f"wait_{col}"] = row.get(f"wait_{col}", 0.0)
        deltas.append(item)
    top = sorted(deltas, key=lambda r: r["delta_e2e_ms"], reverse=True)[: max(1, min(10, len(deltas)))]
    result: dict[str, object] = {
        "mode": mode,
        "label": label,
        "matched_batches": n,
        "mean_delta_op_ms": mean([r["delta_op_ms"] for r in deltas]),
        "p95_delta_op_ms": percentile([r["delta_op_ms"] for r in deltas], 95),
        "p99_delta_op_ms": percentile([r["delta_op_ms"] for r in deltas], 99),
        "mean_delta_start_delay_ms": mean([r["delta_start_delay_ms"] for r in deltas]),
        "p95_delta_start_delay_ms": percentile([r["delta_start_delay_ms"] for r in deltas], 95),
        "p99_delta_start_delay_ms": percentile([r["delta_start_delay_ms"] for r in deltas], 99),
        "mean_delta_e2e_ms": mean([r["delta_e2e_ms"] for r in deltas]),
        "p95_delta_e2e_ms": percentile([r["delta_e2e_ms"] for r in deltas], 95),
        "p99_delta_e2e_ms": percentile([r["delta_e2e_ms"] for r in deltas], 99),
    }
    for col in CACHE_COLS:
        result[f"mean_delta_{col}"] = mean([r[f"delta_{col}"] for r in deltas])
        result[f"top_delta_e2e_mean_{col}"] = mean([r[col] for r in top])
        result[f"mean_wait_delta_{col}"] = mean([r[f"wait_delta_{col}"] for r in deltas])
        result[f"top_delta_e2e_mean_wait_{col}"] = mean([r[f"wait_{col}"] for r in top])
    for col in PHASE_COLS + WORK_COLS:
        result[f"top_delta_e2e_mean_{col}"] = mean([r[col] for r in top])
        result[f"top_delta_e2e_mean_wait_{col}"] = mean([r[f"wait_{col}"] for r in top])
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = ["# M3 Cache Phase Control Delta", ""]
    lines.append("Baseline: `readonly_noop` with the same precomputed producer/search schedule.")
    lines.append("")
    lines.append("| mode | label | matched | p99 delta e2e ms | p99 delta start ms | p99 delta op ms | top wait phase | wait L2/s | top op phase | op L2/s |")
    lines.append("|---|---|---:|---:|---:|---:|---|---:|---|---:|")
    for row in rows:
        phase_means = {
            col: float(row.get(f"top_delta_e2e_mean_{col}", 0.0) or 0.0)
            for col in PHASE_COLS
        }
        top_phase = max(phase_means, key=phase_means.get) if phase_means else ""
        if not top_phase or phase_means.get(top_phase, 0.0) <= 0.0:
            top_phase = "none"
        wait_phase_means = {
            col: float(row.get(f"top_delta_e2e_mean_wait_{col}", 0.0) or 0.0)
            for col in PHASE_COLS
        }
        top_wait_phase = max(wait_phase_means, key=wait_phase_means.get) if wait_phase_means else ""
        if not top_wait_phase or wait_phase_means.get(top_wait_phase, 0.0) <= 0.0:
            top_wait_phase = "none"
        lines.append(
            "| {mode} | {label} | {matched_batches} | {p99_e2e:.3f} | {p99_start:.3f} | {p99_op:.3f} | {wait_phase} | {wait_l2:.3e} | {phase} | {l2:.3e} |".format(
                mode=row["mode"],
                label=display_label(str(row["label"])),
                matched_batches=int(row["matched_batches"]),
                p99_e2e=float(row["p99_delta_e2e_ms"]),
                p99_start=float(row["p99_delta_start_delay_ms"]),
                p99_op=float(row["p99_delta_op_ms"]),
                wait_phase=top_wait_phase,
                wait_l2=float(row.get("top_delta_e2e_mean_wait_perf_l2_rqsts_miss_per_s", 0.0) or 0.0),
                phase=top_phase,
                l2=float(row.get("top_delta_e2e_mean_perf_l2_rqsts_miss_per_s", 0.0) or 0.0),
            )
        )
    lines.append("")
    lines.append("Interpretation rule: a mode is evidence for read/write interference only if its delta over `readonly_noop` is high. Wait columns cover create-to-start; op columns cover start-to-finish. They attribute pressure windows, not exact hardware events to individual source lines.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    cases: dict[tuple[str, str], list[dict[str, float]]] = {}
    for search_wide in args.root.glob("*/*/*/raw_latency/search_wide.csv"):
        case_dir = search_wide.parents[1]
        if not (case_dir / "period_groups.csv").exists():
            continue
        mode, label = mode_label_from_case(case_dir, args.root)
        cases[(mode, label)] = load_case(case_dir)

    labels = sorted({label for _, label in cases})
    rows: list[dict[str, object]] = []
    for label in labels:
        baseline = cases.get((BASELINE_MODE, label))
        if not baseline:
            continue
        for (mode, mode_label), batches in sorted(cases.items()):
            if mode_label != label or mode == BASELINE_MODE:
                continue
            rows.append(summarize_mode(mode, label, batches, baseline))

    out_csv = args.root / "cache_phase_delta_summary.csv"
    out_md = args.root / "cache_phase_delta_summary.md"
    write_csv(out_csv, rows)
    write_md(out_md, rows)
    print(f"[m3-cache-phase-analyze] wrote {out_csv}")
    print(f"[m3-cache-phase-analyze] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
