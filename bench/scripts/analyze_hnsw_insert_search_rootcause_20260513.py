#!/usr/bin/env python3
"""HNSW insert-vs-search P99 root-cause analysis.

This analyzer is deliberately about plain HNSW behavior, not ANNchor policy.
It separates four explanations for search P99 inflation under real insert:

1. direct node-lock wait visible to search;
2. higher per-edge traversal cost with similar traversal volume;
3. overlap with specific writer phases;
4. persistent graph-structure effect after writers stop.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


BATCH_KEY = ("create_raw_ns", "start_raw_ns", "finish_raw_ns")
PHASE_COLUMNS = (
    "diag_phase_existing_update_samples",
    "diag_phase_link_critical_samples",
    "diag_phase_load_scan_samples",
    "diag_phase_append_samples",
    "diag_phase_prune_samples",
    "diag_phase_undo_record_samples",
    "diag_phase_rewrite_samples",
)


def to_float(value: object, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percentile(values: Iterable[float], q: float) -> float:
    xs = sorted(x for x in values if not math.isnan(x))
    if not xs:
        return math.nan
    if q > 1.0:
        q /= 100.0
    k = (len(xs) - 1) * q
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(values: Iterable[float]) -> float:
    xs = [x for x in values if not math.isnan(x)]
    return statistics.mean(xs) if xs else math.nan


def median(values: Iterable[float]) -> float:
    xs = [x for x in values if not math.isnan(x)]
    return statistics.median(xs) if xs else math.nan


def iqr(values: Iterable[float]) -> tuple[float, float]:
    xs = [x for x in values if not math.isnan(x)]
    if not xs:
        return math.nan, math.nan
    return percentile(xs, 25), percentile(xs, 75)


def load_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        for token in line.strip().split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            out[key] = value
    return out


def csv_get(row: list[str], index: dict[str, int], col: str, default: str = "") -> str:
    pos = index.get(col)
    if pos is None or pos >= len(row):
        return default
    return row[pos]


def csv_float(row: list[str], index: dict[str, int], col: str, default: float = 0.0) -> float:
    return to_float(csv_get(row, index, col), default)


def segment_for_finish(finish_ms: float, meta: dict[str, str]) -> str:
    mode = meta.get("mode", "")
    if mode != "periodic_insert":
        skip_ms = to_float(meta.get("skip_pre_ms"), 0.0)
        sample_ms = to_float(meta.get("sample_ms"), 0.0)
        if skip_ms <= finish_ms <= skip_ms + sample_ms:
            return "steady"
        return ""

    start = to_float(meta.get("insert_window_start_ms"), 20_000.0)
    duration = to_float(meta.get("insert_window_duration_ms"), 20_000.0)
    end = start + duration
    guard = to_float(meta.get("segment_guard_ms"), 2_000.0)
    total = to_float(meta.get("duration_ms"), 60_000.0)

    if guard <= finish_ms < start - guard:
        return "pre_bg_search"
    if start + guard <= finish_ms < end - guard:
        return "during_insert"
    if end + guard <= finish_ms < total - guard:
        return "post_bg_search"
    return ""


def load_search_batches(cell: Path) -> list[dict[str, float | str]]:
    meta = load_meta(cell / "cell.meta")
    path = cell / "raw_latency" / "search_wide.csv"
    if not path.exists():
        return []

    batches: dict[tuple[str, str, str], dict[str, float | str]] = {}
    order = 0
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        index = {name: pos for pos, name in enumerate(header)}
        for row in reader:
            measured = csv_get(row, index, "measured", "1")
            if measured not in {"1", "true", "True"}:
                continue
            finish_ms = csv_float(row, index, "finish_ms", math.nan)
            if math.isnan(finish_ms):
                continue
            segment = segment_for_finish(finish_ms, meta)
            if not segment:
                continue
            key = tuple(csv_get(row, index, k) for k in BATCH_KEY)
            if not any(key):
                key = (str(order), "", "")
            if key not in batches:
                start_ns = to_float(csv_get(row, index, "start_raw_ns"), math.nan)
                finish_ns = to_float(csv_get(row, index, "finish_raw_ns"), math.nan)
                raw_wall_ms = (finish_ns - start_ns) / 1e6 if not math.isnan(start_ns + finish_ns) else math.nan
                search_index_ms = csv_float(row, index, "search_index_ms", math.nan)
                if math.isnan(search_index_ms):
                    search_index_ms = csv_float(row, index, "op_ms", math.nan)
                batches[key] = {
                    "cell": cell.name,
                    "mode": meta.get("mode", ""),
                    "n": to_float(meta.get("n_competing"), 0.0),
                    "rep": to_float(meta.get("rep"), 0.0),
                    "segment": segment,
                    "order": float(order),
                    "finish_ms": finish_ms,
                    "raw_wall_ms": raw_wall_ms,
                    "amortized_op_ms": csv_float(row, index, "op_ms", math.nan),
                    "search_index_ms": search_index_ms,
                    "query_rows": 0.0,
                    "lock_wait_sum_ms": 0.0,
                    "lock_wait_max_ms": 0.0,
                    "result_lock_wait_sum_ms": 0.0,
                    "record_lock_wait_sum_ms": 0.0,
                    "l0_edges_sum": 0.0,
                    "l0_expansions_sum": 0.0,
                    "distance_comps_sum": 0.0,
                    "adj_fetch_sum_ms": 0.0,
                    "candidate_loop_sum_ms": 0.0,
                    "dist_compute_sum_ms": 0.0,
                    "thread_cpu_sum_ms": 0.0,
                    "unique_adj_4k_pages_sum": 0.0,
                    "unique_data_4k_pages_sum": 0.0,
                }
                for col in PHASE_COLUMNS:
                    batches[key][col] = 0.0
                order += 1

            item = batches[key]
            lock_wait = csv_float(row, index, "work_upper_lock_wait_ms") + csv_float(
                row, index, "work_level0_lock_wait_ms"
            )
            item["query_rows"] = float(item["query_rows"]) + 1.0
            item["lock_wait_sum_ms"] = float(item["lock_wait_sum_ms"]) + lock_wait
            item["lock_wait_max_ms"] = max(float(item["lock_wait_max_ms"]), lock_wait)
            item["result_lock_wait_sum_ms"] = float(item["result_lock_wait_sum_ms"]) + csv_float(
                row, index, "search_results_lock_wait_ms"
            )
            item["record_lock_wait_sum_ms"] = float(item["record_lock_wait_sum_ms"]) + csv_float(
                row, index, "search_record_lock_wait_ms"
            )
            item["l0_edges_sum"] = float(item["l0_edges_sum"]) + csv_float(row, index, "work_level0_edges_scanned")
            item["l0_expansions_sum"] = float(item["l0_expansions_sum"]) + csv_float(
                row, index, "work_level0_expansions"
            )
            item["distance_comps_sum"] = float(item["distance_comps_sum"]) + csv_float(
                row, index, "work_distance_computations"
            )
            item["adj_fetch_sum_ms"] = float(item["adj_fetch_sum_ms"]) + csv_float(row, index, "work_level0_adj_fetch_ms")
            item["candidate_loop_sum_ms"] = float(item["candidate_loop_sum_ms"]) + csv_float(
                row, index, "work_level0_candidate_loop_ms"
            )
            item["dist_compute_sum_ms"] = float(item["dist_compute_sum_ms"]) + csv_float(
                row, index, "work_distance_compute_ms"
            )
            item["thread_cpu_sum_ms"] = float(item["thread_cpu_sum_ms"]) + csv_float(
                row, index, "work_searchknn_thread_cpu_ms"
            )
            item["unique_adj_4k_pages_sum"] = float(item["unique_adj_4k_pages_sum"]) + csv_float(
                row, index, "work_expand_unique_adj_4k_pages"
            )
            item["unique_data_4k_pages_sum"] = float(item["unique_data_4k_pages_sum"]) + csv_float(
                row, index, "work_expand_unique_data_4k_pages"
            )
            for col in PHASE_COLUMNS:
                item[col] = max(float(item[col]), csv_float(row, index, col))

    out = list(batches.values())
    for item in out:
        raw = float(item["raw_wall_ms"])
        edges = float(item["l0_edges_sum"])
        dist = float(item["distance_comps_sum"])
        lock_max = float(item["lock_wait_max_ms"])
        item["raw_us_per_l0_edge"] = raw * 1000.0 / edges if edges > 0 else math.nan
        item["raw_us_per_dist_comp"] = raw * 1000.0 / dist if dist > 0 else math.nan
        item["residual_after_max_lock_ms"] = max(0.0, raw - lock_max) if not math.isnan(raw) else math.nan
        item["lock_sum_over_wall"] = float(item["lock_wait_sum_ms"]) / raw if raw > 0 else math.nan
    return out


def parse_perf_csv(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        count = to_float(parts[0])
        if math.isnan(count):
            continue
        event = parts[2]
        out[event] = count
        if len(parts) > 4:
            pct = to_float(parts[4])
            if not math.isnan(pct):
                out[f"{event}__run_pct"] = pct
    return out


def scaled_perf(perf: dict[str, float], event: str) -> float:
    count = perf.get(event, math.nan)
    pct = perf.get(f"{event}__run_pct", 100.0)
    if math.isnan(count) or pct <= 0:
        return math.nan
    return count * 100.0 / pct


def load_cell_perf(cell: Path) -> dict[str, float]:
    reader = parse_perf_csv(cell / "perf_reader.txt")
    competing = parse_perf_csv(cell / "perf_competing.txt")
    uncore = parse_perf_csv(cell / "perf_uncore.txt")

    reader_llc_loads = scaled_perf(reader, "LLC-loads")
    reader_llc_misses = scaled_perf(reader, "LLC-load-misses")
    comp_llc_loads = scaled_perf(competing, "LLC-loads")
    comp_llc_misses = scaled_perf(competing, "LLC-load-misses")
    return {
        "reader_llc_loads": reader_llc_loads,
        "reader_llc_load_misses": reader_llc_misses,
        "reader_llc_miss_ratio": reader_llc_misses / reader_llc_loads if reader_llc_loads else math.nan,
        "competing_llc_loads": comp_llc_loads,
        "competing_llc_load_misses": comp_llc_misses,
        "competing_llc_miss_ratio": comp_llc_misses / comp_llc_loads if comp_llc_loads else math.nan,
        "uncore_reads": sum(v for k, v in uncore.items() if "cas_count_read" in k and "__" not in k),
        "uncore_writes": sum(v for k, v in uncore.items() if "cas_count_write" in k and "__" not in k),
    }


def top_fraction(rows: list[dict[str, float | str]], col: str, fraction: float) -> list[dict[str, float | str]]:
    clean = [r for r in rows if not math.isnan(float(r[col]))]
    if not clean:
        return []
    clean.sort(key=lambda r: float(r[col]), reverse=True)
    return clean[: max(1, math.ceil(len(clean) * fraction))]


def aggregate_batches(rows: list[dict[str, float | str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int, str], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["mode"]), int(float(row["n"])), int(float(row["rep"])), str(row["segment"]))].append(row)

    metrics = (
        "raw_wall_ms",
        "amortized_op_ms",
        "search_index_ms",
        "lock_wait_sum_ms",
        "lock_wait_max_ms",
        "residual_after_max_lock_ms",
        "lock_sum_over_wall",
        "l0_edges_sum",
        "l0_expansions_sum",
        "raw_us_per_l0_edge",
        "raw_us_per_dist_comp",
        "adj_fetch_sum_ms",
        "candidate_loop_sum_ms",
        "dist_compute_sum_ms",
        "thread_cpu_sum_ms",
        "unique_adj_4k_pages_sum",
        "unique_data_4k_pages_sum",
    )
    out: list[dict[str, object]] = []
    for (mode, n, rep, segment), group in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0], x[0][2], x[0][3])):
        top1 = top_fraction(group, "raw_wall_ms", 0.01)
        item: dict[str, object] = {
            "mode": mode,
            "n": n,
            "rep": rep,
            "segment": segment,
            "samples": len(group),
        }
        for metric in metrics:
            vals = [float(row[metric]) for row in group]
            item[f"{metric}_p50"] = percentile(vals, 50)
            item[f"{metric}_p95"] = percentile(vals, 95)
            item[f"{metric}_p99"] = percentile(vals, 99)
            item[f"{metric}_mean"] = mean(vals)
        for col in PHASE_COLUMNS:
            item[f"{col}_all_mean"] = mean(float(row[col]) for row in group)
            item[f"{col}_top1_raw_mean"] = mean(float(row[col]) for row in top1)
        item["top1_raw_lock_wait_sum_ms_mean"] = mean(float(row["lock_wait_sum_ms"]) for row in top1)
        item["top1_raw_lock_wait_max_ms_mean"] = mean(float(row["lock_wait_max_ms"]) for row in top1)
        item["top1_raw_us_per_l0_edge_mean"] = mean(float(row["raw_us_per_l0_edge"]) for row in top1)
        out.append(item)
    return out


def aggregate_reps(cell_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in cell_rows:
        groups[(str(row["mode"]), int(row["n"]), str(row["segment"]))].append(row)
    metrics = [k for k in cell_rows[0].keys() if k not in {"mode", "n", "rep", "segment", "samples"}] if cell_rows else []
    out: list[dict[str, object]] = []
    for (mode, n, segment), rows in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        item: dict[str, object] = {
            "mode": mode,
            "n": n,
            "segment": segment,
            "reps": len(rows),
            "samples_min": min(int(r["samples"]) for r in rows),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in rows]
            lo, hi = iqr(vals)
            item[f"{metric}_median"] = median(vals)
            item[f"{metric}_iqr_lo"] = lo
            item[f"{metric}_iqr_hi"] = hi
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def add_perf_to_cell_rows(root: Path, cell_rows: list[dict[str, object]]) -> None:
    perf_by_cell = {cell.name: load_cell_perf(cell) for cell in root.iterdir() if cell.is_dir()}
    for row in cell_rows:
        # Aggregate rows are keyed by mode/n/rep/segment; recover cell by convention.
        mode = row["mode"]
        n = row["n"]
        rep = row["rep"]
        candidates = [
            f"{mode}_n{n}_rep{rep}",
            f"{mode}_n{n}_rep{rep}_phase",
        ]
        perf = {}
        for name in candidates:
            perf = perf_by_cell.get(name, {})
            if perf:
                break
        for key, value in perf.items():
            row[key] = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.root
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_root = args.root / "cells" if (args.root / "cells").exists() else args.root

    batches: list[dict[str, float | str]] = []
    for cell in sorted(cells_root.iterdir()):
        if cell.is_dir():
            batches.extend(load_search_batches(cell))

    cell_rows = aggregate_batches(batches)
    add_perf_to_cell_rows(cells_root, cell_rows)
    rep_rows = aggregate_reps(cell_rows)

    write_csv(out_dir / "hnsw_rootcause_batch_summary.csv", batches)  # type: ignore[arg-type]
    write_csv(out_dir / "hnsw_rootcause_cell_summary.csv", cell_rows)
    write_csv(out_dir / "hnsw_rootcause_aggregate.csv", rep_rows)

    print("mode n segment reps raw_p99 lock_max_p99 residual_p99 edge_p99 l0_edges_p99")
    for row in rep_rows:
        print(
            f"{row['mode']} {row['n']} {row['segment']} {row['reps']} "
            f"{float(row['raw_wall_ms_p99_median']):.3f} "
            f"{float(row['lock_wait_max_ms_p99_median']):.3f} "
            f"{float(row['residual_after_max_lock_ms_p99_median']):.3f} "
            f"{float(row['raw_us_per_l0_edge_p99_median']):.6f} "
            f"{float(row['l0_edges_sum_p99_median']):.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
