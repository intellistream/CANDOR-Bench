#!/usr/bin/env python3
"""Search-only mechanism analysis for the 2026-05-12 M3 RW tail runs.

This script intentionally reports reader/search evidence, not insert p99:
- batch-level search_index_ms / op_ms / post-index latency;
- foreground search lock wait and traversal work;
- reader/competing LLC and uncore counters;
- writer phase activity only as context for what the unmodified insert did.
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
CONTROL_MODES = ("spin", "bg_search")


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


def median(values: Iterable[float]) -> float:
    xs = [x for x in values if not math.isnan(x)]
    return statistics.median(xs) if xs else math.nan


def mean(values: Iterable[float]) -> float:
    xs = [x for x in values if not math.isnan(x)]
    return statistics.mean(xs) if xs else math.nan


def stdev(values: Iterable[float]) -> float:
    xs = [x for x in values if not math.isnan(x)]
    return statistics.stdev(xs) if len(xs) >= 2 else math.nan


def iqr(values: Iterable[float]) -> tuple[float, float]:
    xs = [x for x in values if not math.isnan(x)]
    if not xs:
        return math.nan, math.nan
    return percentile(xs, 25), percentile(xs, 75)


def value(row: dict[str, str], col: str) -> float:
    return to_float(row.get(col), 0.0)


def in_window(row: dict[str, str], lo_ms: float, hi_ms: float) -> bool:
    if row.get("measured", "1") not in {"1", "true", "True"}:
        return False
    finish_ms = to_float(row.get("finish_ms"))
    return math.isnan(finish_ms) or lo_ms <= finish_ms <= hi_ms


def csv_get(row: list[str], index: dict[str, int], col: str, default: str = "") -> str:
    pos = index.get(col)
    if pos is None or pos >= len(row):
        return default
    return row[pos]


def csv_float(row: list[str], index: dict[str, int], col: str, default: float = 0.0) -> float:
    return to_float(csv_get(row, index, col), default)


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


def ratio(num: float, den: float) -> float:
    if math.isnan(num) or math.isnan(den) or den == 0:
        return math.nan
    return num / den


def load_search_batches(cell: Path, skip_ms: float, sample_ms: float) -> list[dict[str, float]]:
    path = cell / "raw_latency" / "search_wide.csv"
    if not path.exists():
        return []
    lo_ms = skip_ms
    hi_ms = skip_ms + sample_ms
    batches: dict[tuple[str, str, str], dict[str, float]] = {}
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
            if not math.isnan(finish_ms) and not (lo_ms <= finish_ms <= hi_ms):
                continue
            key = tuple(csv_get(row, index, k) for k in BATCH_KEY)
            if not any(key):
                key = (str(order), "", "")
            if key not in batches:
                search_index_ms = csv_float(row, index, "search_index_ms", math.nan)
                if math.isnan(search_index_ms):
                    search_index_ms = csv_float(row, index, "op_ms", math.nan)
                batches[key] = {
                    "order": float(order),
                    "op_ms": csv_float(row, index, "op_ms", math.nan),
                    "e2e_ms": csv_float(row, index, "e2e_ms", math.nan),
                    "search_index_ms": search_index_ms,
                    "search_post_index_ms": csv_float(row, index, "search_post_index_ms", 0.0),
                    "query_rows": 0.0,
                    "batch_finish_ms": finish_ms,
                    "batch_lock_wait_sum_ms": 0.0,
                    "batch_lock_wait_max_ms": 0.0,
                    "batch_result_lock_wait_sum_ms": 0.0,
                    "batch_record_lock_wait_sum_ms": 0.0,
                    "batch_l0_edges_sum": 0.0,
                    "batch_l0_expansions_sum": 0.0,
                    "batch_distance_comps_sum": 0.0,
                    "batch_adj_fetch_sum_ms": 0.0,
                    "batch_candidate_loop_sum_ms": 0.0,
                    "batch_dist_compute_sum_ms": 0.0,
                    "batch_query_cpu_sum_ms": 0.0,
                    "batch_unique_adj_4k_pages_sum": 0.0,
                    "batch_unique_data_4k_pages_sum": 0.0,
                    "batch_invisible_expansions_sum": 0.0,
                    "batch_future_skip_hops_sum": 0.0,
                }
                order += 1
            item = batches[key]
            lock_wait = csv_float(row, index, "work_upper_lock_wait_ms") + csv_float(row, index, "work_level0_lock_wait_ms")
            item["query_rows"] += 1.0
            item["batch_lock_wait_sum_ms"] += lock_wait
            item["batch_lock_wait_max_ms"] = max(item["batch_lock_wait_max_ms"], lock_wait)
            item["batch_result_lock_wait_sum_ms"] += csv_float(row, index, "search_results_lock_wait_ms")
            item["batch_record_lock_wait_sum_ms"] += csv_float(row, index, "search_record_lock_wait_ms")
            item["batch_l0_edges_sum"] += csv_float(row, index, "work_level0_edges_scanned")
            item["batch_l0_expansions_sum"] += csv_float(row, index, "work_level0_expansions")
            item["batch_distance_comps_sum"] += csv_float(row, index, "work_distance_computations")
            item["batch_adj_fetch_sum_ms"] += csv_float(row, index, "work_level0_adj_fetch_ms")
            item["batch_candidate_loop_sum_ms"] += csv_float(row, index, "work_level0_candidate_loop_ms")
            item["batch_dist_compute_sum_ms"] += csv_float(row, index, "work_distance_compute_ms")
            item["batch_query_cpu_sum_ms"] += csv_float(row, index, "work_searchknn_thread_cpu_ms")
            item["batch_unique_adj_4k_pages_sum"] += csv_float(row, index, "work_expand_unique_adj_4k_pages")
            item["batch_unique_data_4k_pages_sum"] += csv_float(row, index, "work_expand_unique_data_4k_pages")
            item["batch_invisible_expansions_sum"] += csv_float(row, index, "work_invisible_expansions")
            item["batch_future_skip_hops_sum"] += csv_float(row, index, "work_future_skip_hops")

    out = list(batches.values())
    for item in out:
        edges = item["batch_l0_edges_sum"]
        dist = item["batch_distance_comps_sum"]
        item["search_us_per_l0_edge"] = item["search_index_ms"] * 1000.0 / edges if edges else math.nan
        item["search_us_per_distance_comp"] = item["search_index_ms"] * 1000.0 / dist if dist else math.nan
    return out


def load_insert_rows(cell: Path, skip_ms: float, sample_ms: float) -> list[dict[str, str]]:
    path = cell / "raw_latency" / "insert_wide.csv"
    if not path.exists():
        return []
    lo_ms = skip_ms
    hi_ms = skip_ms + sample_ms
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            finish_ms = to_float(row.get("finish_ms"))
            if math.isnan(finish_ms) or lo_ms <= finish_ms <= hi_ms:
                rows.append(row)
    return rows


def sum_col(rows: list[dict[str, str]], col: str) -> float:
    return sum(value(row, col) for row in rows)


def pctl_col(rows: list[dict[str, float]], col: str, q: float) -> float:
    return percentile((row.get(col, math.nan) for row in rows), q)


def mean_col(rows: list[dict[str, float]], col: str) -> float:
    return mean(row.get(col, math.nan) for row in rows)


def top_fraction_mean(rows: list[dict[str, float]], sort_col: str, value_col: str, fraction: float) -> float:
    clean = [row for row in rows if not math.isnan(row.get(sort_col, math.nan))]
    if not clean:
        return math.nan
    clean.sort(key=lambda row: row[sort_col], reverse=True)
    take = max(1, math.ceil(len(clean) * fraction))
    return mean(row.get(value_col, math.nan) for row in clean[:take])


def analyze_cell(cell: Path) -> dict[str, object]:
    meta = load_meta(cell / "cell.meta")
    skip_ms = to_float(meta.get("skip_pre_ms"), 0.0)
    sample_ms = to_float(meta.get("sample_ms"), 0.0)
    sample_s = sample_ms / 1000.0 if sample_ms > 0 else math.nan
    batch_size = int(to_float(meta.get("batch_size"), 20.0))
    mode = meta.get("mode", cell.name.split("_n", 1)[0])
    n_competing = int(to_float(meta.get("n_competing", meta.get("n_writers")), 0.0))
    rep = int(to_float(meta.get("rep"), 0.0))

    search = load_search_batches(cell, skip_ms, sample_ms)
    inserts = load_insert_rows(cell, skip_ms, sample_ms)
    reader_perf = parse_perf_csv(cell / "perf_reader.txt")
    competing_perf = parse_perf_csv(cell / "perf_competing.txt")
    uncore_perf = parse_perf_csv(cell / "perf_uncore.txt")

    reader_llc_loads = scaled_perf(reader_perf, "LLC-loads")
    reader_llc_load_misses = scaled_perf(reader_perf, "LLC-load-misses")
    reader_llc_stores = scaled_perf(reader_perf, "LLC-stores")
    reader_llc_store_misses = scaled_perf(reader_perf, "LLC-store-misses")
    competing_llc_loads = scaled_perf(competing_perf, "LLC-loads")
    competing_llc_load_misses = scaled_perf(competing_perf, "LLC-load-misses")
    competing_llc_stores = scaled_perf(competing_perf, "LLC-stores")
    competing_llc_store_misses = scaled_perf(competing_perf, "LLC-store-misses")
    uncore_reads_mib = scaled_perf(uncore_perf, "uncore_imc/cas_count_read/")
    uncore_writes_mib = scaled_perf(uncore_perf, "uncore_imc/cas_count_write/")

    insert_op_sum = sum_col(inserts, "op_ms")
    writer_existing_loop_sum = sum_col(inserts, "graph_existing_neighbor_update_loop_ms")
    writer_link_critical_sum = sum_col(inserts, "graph_link_critical_ms")
    writer_lock_wait_sum = sum_col(inserts, "graph_unique_lock_wait_ms")
    writer_edges_written = sum_col(inserts, "graph_existing_neighbor_edges_written")
    writer_edges_pruned = sum_col(inserts, "graph_existing_neighbor_edges_pruned")
    writer_edges_loaded = sum_col(inserts, "graph_existing_neighbor_loaded_edges")

    return {
        "cell": cell.name,
        "mode": mode,
        "n": n_competing,
        "rep": rep,
        "dataset_name": meta.get("dataset_name", ""),
        "begin_num": meta.get("begin_num", ""),
        "max_elements": meta.get("max_elements", ""),
        "lock_variant": meta.get("lock_variant", ""),
        "search_batches": len(search),
        "search_op_p50_ms": pctl_col(search, "op_ms", 50),
        "search_op_p95_ms": pctl_col(search, "op_ms", 95),
        "search_op_p99_ms": pctl_col(search, "op_ms", 99),
        "search_index_p50_ms": pctl_col(search, "search_index_ms", 50),
        "search_index_p95_ms": pctl_col(search, "search_index_ms", 95),
        "search_index_p99_ms": pctl_col(search, "search_index_ms", 99),
        "search_index_p999_ms": pctl_col(search, "search_index_ms", 99.9),
        "search_index_max_ms": max((row["search_index_ms"] for row in search), default=math.nan),
        "search_post_index_p99_ms": pctl_col(search, "search_post_index_ms", 99),
        "search_batch_lock_wait_sum_p99_ms": pctl_col(search, "batch_lock_wait_sum_ms", 99),
        "search_batch_lock_wait_max_p99_ms": pctl_col(search, "batch_lock_wait_max_ms", 99),
        "search_result_lock_wait_sum_p99_ms": pctl_col(search, "batch_result_lock_wait_sum_ms", 99),
        "search_record_lock_wait_sum_p99_ms": pctl_col(search, "batch_record_lock_wait_sum_ms", 99),
        "search_l0_edges_sum_p99": pctl_col(search, "batch_l0_edges_sum", 99),
        "search_l0_expansions_sum_p99": pctl_col(search, "batch_l0_expansions_sum", 99),
        "search_distance_comps_sum_p99": pctl_col(search, "batch_distance_comps_sum", 99),
        "search_adj_fetch_sum_p99_ms": pctl_col(search, "batch_adj_fetch_sum_ms", 99),
        "search_candidate_loop_sum_p99_ms": pctl_col(search, "batch_candidate_loop_sum_ms", 99),
        "search_dist_compute_sum_p99_ms": pctl_col(search, "batch_dist_compute_sum_ms", 99),
        "search_query_cpu_sum_p99_ms": pctl_col(search, "batch_query_cpu_sum_ms", 99),
        "search_unique_adj_4k_pages_sum_p99": pctl_col(search, "batch_unique_adj_4k_pages_sum", 99),
        "search_unique_data_4k_pages_sum_p99": pctl_col(search, "batch_unique_data_4k_pages_sum", 99),
        "search_us_per_l0_edge_p50": pctl_col(search, "search_us_per_l0_edge", 50),
        "search_us_per_l0_edge_p99": pctl_col(search, "search_us_per_l0_edge", 99),
        "slow1pct_lock_wait_sum_mean_ms": top_fraction_mean(search, "search_index_ms", "batch_lock_wait_sum_ms", 0.01),
        "slow1pct_l0_edges_sum_mean": top_fraction_mean(search, "search_index_ms", "batch_l0_edges_sum", 0.01),
        "slow1pct_us_per_l0_edge_mean": top_fraction_mean(search, "search_index_ms", "search_us_per_l0_edge", 0.01),
        "insert_batches_window": len(inserts),
        "insert_iqps_vec_s": len(inserts) * batch_size / sample_s if sample_s else math.nan,
        "writer_existing_loop_fraction": ratio(writer_existing_loop_sum, insert_op_sum),
        "writer_link_critical_fraction": ratio(writer_link_critical_sum, insert_op_sum),
        "writer_lock_wait_fraction": ratio(writer_lock_wait_sum, insert_op_sum),
        "writer_existing_edges_written_s": writer_edges_written / sample_s if sample_s else math.nan,
        "writer_existing_edges_pruned_s": writer_edges_pruned / sample_s if sample_s else math.nan,
        "writer_existing_edges_loaded_s": writer_edges_loaded / sample_s if sample_s else math.nan,
        "reader_llc_loads": reader_llc_loads,
        "reader_llc_load_misses": reader_llc_load_misses,
        "reader_llc_load_miss_ratio": ratio(reader_llc_load_misses, reader_llc_loads),
        "reader_llc_stores": reader_llc_stores,
        "reader_llc_store_misses": reader_llc_store_misses,
        "reader_llc_store_miss_ratio": ratio(reader_llc_store_misses, reader_llc_stores),
        "competing_llc_loads": competing_llc_loads,
        "competing_llc_load_misses": competing_llc_load_misses,
        "competing_llc_load_miss_ratio": ratio(competing_llc_load_misses, competing_llc_loads),
        "competing_llc_stores": competing_llc_stores,
        "competing_llc_store_misses": competing_llc_store_misses,
        "competing_llc_store_miss_ratio": ratio(competing_llc_store_misses, competing_llc_stores),
        "uncore_reads_mib": uncore_reads_mib,
        "uncore_writes_mib": uncore_writes_mib,
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("lock_variant", "")), str(row["mode"]), int(row["n"]))].append(row)
    out: list[dict[str, object]] = []
    metrics = [
        "search_op_p99_ms",
        "search_index_p99_ms",
        "search_index_p999_ms",
        "search_post_index_p99_ms",
        "search_batch_lock_wait_sum_p99_ms",
        "search_batch_lock_wait_max_p99_ms",
        "search_l0_edges_sum_p99",
        "search_l0_expansions_sum_p99",
        "search_distance_comps_sum_p99",
        "search_adj_fetch_sum_p99_ms",
        "search_candidate_loop_sum_p99_ms",
        "search_dist_compute_sum_p99_ms",
        "search_query_cpu_sum_p99_ms",
        "search_unique_adj_4k_pages_sum_p99",
        "search_unique_data_4k_pages_sum_p99",
        "search_us_per_l0_edge_p50",
        "search_us_per_l0_edge_p99",
        "slow1pct_lock_wait_sum_mean_ms",
        "slow1pct_l0_edges_sum_mean",
        "slow1pct_us_per_l0_edge_mean",
        "insert_iqps_vec_s",
        "writer_existing_loop_fraction",
        "writer_link_critical_fraction",
        "writer_lock_wait_fraction",
        "writer_existing_edges_written_s",
        "writer_existing_edges_pruned_s",
        "writer_existing_edges_loaded_s",
        "reader_llc_load_miss_ratio",
        "reader_llc_load_misses",
        "reader_llc_store_miss_ratio",
        "competing_llc_load_miss_ratio",
        "competing_llc_load_misses",
        "competing_llc_store_misses",
        "uncore_reads_mib",
        "uncore_writes_mib",
    ]
    for key, rs in sorted(groups.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])):
        lock_variant, mode, n = key
        item: dict[str, object] = {
            "lock_variant": lock_variant,
            "mode": mode,
            "n": n,
            "reps": len(rs),
            "min_search_batches": min(int(row.get("search_batches", 0)) for row in rs),
            "bad_cells": sum(1 for row in rs if int(row.get("search_batches", 0)) < 100),
        }
        for metric in metrics:
            values = [float(row.get(metric, math.nan)) for row in rs]
            lo, hi = iqr(values)
            item[f"{metric}_median"] = median(values)
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_stdev"] = stdev(values)
            item[f"{metric}_iqr_lo"] = lo
            item[f"{metric}_iqr_hi"] = hi
        out.append(item)
    return out


def build_contrasts(agg: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {(str(row["lock_variant"]), str(row["mode"]), int(row["n"])): row for row in agg}
    out: list[dict[str, object]] = []
    variants = sorted({key[0] for key in by_key})
    ns = sorted({key[2] for key in by_key if key[2] > 0})
    metrics = [
        "search_index_p99_ms",
        "search_op_p99_ms",
        "search_batch_lock_wait_sum_p99_ms",
        "search_l0_edges_sum_p99",
        "search_l0_expansions_sum_p99",
        "search_us_per_l0_edge_p99",
        "slow1pct_lock_wait_sum_mean_ms",
        "slow1pct_l0_edges_sum_mean",
        "slow1pct_us_per_l0_edge_mean",
        "reader_llc_load_miss_ratio",
        "reader_llc_load_misses",
        "competing_llc_load_misses",
        "uncore_reads_mib",
        "uncore_writes_mib",
    ]
    for variant in variants:
        for n in ns:
            real = by_key.get((variant, "insert_real", n))
            if not real:
                continue
            for control in CONTROL_MODES:
                ctrl = by_key.get((variant, control, n))
                if not ctrl:
                    continue
                row: dict[str, object] = {
                    "lock_variant": variant,
                    "n": n,
                    "contrast": f"insert_real_vs_{control}",
                    "control": control,
                }
                for metric in metrics:
                    r = float(real.get(f"{metric}_median", math.nan))
                    c = float(ctrl.get(f"{metric}_median", math.nan))
                    row[f"{metric}_real"] = r
                    row[f"{metric}_control"] = c
                    row[f"{metric}_delta"] = r - c if not math.isnan(r) and not math.isnan(c) else math.nan
                    row[f"{metric}_ratio"] = ratio(r, c)
                out.append(row)
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
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(agg: list[dict[str, object]], contrasts: list[dict[str, object]]) -> None:
    print("mode                 N reps bad  search_idx_p99  lock_p99  l0_edges_p99  us/edge_p99  reader_llc_mr")
    print("-" * 104)
    for row in agg:
        print(
            f"{str(row['mode']):<20s} {int(row['n']):>2d} {int(row['reps']):>4d} {int(row['bad_cells']):>3d} "
            f"{float(row['search_index_p99_ms_median']):>14.3f} "
            f"{float(row['search_batch_lock_wait_sum_p99_ms_median']):>9.3f} "
            f"{float(row['search_l0_edges_sum_p99_median']):>13.0f} "
            f"{float(row['search_us_per_l0_edge_p99_median']):>12.4f} "
            f"{float(row['reader_llc_load_miss_ratio_median']):>13.3f}"
        )
    if not contrasts:
        return
    print()
    print("Key contrasts, positive means real insert is higher than the control:")
    for row in contrasts:
        print(
            f"N={int(row['n']):>2d} {row['contrast']:<34s} "
            f"d_idx_p99={float(row['search_index_p99_ms_delta']):>7.3f}ms "
            f"d_lock_p99={float(row['search_batch_lock_wait_sum_p99_ms_delta']):>7.3f}ms "
            f"d_edges={float(row['search_l0_edges_sum_p99_delta']):>8.0f} "
            f"d_llc_mr={float(row['reader_llc_load_miss_ratio_delta']):>7.3f} "
            f"d_comp_llc_miss={float(row['competing_llc_load_misses_delta']):>12.0f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    root = args.root
    out_dir = args.out_dir or root
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for cell in sorted(root.iterdir()):
        if not cell.is_dir() or not (cell / "cell.meta").exists():
            continue
        meta = load_meta(cell / "cell.meta")
        if meta.get("completed") != "1":
            continue
        if not (cell / "raw_latency" / "search_wide.csv").exists():
            continue
        rows.append(analyze_cell(cell))
    if not rows:
        raise SystemExit("no completed cells with raw_latency/search_wide.csv")

    agg = aggregate(rows)
    contrasts = build_contrasts(agg)
    write_csv(out_dir / "clean_search_cell_summary.csv", rows)
    write_csv(out_dir / "clean_search_aggregate.csv", agg)
    write_csv(out_dir / "clean_search_contrasts.csv", contrasts)
    print_summary(agg, contrasts)
    print(f"\nwrote {out_dir / 'clean_search_cell_summary.csv'}")
    print(f"wrote {out_dir / 'clean_search_aggregate.csv'}")
    print(f"wrote {out_dir / 'clean_search_contrasts.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
