#!/usr/bin/env python3
"""Bin ANNchor insert phase trace into dynamic per-period work metrics."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable


PHASES = [
    "insert_path_search",
    "existing_neighbor_load_scan",
    "existing_neighbor_prune",
    "existing_neighbor_undo_record",
    "existing_neighbor_rewrite",
]

WORK_COLUMNS = [
    "insert_path_edges_scanned",
    "insert_path_dist_comps",
    "existing_neighbor_visits",
    "existing_neighbor_edges_loaded",
    "existing_neighbor_prune_calls",
    "existing_neighbor_prune_candidates",
    "existing_neighbor_prune_dist_comps",
    "existing_neighbor_edges_written",
    "existing_neighbor_edges_pruned",
    "existing_neighbor_undo_edges_recorded",
]

SEARCH_COLUMNS = [
    "search_finish_count",
    "search_finish_op_ms_sum",
    "search_finish_e2e_ms_sum",
    "search_finish_op_ms_mean",
    "search_finish_e2e_ms_mean",
    "search_finish_op_ms_max",
    "search_finish_e2e_ms_max",
    "search_op_active_ns",
    "search_op_active_workers",
    "search_e2e_active_ns",
    "search_e2e_active_workers",
]


def read_uint(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(float(value))


def read_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


def read_meta_start_raw(path: Path | None) -> int | None:
    if path is None:
        return None
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("key") == "bench_start_raw_ns":
                return int(row["value"])
    return None


def load_insert_events(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        required = {"start_ns", "end_ns", "phase"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            start_ns = read_uint(row, "start_ns")
            end_ns = read_uint(row, "end_ns")
            if start_ns <= 0:
                continue
            if end_ns < start_ns:
                end_ns = start_ns
            events.append(
                {
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "phase": row["phase"],
                    "edges_loaded": read_float(row, "edges_loaded"),
                    "prune_candidates": read_float(row, "prune_candidates"),
                    "dist_comps": read_float(row, "dist_comps"),
                    "edges_written": read_float(row, "edges_written"),
                    "edges_pruned": read_float(row, "edges_pruned"),
                }
            )
    return events


def load_search_rows(path: Path, meta_start_raw_ns: int | None) -> list[dict[str, float]]:
    rows_by_finish: dict[tuple[float, float], dict[str, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "finish_raw_ns" not in fields and meta_start_raw_ns is None:
            raise ValueError(
                f"{path} has no finish_raw_ns; rerun the updated bench or pass a meta.csv with bench_start_raw_ns"
            )
        for row in reader:
            if "finish_raw_ns" in fields and row.get("finish_raw_ns"):
                finish_raw_ns = float(read_uint(row, "finish_raw_ns"))
            else:
                finish_raw_ns = float(meta_start_raw_ns or 0) + read_float(row, "finish_ms") * 1_000_000.0

            start_raw_ns = float(read_uint(row, "start_raw_ns")) if "start_raw_ns" in fields else 0.0
            create_raw_ns = float(read_uint(row, "create_raw_ns")) if "create_raw_ns" in fields else 0.0
            key = (finish_raw_ns, start_raw_ns)
            agg = rows_by_finish.setdefault(
                key,
                {
                    "finish_raw_ns": finish_raw_ns,
                    "start_raw_ns": start_raw_ns,
                    "create_raw_ns": create_raw_ns,
                    "op_ms": 0.0,
                    "e2e_ms": 0.0,
                    "sample_rows": 0.0,
                },
            )
            if start_raw_ns > 0 and (agg["start_raw_ns"] <= 0 or start_raw_ns < agg["start_raw_ns"]):
                agg["start_raw_ns"] = start_raw_ns
            if create_raw_ns > 0 and (agg["create_raw_ns"] <= 0 or create_raw_ns < agg["create_raw_ns"]):
                agg["create_raw_ns"] = create_raw_ns
            agg["op_ms"] += read_float(row, "op_ms")
            agg["e2e_ms"] += read_float(row, "e2e_ms")
            agg["sample_rows"] += 1.0
    return sorted(rows_by_finish.values(), key=lambda r: r["finish_raw_ns"])


def make_bin() -> dict[str, float]:
    row: dict[str, float] = defaultdict(float)
    for phase in PHASES:
        row[f"{phase}_active_ns"] = 0.0
    for col in WORK_COLUMNS + SEARCH_COLUMNS:
        row[col] = 0.0
    return row


def allocate_interval(
    rows: dict[int, dict[str, float]],
    origin_ns: int,
    bin_ns: int,
    start_ns: int,
    end_ns: int,
    apply: Callable[[dict[str, float], int, float], None],
) -> None:
    span_end_ns = end_ns if end_ns > start_ns else start_ns + 1
    duration_ns = max(1, span_end_ns - start_ns)
    first_bin = (start_ns - origin_ns) // bin_ns
    last_bin = ((span_end_ns - 1) - origin_ns) // bin_ns
    for bin_idx in range(int(first_bin), int(last_bin) + 1):
        bin_start = origin_ns + bin_idx * bin_ns
        bin_end = bin_start + bin_ns
        overlap = min(span_end_ns, bin_end) - max(start_ns, bin_start)
        if overlap <= 0:
            continue
        apply(rows[bin_idx], int(overlap), overlap / duration_ns)


def add_insert_event(
    rows: dict[int, dict[str, float]],
    origin_ns: int,
    bin_ns: int,
    event: dict[str, object],
) -> None:
    phase = str(event["phase"])
    if phase not in PHASES:
        return

    def apply(row: dict[str, float], overlap_ns: int, weight: float) -> None:
        row[f"{phase}_active_ns"] += overlap_ns
        if phase == "insert_path_search":
            row["insert_path_edges_scanned"] += float(event["edges_loaded"]) * weight
            row["insert_path_dist_comps"] += float(event["dist_comps"]) * weight
        elif phase == "existing_neighbor_load_scan":
            row["existing_neighbor_visits"] += weight
            row["existing_neighbor_edges_loaded"] += float(event["edges_loaded"]) * weight
        elif phase == "existing_neighbor_prune":
            row["existing_neighbor_prune_calls"] += weight
            row["existing_neighbor_prune_candidates"] += float(event["prune_candidates"]) * weight
            row["existing_neighbor_prune_dist_comps"] += float(event["dist_comps"]) * weight
        elif phase == "existing_neighbor_undo_record":
            row["existing_neighbor_undo_edges_recorded"] += float(event["edges_pruned"]) * weight
        elif phase == "existing_neighbor_rewrite":
            row["existing_neighbor_edges_written"] += float(event["edges_written"]) * weight
            row["existing_neighbor_edges_pruned"] += float(event["edges_pruned"]) * weight

    allocate_interval(
        rows,
        origin_ns,
        bin_ns,
        int(event["start_ns"]),
        int(event["end_ns"]),
        apply,
    )


def add_search_row(
    rows: dict[int, dict[str, float]],
    origin_ns: int,
    bin_ns: int,
    search: dict[str, float],
) -> None:
    finish_ns = int(search["finish_raw_ns"])
    op_ms = search["op_ms"]
    e2e_ms = search["e2e_ms"]
    finish_bin = int((finish_ns - origin_ns) // bin_ns)
    row = rows[finish_bin]
    row["search_finish_count"] += 1.0
    row["search_finish_op_ms_sum"] += op_ms
    row["search_finish_e2e_ms_sum"] += e2e_ms
    row["search_finish_op_ms_max"] = max(row["search_finish_op_ms_max"], op_ms)
    row["search_finish_e2e_ms_max"] = max(row["search_finish_e2e_ms_max"], e2e_ms)

    op_start = int(search.get("start_raw_ns", 0.0)) or finish_ns - int(op_ms * 1_000_000.0)
    e2e_start = int(search.get("create_raw_ns", 0.0)) or finish_ns - int(e2e_ms * 1_000_000.0)

    def add_op_active(active_row: dict[str, float], overlap_ns: int, _: float) -> None:
        active_row["search_op_active_ns"] += overlap_ns

    def add_e2e_active(active_row: dict[str, float], overlap_ns: int, _: float) -> None:
        active_row["search_e2e_active_ns"] += overlap_ns

    allocate_interval(rows, origin_ns, bin_ns, op_start, finish_ns, add_op_active)
    allocate_interval(rows, origin_ns, bin_ns, e2e_start, finish_ns, add_e2e_active)


def choose_origin(
    events: list[dict[str, object]],
    searches: list[dict[str, float]],
    explicit_origin: int | None,
    meta_origin: int | None,
) -> int:
    if explicit_origin is not None:
        return explicit_origin
    if meta_origin is not None:
        return meta_origin
    starts: list[int] = [int(e["start_ns"]) for e in events]
    starts.extend(int(s["finish_raw_ns"] - s["e2e_ms"] * 1_000_000.0) for s in searches)
    if not starts:
        raise ValueError("no insert events or search rows to bin")
    return min(starts)


def output_rows(
    rows: dict[int, dict[str, float]],
    origin_ns: int,
    bin_ns: int,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    if not rows:
        return out
    for bin_idx in range(min(rows), max(rows) + 1):
        row = rows[bin_idx]
        result: dict[str, object] = {
            "bin_idx": bin_idx,
            "bin_start_ns": origin_ns + bin_idx * bin_ns,
            "bin_end_ns": origin_ns + (bin_idx + 1) * bin_ns,
            "bin_start_ms": bin_idx * bin_ns / 1_000_000.0,
            "bin_end_ms": (bin_idx + 1) * bin_ns / 1_000_000.0,
        }

        existing_update_active_ns = 0.0
        for phase in PHASES:
            active_ns = row[f"{phase}_active_ns"]
            result[f"{phase}_active_ns"] = active_ns
            result[f"{phase}_active_workers"] = active_ns / bin_ns
            if phase.startswith("existing_neighbor_"):
                existing_update_active_ns += active_ns
        result["existing_neighbor_update_active_ns"] = existing_update_active_ns
        result["existing_neighbor_update_active_workers"] = existing_update_active_ns / bin_ns

        for col in WORK_COLUMNS:
            result[col] = row[col]

        finish_count = row["search_finish_count"]
        if finish_count > 0:
            row["search_finish_op_ms_mean"] = row["search_finish_op_ms_sum"] / finish_count
            row["search_finish_e2e_ms_mean"] = row["search_finish_e2e_ms_sum"] / finish_count
        row["search_op_active_workers"] = row["search_op_active_ns"] / bin_ns
        row["search_e2e_active_workers"] = row["search_e2e_active_ns"] / bin_ns
        for col in SEARCH_COLUMNS:
            result[col] = row[col]
        out.append(result)
    return out


def write_csv(path: Path | None, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    if path is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ANNCHOR_INSERT_PHASE_TRACE event CSV into fixed-size dynamic "
            "period metrics. Event work is split across bins by exact time overlap."
        )
    )
    parser.add_argument("trace", type=Path, help="CSV written by ANNCHOR_INSERT_PHASE_TRACE")
    parser.add_argument("--out", type=Path, help="output period CSV; default writes to stdout")
    parser.add_argument("--bin-ms", type=float, default=5.0, help="period width in ms")
    parser.add_argument("--search-wide", type=Path, help="optional raw_latency/search_wide.csv to bin on the same raw clock")
    parser.add_argument("--meta", type=Path, help="optional raw_latency/meta.csv containing bench_start_raw_ns")
    parser.add_argument("--origin-raw-ns", type=int, help="explicit raw-clock origin for bin_idx=0")
    parser.add_argument(
        "--include-pre-origin",
        action="store_true",
        help="keep trace events before the origin; default drops initial-build events before bench_start_raw_ns",
    )
    args = parser.parse_args()

    if args.bin_ms <= 0:
        raise ValueError("--bin-ms must be positive")
    bin_ns = int(args.bin_ms * 1_000_000.0)
    if bin_ns <= 0:
        raise ValueError("--bin-ms is too small")

    events = load_insert_events(args.trace)
    meta_origin = read_meta_start_raw(args.meta)
    searches = load_search_rows(args.search_wide, meta_origin) if args.search_wide else []
    origin_ns = choose_origin(events, searches, args.origin_raw_ns, meta_origin)

    rows: dict[int, dict[str, float]] = defaultdict(make_bin)
    for event in events:
        if not args.include_pre_origin:
            if int(event["end_ns"]) <= origin_ns:
                continue
            if int(event["start_ns"]) < origin_ns:
                event = {**event, "start_ns": origin_ns}
        add_insert_event(rows, origin_ns, bin_ns, event)
    for search in searches:
        add_search_row(rows, origin_ns, bin_ns, search)

    write_csv(args.out, output_rows(rows, origin_ns, bin_ns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
