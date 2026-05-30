#!/usr/bin/env python3
"""Attribute perf c2c cache lines to plain-HNSW memory ranges."""
from __future__ import annotations

import argparse
import csv
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


ROW_RE = re.compile(r"^\s*(\d+)\s+(0x[0-9a-fA-F]+)\s+")


def parse_stats(path: Path) -> dict[str, str]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {}
    stats = rows[-1].get("stats", "")
    out: dict[str, str] = {}
    for item in stats.split(","):
        if ":" not in item:
            continue
        key, value = item.strip().split(":", 1)
        out[key.strip()] = value.strip()
    return out


def parse_range_dump(path: Path | None) -> tuple[list[tuple[int, int, str]], list[int]]:
    if path is None or not path.exists():
        return [], []
    ranges: list[tuple[int, int, str]] = []
    for row in csv.DictReader(path.open()):
        try:
            begin = parse_int(row.get("begin", "0"))
            end = parse_int(row.get("end", "0"))
        except ValueError:
            continue
        cls = row.get("class", "")
        if cls and begin < end:
            ranges.append((begin, end, cls))
    ranges.sort(key=lambda item: item[0])
    return ranges, [begin for begin, _, _ in ranges]


def parse_int(value: str, default: int = 0) -> int:
    if not value:
        return default
    return int(value, 16) if value.startswith("0x") else int(float(value))


def classify_dump_range(addr: int, ranges: list[tuple[int, int, str]], begins: list[int]) -> str | None:
    idx = bisect_right(begins, addr) - 1
    if idx >= 0:
        begin, end, cls = ranges[idx]
        if begin <= addr < end:
            return cls
    return None


def classify(addr: int, stats: dict[str, str], ranges: list[tuple[int, int, str]], begins: list[int]) -> str:
    level0_begin = parse_int(stats.get("hnsw_level0_begin", "0"))
    level0_end = parse_int(stats.get("hnsw_level0_end", "0"))
    locks_begin = parse_int(stats.get("hnsw_locks_begin", "0"))
    locks_end = parse_int(stats.get("hnsw_locks_end", "0"))
    ptr_begin = parse_int(stats.get("hnsw_linklists_ptr_begin", "0"))
    ptr_end = parse_int(stats.get("hnsw_linklists_ptr_end", "0"))
    element_levels_begin = parse_int(stats.get("hnsw_element_levels_begin", "0"))
    element_levels_end = parse_int(stats.get("hnsw_element_levels_end", "0"))
    object_begin = parse_int(stats.get("hnsw_object_begin", "0"))
    object_end = parse_int(stats.get("hnsw_object_end", "0"))

    if level0_begin <= addr < level0_end:
        size_data = parse_int(stats.get("hnsw_size_data_per_element", "0"))
        offset_level0 = parse_int(stats.get("hnsw_offset_level0", "0"))
        size_links0 = parse_int(stats.get("hnsw_size_links_level0", "0"))
        offset_data = parse_int(stats.get("hnsw_offset_data", "0"))
        off = (addr - level0_begin) % size_data if size_data > 0 else -1
        if offset_level0 <= off < offset_level0 + size_links0:
            return "level0_adjacency"
        if off >= offset_data:
            return "vector_payload"
        return "level0_record_metadata"
    if locks_begin <= addr < locks_end:
        return "node_lock_array"
    if ptr_begin <= addr < ptr_end:
        return "upper_link_pointer_array"
    if element_levels_begin <= addr < element_levels_end:
        return "element_levels_array"
    if object_begin <= addr < object_end:
        return "hnsw_object_metadata"
    dumped = classify_dump_range(addr, ranges, begins)
    if dumped is not None and dumped != "level0_records":
        return dumped
    return "other_or_upper_heap"


def parse_c2c_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    in_table = False
    for line in path.read_text(errors="ignore").splitlines():
        if "Shared Data Cache Line Table" in line:
            in_table = True
            continue
        if in_table and "Shared Cache Line Distribution Pareto" in line:
            break
        if not in_table:
            continue
        match = ROW_RE.match(line)
        if not match:
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        try:
            rows.append(
                {
                    "idx": int(parts[0]),
                    "addr": int(parts[1], 16),
                    "node": parts[2],
                    "pa_cnt": int(parts[3]),
                    "hitm_total": int(parts[5]),
                    "hitm_local": int(parts[6]),
                    "hitm_remote": int(parts[7]),
                    "records": int(parts[8]),
                    "loads": int(parts[9]),
                    "stores": int(parts[10]),
                }
            )
        except (ValueError, IndexError):
            continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, type=Path)
    parser.add_argument("--report", default="perf_c2c_report.txt")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--ranges",
        type=Path,
        help="Optional address range CSV dumped by ANNCHOR_M3_ADDR_RANGE_DUMP",
    )
    args = parser.parse_args()

    stats = parse_stats(args.cell / "benchmark_results.csv")
    ranges, begins = parse_range_dump(args.ranges or (args.cell / "address_ranges.csv"))
    report = args.cell / args.report
    rows = parse_c2c_rows(report)
    for row in rows:
        row["class"] = classify(int(row["addr"]), stats, ranges, begins)

    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        bucket = str(row["class"])
        for key in ("hitm_total", "hitm_local", "hitm_remote", "records", "loads", "stores"):
            totals[bucket][key] += int(row[key])
        totals[bucket]["lines"] += 1

    out = args.out or args.cell / "perf_c2c_address_attribution.csv"
    with out.open("w", newline="") as f:
        fieldnames = ["class", "lines", "records", "loads", "stores", "hitm_total", "hitm_local", "hitm_remote"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cls, vals in sorted(totals.items(), key=lambda item: item[1]["records"], reverse=True):
            writer.writerow({"class": cls, **{k: vals.get(k, 0) for k in fieldnames if k != "class"}})

    print(f"cell={args.cell.name}")
    print("class lines records loads stores hitm_total hitm_local hitm_remote")
    for cls, vals in sorted(totals.items(), key=lambda item: item[1]["records"], reverse=True):
        print(
            cls,
            vals.get("lines", 0),
            vals.get("records", 0),
            vals.get("loads", 0),
            vals.get("stores", 0),
            vals.get("hitm_total", 0),
            vals.get("hitm_local", 0),
            vals.get("hitm_remote", 0),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
