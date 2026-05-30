#!/usr/bin/env python3
"""Attribute M3 batch tails to hard/easy query tags."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean


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


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return fmean(vals) if vals else math.nan


def ratio(num: float, den: float) -> float:
    if not den or not math.isfinite(num) or not math.isfinite(den):
        return math.nan
    return num / den


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    mx = fmean(x for x, _ in pairs)
    my = fmean(y for _, y in pairs)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    denx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    deny = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    return num / (denx * deny) if denx and deny else math.nan


def row_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def row_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "") or 0.0))
    except ValueError:
        return 0


def read_tags(path: Path | None) -> set[int]:
    if path is None:
        return set()
    tags: set[int] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                tags.add(int(line))
    return tags


def load_hardness_scores(path: Path | None) -> dict[int, float]:
    if path is None:
        return {}
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("measured") != "1":
                continue
            if row_float(row, "finish_ms") < POST_START_MS:
                continue
            tag = row_int(row, "query_tag")
            sums[tag] = sums.get(tag, 0.0) + row_float(row, "work_base_search_ms")
            counts[tag] = counts.get(tag, 0) + 1
    return {tag: sums[tag] / counts[tag] for tag in sums if counts[tag]}


def find_case_dirs(root: Path) -> list[Path]:
    if (root / "raw_latency" / "search_wide.csv").exists():
        return [root]
    return sorted(p.parents[1] for p in root.glob("**/raw_latency/search_wide.csv"))


def load_batches(
    case_dir: Path,
    slow_tags: set[int],
    fast_tags: set[int],
    hardness_scores: dict[int, float],
) -> list[dict[str, float]]:
    path = case_dir / "raw_latency" / "search_wide.csv"
    batches: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("measured") != "1":
                continue
            finish_ms = row_float(row, "finish_ms")
            if finish_ms < POST_START_MS:
                continue
            key = (row["start_raw_ns"], row["finish_raw_ns"])
            item = batches.setdefault(
                key,
                {
                    "finish_ms": finish_ms,
                    "op_ms": row_float(row, "op_ms"),
                    "rows": 0.0,
                    "slow_count": 0.0,
                    "fast_count": 0.0,
                    "base_sum": 0.0,
                    "base_max": 0.0,
                    "thread_sum": 0.0,
                    "thread_max": 0.0,
                    "dist_sum": 0.0,
                    "dist_max": 0.0,
                    "edges_sum": 0.0,
                    "edges_max": 0.0,
                    "visited_sum": 0.0,
                    "visited_max": 0.0,
                    "hardness_sum": 0.0,
                    "hardness_max": 0.0,
                },
            )
            tag = row_int(row, "query_tag")
            base = row_float(row, "work_base_search_ms")
            thread = row_float(row, "work_searchknn_thread_cpu_ms")
            dist = row_float(row, "work_distance_computations")
            edges = row_float(row, "work_level0_edges_scanned")
            visited = row_float(row, "work_visited_nodes")

            item["rows"] += 1.0
            item["slow_count"] += 1.0 if tag in slow_tags else 0.0
            item["fast_count"] += 1.0 if tag in fast_tags else 0.0
            item["base_sum"] += base
            item["base_max"] = max(item["base_max"], base)
            item["thread_sum"] += thread
            item["thread_max"] = max(item["thread_max"], thread)
            item["dist_sum"] += dist
            item["dist_max"] = max(item["dist_max"], dist)
            item["edges_sum"] += edges
            item["edges_max"] = max(item["edges_max"], edges)
            item["visited_sum"] += visited
            item["visited_max"] = max(item["visited_max"], visited)
            hardness = hardness_scores.get(tag, 0.0)
            item["hardness_sum"] += hardness
            item["hardness_max"] = max(item["hardness_max"], hardness)

    out = list(batches.values())
    for item in out:
        rows = max(item["rows"], 1.0)
        item["base_mean"] = item["base_sum"] / rows
        item["thread_mean"] = item["thread_sum"] / rows
        item["dist_mean"] = item["dist_sum"] / rows
        item["edges_mean"] = item["edges_sum"] / rows
        item["visited_mean"] = item["visited_sum"] / rows
        item["hardness_mean"] = item["hardness_sum"] / rows
        item["has_slow"] = 1.0 if item["slow_count"] else 0.0
        item["has_fast"] = 1.0 if item["fast_count"] else 0.0
    return sorted(out, key=lambda b: b["finish_ms"])


def summarize(
    case_dir: Path,
    slow_tags: set[int],
    fast_tags: set[int],
    hardness_scores: dict[int, float],
) -> dict[str, object]:
    batches = load_batches(case_dir, slow_tags, fast_tags, hardness_scores)
    p99 = percentile([b["op_ms"] for b in batches], 0.99)
    top = [b for b in batches if b["op_ms"] >= p99]
    rest = [b for b in batches if b["op_ms"] < p99]
    has_slow = [b for b in batches if b["has_slow"]]
    no_slow = [b for b in batches if not b["has_slow"]]
    has_fast = [b for b in batches if b["has_fast"]]
    no_fast = [b for b in batches if not b["has_fast"]]
    tail_has_slow = [b for b in has_slow if b["op_ms"] >= p99]
    tail_no_slow = [b for b in no_slow if b["op_ms"] >= p99]
    tail_has_fast = [b for b in has_fast if b["op_ms"] >= p99]
    tail_no_fast = [b for b in no_fast if b["op_ms"] >= p99]

    def pct(items: list[dict[str, float]], pred_key: str) -> float:
        if not items:
            return math.nan
        return 100.0 * sum(1 for b in items if b[pred_key]) / len(items)

    def m(items: list[dict[str, float]], key: str) -> float:
        return mean([b[key] for b in items])

    has_slow_tail_rate = 100.0 * len(tail_has_slow) / max(1, len(has_slow))
    no_slow_tail_rate = 100.0 * len(tail_no_slow) / max(1, len(no_slow))
    has_fast_tail_rate = 100.0 * len(tail_has_fast) / max(1, len(has_fast))
    no_fast_tail_rate = 100.0 * len(tail_no_fast) / max(1, len(no_fast))
    op = [b["op_ms"] for b in batches]

    return {
        "case_dir": str(case_dir),
        "batches": len(batches),
        "post_p50_ms": percentile(op, 0.50),
        "post_p95_ms": percentile(op, 0.95),
        "post_p99_ms": p99,
        "all_has_slow_pct": pct(batches, "has_slow"),
        "top1_has_slow_pct": pct(top, "has_slow"),
        "has_slow_tail_rate_pct": has_slow_tail_rate,
        "no_slow_tail_rate_pct": no_slow_tail_rate,
        "has_slow_tail_enrichment": ratio(has_slow_tail_rate, no_slow_tail_rate),
        "all_has_fast_pct": pct(batches, "has_fast"),
        "top1_has_fast_pct": pct(top, "has_fast"),
        "has_fast_tail_rate_pct": has_fast_tail_rate,
        "no_fast_tail_rate_pct": no_fast_tail_rate,
        "has_fast_tail_enrichment": ratio(has_fast_tail_rate, no_fast_tail_rate),
        "top1_slow_count_mean": m(top, "slow_count"),
        "rest_slow_count_mean": m(rest, "slow_count"),
        "top1_fast_count_mean": m(top, "fast_count"),
        "rest_fast_count_mean": m(rest, "fast_count"),
        "top1_base_max_mean": m(top, "base_max"),
        "rest_base_max_mean": m(rest, "base_max"),
        "top1_base_sum_mean": m(top, "base_sum"),
        "rest_base_sum_mean": m(rest, "base_sum"),
        "top1_hardness_max_mean": m(top, "hardness_max"),
        "rest_hardness_max_mean": m(rest, "hardness_max"),
        "top1_hardness_sum_mean": m(top, "hardness_sum"),
        "rest_hardness_sum_mean": m(rest, "hardness_sum"),
        "top1_dist_sum_mean": m(top, "dist_sum"),
        "rest_dist_sum_mean": m(rest, "dist_sum"),
        "top1_edges_sum_mean": m(top, "edges_sum"),
        "rest_edges_sum_mean": m(rest, "edges_sum"),
        "top1_visited_sum_mean": m(top, "visited_sum"),
        "rest_visited_sum_mean": m(rest, "visited_sum"),
        "corr_op_slow_count": pearson(op, [b["slow_count"] for b in batches]),
        "corr_op_base_max": pearson(op, [b["base_max"] for b in batches]),
        "corr_op_base_sum": pearson(op, [b["base_sum"] for b in batches]),
        "corr_op_hardness_max": pearson(op, [b["hardness_max"] for b in batches]),
        "corr_op_hardness_sum": pearson(op, [b["hardness_sum"] for b in batches]),
        "corr_op_hardness_mean": pearson(op, [b["hardness_mean"] for b in batches]),
        "corr_op_dist_sum": pearson(op, [b["dist_sum"] for b in batches]),
        "corr_op_edges_sum": pearson(op, [b["edges_sum"] for b in batches]),
        "corr_op_visited_sum": pearson(op, [b["visited_sum"] for b in batches]),
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
    parser.add_argument("root", type=Path)
    parser.add_argument("--slow-tags", type=Path, required=True)
    parser.add_argument("--fast-tags", type=Path, default=None)
    parser.add_argument(
        "--hardness-source",
        type=Path,
        default=None,
        help="raw_latency/search_wide.csv used to score each query tag by mean base-search time",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    slow_tags = read_tags(args.slow_tags)
    fast_tags = read_tags(args.fast_tags)
    hardness_scores = load_hardness_scores(args.hardness_source)
    rows = [
        summarize(case_dir, slow_tags, fast_tags, hardness_scores)
        for case_dir in find_case_dirs(args.root)
    ]
    if not rows:
        raise SystemExit(f"no search_wide.csv found under {args.root}")

    out = args.out or (args.root / "query_hardness_batch_driver_summary.csv")
    write_csv(out, rows)
    for row in rows:
        print(
            f"{row['case_dir']}: p99={row['post_p99_ms']:.3f}ms "
            f"top_has_slow={row['top1_has_slow_pct']:.1f}% "
            f"all_has_slow={row['all_has_slow_pct']:.1f}% "
            f"slow_tail_enrich={row['has_slow_tail_enrichment']:.2f}x "
            f"corr(op,base_sum)={row['corr_op_base_sum']:.3f} "
            f"corr(op,hardness_sum)={row['corr_op_hardness_sum']:.3f} "
            f"corr(op,slow_count)={row['corr_op_slow_count']:.3f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
