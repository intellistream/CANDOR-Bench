#!/usr/bin/env python3
"""Summarize fixed-worker closed-loop M3 tail runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def pct(values: list[float], q: float) -> float:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def benchmark_row(run_dir: Path) -> dict[str, str]:
    path = run_dir / "benchmark_results.csv"
    if not path.exists():
        return {}
    rows = load_csv(path)
    return rows[-1] if rows else {}


def raw_search(run_dir: Path) -> list[dict[str, str]]:
    wide = run_dir / "raw_latency" / "search_wide.csv"
    compact = run_dir / "raw_latency" / "search_compact.csv"
    path = wide if wide.exists() else compact
    return load_csv(path)


def active_workers(rows: list[dict[str, str]], measured: bool, duration_ms: float) -> float:
    if duration_ms <= 0:
        return float("nan")
    total_ms = 0.0
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        is_measured = f(row.get("measured")) >= 0.5
        if is_measured != measured:
            continue
        key = (
            row.get("measured", ""),
            row.get("create_raw_ns", ""),
            row.get("start_raw_ns", ""),
            row.get("finish_raw_ns", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        total_ms += max(0.0, (f(row.get("finish_raw_ns")) - f(row.get("start_raw_ns"))) / 1_000_000.0)
    return total_ms / duration_ms


def insert_active_workers(run_dir: Path, duration_ms: float) -> float:
    path = run_dir / "raw_latency" / "insert_wide.csv"
    if not path.exists() or duration_ms <= 0:
        return 0.0
    total_ms = 0.0
    for row in load_csv(path):
        total_ms += max(0.0, (f(row.get("finish_raw_ns")) - f(row.get("start_raw_ns"))) / 1_000_000.0)
    return total_ms / duration_ms


def summarize_case(case: dict[str, str], queue_max_ms: float) -> dict[str, object]:
    run_dir = Path(case["run_dir"])
    duration_ms = f(case.get("duration_ms"))
    rows = raw_search(run_dir)
    measured = [row for row in rows if f(row.get("measured")) >= 0.5]
    lowq = [
        row for row in measured
        if max(0.0, f(row.get("e2e_ms")) - f(row.get("op_ms"))) <= queue_max_ms
    ]
    bench = benchmark_row(run_dir)
    out: dict[str, object] = {
        "case": case["case"],
        "group": case["group"],
        "workers": case["workers"],
        "ef_construction": case.get("ef_construction", ""),
        "mutation": case.get("mutation", ""),
        "duration_ms": duration_ms,
        "measured_rows": len(measured),
        "lowq_rows": len(lowq),
        "op_p50_lowq_ms": pct([f(row.get("op_ms")) for row in lowq], 50),
        "op_p95_lowq_ms": pct([f(row.get("op_ms")) for row in lowq], 95),
        "op_p99_lowq_ms": pct([f(row.get("op_ms")) for row in lowq], 99),
        "op_p999_lowq_ms": pct([f(row.get("op_ms")) for row in lowq], 99.9),
        "queue_p95_all_ms": pct([max(0.0, f(row.get("e2e_ms")) - f(row.get("op_ms"))) for row in measured], 95),
        "dist_p95_lowq": pct([f(row.get("diag_dist_comps")) for row in lowq], 95),
        "hops_p95_lowq": pct([f(row.get("diag_hops")) for row in lowq], 95),
        "measured_active_workers": active_workers(rows, True, duration_ms),
        "bg_search_active_workers": active_workers(rows, False, duration_ms),
        "insert_active_workers": insert_active_workers(run_dir, duration_ms),
        "search_qps_per_point": f(bench.get("search_qps (per-point)")),
        "insert_qps_per_point": f(bench.get("insert_qps (per-point)")),
        "recall": f(bench.get("recall")),
        "run_dir": str(run_dir),
    }
    return out


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def make_pairs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baselines = {str(row["workers"]): row for row in rows if row["group"] == "bg_search"}
    idle = next((row for row in rows if row["group"] == "baseline"), None)
    pairs: list[dict[str, object]] = []
    for row in rows:
        if not str(row["group"]).startswith("insert"):
            continue
        bg = baselines.get(str(row["workers"]))
        if bg is None:
            continue
        item: dict[str, object] = {
            "insert_case": row["case"],
            "bg_case": bg["case"],
            "workers": row["workers"],
            "insert_p95_ms": row["op_p95_lowq_ms"],
            "bg_p95_ms": bg["op_p95_lowq_ms"],
            "delta_p95_ms": f(row["op_p95_lowq_ms"]) - f(bg["op_p95_lowq_ms"]),
            "insert_p99_ms": row["op_p99_lowq_ms"],
            "bg_p99_ms": bg["op_p99_lowq_ms"],
            "delta_p99_ms": f(row["op_p99_lowq_ms"]) - f(bg["op_p99_lowq_ms"]),
            "insert_p999_ms": row["op_p999_lowq_ms"],
            "bg_p999_ms": bg["op_p999_lowq_ms"],
            "delta_p999_ms": f(row["op_p999_lowq_ms"]) - f(bg["op_p999_lowq_ms"]),
            "insert_queue_p95_ms": row["queue_p95_all_ms"],
            "bg_queue_p95_ms": bg["queue_p95_all_ms"],
            "insert_active_workers": row["insert_active_workers"],
            "bg_search_active_workers": bg["bg_search_active_workers"],
            "insert_qps_per_point": row["insert_qps_per_point"],
            "bg_search_qps_per_point": bg["search_qps_per_point"],
        }
        if idle is not None:
            item["baseline_p99_ms"] = idle["op_p99_lowq_ms"]
            item["delta_p99_vs_baseline_ms"] = f(row["op_p99_lowq_ms"]) - f(idle["op_p99_lowq_ms"])
        pairs.append(item)
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-pairs", type=Path, required=True)
    parser.add_argument("--queue-max-ms", type=float, default=0.1)
    args = parser.parse_args()

    cases = load_csv(args.manifest)
    summary = [summarize_case(case, args.queue_max_ms) for case in cases]
    fields = [
        "case", "group", "workers", "ef_construction", "mutation", "duration_ms",
        "measured_rows", "lowq_rows", "op_p50_lowq_ms", "op_p95_lowq_ms",
        "op_p99_lowq_ms", "op_p999_lowq_ms", "queue_p95_all_ms",
        "dist_p95_lowq", "hops_p95_lowq", "measured_active_workers",
        "bg_search_active_workers", "insert_active_workers", "search_qps_per_point",
        "insert_qps_per_point", "recall", "run_dir",
    ]
    write_csv(args.out_summary, summary, fields)
    pairs = make_pairs(summary)
    pair_fields = [
        "insert_case", "bg_case", "workers", "insert_p95_ms", "bg_p95_ms",
        "delta_p95_ms", "insert_p99_ms", "bg_p99_ms", "delta_p99_ms",
        "insert_p999_ms", "bg_p999_ms", "delta_p999_ms",
        "insert_queue_p95_ms", "bg_queue_p95_ms", "insert_active_workers",
        "bg_search_active_workers", "insert_qps_per_point",
        "bg_search_qps_per_point", "baseline_p99_ms", "delta_p99_vs_baseline_ms",
    ]
    write_csv(args.out_pairs, pairs, pair_fields)
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
