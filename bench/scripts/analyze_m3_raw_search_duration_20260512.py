#!/usr/bin/env python3
"""Aggregate raw timestamp search durations for M3 RW-tail runs.

The benchmark's op_ms/search_index_ms columns are per-query amortized values
for a batch. The raw timestamps record batch wall duration. This analyzer keeps
both views explicit so mechanism counters are compared to the right unit.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_meta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            out[key] = value
    return out


def percentile(values: list[float], q: float) -> float:
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


def median(values: list[float]) -> float:
    xs = [x for x in values if not math.isnan(x)]
    return statistics.median(xs) if xs else math.nan


def iqr(values: list[float]) -> tuple[float, float]:
    xs = [x for x in values if not math.isnan(x)]
    if not xs:
        return math.nan, math.nan
    return percentile(xs, 25), percentile(xs, 75)


def to_float(value: str | None, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def analyze_cell(cell: Path) -> dict[str, object] | None:
    meta = load_meta(cell / "cell.meta")
    if meta.get("completed") != "1":
        return None
    compact = cell / "raw_latency" / "search_compact.csv"
    if not compact.exists():
        return None
    skip_ms = to_float(meta.get("skip_pre_ms"), 0.0)
    sample_ms = to_float(meta.get("sample_ms"), 0.0)
    lo_ms = skip_ms
    hi_ms = skip_ms + sample_ms
    mode = meta.get("mode", cell.name.split("_n", 1)[0])
    n = int(to_float(meta.get("n_competing"), 0.0))
    rep = int(to_float(meta.get("rep"), 0.0))
    batch_size = int(to_float(meta.get("batch_size"), 20.0))
    raw_op: list[float] = []
    raw_e2e: list[float] = []
    col_op: list[float] = []
    seen: set[tuple[str, str, str]] = set()
    with compact.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("measured") != "1":
                continue
            finish_ms = to_float(row.get("finish_ms"))
            if not (lo_ms <= finish_ms <= hi_ms):
                continue
            key = (row.get("create_raw_ns", ""), row.get("start_raw_ns", ""), row.get("finish_raw_ns", ""))
            if key in seen:
                continue
            seen.add(key)
            create = int(row["create_raw_ns"])
            start = int(row["start_raw_ns"])
            finish = int(row["finish_raw_ns"])
            raw_op.append((finish - start) / 1e6)
            raw_e2e.append((finish - create) / 1e6)
            col_op.append(to_float(row.get("op_ms")))
    return {
        "cell": cell.name,
        "mode": mode,
        "n": n,
        "rep": rep,
        "batch_size": batch_size,
        "samples": len(raw_op),
        "raw_batch_op_p50_ms": percentile(raw_op, 50),
        "raw_batch_op_p99_ms": percentile(raw_op, 99),
        "raw_batch_op_p999_ms": percentile(raw_op, 99.9),
        "raw_batch_op_max_ms": max(raw_op) if raw_op else math.nan,
        "raw_batch_e2e_p99_ms": percentile(raw_e2e, 99),
        "amortized_op_p99_ms": percentile(col_op, 99),
        "raw_to_amortized_p99_ratio": percentile(raw_op, 99) / percentile(col_op, 99) if percentile(col_op, 99) else math.nan,
    }


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["mode"]), int(row["n"]))].append(row)
    metrics = [
        "raw_batch_op_p50_ms",
        "raw_batch_op_p99_ms",
        "raw_batch_op_p999_ms",
        "raw_batch_op_max_ms",
        "raw_batch_e2e_p99_ms",
        "amortized_op_p99_ms",
        "raw_to_amortized_p99_ratio",
    ]
    out: list[dict[str, object]] = []
    for (mode, n), rs in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        item: dict[str, object] = {
            "mode": mode,
            "n": n,
            "reps": len(rs),
            "samples_min": min(int(r["samples"]) for r in rs),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in rs]
            lo, hi = iqr(vals)
            item[f"{metric}_median"] = median(vals)
            item[f"{metric}_iqr_lo"] = lo
            item[f"{metric}_iqr_hi"] = hi
        out.append(item)
    return out


def contrasts(agg: list[dict[str, object]]) -> list[dict[str, object]]:
    by = {(str(r["mode"]), int(r["n"])): r for r in agg}
    out: list[dict[str, object]] = []
    for n in sorted({int(r["n"]) for r in agg if int(r["n"]) > 0}):
        real = by.get(("insert_real", n))
        if not real:
            continue
        for control in ("spin", "bg_search"):
            ctrl = by.get((control, n))
            if not ctrl:
                continue
            r = float(real["raw_batch_op_p99_ms_median"])
            c = float(ctrl["raw_batch_op_p99_ms_median"])
            out.append({
                "n": n,
                "contrast": f"insert_real_vs_{control}",
                "raw_batch_op_p99_real_ms": r,
                "raw_batch_op_p99_control_ms": c,
                "raw_batch_op_p99_delta_ms": r - c,
                "raw_batch_op_p99_ratio": r / c if c else math.nan,
            })
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.root
    rows = [row for cell in sorted(args.root.iterdir()) if cell.is_dir() for row in [analyze_cell(cell)] if row]
    agg = aggregate(rows)
    cont = contrasts(agg)
    write_csv(out_dir / "raw_search_duration_cell_summary.csv", rows)
    write_csv(out_dir / "raw_search_duration_aggregate.csv", agg)
    write_csv(out_dir / "raw_search_duration_contrasts.csv", cont)
    print("mode n reps raw_batch_p99 raw_batch_p999 amortized_p99 ratio")
    for row in agg:
        print(
            f"{row['mode']} {row['n']} {row['reps']} "
            f"{float(row['raw_batch_op_p99_ms_median']):.3f} "
            f"{float(row['raw_batch_op_p999_ms_median']):.3f} "
            f"{float(row['amortized_op_p99_ms_median']):.3f} "
            f"{float(row['raw_to_amortized_p99_ratio_median']):.2f}"
        )
    print("contrasts")
    for row in cont:
        print(
            f"N={row['n']} {row['contrast']} "
            f"delta={float(row['raw_batch_op_p99_delta_ms']):.3f}ms "
            f"ratio={float(row['raw_batch_op_p99_ratio']):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
