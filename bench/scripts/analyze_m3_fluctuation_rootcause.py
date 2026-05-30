#!/usr/bin/env python3
"""Summarize true batch-level search E2E fluctuation from raw latency dumps."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[int((len(s) - 1) * p)]


def batch_size_from_name(name: str) -> int:
    match = re.search(r"_b(\d+)_", name)
    if not match:
        raise ValueError(f"cannot infer batch size from {name}")
    return int(match.group(1))


def true_batch_series(path: Path, batch_size: int) -> tuple[list[float], list[float], list[float], list[float]]:
    e2e_batches: list[float] = []
    op_batches: list[float] = []
    active_batches: list[float] = []
    missed_batches: list[float] = []

    cur_e2e: list[float] = []
    cur_op: list[float] = []
    cur_active: list[float] = []
    cur_missed: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur_e2e.append(float(row["e2e_ms"]))
            cur_op.append(float(row["op_ms"]))
            cur_active.append(float(row.get("freshness_active_front_pts", 0.0)))
            cur_missed.append(float(row.get("freshness_missed_e2e_pts", 0.0)))
            if len(cur_e2e) == batch_size:
                # run_m3_fluctuation_rootcause.sh enables per_query_latency, so each
                # task contributes batch_size amortized rows. Sum them back to the
                # original batch-level latency.
                e2e_batches.append(sum(cur_e2e))
                op_batches.append(sum(cur_op))
                active_batches.append(sum(cur_active) / len(cur_active))
                missed_batches.append(sum(cur_missed) / len(cur_missed))
                cur_e2e.clear()
                cur_op.clear()
                cur_active.clear()
                cur_missed.clear()
    return e2e_batches, op_batches, active_batches, missed_batches


def worst_window(values: list[float], width: int) -> tuple[float, int]:
    best = 0.0
    best_idx = 0
    for i in range(0, len(values), width):
        chunk = values[i : i + width]
        if len(chunk) < max(10, width // 5):
            break
        avg = sum(chunk) / len(chunk)
        if avg > best:
            best = avg
            best_idx = i
    return best, best_idx


def parse_label(label: str) -> dict[str, str]:
    parts = label.split("_")
    out = {"label": label}
    for part in parts:
        if part.startswith("b") and part[1:].isdigit():
            out["batch"] = part[1:]
        elif part in {"chasing", "round"}:
            out["query_mode_part"] = part
        elif part == "robin" and out.get("query_mode_part") == "round":
            out["query_mode"] = "round_robin"
        elif part.startswith("w") and part[1:].isdigit():
            out["write_rate"] = part[1:]
        elif part.startswith("r") and part[1:].isdigit():
            out["read_rate"] = part[1:]
        elif part in {"nodelock", "nolock", "rwlock"}:
            out["sync_mode"] = part
    if out.get("query_mode_part") == "chasing":
        out["query_mode"] = "chasing"
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: analyze_m3_fluctuation_rootcause.py <result_dir>", file=sys.stderr)
        return 2

    result_dir = Path(sys.argv[1])
    rows: list[dict[str, object]] = []
    for wide in sorted(result_dir.glob("*/raw_latency/search_wide.csv")):
        label = wide.parents[1].name
        batch_size = batch_size_from_name(label)
        e2e, op, active, missed = true_batch_series(wide, batch_size)
        if not e2e:
            continue
        worst, worst_idx = worst_window(e2e, 100)
        worst_op, _ = worst_window(op, 100)
        row: dict[str, object] = {
            **parse_label(label),
            "batch": batch_size,
            "num_batches": len(e2e),
            "mean_batch_e2e_ms": sum(e2e) / len(e2e),
            "p95_batch_e2e_ms": percentile(e2e, 0.95),
            "p99_batch_e2e_ms": percentile(e2e, 0.99),
            "max_batch_e2e_ms": max(e2e),
            "worst100_batch_e2e_ms": worst,
            "worst100_over_mean": worst / (sum(e2e) / len(e2e)),
            "worst100_batch_op_ms": worst_op,
            "worst100_queue_ms_est": worst - worst_op,
            "avg_active_front_pts": sum(active) / len(active),
            "max_active_front_pts": max(active),
            "avg_missed_e2e_pts": sum(missed) / len(missed),
            "max_missed_e2e_pts": max(missed),
            "worst100_start_batch": worst_idx,
        }
        rows.append(row)

    out = result_dir / "m3_fluctuation_rootcause_summary.csv"
    fieldnames = [
        "label",
        "sync_mode",
        "batch",
        "query_mode",
        "write_rate",
        "read_rate",
        "num_batches",
        "mean_batch_e2e_ms",
        "p95_batch_e2e_ms",
        "p99_batch_e2e_ms",
        "max_batch_e2e_ms",
        "worst100_batch_e2e_ms",
        "worst100_over_mean",
        "worst100_batch_op_ms",
        "worst100_queue_ms_est",
        "avg_active_front_pts",
        "max_active_front_pts",
        "avg_missed_e2e_pts",
        "max_missed_e2e_pts",
        "worst100_start_batch",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    for row in sorted(rows, key=lambda r: float(r["p99_batch_e2e_ms"]), reverse=True):
        print(
            "{label}: sync={sync_mode} b={batch} mode={query_mode} w/r={write_rate}/{read_rate} "
            "mean={mean_batch_e2e_ms:.2f} p99={p99_batch_e2e_ms:.2f} "
            "worst100={worst100_batch_e2e_ms:.2f} op100={worst100_batch_op_ms:.2f} "
            "queue_est={worst100_queue_ms_est:.2f} active_max={max_active_front_pts:.0f}".format(**row)
        )
    print(f"[m3-rootcause] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
