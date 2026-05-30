#!/usr/bin/env python3
"""Summarize how much of tail batch latency is node-lock wait."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = [(2500.0, 3500.0), (6000.0, 7000.0), (9500.0, 10500.0)]
COLS = [
    "measured",
    "finish_ms",
    "start_raw_ns",
    "finish_raw_ns",
    "op_ms",
    "work_level0_lock_wait_ms",
    "work_searchknn_thread_cpu_ms",
    "work_distance_computations",
]


def in_insert(ms: float) -> bool:
    return any(start <= ms < end for start, end in WINDOWS)


def load_batches(path: Path) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(path, usecols=COLS, chunksize=300000):
        chunks.append(chunk[chunk["measured"] == 1])
    if not chunks:
        return pd.DataFrame()
    rows = pd.concat(chunks, ignore_index=True)
    grouped = rows.groupby(["start_raw_ns", "finish_raw_ns"], sort=False)
    batches = grouped.agg(
        finish_ms=("finish_ms", "max"),
        n=("op_ms", "size"),
        op_per_query_ms=("op_ms", "first"),
        lock_sum_ms=("work_level0_lock_wait_ms", "sum"),
        lock_max_ms=("work_level0_lock_wait_ms", "max"),
        cpu_sum_ms=("work_searchknn_thread_cpu_ms", "sum"),
        distance_sum=("work_distance_computations", "sum"),
    ).reset_index()
    batches["batch_op_ms"] = batches["op_per_query_ms"] * batches["n"]
    batches["insert_active"] = batches["finish_ms"].map(in_insert)
    return batches


def summarize_tail(frame: pd.DataFrame, scope: str, percentile: float) -> dict[str, object]:
    threshold = float(frame["batch_op_ms"].quantile(percentile))
    tail = frame[frame["batch_op_ms"] >= threshold].copy()
    lock_max_frac = tail["lock_max_ms"] / tail["batch_op_ms"].replace(0.0, np.nan)
    lock_sum_frac = tail["lock_sum_ms"] / tail["batch_op_ms"].replace(0.0, np.nan)
    return {
        "scope": scope,
        "tail": f"top{(1.0 - percentile) * 100:.1f}%",
        "threshold_batch_ms": threshold,
        "tail_batches": int(len(tail)),
        "batch_op_median_ms": float(tail["batch_op_ms"].median()),
        "batch_op_max_ms": float(tail["batch_op_ms"].max()),
        "lock_max_median_ms": float(tail["lock_max_ms"].median()),
        "lock_max_p95_ms": float(tail["lock_max_ms"].quantile(0.95)),
        "lock_max_max_ms": float(tail["lock_max_ms"].max()),
        "lock_sum_median_ms": float(tail["lock_sum_ms"].median()),
        "lock_sum_p95_ms": float(tail["lock_sum_ms"].quantile(0.95)),
        "lock_sum_max_ms": float(tail["lock_sum_ms"].max()),
        "lock_max_frac_median": float(lock_max_frac.median()),
        "lock_max_frac_p95": float(lock_max_frac.quantile(0.95)),
        "lock_sum_frac_median": float(lock_sum_frac.median()),
        "lock_sum_frac_p95": float(lock_sum_frac.quantile(0.95)),
    }


def fmt(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("bench/result/m2_node_lock_hardware_matrix_20260507"))
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "sift200k_node_lock_on",
            "sift200k_node_lock_off",
            "gist200k_node_lock_on",
            "gist200k_node_lock_off",
        ],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper-progress/m3/search-tail-rootcause/tables/26_tail_batch_lock_wait_summary.csv"),
    )
    args = parser.parse_args()

    rows = []
    for case in args.cases:
        search = args.result_root / case / "raw_latency" / "search_wide.csv"
        batches = load_batches(search)
        if batches.empty:
            continue
        parts = [
            ("all", batches),
            ("insert", batches[batches["insert_active"]]),
            ("outside", batches[~batches["insert_active"]]),
        ]
        for scope, frame in parts:
            if frame.empty:
                continue
            for percentile in (0.99, 0.999):
                rows.append({"case": case, **summarize_tail(frame, scope, percentile)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).applymap(fmt).to_csv(args.out, index=False)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
