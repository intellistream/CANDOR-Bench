#!/usr/bin/env python3
"""Paper-style period alignment figure for M3 latency/cache pressure."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def load_batches(path: Path) -> list[dict[str, float]]:
    seen: dict[tuple[int, int, int], dict[str, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            create = int(to_float(row.get("create_raw_ns")))
            start = int(to_float(row.get("start_raw_ns")))
            finish = int(to_float(row.get("finish_raw_ns")))
            if finish <= 0:
                continue
            seen[(create, start, finish)] = {
                "create_raw_ns": float(create),
                "start_raw_ns": float(start),
                "finish_raw_ns": float(finish),
                "op_ms": max(0.0, (finish - start) / 1_000_000.0),
                "e2e_ms": max(0.0, (finish - create) / 1_000_000.0),
            }
    return sorted(seen.values(), key=lambda r: (r["create_raw_ns"], r["finish_raw_ns"]))


def load_periods(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        return [{k: to_float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def rel_s(ns: float, origin_ns: float) -> float:
    return (ns - origin_ns) / 1_000_000_000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--xmax-s", type=float, default=2.25)
    args = parser.parse_args()

    periods = load_periods(args.target / "period_groups.csv")
    target = load_batches(args.target / "raw_latency" / "search_wide.csv")
    baseline = load_batches(args.baseline / "raw_latency" / "search_wide.csv")
    if not periods or not target or not baseline:
        raise SystemExit("missing period or search data")

    origin_ns = periods[0]["bin_start_ns"]
    n = min(len(target), len(baseline))
    batches: list[dict[str, float]] = []
    for i in range(n):
        row = dict(target[i])
        row["idx"] = float(i)
        row["e2e_delta_ms"] = target[i]["e2e_ms"] - baseline[i]["e2e_ms"]
        row["op_delta_ms"] = target[i]["op_ms"] - baseline[i]["op_ms"]
        row["finish_s"] = rel_s(row["finish_raw_ns"], origin_ns)
        batches.append(row)

    period_x = [rel_s((r["bin_start_ns"] + r["bin_end_ns"]) / 2.0, origin_ns) for r in periods]
    l2 = [r["perf_l2_rqsts_miss_per_s"] / 1.0e9 for r in periods]
    path = [r["insert_path_search_active_workers"] for r in periods]
    path_p90 = percentile(path, 90)
    baseline_periods = load_periods(args.baseline / "period_groups.csv")
    baseline_l2 = [
        r["perf_l2_rqsts_miss_per_s"] / 1.0e9
        for r in baseline_periods
        if r.get("perf_l2_rqsts_miss_per_s", 0.0) > 0.0
    ]
    baseline_l2_mean = sum(baseline_l2) / len(baseline_l2) if baseline_l2 else 0.0

    batches = [b for b in batches if 0.0 <= b["finish_s"] <= args.xmax_s]
    top = sorted(batches, key=lambda r: r["e2e_delta_ms"], reverse=True)[: args.top_n]
    top_times = [r["finish_s"] for r in top]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True)

    for ax in axes:
        for t in top_times:
            ax.axvline(t, color="#b84242", linewidth=0.9, alpha=0.65)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.8)

    axes[0].scatter(
        [b["finish_s"] for b in batches],
        [b["e2e_delta_ms"] for b in batches],
        s=12,
        color="#2f6db2",
        alpha=0.75,
        label="batch latency delta",
    )
    axes[0].scatter(
        top_times,
        [b["e2e_delta_ms"] for b in top],
        s=20,
        color="#b84242",
        zorder=3,
        label="top latency batches",
    )
    axes[0].set_ylabel("latency\nΔ ms")
    axes[0].legend(loc="upper right", fontsize=7, frameon=False)

    axes[1].plot(period_x, l2, color="#b84242", linewidth=1.1, label="full insert")
    axes[1].axhline(
        baseline_l2_mean,
        color="#333333",
        linestyle="--",
        linewidth=0.9,
        alpha=0.8,
        label="read-only mean",
    )
    axes[1].set_ylabel("L2 miss\n1e9/s")
    axes[1].legend(loc="upper right", fontsize=7, frameon=False)

    axes[2].plot(period_x, path, color="#2f6db2", linewidth=1.1)
    axes[2].axhline(path_p90, color="#2f6db2", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[2].set_ylabel("insert path\nworkers")
    axes[2].set_xlabel("time in run (s)")

    axes[0].set_title("Pre-experiment: latency spikes coincide with cache pressure and insert path search", fontsize=10.5)
    axes[-1].set_xlim(0, args.xmax_s)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    print(f"[m3-period-align] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
