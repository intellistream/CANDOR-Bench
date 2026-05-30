#!/usr/bin/env python3
"""Plot a time-aligned latency/cache/insert-phase trace for one M3 run."""

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
                "start_delay_ms": max(0.0, (start - create) / 1_000_000.0),
                "op_ms": max(0.0, (finish - start) / 1_000_000.0),
                "e2e_ms": max(0.0, (finish - create) / 1_000_000.0),
            }
    return sorted(seen.values(), key=lambda r: (r["create_raw_ns"], r["finish_raw_ns"]))


def load_periods(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        return [{k: to_float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def paired_deltas(target: list[dict[str, float]], baseline: list[dict[str, float]]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    n = min(len(target), len(baseline))
    for i in range(n):
        row = dict(target[i])
        base = baseline[i]
        row["batch_idx"] = float(i)
        row["delta_start_delay_ms"] = row["start_delay_ms"] - base["start_delay_ms"]
        row["delta_op_ms"] = row["op_ms"] - base["op_ms"]
        row["delta_e2e_ms"] = row["e2e_ms"] - base["e2e_ms"]
        out.append(row)
    return out


def rel_ms(ns: float, origin_ns: float) -> float:
    return (ns - origin_ns) / 1_000_000.0


def shade_top_windows(axes: list[plt.Axes], rows: list[dict[str, float]], origin_ns: float, top_n: int) -> list[dict[str, float]]:
    top = sorted(rows, key=lambda r: r["delta_e2e_ms"], reverse=True)[:top_n]
    for row in top:
        x0 = rel_ms(row["create_raw_ns"], origin_ns)
        x1 = rel_ms(row["finish_raw_ns"], origin_ns)
        for ax in axes:
            ax.axvspan(x0, x1, color="#c44949", alpha=0.10, linewidth=0)
    return top


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="target run directory")
    parser.add_argument("--baseline", type=Path, required=True, help="readonly baseline run directory")
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    target_batches = load_batches(args.target / "raw_latency" / "search_wide.csv")
    base_batches = load_batches(args.baseline / "raw_latency" / "search_wide.csv")
    periods = load_periods(args.target / "period_groups.csv")
    rows = paired_deltas(target_batches, base_batches)
    if not rows or not periods:
        raise SystemExit("missing rows or periods")

    origin_ns = min(periods[0]["bin_start_ns"], rows[0]["create_raw_ns"])
    x_batch = [rel_ms(r["finish_raw_ns"], origin_ns) for r in rows]
    x_period = [rel_ms((r["bin_start_ns"] + r["bin_end_ns"]) / 2.0, origin_ns) for r in periods]

    l2 = [r.get("perf_l2_rqsts_miss_per_s", 0.0) / 1.0e9 for r in periods]
    llc = [r.get("perf_llc_load_misses_per_s", 0.0) / 1.0e8 for r in periods]
    dtlb = [r.get("perf_dtlb_load_misses_walk_completed_per_s", 0.0) / 1.0e8 for r in periods]
    path_workers = [r.get("insert_path_search_active_workers", 0.0) for r in periods]
    l2_threshold = percentile(l2, 90)
    path_threshold = percentile(path_workers, 90)

    fig, axes = plt.subplots(4, 1, figsize=(13, 8.5), sharex=True)
    fig.suptitle(args.title, fontsize=14)
    top = shade_top_windows(list(axes), rows, origin_ns, args.top_n)

    axes[0].scatter(
        x_batch,
        [r["delta_e2e_ms"] for r in rows],
        color="#2f6db2",
        s=14,
        alpha=0.75,
        label="batch e2e delta",
    )
    axes[0].scatter(
        [rel_ms(r["finish_raw_ns"], origin_ns) for r in top],
        [r["delta_e2e_ms"] for r in top],
        color="#c44949",
        s=18,
        zorder=3,
        label=f"top {args.top_n} e2e deltas",
    )
    axes[0].set_ylabel("e2e delta\n(ms)")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)

    axes[1].scatter(
        x_batch,
        [r["delta_start_delay_ms"] for r in rows],
        color="#7b8794",
        s=12,
        alpha=0.70,
        label="start-delay delta",
    )
    axes[1].scatter(
        x_batch,
        [r["delta_op_ms"] for r in rows],
        color="#2f9c6b",
        s=12,
        alpha=0.70,
        label="op delta",
    )
    axes[1].set_ylabel("batch delta\n(ms)")
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)

    axes[2].plot(x_period, l2, color="#c44949", linewidth=1.0, label="L2 miss/s / 1e9")
    axes[2].plot(x_period, llc, color="#7a4fb3", linewidth=1.0, label="LLC load miss/s / 1e8")
    axes[2].plot(x_period, dtlb, color="#d08a28", linewidth=1.0, label="DTLB walks/s / 1e8")
    axes[2].axhline(l2_threshold, color="#c44949", linestyle="--", linewidth=0.8, alpha=0.7, label="L2 p90")
    axes[2].set_ylabel("cache pressure")
    axes[2].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)

    axes[3].plot(x_period, path_workers, color="#2f6db2", linewidth=1.1, label="insert_path_search active workers")
    axes[3].axhline(path_threshold, color="#2f6db2", linestyle="--", linewidth=0.8, alpha=0.7, label="path p90")
    axes[3].set_ylabel("path workers")
    axes[3].set_xlabel("time since run start (ms)")
    axes[3].legend(loc="upper left", fontsize=8, frameon=False)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    caption = (
        f"Red shaded bands are the top {args.top_n} batch e2e-delta windows. "
        "A useful causal picture should show these windows overlapping elevated cache pressure and insert path-search activity."
    )
    fig.text(0.01, 0.01, caption, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"[m3-cache-phase-timeline] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
