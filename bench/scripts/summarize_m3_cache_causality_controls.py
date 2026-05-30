#!/usr/bin/env python3
"""Summarize M3 writer-control runs against a read-only replay baseline."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODES = [
    ("cpu_dummy", "CPU dummy"),
    ("ann_search_dummy", "ANN-search dummy"),
    ("path_only_insert_control", "path-only insert"),
    ("full_insert", "full insert"),
]
BASELINE = "readonly_noop"


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * pct)]


def load_search_batches(path: Path, batch_size: int) -> dict[int, dict[str, float]]:
    batches: dict[int, dict[str, list[float] | float]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            idx = int(f(row.get("sample_idx"))) // batch_size
            batch = batches.setdefault(idx, {"op": [], "e2e": [], "finish": 0.0})
            batch["op"].append(f(row.get("op_ms")))  # type: ignore[union-attr]
            batch["e2e"].append(f(row.get("e2e_ms")))  # type: ignore[union-attr]
            batch["finish"] = max(float(batch["finish"]), f(row.get("finish_raw_ns")))
    return {
        idx: {
            "op_ms": statistics.mean(batch["op"]),  # type: ignore[arg-type]
            "e2e_ms": statistics.mean(batch["e2e"]),  # type: ignore[arg-type]
            "finish_raw_ns": float(batch["finish"]),
        }
        for idx, batch in batches.items()
    }


def load_period_summary(path: Path) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows.append({key: f(value) for key, value in row.items()})
    if not rows:
        return {}

    def mean(key: str) -> float:
        return statistics.mean(row.get(key, 0.0) for row in rows)

    def p95(key: str) -> float:
        return percentile([row.get(key, 0.0) for row in rows], 0.95)

    return {
        "writer_graph_occupancy_mean": mean("insert_path_search_active_workers"),
        "l2_miss_rate_mean": mean("perf_l2_rqsts_miss_per_s"),
        "llc_load_miss_rate_mean": mean("perf_llc_load_misses_per_s"),
        "dtlb_walk_rate_mean": mean("perf_dtlb_load_misses_walk_completed_per_s"),
        "llc_load_miss_rate_p95": p95("perf_llc_load_misses_per_s"),
    }


def load_benchmark(path: Path) -> dict[str, float]:
    with path.open(newline="") as file:
        row = next(csv.DictReader(file))
    return {
        "insert_qps": f(row.get("insert_qps (per-point)")),
        "insert_e2e_mean_ms": f(row.get("insert_mean_e2e_latency_ms (per-batch)")),
        "search_e2e_mean_ms": f(row.get("search_mean_e2e_latency_ms (per-batch)")),
        "search_e2e_p95_ms": f(row.get("search_p95_e2e_latency_ms (per-batch)")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--case", default="high_writer_concurrency/sift1m_b200_round_robin_w40000_r5000_nolock")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-fig", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_search_batches(args.root / BASELINE / args.case / "raw_latency/search_wide.csv", args.batch_size)
    rows: list[dict[str, float | str]] = []
    for mode, label in MODES:
        run_dir = args.root / mode / args.case
        batches = load_search_batches(run_dir / "raw_latency/search_wide.csv", args.batch_size)
        common = sorted(set(baseline) & set(batches))
        e2e_delta = [batches[idx]["e2e_ms"] - baseline[idx]["e2e_ms"] for idx in common]
        op_delta = [batches[idx]["op_ms"] - baseline[idx]["op_ms"] for idx in common]
        period = load_period_summary(run_dir / "period_groups.csv")
        bench = load_benchmark(run_dir / "benchmark_results.csv")
        row: dict[str, float | str] = {
            "mode": mode,
            "label": label,
            "matched_batches": len(common),
            "e2e_overhead_mean_ms": statistics.mean(e2e_delta) if e2e_delta else 0.0,
            "e2e_overhead_p95_ms": percentile(e2e_delta, 0.95),
            "e2e_overhead_p99_ms": percentile(e2e_delta, 0.99),
            "op_overhead_mean_ms": statistics.mean(op_delta) if op_delta else 0.0,
            "op_overhead_p95_ms": percentile(op_delta, 0.95),
            "wait_overhead_mean_ms": (statistics.mean(e2e_delta) - statistics.mean(op_delta)) if e2e_delta and op_delta else 0.0,
            "wait_overhead_p95_ms": percentile([e - o for e, o in zip(e2e_delta, op_delta)], 0.95),
        }
        row.update(period)
        row.update(bench)
        rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with args.out_md.open("w") as file:
        file.write("| mode | e2e overhead mean ms | wait overhead mean ms | op overhead mean ms | LLC load miss rate mean /s | DTLB walk rate mean /s | insert QPS |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                f"| {row['label']} | {row['e2e_overhead_mean_ms']:.3f} | {row['wait_overhead_mean_ms']:.3f} | {row['op_overhead_mean_ms']:.3f} | "
                f"{row['llc_load_miss_rate_mean'] / 1e6:.1f}M | {row['dtlb_walk_rate_mean'] / 1e6:.1f}M | {row['insert_qps']:.0f} |\n"
            )

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    fig.patch.set_facecolor("white")
    labels = [str(row["label"]) for row in rows]
    x = range(len(rows))
    e2e_mean = [float(row["e2e_overhead_mean_ms"]) for row in rows]
    e2e_p95 = [float(row["e2e_overhead_p95_ms"]) for row in rows]
    llc = [float(row["llc_load_miss_rate_mean"]) / 1e6 for row in rows]
    dtlb = [float(row["dtlb_walk_rate_mean"]) / 1e6 for row in rows]

    width = 0.36
    wait_mean = [float(row["wait_overhead_mean_ms"]) for row in rows]
    op_mean = [float(row["op_overhead_mean_ms"]) for row in rows]
    axes[0].bar(x, wait_mean, width=0.56, color="#2563eb", label="wait")
    axes[0].bar(x, op_mean, width=0.56, bottom=wait_mean, color="#93c5fd", label="search op")
    axes[0].set_ylabel("matched search e2e\noverhead (ms)")
    axes[0].set_title("Search e2e overhead decomposition", loc="left")
    axes[0].legend(frameon=False)

    axes[1].bar([i - width / 2 for i in x], llc, width=width, color="#0891b2", label="LLC load misses")
    axes[1].bar([i + width / 2 for i in x], dtlb, width=width, color="#9333ea", label="DTLB walks")
    axes[1].set_ylabel("event rate (M/s)")
    axes[1].set_title("Cache/TLB pressure", loc="left")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.75)
        ax.set_axisbelow(True)

    fig.suptitle("Cache-heavy writer work alone does not explain search e2e latency", x=0.02, y=1.02, ha="left", fontsize=12)
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=240, bbox_inches="tight")
    print(f"[m3-cache-causality-summary] wrote {args.out_csv}")
    print(f"[m3-cache-causality-summary] wrote {args.out_md}")
    print(f"[m3-cache-causality-summary] wrote {args.out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
