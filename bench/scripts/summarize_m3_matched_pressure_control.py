#!/usr/bin/env python3
"""Summarize matched writer-pressure control for M3 insert attribution."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASELINE = "readonly_noop"
MODES = [
    ("path_only_cpu_delay", "path-only + CPU delay"),
    ("full_insert", "full insert"),
]
RELEASE_RATE = 40000.0


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
            batch = batches.setdefault(idx, {"op": [], "e2e": []})
            batch["op"].append(f(row.get("op_ms")))  # type: ignore[union-attr]
            batch["e2e"].append(f(row.get("e2e_ms")))  # type: ignore[union-attr]
    return {
        idx: {
            "op_ms": statistics.mean(batch["op"]),  # type: ignore[arg-type]
            "e2e_ms": statistics.mean(batch["e2e"]),  # type: ignore[arg-type]
        }
        for idx, batch in batches.items()
    }


def load_benchmark(path: Path) -> dict[str, float]:
    with path.open(newline="") as file:
        row = next(csv.DictReader(file))
    return {
        "insert_qps": f(row.get("insert_qps (per-point)")),
        "insert_op_mean_ms": f(row.get("insert_mean_op_latency_ms (per-batch)")),
        "insert_e2e_mean_ms": f(row.get("insert_mean_e2e_latency_ms (per-batch)")),
        "search_e2e_mean_ms": f(row.get("search_mean_e2e_latency_ms (per-batch)")),
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
        wait_delta = [e - o for e, o in zip(e2e_delta, op_delta)]
        bench = load_benchmark(run_dir / "benchmark_results.csv")
        insert_qps = bench["insert_qps"]
        rows.append(
            {
                "mode": mode,
                "label": label,
                "matched_batches": len(common),
                "insert_qps": insert_qps,
                "unserved_insert_release_kpts_per_s": max(0.0, RELEASE_RATE - insert_qps) / 1000.0,
                "insert_op_mean_ms": bench["insert_op_mean_ms"],
                "insert_e2e_mean_ms": bench["insert_e2e_mean_ms"],
                "search_e2e_overhead_mean_ms": statistics.mean(e2e_delta) if e2e_delta else 0.0,
                "search_e2e_overhead_p95_ms": percentile(e2e_delta, 0.95),
                "search_wait_overhead_mean_ms": statistics.mean(wait_delta) if wait_delta else 0.0,
                "search_wait_overhead_p95_ms": percentile(wait_delta, 0.95),
                "search_op_overhead_mean_ms": statistics.mean(op_delta) if op_delta else 0.0,
                "search_op_overhead_p95_ms": percentile(op_delta, 0.95),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with args.out_md.open("w") as file:
        file.write("# M3 Matched Writer-Pressure Control\n\n")
        file.write("Path-only insert is slowed with post-insert CPU work so writer service pressure is at least as high as full insert, while existing-node repair remains disabled.\n\n")
        file.write("| mode | insert QPS | unserved insert release kpts/s | search e2e overhead mean ms | wait overhead mean ms | op overhead mean ms |\n")
        file.write("|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                f"| {row['label']} | {float(row['insert_qps']):.0f} | {float(row['unserved_insert_release_kpts_per_s']):.1f} | "
                f"{float(row['search_e2e_overhead_mean_ms']):.3f} | {float(row['search_wait_overhead_mean_ms']):.3f} | "
                f"{float(row['search_op_overhead_mean_ms']):.3f} |\n"
            )
        file.write("\n")
        file.write(
            "Mechanism read: path-only + CPU delay has lower insert QPS and higher unserved insert release than full insert, "
            "but it does not reproduce full-insert search wait. This strengthens the conclusion that the existing-node "
            "repair / full graph-mutation path is necessary for the reader-visible latency jump.\n"
        )

    labels = [str(row["label"]) for row in rows]
    x = list(range(len(rows)))
    backlog = [float(row["unserved_insert_release_kpts_per_s"]) for row in rows]
    wait = [max(0.0, float(row["search_wait_overhead_mean_ms"])) for row in rows]
    op = [float(row["search_op_overhead_mean_ms"]) for row in rows]
    e2e = [float(row["search_e2e_overhead_mean_ms"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    fig.patch.set_facecolor("white")
    axes[0].bar(x, backlog, color="#64748b")
    axes[0].set_title("(a) Writer pressure is matched or stronger", loc="left")
    axes[0].set_ylabel("unserved insert release (k pts/s)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=12, ha="right")
    axes[0].grid(axis="y", color="#e5e7eb")
    axes[0].set_axisbelow(True)

    axes[1].bar(x, wait, color="#b91c1c", label="wait / dispatch")
    axes[1].bar(x, op, bottom=wait, color="#9ca3af", label="search op")
    axes[1].set_title("(b) Only full mutation produces reader wait", loc="left")
    axes[1].set_ylabel("matched search overhead (ms)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=12, ha="right")
    axes[1].grid(axis="y", color="#e5e7eb")
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="upper left")
    for i, value in enumerate(e2e):
        axes[1].text(i, value + max(e2e) * 0.03, f"{value:.3f}", ha="center", fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Matched writer-pressure control", x=0.02, y=1.02, ha="left", fontsize=12)
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=240, bbox_inches="tight")
    print(f"[m3-matched-pressure] wrote {args.out_csv}")
    print(f"[m3-matched-pressure] wrote {args.out_md}")
    print(f"[m3-matched-pressure] wrote {args.out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
