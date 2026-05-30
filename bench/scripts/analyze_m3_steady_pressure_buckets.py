#!/usr/bin/env python3
"""Analyze steady M3 latency by writer/resource-pressure buckets."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def quantile_edges(values: list[float]) -> tuple[float, float]:
    return percentile(values, 1.0 / 3.0), percentile(values, 2.0 / 3.0)


def bucket(value: float, lo: float, hi: float) -> str:
    if value <= lo:
        return "low"
    if value <= hi:
        return "mid"
    return "high"


def corr(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return 0.0
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    mx = statistics.mean(xvals)
    my = statistics.mean(yvals)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xvals))
    sy = math.sqrt(sum((y - my) ** 2 for y in yvals))
    if sx <= 0 or sy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy)


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as file:
        return [{key: f(value) for key, value in row.items()} for row in csv.DictReader(file)]


def summarize_bucket(rows: list[dict[str, float]], bucket_key: str) -> list[dict[str, float | str]]:
    groups = ["low", "mid", "high"]
    out = []
    for group in groups:
        subset = [row for row in rows if row[bucket_key] == group]
        exec_vals = [row["execution_overhead_ms"] for row in subset]
        e2e_vals = [row["e2e_overhead_ms"] for row in subset]
        out.append(
            {
                "bucket_by": bucket_key,
                "bucket": group,
                "batches": len(subset),
                "exec_mean_ms": statistics.mean(exec_vals) if exec_vals else 0.0,
                "exec_p50_ms": statistics.median(exec_vals) if exec_vals else 0.0,
                "exec_p95_ms": percentile(exec_vals, 0.95),
                "e2e_mean_ms": statistics.mean(e2e_vals) if e2e_vals else 0.0,
                "traversal_workers_mean": statistics.mean([row["path_workers"] for row in subset]) if subset else 0.0,
                "llc_miss_rate_mps_mean": statistics.mean([row["llc_miss_rate"] for row in subset]) / 1e6 if subset else 0.0,
                "dtlb_walk_rate_mps_mean": statistics.mean([row["dtlb_walk_rate"] for row in subset]) / 1e6 if subset else 0.0,
            }
        )
    return out


def summarize_joint(rows: list[dict[str, float]]) -> list[dict[str, float | str]]:
    out = []
    for traversal_bucket in ("low", "mid", "high"):
        for resource_bucket in ("low", "mid", "high"):
            subset = [
                row
                for row in rows
                if row["traversal_bucket"] == traversal_bucket and row["resource_bucket"] == resource_bucket
            ]
            exec_vals = [row["execution_overhead_ms"] for row in subset]
            out.append(
                {
                    "traversal_bucket": traversal_bucket,
                    "resource_bucket": resource_bucket,
                    "batches": len(subset),
                    "exec_mean_ms": statistics.mean(exec_vals) if exec_vals else 0.0,
                    "exec_p50_ms": statistics.median(exec_vals) if exec_vals else 0.0,
                    "exec_p95_ms": percentile(exec_vals, 0.95),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(
    path: Path,
    rows: list[dict[str, float]],
    single: list[dict[str, float | str]],
    joint: list[dict[str, float | str]],
) -> None:
    exec_vals = [row["execution_overhead_ms"] for row in rows]
    with path.open("w") as file:
        file.write("# M3 Steady Pressure Bucket Analysis\n\n")
        file.write("Scope: steady R5k/W40k full-insert run, matched read-only replay. Rows are completed search batches only; no interpolation is used.\n\n")
        file.write("## Correlation\n\n")
        for key, label in [
            ("path_workers", "writer graph traversal occupancy"),
            ("llc_miss_rate", "LLC load miss rate"),
            ("dtlb_walk_rate", "DTLB walk rate"),
            ("l2_miss_rate", "L2 miss rate"),
            ("resource_score", "combined LLC/DTLB resource score"),
        ]:
            file.write(f"- {label}: {corr([row[key] for row in rows], exec_vals):.3f}\n")
        file.write("\n## Single-Pressure Buckets\n\n")
        file.write("| bucket by | bucket | batches | exec mean ms | exec p50 ms | exec p95 ms | traversal workers | LLC M/s | DTLB M/s |\n")
        file.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in single:
            file.write(
                f"| {row['bucket_by']} | {row['bucket']} | {int(row['batches'])} | "
                f"{float(row['exec_mean_ms']):.2f} | {float(row['exec_p50_ms']):.2f} | {float(row['exec_p95_ms']):.2f} | "
                f"{float(row['traversal_workers_mean']):.2f} | {float(row['llc_miss_rate_mps_mean']):.1f} | {float(row['dtlb_walk_rate_mps_mean']):.1f} |\n"
            )
        file.write("\n## Joint Traversal/Resource Buckets\n\n")
        file.write("| traversal | resource | batches | exec mean ms | exec p50 ms | exec p95 ms |\n")
        file.write("|---|---|---:|---:|---:|---:|\n")
        for row in joint:
            file.write(
                f"| {row['traversal_bucket']} | {row['resource_bucket']} | {int(row['batches'])} | "
                f"{float(row['exec_mean_ms']):.2f} | {float(row['exec_p50_ms']):.2f} | {float(row['exec_p95_ms']):.2f} |\n"
            )
        file.write("\n## Read\n\n")
        file.write(
            "The bucket analysis should be read as association evidence, not proof of a single sufficient cause. "
            "If the high-pressure buckets do not monotonically raise latency, the conclusion must stay weak: the steady workload shows sustained writer/resource pressure and latency outliers, but not a clean local causal ordering.\n"
        )


def plot(out: Path, single: list[dict[str, float | str]], joint: list[dict[str, float | str]]) -> None:
    fig = plt.figure(figsize=(6.5, 3.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.24)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    labels = ["low", "mid", "high"]
    x = list(range(3))
    colors = {
        "traversal_bucket": "#4f46e5",
        "resource_bucket": "#0891b2",
    }
    for key, offset in [("traversal_bucket", -0.18), ("resource_bucket", 0.18)]:
        rows = [row for row in single if row["bucket_by"] == key]
        means = [float(next(row for row in rows if row["bucket"] == label)["exec_mean_ms"]) for label in labels]
        ax0.bar([i + offset for i in x], means, width=0.34, color=colors[key], label=key.replace("_bucket", ""))
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("mean execution overhead (ms)")
    ax0.set_title("(a) Single pressure bucket", loc="left")
    ax0.legend(frameon=False)

    grid = [[0.0 for _ in labels] for _ in labels]
    for i, traversal in enumerate(labels):
        for j, resource in enumerate(labels):
            row = next(item for item in joint if item["traversal_bucket"] == traversal and item["resource_bucket"] == resource)
            grid[i][j] = float(row["exec_mean_ms"])
    im = ax1.imshow(grid, cmap="Blues", aspect="auto")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_yticks(x)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("resource bucket")
    ax1.set_ylabel("traversal bucket")
    ax1.set_title("(b) Joint bucket", loc="left")
    for i in x:
        for j in x:
            ax1.text(j, i, f"{grid[i][j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="mean ms")

    for ax in (ax0, ax1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.rows)
    llc_lo, llc_hi = quantile_edges([row["llc_miss_rate"] for row in rows])
    dtlb_lo, dtlb_hi = quantile_edges([row["dtlb_walk_rate"] for row in rows])
    trav_lo, trav_hi = quantile_edges([row["path_workers"] for row in rows])
    for row in rows:
        llc_norm = (row["llc_miss_rate"] - llc_lo) / (llc_hi - llc_lo) if llc_hi > llc_lo else 0.0
        dtlb_norm = (row["dtlb_walk_rate"] - dtlb_lo) / (dtlb_hi - dtlb_lo) if dtlb_hi > dtlb_lo else 0.0
        row["resource_score"] = (llc_norm + dtlb_norm) / 2.0
    res_lo, res_hi = quantile_edges([row["resource_score"] for row in rows])
    for row in rows:
        row["traversal_bucket"] = bucket(row["path_workers"], trav_lo, trav_hi)  # type: ignore[assignment]
        row["resource_bucket"] = bucket(row["resource_score"], res_lo, res_hi)  # type: ignore[assignment]

    single = summarize_bucket(rows, "traversal_bucket") + summarize_bucket(rows, "resource_bucket")
    joint = summarize_joint(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "m3_steady_pressure_bucket_rows.csv", rows)
    write_csv(args.out_dir / "m3_steady_pressure_bucket_single.csv", single)
    write_csv(args.out_dir / "m3_steady_pressure_bucket_joint.csv", joint)
    write_md(args.out_dir / "m3_steady_pressure_bucket_analysis.md", rows, single, joint)
    plot(args.out_dir / "m3_steady_pressure_bucket_analysis.png", single, joint)
    print(f"[m3-pressure-buckets] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
