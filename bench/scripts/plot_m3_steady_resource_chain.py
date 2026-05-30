#!/usr/bin/env python3
"""Plot steady read/write resource-chain evidence for the main benchmark scenario."""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


CASE = "high_writer_concurrency/sift1m_b200_round_robin_w40000_r5000_nolock"

plt.rcParams.update(
    {
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.75,
    }
)


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


def normalize(values: list[float]) -> list[float]:
    scale = percentile([v for v in values if v > 0 and math.isfinite(v)], 0.95)
    if scale <= 0:
        scale = max([v for v in values if math.isfinite(v)] or [1.0])
    return [min(v / scale, 1.25) if scale > 0 and math.isfinite(v) else float("nan") for v in values]


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


def load_rates(config: Path) -> tuple[float, float]:
    with config.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    groups = workload.get("rate_groups(r/w)") or []
    if groups:
        return f(groups[0][0]) / 1000.0, f(groups[0][1]) / 1000.0
    return f(workload.get("search_event_rate")) / 1000.0, f(workload.get("insert_event_rate")) / 1000.0


def load_search_batches(path: Path, batch_size: int) -> dict[int, dict[str, float]]:
    batches: dict[int, dict[str, float]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            idx = int(f(row.get("sample_idx"))) // batch_size
            batch = batches.setdefault(idx, {"op_ms": 0.0, "e2e_ms": 0.0, "finish_ms": 0.0, "count": 0.0})
            batch["op_ms"] += f(row.get("op_ms"))
            batch["e2e_ms"] += f(row.get("e2e_ms"))
            batch["finish_ms"] = max(batch["finish_ms"], f(row.get("finish_ms")))
            batch["count"] += 1.0
    for batch in batches.values():
        if batch["count"] > 0:
            batch["op_ms"] /= batch["count"]
            batch["e2e_ms"] /= batch["count"]
    return batches


def load_period_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows.append(
                {
                    "elapsed_s": (f(row.get("bin_start_ms")) + f(row.get("bin_end_ms"))) / 2000.0,
                    "path_workers": f(row.get("insert_path_search_active_workers")),
                    "existing_node_repair_workers": f(row.get("existing_neighbor_update_active_workers")),
                    "l2_miss_rate": f(row.get("perf_l2_rqsts_miss_per_s")),
                    "llc_miss_rate": f(row.get("perf_llc_load_misses_per_s")),
                    "dtlb_walk_rate": f(row.get("perf_dtlb_load_misses_walk_completed_per_s")),
                }
            )
    return rows


def load_bench(path: Path) -> dict[str, float]:
    with path.open(newline="") as file:
        row = next(csv.DictReader(file))
    return {
        "insert_qps": f(row.get("insert_qps (per-point)")),
        "search_qps": f(row.get("search_qps (per-point)")),
        "recall": f(row.get("recall")),
    }


def build_rows(root: Path, batch_size: int) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, float], float]:
    full_dir = root / "full_insert" / CASE
    readonly_dir = root / "readonly_noop" / CASE
    full = load_search_batches(full_dir / "raw_latency/search_wide.csv", batch_size)
    readonly = load_search_batches(readonly_dir / "raw_latency/search_wide.csv", batch_size)
    periods = load_period_rows(full_dir / "period_groups.csv")
    starts = [row["elapsed_s"] for row in periods]
    rows = []
    for idx in sorted(set(full) & set(readonly)):
        elapsed = full[idx]["finish_ms"] / 1000.0
        period_idx = bisect.bisect_right(starts, elapsed) - 1
        if period_idx < 0:
            continue
        period = periods[min(period_idx, len(periods) - 1)]
        rows.append(
            {
                "batch_idx": float(idx),
                "elapsed_s": elapsed,
                "execution_overhead_ms": (full[idx]["op_ms"] - readonly[idx]["op_ms"]) * batch_size,
                "e2e_overhead_ms": (full[idx]["e2e_ms"] - readonly[idx]["e2e_ms"]) * batch_size,
                **{k: v for k, v in period.items() if k != "elapsed_s"},
            }
        )
    bench = load_bench(full_dir / "benchmark_results.csv")
    horizon = max(row["elapsed_s"] for row in rows) if rows else 0.0
    return rows, periods, bench, horizon


def bin_rows(rows: list[dict[str, float]], horizon: float, bin_s: float, key: str, op: str = "median") -> tuple[list[float], list[float]]:
    nbins = int(horizon / bin_s) + 1
    buckets: list[list[float]] = [[] for _ in range(nbins)]
    for row in rows:
        elapsed = row["elapsed_s"]
        if elapsed < 0 or elapsed > horizon:
            continue
        idx = min(nbins - 1, int(elapsed / bin_s))
        buckets[idx].append(row[key])
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    if op == "mean":
        ys = [statistics.mean(bucket) if bucket else float("nan") for bucket in buckets]
    else:
        ys = [statistics.median(bucket) if bucket else float("nan") for bucket in buckets]
    return xs, ys


def summarize(rows: list[dict[str, float]], bench: dict[str, float], read_rate: float, write_rate: float) -> list[dict[str, float | str]]:
    exec_vals = [row["execution_overhead_ms"] for row in rows]
    e2e_vals = [row["e2e_overhead_ms"] for row in rows]
    out: list[dict[str, float | str]] = [
        {
            "metric": "scheduled read rate kpts/s",
            "value": read_rate,
        },
        {
            "metric": "scheduled write rate kpts/s",
            "value": write_rate,
        },
        {
            "metric": "actual insert qps",
            "value": bench["insert_qps"],
        },
        {
            "metric": "actual search qps",
            "value": bench["search_qps"],
        },
        {
            "metric": "execution overhead mean ms",
            "value": statistics.mean(exec_vals) if exec_vals else 0.0,
        },
        {
            "metric": "execution overhead p50 ms",
            "value": statistics.median(exec_vals) if exec_vals else 0.0,
        },
        {
            "metric": "execution overhead p95 ms",
            "value": percentile(exec_vals, 0.95),
        },
        {
            "metric": "e2e overhead mean ms",
            "value": statistics.mean(e2e_vals) if e2e_vals else 0.0,
        },
    ]
    for key, label in [
        ("path_workers", "corr execution vs writer traversal"),
        ("llc_miss_rate", "corr execution vs LLC load miss rate"),
        ("dtlb_walk_rate", "corr execution vs DTLB walk rate"),
        ("l2_miss_rate", "corr execution vs L2 miss rate"),
    ]:
        out.append({"metric": label, "value": corr([row[key] for row in rows], exec_vals)})
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, object]]) -> None:
    by_metric = {str(row["metric"]): float(row["value"]) for row in rows}
    with path.open("w") as file:
        file.write("# M3 Steady Read/Write Resource Chain\n\n")
        file.write("Scope: main-benchmark steady read/write scenario. Reads and writes are continuously released using `rate_groups(r/w) = [5000, 40000]`; this is not a burst probe.\n\n")
        file.write("| metric | value |\n|---|---:|\n")
        for row in rows:
            file.write(f"| {row['metric']} | {float(row['value']):.3f} |\n")
        file.write("\n")
        file.write(
            "Interpretation: this figure is the comparable before/after artifact for M3 optimization. "
            "It uses the same steady workload as the benchmark, so later optimized runs should regenerate the same figure and summary.\n"
        )
        file.write(
            f"Mean matched execution overhead is {by_metric.get('execution overhead mean ms', 0.0):.2f} ms; "
            f"p95 is {by_metric.get('execution overhead p95 ms', 0.0):.2f} ms.\n"
        )


def plot(out: Path, rows: list[dict[str, float]], periods: list[dict[str, float]], horizon: float, read_rate: float, write_rate: float) -> None:
    # The steady run can have intervals with no completed search batch because
    # work is queued behind writer pressure. Plot completed search batches as
    # points instead of connecting across no-sample intervals.
    rows = sorted(rows, key=lambda row: row["elapsed_s"])
    x_exec = [row["elapsed_s"] for row in rows]
    y_exec = [row["execution_overhead_ms"] for row in rows]
    periods = sorted(periods, key=lambda row: row["elapsed_s"])
    x_path = [row["elapsed_s"] for row in periods]
    y_path = [row["path_workers"] for row in periods]
    y_repair = [row["existing_node_repair_workers"] for row in periods]
    x_llc = x_path
    y_llc = [row["llc_miss_rate"] for row in periods]
    y_dtlb = [row["dtlb_walk_rate"] for row in periods]
    y_l2 = [row["l2_miss_rate"] for row in periods]
    y_llc_n = normalize(y_llc)
    y_dtlb_n = normalize(y_dtlb)
    y_l2_n = normalize(y_l2)

    fig = plt.figure(figsize=(5.35, 3.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.10)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)

    ax0.scatter(x_exec, y_exec, color="#4f46e5", s=8, alpha=0.85, linewidths=0)
    ax0.set_title(f"(a) Search P50 execution overhead, steady R{read_rate:.0f}k/W{write_rate:.0f}k", loc="left", pad=1)
    ax0.set_ylabel("Latency\n(ms)")

    ax1.plot(x_path, y_path, color="#374151", lw=1.05, label="writer graph traversal")
    ax1.plot(x_path, y_repair, color="#dc2626", lw=0.95, label="existing-node repair")
    ax1.set_title("(b) Writer phases", loc="left", pad=1)
    ax1.set_ylabel("Phase\nworkers")
    ax1.legend(frameon=False, loc="upper right", handlelength=1.2)

    ax2.plot(x_llc, y_l2_n, color="#059669", lw=0.95, label="L2 miss rate")
    ax2.plot(x_llc, y_llc_n, color="#0891b2", lw=0.95, label="LLC load miss rate")
    ax2.plot(x_llc, y_dtlb_n, color="#9333ea", lw=0.95, label="DTLB walk rate")
    ax2.set_title("(c) Memory-hierarchy pressure", loc="left", pad=1)
    ax2.set_ylabel("Normalized\nrate")
    ax2.set_xlabel("Elapsed time (s)")
    ax2.legend(frameon=False, loc="upper right", ncol=3, handlelength=1.1, columnspacing=0.8)

    for ax in (ax0, ax1, ax2):
        ax.grid(axis="y", color="#e5e7eb", lw=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, horizon)
    ax0.tick_params(labelbottom=False)
    ax1.tick_params(labelbottom=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    args = parser.parse_args()

    read_rate, write_rate = load_rates(args.config)
    rows, periods, bench, horizon = build_rows(args.root, args.batch_size)
    summary = summarize(rows, bench, read_rate, write_rate)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "m3_steady_resource_chain_rows.csv", rows)
    write_csv(args.out_dir / "m3_steady_resource_chain_summary.csv", summary)
    write_md(args.out_dir / "m3_steady_resource_chain.md", summary)
    plot(args.out_dir / "m3_steady_resource_chain.png", rows, periods, horizon, read_rate, write_rate)
    print(f"[m3-steady-resource-chain] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
