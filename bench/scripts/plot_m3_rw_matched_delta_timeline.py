#!/usr/bin/env python3
"""Plot matched read-latency overhead on writer/cache timeline bins."""

from __future__ import annotations

import argparse
import bisect
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 7.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
    }
)


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_period_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as file:
        return [{k: f(v) for k, v in row.items()} for row in csv.DictReader(file)]


def load_insert_finishes(path: Path) -> list[float]:
    out: list[float] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            raw = f(row.get("finish_raw_ns"))
            if raw > 0:
                out.append(raw)
    return sorted(out)


def load_search_batches(path: Path, batch_size: int) -> dict[int, dict[str, float]]:
    batches: dict[int, dict[str, float]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            idx = int(f(row.get("sample_idx"))) // batch_size
            batch = batches.setdefault(idx, {"op_ms": 0.0, "e2e_ms": 0.0, "finish_raw_ns": 0.0, "count": 0.0})
            batch["op_ms"] += f(row.get("op_ms"))
            batch["e2e_ms"] += f(row.get("e2e_ms"))
            batch["finish_raw_ns"] = max(batch["finish_raw_ns"], f(row.get("finish_raw_ns")))
            batch["count"] += 1.0
    for batch in batches.values():
        if batch["count"] > 0:
            batch["op_ms"] /= batch["count"]
            batch["e2e_ms"] /= batch["count"]
    return batches


def aggregate_rows(rows: list[dict[str, float]], group_size: int) -> list[dict[str, float]]:
    if group_size <= 1:
        return rows
    out: list[dict[str, float]] = []
    for start in range(0, len(rows), group_size):
        chunk = rows[start : start + group_size]
        if not chunk:
            continue
        width_ns = sum(r["bin_end_ns"] - r["bin_start_ns"] for r in chunk)
        if width_ns <= 0:
            continue
        row = {
            "bin_start_ns": chunk[0]["bin_start_ns"],
            "bin_end_ns": chunk[-1]["bin_end_ns"],
            "bin_start_ms": chunk[0]["bin_start_ms"],
            "bin_end_ms": chunk[-1]["bin_end_ms"],
        }
        for key in [
            "insert_path_search_active_workers",
            "existing_neighbor_update_active_workers",
            "existing_neighbor_prune_active_workers",
            "existing_neighbor_rewrite_active_workers",
            "perf_l2_rqsts_miss_per_s",
            "perf_llc_load_misses_per_s",
            "perf_dtlb_load_misses_walk_completed_per_s",
        ]:
            row[key] = sum(r.get(key, 0.0) * (r["bin_end_ns"] - r["bin_start_ns"]) for r in chunk) / width_ns
        out.append(row)
    return out


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * pct)]


def normalize(values: list[float]) -> list[float]:
    base = percentile([v for v in values if v > 0], 0.95)
    if base <= 0:
        base = max(values) if values else 1.0
    if base <= 0:
        return [0.0 for _ in values]
    return [min(v / base, 1.25) for v in values]


def load_burst_windows(path: Path | None) -> list[tuple[float, float]]:
    if path is None:
        return []
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    windows = []
    for item in workload.get("insert_burst_schedule") or []:
        start = f(item.get("start_ms")) / 1000.0
        end = start + f(item.get("duration_ms")) / 1000.0
        if end > start:
            windows.append((start, end))
    return windows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("full_run", type=Path)
    parser.add_argument("readonly_run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--coarsen-bins", type=int, default=10)
    parser.add_argument("--drop-post-insert-drain", action="store_true")
    parser.add_argument("--burst-config", type=Path, help="YAML config containing workload.insert_burst_schedule")
    parser.add_argument("--title", default="Writer bursts and matched read-latency overhead")
    args = parser.parse_args()

    rows = load_period_rows(args.full_run / "period_groups.csv")
    if args.drop_post_insert_drain:
        finishes = load_insert_finishes(args.full_run / "raw_latency" / "insert_wide.csv")
        if finishes:
            last_finish = finishes[-1]
            rows = [r for r in rows if (r["bin_start_ns"] + r["bin_end_ns"]) / 2 <= last_finish]
    rows = aggregate_rows(rows, args.coarsen_bins)

    full = load_search_batches(args.full_run / "raw_latency" / "search_wide.csv", args.batch_size)
    readonly = load_search_batches(args.readonly_run / "raw_latency" / "search_wide.csv", args.batch_size)

    starts = [r["bin_start_ns"] for r in rows]
    per_bin: list[dict[str, list[float]]] = [{"op_delta": [], "e2e_delta": []} for _ in rows]
    for idx in sorted(set(full) & set(readonly)):
        frow = full[idx]
        brow = readonly[idx]
        bin_idx = bisect.bisect_right(starts, frow["finish_raw_ns"]) - 1
        if bin_idx < 0 or bin_idx >= len(rows):
            continue
        if frow["finish_raw_ns"] > rows[bin_idx]["bin_end_ns"]:
            continue
        per_bin[bin_idx]["op_delta"].append(frow["op_ms"] - brow["op_ms"])
        per_bin[bin_idx]["e2e_delta"].append(frow["e2e_ms"] - brow["e2e_ms"])

    x = [(r["bin_start_ms"] + r["bin_end_ms"]) / 2000.0 for r in rows]
    op_mean = [sum(b["op_delta"]) / len(b["op_delta"]) if b["op_delta"] else None for b in per_bin]
    op_max = [max(b["op_delta"]) if b["op_delta"] else None for b in per_bin]
    e2e_mean = [sum(b["e2e_delta"]) / len(b["e2e_delta"]) if b["e2e_delta"] else None for b in per_bin]
    op_x = [t for t, v in zip(x, op_mean) if v is not None]
    op_mean_y = [v for v in op_mean if v is not None]
    op_max_y = [v for v in op_max if v is not None]
    e2e_mean_y = [v for v in e2e_mean if v is not None]

    graph = [r["insert_path_search_active_workers"] for r in rows]
    repair = [r["existing_neighbor_update_active_workers"] for r in rows]
    prune = [r["existing_neighbor_prune_active_workers"] for r in rows]
    rewrite = [r["existing_neighbor_rewrite_active_workers"] for r in rows]
    l2_n = normalize([r["perf_l2_rqsts_miss_per_s"] for r in rows])
    llc_n = normalize([r["perf_llc_load_misses_per_s"] for r in rows])
    dtlb_n = normalize([r["perf_dtlb_load_misses_walk_completed_per_s"] for r in rows])

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.4, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]},
    )
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.75)
        ax.set_axisbelow(True)
    burst_windows = load_burst_windows(args.burst_config)
    for ax in axes:
        for idx, (start_s, end_s) in enumerate(burst_windows):
            ax.axvspan(
                start_s,
                end_s,
                color="#d1d5db",
                alpha=0.22,
                linewidth=0,
                label="insert release window" if ax is axes[0] and idx == 0 else None,
            )
            ax.axvline(start_s, color="#9ca3af", linewidth=0.55, alpha=0.45)
            ax.axvline(end_s, color="#9ca3af", linewidth=0.55, alpha=0.45)

    axes[0].plot(op_x, op_max_y, color="#111827", linewidth=1.05, label="execution overhead (max)")
    axes[0].plot(op_x, op_mean_y, color="#2563eb", linewidth=1.05, label="execution overhead (mean)")
    axes[0].plot(op_x, e2e_mean_y, color="#94a3b8", linewidth=0.9, linestyle="--", label="end-to-end overhead (mean)")
    axes[0].set_ylabel("matched read-latency\noverhead (ms)")
    axes[0].set_title(args.title, loc="left", pad=8)
    axes[0].legend(frameon=False, loc="upper right", ncol=4, handlelength=1.7, columnspacing=1.0)

    axes[1].plot(x, graph, linewidth=1.15, color="#dc2626", label="insert graph traversal")
    axes[1].plot(x, repair, linewidth=0.85, color="#d97706", label="neighbor update")
    axes[1].plot(x, prune, linewidth=0.85, color="#7c3aed", label="prune scan")
    axes[1].plot(x, rewrite, linewidth=0.85, color="#64748b", label="rewrite")
    axes[1].set_ylabel("writer phase occupancy\n(worker-equivalent)")
    axes[1].legend(frameon=False, loc="upper right", ncol=2, handlelength=1.7, columnspacing=1.0)

    axes[2].plot(x, l2_n, linewidth=1.05, color="#059669", label="L2 miss rate")
    axes[2].plot(x, llc_n, linewidth=1.05, color="#0891b2", label="LLC load miss rate")
    axes[2].plot(x, dtlb_n, linewidth=1.05, color="#9333ea", label="DTLB walk rate")
    axes[2].set_ylabel("cache/TLB event rate\n(normalized for display)")
    axes[2].set_xlabel("elapsed time (s)")
    axes[2].legend(frameon=False, loc="upper right", ncol=3, handlelength=1.7, columnspacing=1.0)

    panel_labels = ["(a)", "(b)", "(c)"]
    for label, ax in zip(panel_labels, axes):
        ax.text(
            0.006,
            0.92,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#374151",
        )

    for ax in axes:
        ax.set_xlim(min(x), max(x))
    ys = [v for v in op_max_y + op_mean_y + e2e_mean_y if v is not None]
    if ys:
        lo = min(0.0, min(ys) * 1.05)
        hi = max(1.0, max(ys) * 1.08)
        axes[0].set_ylim(lo, hi)
    axes[1].set_ylim(0, max(1.0, max(graph + repair + prune + rewrite) * 1.2))
    axes[2].set_ylim(0, 1.3)

    fig.tight_layout(h_pad=1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_out.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "elapsed_s",
                    "matched_search_batches",
                    "execution_delta_ms_mean",
                    "execution_delta_ms_max",
                    "observed_delta_ms_mean",
                    "insert_graph_traversal_occupancy_worker_equiv",
                    "neighbor_update_occupancy_worker_equiv",
                    "l2_miss_rate",
                    "llc_load_miss_rate",
                    "dtlb_walk_rate",
                ],
            )
            writer.writeheader()
            for i, row in enumerate(rows):
                writer.writerow(
                    {
                        "elapsed_s": x[i],
                        "matched_search_batches": len(per_bin[i]["op_delta"]),
                        "execution_delta_ms_mean": "" if op_mean[i] is None else op_mean[i],
                        "execution_delta_ms_max": "" if op_max[i] is None else op_max[i],
                        "observed_delta_ms_mean": "" if e2e_mean[i] is None else e2e_mean[i],
                        "insert_graph_traversal_occupancy_worker_equiv": graph[i],
                        "neighbor_update_occupancy_worker_equiv": repair[i],
                        "l2_miss_rate": row["perf_l2_rqsts_miss_per_s"],
                        "llc_load_miss_rate": row["perf_llc_load_misses_per_s"],
                        "dtlb_walk_rate": row["perf_dtlb_load_misses_walk_completed_per_s"],
                    }
                )
    print(f"[m3-matched-delta] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
