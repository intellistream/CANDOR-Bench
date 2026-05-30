#!/usr/bin/env python3
"""Plot an Odin-style full RW timeline: writer phase, cache pressure, search latency."""

from __future__ import annotations

import argparse
import csv
import bisect
import math
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
    vals = sorted(values)
    return vals[int((len(vals) - 1) * pct)]


def normalize(values: list[float]) -> list[float]:
    base = percentile([v for v in values if v > 0], 0.95)
    if base <= 0:
        base = max(values) if values else 1.0
    if base <= 0:
        return [0.0 for _ in values]
    return [min(v / base, 1.25) for v in values]


def load_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows.append({k: f(v) for k, v in row.items()})
    return rows


def load_insert_progress(path: Path) -> tuple[list[float], int]:
    finishes: list[float] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            finishes.append(f(row.get("finish_raw_ns")))
    finishes = sorted(v for v in finishes if v > 0)
    return finishes, len(finishes)


def progress_at(finish_times: list[float], total: int, raw_ns: float) -> float:
    if total <= 0:
        return 0.0
    completed = bisect.bisect_right(finish_times, raw_ns)
    return min(1.0, max(0.0, completed / total))


def elapsed_at_progress(finish_times: list[float], total: int, origin_ns: float, progress: float) -> float | None:
    if total <= 0 or not finish_times:
        return None
    if progress <= 0:
        return 0.0
    idx = min(total - 1, max(0, math.ceil(progress * total) - 1))
    return max(0.0, (finish_times[idx] - origin_ns) / 1_000_000_000.0)


def aggregate_rows(rows: list[dict[str, float]], group_size: int, perf_events: list[str]) -> list[dict[str, float]]:
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
        row: dict[str, float] = {
            "bin_idx": chunk[0]["bin_idx"],
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
        ]:
            row[key] = sum(r.get(key, 0.0) * (r["bin_end_ns"] - r["bin_start_ns"]) for r in chunk) / width_ns

        finish_count = sum(r.get("search_finish_count", 0.0) for r in chunk)
        row["search_finish_count"] = finish_count
        if finish_count > 0:
            row["search_finish_op_ms_mean"] = (
                sum(r.get("search_finish_op_ms_sum", 0.0) for r in chunk) / finish_count
            )
            row["search_finish_e2e_ms_mean"] = (
                sum(r.get("search_finish_e2e_ms_sum", 0.0) for r in chunk) / finish_count
            )
        else:
            row["search_finish_op_ms_mean"] = 0.0
            row["search_finish_e2e_ms_mean"] = 0.0
        row["search_finish_op_ms_max"] = max(r.get("search_finish_op_ms_max", 0.0) for r in chunk)
        row["search_finish_e2e_ms_max"] = max(r.get("search_finish_e2e_ms_max", 0.0) for r in chunk)

        width_s = width_ns / 1_000_000_000.0
        for event_key in perf_events:
            count_key = f"perf_{event_key}"
            rate_key = f"{count_key}_per_s"
            if any(count_key in r for r in chunk):
                row[rate_key] = sum(r.get(count_key, 0.0) for r in chunk) / width_s
            else:
                row[rate_key] = (
                    sum(r.get(rate_key, 0.0) * (r["bin_end_ns"] - r["bin_start_ns"]) for r in chunk)
                    / width_ns
                )
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing period_groups.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--x-axis", choices=["elapsed", "index_progress"], default="elapsed")
    parser.add_argument(
        "--latency-mode",
        choices=["execution", "observed", "decomposed"],
        default="execution",
        help="execution=start-to-finish search op; observed=create-to-finish e2e; decomposed adds queue wait.",
    )
    parser.add_argument("--insert-wide", type=Path)
    parser.add_argument("--drop-post-insert-drain", action="store_true")
    parser.add_argument(
        "--coarsen-bins",
        type=int,
        default=1,
        help="Aggregate this many period bins for plotting/summary; use 10 for 100ms if source bins are 10ms.",
    )
    parser.add_argument(
        "--progress-top-axis",
        action="store_true",
        help="With elapsed x-axis, add a top axis showing completed-insert progress ticks.",
    )
    parser.add_argument("--title", default="Read/write timeline: insert graph search, cache pressure, search latency")
    args = parser.parse_args()

    period_path = args.run_dir / "period_groups.csv"
    if not period_path.exists():
        raise FileNotFoundError(period_path)

    rows = load_rows(period_path)
    insert_finishes: list[float] = []
    total_insert_batches = 0
    if args.x_axis == "index_progress" or args.drop_post_insert_drain or args.progress_top_axis:
        insert_wide = args.insert_wide or (args.run_dir / "raw_latency" / "insert_wide.csv")
        if not insert_wide.exists():
            raise FileNotFoundError(insert_wide)
        insert_finishes, total_insert_batches = load_insert_progress(insert_wide)
        if args.drop_post_insert_drain and insert_finishes:
            last_finish = insert_finishes[-1]
            rows = [r for r in rows if (r["bin_start_ns"] + r["bin_end_ns"]) / 2.0 <= last_finish]

    perf_event_keys = [
        "l2_rqsts_miss",
        "llc_load_misses",
        "dtlb_load_misses_walk_completed",
    ]
    rows = aggregate_rows(rows, args.coarsen_bins, perf_event_keys)

    elapsed_x = [(r["bin_start_ms"] + r["bin_end_ms"]) / 2000.0 for r in rows]
    if args.x_axis == "index_progress":
        x = [
            progress_at(insert_finishes, total_insert_batches, (r["bin_start_ns"] + r["bin_end_ns"]) / 2.0)
            for r in rows
        ]
        x_label = "index progress (completed inserts / scheduled inserts)"
    else:
        x = elapsed_x
        x_label = "elapsed time (s)"

    search_op = [r["search_finish_op_ms_max"] if r["search_finish_count"] > 0 else None for r in rows]
    search_mean = [r["search_finish_op_ms_mean"] if r["search_finish_count"] > 0 else None for r in rows]
    search_e2e = [r["search_finish_e2e_ms_max"] if r["search_finish_count"] > 0 else None for r in rows]
    search_e2e_mean = [r["search_finish_e2e_ms_mean"] if r["search_finish_count"] > 0 else None for r in rows]
    queue_wait_mean = [
        max(0.0, r["search_finish_e2e_ms_mean"] - r["search_finish_op_ms_mean"])
        if r["search_finish_count"] > 0
        else None
        for r in rows
    ]
    search_op_x = [t for t, v in zip(x, search_op) if v is not None]
    search_op_y = [v for v in search_op if v is not None]
    search_mean_x = [t for t, v in zip(x, search_mean) if v is not None]
    search_mean_y = [v for v in search_mean if v is not None]
    search_e2e_x = [t for t, v in zip(x, search_e2e) if v is not None]
    search_e2e_y = [v for v in search_e2e if v is not None]
    search_e2e_mean_x = [t for t, v in zip(x, search_e2e_mean) if v is not None]
    search_e2e_mean_y = [v for v in search_e2e_mean if v is not None]
    queue_wait_x = [t for t, v in zip(x, queue_wait_mean) if v is not None]
    queue_wait_y = [v for v in queue_wait_mean if v is not None]

    graph = [r["insert_path_search_active_workers"] for r in rows]
    repair = [r["existing_neighbor_update_active_workers"] for r in rows]
    prune = [r["existing_neighbor_prune_active_workers"] for r in rows]
    rewrite = [r["existing_neighbor_rewrite_active_workers"] for r in rows]

    l2 = [r["perf_l2_rqsts_miss_per_s"] for r in rows]
    llc = [r["perf_llc_load_misses_per_s"] for r in rows]
    dtlb = [r["perf_dtlb_load_misses_walk_completed_per_s"] for r in rows]
    l2_n = normalize(l2)
    llc_n = normalize(llc)
    dtlb_n = normalize(dtlb)

    panel_count = 4 if args.latency_mode == "decomposed" else 3
    height_ratios = [1.25, 0.85, 1.0, 1.0] if args.latency_mode == "decomposed" else [1.25, 1.0, 1.0]
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(12.8, 8.7 if args.latency_mode == "decomposed" else 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)

    if args.latency_mode == "observed":
        axes[0].plot(search_e2e_x, search_e2e_y, color="#111827", linewidth=1.25, label="observed search latency max")
        axes[0].scatter(search_e2e_x, search_e2e_y, s=10, color="#111827", edgecolor="white", linewidth=0.25)
        axes[0].plot(search_e2e_mean_x, search_e2e_mean_y, color="#2563eb", linewidth=1.0, label="observed search latency mean")
        axes[0].set_ylabel("observed search\nlatency (ms)")
    elif args.latency_mode == "decomposed":
        axes[0].plot(search_e2e_x, search_e2e_y, color="#111827", linewidth=1.25, label="observed search latency max")
        axes[0].scatter(search_e2e_x, search_e2e_y, s=10, color="#111827", edgecolor="white", linewidth=0.25)
        axes[0].plot(search_e2e_mean_x, search_e2e_mean_y, color="#2563eb", linewidth=1.0, label="observed search latency mean")
        axes[0].set_ylabel("observed search\nlatency (ms)")
        axes[1].plot(queue_wait_x, queue_wait_y, color="#dc2626", linewidth=1.15, label="queue wait mean")
        axes[1].plot(search_mean_x, search_mean_y, color="#2563eb", linewidth=1.0, label="execution mean")
        axes[1].set_ylabel("latency split\n(ms)")
        axes[1].legend(frameon=False, loc="upper right", ncol=2, fontsize=8)
    else:
        axes[0].plot(search_op_x, search_op_y, color="#2563eb", linewidth=1.25, label="search execution max")
        axes[0].scatter(search_op_x, search_op_y, s=10, color="#2563eb", edgecolor="white", linewidth=0.25)
        axes[0].plot(search_mean_x, search_mean_y, color="#93c5fd", linewidth=1.0, label="search execution mean")
        axes[0].set_ylabel("search execution\nlatency (ms)")
    axes[0].set_title(args.title, loc="left", fontsize=13)
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)

    phase_ax = axes[2] if args.latency_mode == "decomposed" else axes[1]
    cache_ax = axes[3] if args.latency_mode == "decomposed" else axes[2]

    phase_ax.plot(x, graph, linewidth=1.35, color="#dc2626", label="insert graph traversal")
    phase_ax.plot(x, repair, linewidth=1.0, color="#d97706", label="neighbor update")
    phase_ax.plot(x, prune, linewidth=1.0, color="#7c3aed", label="prune scan")
    phase_ax.plot(x, rewrite, linewidth=1.0, color="#64748b", label="rewrite")
    phase_ax.set_ylabel("writer phase occupancy\n(worker-equivalent)")
    phase_ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8)

    cache_ax.plot(x, l2_n, linewidth=1.25, color="#059669", label="L2 miss rate")
    cache_ax.plot(x, llc_n, linewidth=1.25, color="#0891b2", label="LLC load miss rate")
    cache_ax.plot(x, dtlb_n, linewidth=1.25, color="#9333ea", label="DTLB walk rate")
    cache_ax.set_ylabel("cache/TLB event rate\n(normalized for display)")
    cache_ax.set_xlabel(x_label)
    cache_ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=8)

    for ax in axes:
        ax.set_xlim(min(x), max(x))

    if args.latency_mode in {"observed", "decomposed"}:
        top_vals = search_e2e_y
    else:
        top_vals = search_op_y
    if top_vals:
        top = max(top_vals)
        axes[0].set_ylim(0, max(1.0, top * 1.08))
    if args.latency_mode == "decomposed" and queue_wait_y and search_mean_y:
        axes[1].set_ylim(0, max(1.0, max(queue_wait_y + search_mean_y) * 1.15))
    phase_ax.set_ylim(0, max(1.0, max(graph + repair + prune + rewrite) * 1.2))
    cache_ax.set_ylim(0, 1.3)

    if args.x_axis == "elapsed" and args.progress_top_axis and insert_finishes:
        origin_ns = rows[0]["bin_start_ns"]
        progress_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        min_x = min(x)
        max_x = max(x)
        for progress in progress_ticks:
            pos = elapsed_at_progress(insert_finishes, total_insert_batches, origin_ns, progress)
            if pos is None or pos < min_x or pos > max_x:
                continue
            tick_positions.append(pos)
            tick_labels.append(f"{int(progress * 100)}%")
        if tick_positions:
            top_axis = axes[0].twiny()
            top_axis.set_xlim(axes[0].get_xlim())
            top_axis.set_xticks(tick_positions)
            top_axis.set_xticklabels(tick_labels)
            top_axis.set_xlabel("insert progress (completed inserts)")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_out.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "elapsed_s",
                    "index_progress",
                    "search_op_ms",
                    "search_op_ms_mean",
                    "search_e2e_ms_max",
                    "search_e2e_ms",
                    "search_queue_wait_ms_mean",
                    "insert_graph_traversal_occupancy_worker_equiv",
                    "neighbor_update_occupancy_worker_equiv",
                    "prune_scan_occupancy_worker_equiv",
                    "rewrite_occupancy_worker_equiv",
                    "l2_miss_rate",
                    "llc_load_miss_rate",
                    "dtlb_walk_rate",
                ],
            )
            writer.writeheader()
            for i, t in enumerate(x):
                row = rows[i]
                writer.writerow(
                    {
                        "elapsed_s": elapsed_x[i],
                        "index_progress": (
                            progress_at(
                                insert_finishes,
                                total_insert_batches,
                                (rows[i]["bin_start_ns"] + rows[i]["bin_end_ns"]) / 2.0,
                            )
                            if total_insert_batches
                            else ""
                        ),
                        "search_op_ms": "" if search_op[i] is None else search_op[i],
                        "search_op_ms_mean": "" if search_mean[i] is None else search_mean[i],
                        "search_e2e_ms_max": "" if search_e2e[i] is None else search_e2e[i],
                        "search_e2e_ms": "" if row["search_finish_count"] <= 0 else row["search_finish_e2e_ms_mean"],
                        "search_queue_wait_ms_mean": (
                            "" if queue_wait_mean[i] is None else queue_wait_mean[i]
                        ),
                        "insert_graph_traversal_occupancy_worker_equiv": graph[i],
                        "neighbor_update_occupancy_worker_equiv": repair[i],
                        "prune_scan_occupancy_worker_equiv": prune[i],
                        "rewrite_occupancy_worker_equiv": rewrite[i],
                        "l2_miss_rate": l2[i],
                        "llc_load_miss_rate": llc[i],
                        "dtlb_walk_rate": dtlb[i],
                    }
                )
    print(f"[m3-rw-timeline] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
