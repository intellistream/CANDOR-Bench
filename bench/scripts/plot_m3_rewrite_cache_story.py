#!/usr/bin/env python3
"""Plot the no-lock HNSW rewrite-period cache-pressure story."""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


BATCH_SIZE = 200


def read_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def load_search_batches(path: Path, batch_size: int = BATCH_SIZE) -> list[dict[str, float]]:
    rows: list[dict[str, str]] = list(csv.DictReader(path.open()))
    batches: list[dict[str, float]] = []
    chunks: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        chunks.setdefault(row["finish_ms"], []).append(row)
    for finish_ms in sorted(chunks, key=lambda x: float(x)):
        chunk = chunks[finish_ms]
        hops = sum(read_float(r, "diag_hops") for r in chunk)
        period_exp = sum(read_float(r, "diag_rewrite_period_expansions") for r in chunk)
        period_active_sum = sum(read_float(r, "diag_rewrite_period_active_sum") for r in chunk)
        period_active_max = max((read_float(r, "diag_rewrite_period_active_max") for r in chunk), default=0.0)
        n = max(1, len(chunk))
        period_query_raw = sum(read_float(r, "diag_rewrite_period_queries") for r in chunk) / n
        active_node_raw = sum(read_float(r, "diag_rewrite_active_queries") for r in chunk) / n
        recent_node_raw = sum(read_float(r, "diag_rewrite_recent_queries") for r in chunk) / n
        batches.append(
            {
                "batch_idx": len(batches),
                "finish_ms": max(read_float(r, "finish_ms") for r in chunk),
                "e2e_ms": sum(read_float(r, "e2e_ms") for r in chunk),
                "op_ms": sum(read_float(r, "op_ms") for r in chunk),
                "period_query_hit_prob": min(1.0, period_query_raw),
                "period_query_hit_intensity": period_query_raw,
                "active_node_query_hit_prob": min(1.0, active_node_raw),
                "recent_node_query_hit_prob": min(1.0, recent_node_raw),
                "period_hits_per_hop": period_exp / hops if hops > 0 else 0.0,
                "period_active_writers_per_expansion": period_active_sum / period_exp if period_exp > 0 else 0.0,
                "period_active_max": period_active_max,
            }
        )
    return batches


def load_meta(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows = csv.DictReader(path.open())
    return {r["key"]: float(r["value"]) for r in rows}


def perf_to_bench_offset_ms(case_dir: Path) -> float:
    perf_start = case_dir / "perf_start_unix_ns.txt"
    meta = load_meta(case_dir / "raw_latency" / "meta.csv")
    if not perf_start.exists() or "bench_start_unix_ns" not in meta:
        return 0.0
    try:
        start_ns = float(perf_start.read_text().strip())
    except ValueError:
        return 0.0
    return (meta["bench_start_unix_ns"] - start_ns) / 1e6


def load_perf_intervals(path: Path, event: str, offset_ms: float = 0.0) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for line in path.open(errors="ignore"):
        if not line.strip() or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or parts[3] != event:
            continue
        if parts[1] == "<not counted>":
            continue
        try:
            out.append((float(parts[0]) * 1000.0 - offset_ms, float(parts[1])))
        except ValueError:
            continue
    return out


def nearest_value(series: list[tuple[float, float]], t_ms: float) -> float:
    if not series:
        return 0.0
    return min(series, key=lambda p: abs(p[0] - t_ms))[1]


def load_insert_progress(case_dir: Path, batch_size: int = BATCH_SIZE) -> list[tuple[float, float]]:
    path = case_dir / "raw_latency" / "insert_wide.csv"
    if not path.exists():
        return []
    out: list[tuple[float, float]] = []
    for row in csv.DictReader(path.open()):
        idx = int(float(row["sample_idx"]))
        out.append((float(row["finish_ms"]), float((idx + 1) * batch_size)))
    return out


def progress_at_time(progress: list[tuple[float, float]], t_ms: float) -> float:
    if not progress:
        return t_ms
    last = 0.0
    for ts, value in progress:
        if ts <= t_ms:
            last = value
        else:
            break
    return last


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


def plot_alignment(root: Path, out: Path) -> None:
    cases = [
        ("baseline", "Baseline HNSW insert", "#2563eb"),
        ("no_old_rewire", "Existing-neighbor list update/prune disabled", "#059669"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2), sharex=False)
    fig.patch.set_facecolor("white")

    for ax, (case, title, color) in zip(axes, cases):
        case_dir = root / case
        batches = load_search_batches(case_dir / "raw_latency" / "search_wide.csv")
        offset_ms = perf_to_bench_offset_ms(case_dir)
        llc = load_perf_intervals(case_dir / "perf_interval.csv", "LLC-load-misses", offset_ms)
        l2 = load_perf_intervals(case_dir / "perf_interval.csv", "l2_rqsts.miss", offset_ms)
        x_max = max((b["finish_ms"] for b in batches), default=0.0)
        llc = [(t, v) for t, v in llc if -100 <= t <= x_max + 250]
        l2 = [(t, v) for t, v in l2 if -100 <= t <= x_max + 250]

        x = [b["finish_ms"] for b in batches]
        e2e = [b["e2e_ms"] for b in batches]
        period = [b["period_query_hit_prob"] for b in batches]
        active = [b["active_node_query_hit_prob"] for b in batches]
        llc_nearest = [nearest_value(llc, t) for t in x]
        corr = pearson(e2e, llc_nearest)

        ax2 = ax.twinx()
        if llc:
            ax2.bar(
                [t for t, _ in llc],
                [v / 1e6 for _, v in llc],
                width=85,
                color="#f97316",
                alpha=0.22,
                label="LLC load misses / 100ms",
            )
        if l2:
            ax2.plot(
                [t for t, _ in l2],
                [v / 1e6 for _, v in l2],
                color="#ea580c",
                alpha=0.45,
                linewidth=1.2,
                label="L2 misses / 100ms",
            )

        ax.plot(x, e2e, color=color, linewidth=1.6, marker="o", markersize=3.2, label="search batch e2e")
        ax.fill_between(
            x,
            0,
            [p * max(e2e) if e2e else 0 for p in period],
            color="#94a3b8",
            alpha=0.18,
            label="rewrite-period overlap",
        )
        ax.plot(
            x,
            [a * max(e2e) if e2e else 0 for a in active],
            color="#7c3aed",
            linewidth=1.0,
            linestyle="--",
            label="exact active-node hit",
        )

        ax.set_title(
            f"{title}: p99={percentile(e2e, 0.99):.1f}ms, "
            f"period-hit={sum(period)/len(period):.2f}, active-node-hit={sum(active)/len(active):.5f}, "
            f"corr(e2e, nearest LLC)={corr:.2f}",
            loc="left",
            fontsize=11,
        )
        ax.set_ylabel("search batch e2e ms")
        ax2.set_ylabel("misses / 100ms (millions)")
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8, frameon=False, ncol=2)

    axes[-1].set_xlabel("elapsed time (ms)")
    fig.suptitle("No-lock HNSW insert rewrite period aligns with cache pressure and search latency", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * p)]


def plot_perf_ratios(summary: Path, out: Path) -> None:
    rows = list(csv.DictReader(summary.open()))
    metrics = [
        ("L1-dcache-load-misses", "L1D miss"),
        ("l2_rqsts.miss", "L2 miss"),
        ("LLC-load-misses", "LLC load miss"),
        ("cache-misses", "cache miss"),
        ("dtlb_load_misses.walk_completed", "DTLB walk"),
    ]
    ratios: dict[str, list[float]] = {label: [] for _, label in metrics}
    for mode in ("chasing", "round_robin"):
        normal_by_metric = {}
        skip_by_metric = {}
        for r in rows:
            if r.get("query_mode") != mode:
                continue
            target = normal_by_metric if r.get("case", "").startswith("normal") else skip_by_metric
            for key, label in metrics:
                v = read_float(r, key)
                if v > 0:
                    target[label] = v
        for _, label in metrics:
            if normal_by_metric.get(label) and skip_by_metric.get(label):
                ratios[label].append(normal_by_metric[label] / skip_by_metric[label])

    labels = [label for _, label in metrics]
    vals = [sum(ratios[label]) / len(ratios[label]) if ratios[label] else 0 for label in labels]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = ["#1d4ed8", "#0f766e", "#ea580c", "#9333ea", "#be123c"]
    ax.bar(labels, vals, color=colors, alpha=0.88)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.025, f"{v:.1f}x", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("baseline / no-old-rewire ratio")
    ax.set_title("Existing-node rewrite/prune amplifies cache and TLB miss activity", loc="left", fontsize=13)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_binned_latency_cache_alignment(root: Path, out: Path) -> None:
    cases = [
        ("baseline", "baseline insert: existing-neighbor list update/prune enabled", "#2563eb"),
        ("no_old_rewire", "existing-neighbor list update/prune disabled", "#059669"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.8), sharex=False)
    fig.patch.set_facecolor("white")

    for ax, (case, title, color) in zip(axes, cases):
        case_dir = root / case
        batches = load_search_batches(case_dir / "raw_latency" / "search_wide.csv")
        offset_ms = perf_to_bench_offset_ms(case_dir)
        llc = load_perf_intervals(case_dir / "perf_interval.csv", "LLC-load-misses", offset_ms)
        l2 = load_perf_intervals(case_dir / "perf_interval.csv", "l2_rqsts.miss", offset_ms)

        # perf stat -I emits cumulative interval endpoints; use endpoint time as
        # the bucket label and aggregate search batches that finished since the
        # previous endpoint.
        perf_times = sorted({t for t, _ in llc} | {t for t, _ in l2})
        perf_times = [t for t in perf_times if t >= 0]
        if not perf_times:
            continue
        x: list[float] = []
        p99s: list[float] = []
        means: list[float] = []
        period_means: list[float] = []
        llc_m: list[float] = []
        l2_m: list[float] = []

        llc_by_t = {t: v for t, v in llc}
        l2_by_t = {t: v for t, v in l2}
        prev = min(0.0, perf_times[0] - 100.0)
        for t in perf_times:
            bucket = [b for b in batches if prev < b["finish_ms"] <= t]
            if bucket:
                e2es = [b["e2e_ms"] for b in bucket]
                p99s.append(percentile(e2es, 0.99))
                means.append(sum(e2es) / len(e2es))
                period_means.append(sum(b["period_query_hit_prob"] for b in bucket) / len(bucket))
            else:
                p99s.append(float("nan"))
                means.append(float("nan"))
                period_means.append(0.0)
            x.append(t)
            llc_m.append(llc_by_t.get(t, 0.0) / 1e6)
            l2_m.append(l2_by_t.get(t, 0.0) / 1e6)
            prev = t

        ax2 = ax.twinx()
        ax2.bar(x, llc_m, width=70, color="#f97316", alpha=0.24, label="LLC load misses / 100ms")
        ax2.plot(x, l2_m, color="#ea580c", linewidth=1.8, alpha=0.65, label="L2 misses / 100ms")

        ymax = max([v for v in p99s if not math.isnan(v)] + [1.0])
        ax.fill_between(
            x,
            0,
            [p * ymax for p in period_means],
            color="#94a3b8",
            alpha=0.16,
            step="mid",
            label="rewrite-period overlap",
        )
        ax.plot(x, p99s, color=color, linewidth=2.0, marker="o", markersize=4, label="search e2e p99 in same 100ms bin")
        ax.plot(x, means, color=color, linewidth=1.2, linestyle="--", alpha=0.75, label="search e2e mean in same bin")

        finite = [(a, b) for a, b in zip(p99s, llc_m) if not math.isnan(a)]
        corr = pearson([a for a, _ in finite], [b for _, b in finite]) if finite else 0.0
        ax.set_title(f"{title}  |  corr(bin p99, LLC miss)={corr:.2f}", loc="left", fontsize=12)
        ax.set_ylabel("search batch e2e (ms)")
        ax2.set_ylabel("misses per 100ms (millions)")
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8, frameon=False, ncol=2)

    axes[-1].set_xlabel("benchmark elapsed time, aligned to perf intervals (ms)")
    fig.suptitle("Search latency vs cache-miss pressure in the same 100ms time bins", fontsize=14, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_normal_period_cache_latency(root: Path, out: Path) -> None:
    case_dir = root / "normal"
    batches = load_search_batches(case_dir / "raw_latency" / "search_wide.csv")
    insert_progress = load_insert_progress(case_dir)
    offset_ms = perf_to_bench_offset_ms(case_dir)
    l1 = load_perf_intervals(case_dir / "perf_interval.csv", "L1-dcache-load-misses", offset_ms)
    llc = load_perf_intervals(case_dir / "perf_interval.csv", "LLC-load-misses", offset_ms)
    l2 = load_perf_intervals(case_dir / "perf_interval.csv", "l2_rqsts.miss", offset_ms)
    x_max = max((b["finish_ms"] for b in batches), default=0.0)
    l1 = [(t, v) for t, v in l1 if 0 <= t <= x_max + 150]
    llc = [(t, v) for t, v in llc if 0 <= t <= x_max + 150]
    l2 = [(t, v) for t, v in l2 if 0 <= t <= x_max + 150]

    perf_times = sorted({t for t, _ in l1} | {t for t, _ in llc} | {t for t, _ in l2})
    l1_by_t = {t: v for t, v in l1}
    llc_by_t = {t: v for t, v in llc}
    l2_by_t = {t: v for t, v in l2}
    x_ms: list[float] = []
    x_progress: list[float] = []
    p99s: list[float] = []
    means: list[float] = []
    period_intensity: list[float] = []
    exact_hit: list[float] = []
    l1_m: list[float] = []
    llc_m: list[float] = []
    l2_m: list[float] = []

    prev = 0.0
    for t in perf_times:
        bucket = [b for b in batches if prev < b["finish_ms"] <= t]
        if bucket:
            e2es = [b["e2e_ms"] for b in bucket]
            p99s.append(percentile(e2es, 0.99))
            means.append(sum(e2es) / len(e2es))
            period_intensity.append(sum(b["period_active_writers_per_expansion"] for b in bucket) / len(bucket))
            exact_hit.append(sum(b["active_node_query_hit_prob"] for b in bucket) / len(bucket))
        else:
            p99s.append(float("nan"))
            means.append(float("nan"))
            period_intensity.append(0.0)
            exact_hit.append(0.0)
        x_ms.append(t)
        x_progress.append(progress_at_time(insert_progress, t) / 1000.0)
        l1_m.append(l1_by_t.get(t, 0.0) / 1e6)
        llc_m.append(llc_by_t.get(t, 0.0) / 1e6)
        l2_m.append(l2_by_t.get(t, 0.0) / 1e6)
        prev = t

    fig, axes = plt.subplots(3, 1, figsize=(13.2, 8.6), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0, 0.85]})
    fig.patch.set_facecolor("white")
    ax = axes[0]
    ax.plot(x_progress, p99s, color="#2563eb", linewidth=2.2, marker="o", markersize=4, label="search batch e2e p99 / 100ms bin")
    ax.plot(x_progress, means, color="#2563eb", linewidth=1.3, linestyle="--", alpha=0.7, label="search batch e2e mean / 100ms bin")
    ax.set_title("Normal SIFT insert: search latency rises during rewrite/prune pressure", loc="left", fontsize=13)
    ax.set_ylabel("search batch e2e latency (ms)")
    ax.grid(True, axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

    ax = axes[1]
    ax.plot(x_progress, l1_m, color="#9333ea", linewidth=1.9, label="L1D load misses / 100ms")
    ax.plot(x_progress, l2_m, color="#ea580c", linewidth=1.9, label="L2 misses / 100ms")
    ax.bar(x_progress, llc_m, width=1.2, color="#64748b", alpha=0.28, label="LLC load misses / 100ms")
    ax.set_ylabel("misses / 100ms (millions)")
    ax.set_title("Hardware miss pressure in the same 100ms bins", loc="left", fontsize=12)
    ax.grid(True, axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=3)

    ax = axes[2]
    ax.plot(
        x_progress,
        period_intensity,
        color="#0f766e",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="rewrite active writers per search expansion",
    )
    ax.plot(
        x_progress,
        exact_hit,
        color="#7c3aed",
        linewidth=1.3,
        linestyle="--",
        label="exact active-node query hit probability",
    )
    ax.set_ylabel("rewrite intensity")
    ax.set_xlabel("index progress in this run: inserted points since 300K SIFT base (K)")
    ax.set_title("Not a flat overlap flag: rewrite pressure intensity varies by insert phase", loc="left", fontsize=12)
    ax.grid(True, axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

    finite_progress = [x for x in x_progress if x > 0]
    if finite_progress:
        fig.text(
            0.01,
            0.01,
            f"x-axis covers aligned search/perf window, not full dataset size; visible insert progress ~= {min(finite_progress):.1f}K to {max(finite_progress):.1f}K of the 300K->340K SIFT insert.",
            fontsize=9,
            color="#475569",
        )
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def binned_progress_case(root: Path, case: str) -> dict[str, list[float]]:
    case_dir = root / case
    batches = load_search_batches(case_dir / "raw_latency" / "search_wide.csv")
    insert_rows = load_insert_rows(case_dir)
    insert_progress = load_insert_progress(case_dir)
    offset_ms = perf_to_bench_offset_ms(case_dir)
    events = {
        "l1": load_perf_intervals(case_dir / "perf_interval.csv", "L1-dcache-load-misses", offset_ms),
        "l2": load_perf_intervals(case_dir / "perf_interval.csv", "l2_rqsts.miss", offset_ms),
        "llc": load_perf_intervals(case_dir / "perf_interval.csv", "LLC-load-misses", offset_ms),
    }
    x_max = max((b["finish_ms"] for b in batches), default=0.0)
    for key, series in events.items():
        events[key] = [(t, v) for t, v in series if 0 <= t <= x_max + 150]
    perf_times = sorted({t for series in events.values() for t, _ in series})
    by_event = {key: {t: v for t, v in series} for key, series in events.items()}
    out = {
        key: []
        for key in (
            "x",
            "p99",
            "mean",
            "l1",
            "l2",
            "llc",
            "rewrite_intensity",
            "neighbor_update_prune_ms",
            "neighbor_visits_k",
        )
    }

    prev = 0.0
    for t in perf_times:
        bucket = [b for b in batches if prev < b["finish_ms"] <= t]
        if bucket:
            e2es = [b["e2e_ms"] for b in bucket]
            p99 = percentile(e2es, 0.99)
            mean = sum(e2es) / len(e2es)
            intensity = sum(b["period_active_writers_per_expansion"] for b in bucket) / len(bucket)
        else:
            p99 = float("nan")
            mean = float("nan")
            intensity = float("nan")
        insert_bucket = [r for r in insert_rows if prev < r["finish_ms"] <= t]
        if insert_bucket:
            update_prune_ms = sum(r["neighbor_update_ms"] + r["prune_ms"] for r in insert_bucket)
            visits_k = sum(r["visits"] for r in insert_bucket) / 1000.0
        else:
            update_prune_ms = float("nan")
            visits_k = float("nan")
        out["x"].append(progress_at_time(insert_progress, t) / 1000.0)
        out["p99"].append(p99)
        out["mean"].append(mean)
        out["rewrite_intensity"].append(intensity)
        out["neighbor_update_prune_ms"].append(update_prune_ms)
        out["neighbor_visits_k"].append(visits_k)
        out["l1"].append(by_event["l1"].get(t, 0.0) / 1e6)
        out["l2"].append(by_event["l2"].get(t, 0.0) / 1e6)
        out["llc"].append(by_event["llc"].get(t, 0.0) / 1e6)
        prev = t
    return out


def finite_pairs(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys):
        if not math.isnan(y):
            out_x.append(x)
            out_y.append(y)
    return out_x, out_y


def plot_normal_vs_skip_progress(root: Path, out: Path) -> None:
    normal = binned_progress_case(root, "baseline")
    skip = binned_progress_case(root, "no_old_rewire")

    fig, axes = plt.subplots(3, 1, figsize=(13.2, 8.8), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.25, 0.85]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    for data, label, color, style in (
        (normal, "normal HNSW insert", "#2563eb", "-"),
        (skip, "skip neighbor-list update/prune", "#059669", "--"),
    ):
        x, y = finite_pairs(data["x"], data["p99"])
        ax.plot(x, y, color=color, linestyle=style, linewidth=2.2, marker="o", markersize=3.8, label=f"{label}: search e2e p99")
    ax.set_ylabel("search p99 (ms)")
    ax.set_title("SIFT 300K->340K: existing-neighbor list update/prune creates the latency spike", loc="left", fontsize=13)
    ax.grid(True, axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

    ax = axes[1]
    metric_styles = [
        ("l1", "L1D miss", "#9333ea"),
        ("l2", "L2 miss", "#ea580c"),
        ("llc", "LLC load miss", "#64748b"),
    ]
    for metric, label, color in metric_styles:
        ax.plot(normal["x"], normal[metric], color=color, linewidth=2.0, label=f"normal {label}")
        ax.plot(skip["x"], skip[metric], color=color, linewidth=1.5, linestyle="--", alpha=0.85, label=f"skip {label}")
    ax.set_ylabel("misses / 100ms (millions)")
    ax.set_title("L1/L2/LLC miss pressure collapses when neighbor-list update/prune is disabled", loc="left", fontsize=12)
    ax.grid(True, axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=3)

    ax = axes[2]
    for data, label, color, style in (
        (normal, "normal HNSW insert", "#0f766e", "-"),
        (skip, "skip neighbor-list update/prune", "#059669", "--"),
    ):
        x, y = finite_pairs(data["x"], data["rewrite_intensity"])
        ax.plot(x, y, color=color, linestyle=style, linewidth=2.0, marker="o", markersize=3.5, label=label)
    ax.set_ylabel("active writers / expansion")
    ax.set_xlabel("index progress in this run: inserted points since 300K SIFT base (K)")
    ax.set_title("Rewrite pressure is an intensity signal, not a flat overlap flag", loc="left", fontsize=12)
    ax.grid(True, axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

    fig.text(
        0.01,
        0.01,
        "The x-axis is insert progress within the measured run. Dataset is SIFT; base index is 300K and the run inserts up to 340K.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def load_insert_rows(case_dir: Path) -> list[dict[str, float]]:
    path = case_dir / "raw_latency" / "insert_wide.csv"
    if not path.exists():
        return []
    out: list[dict[str, float]] = []
    for row in csv.DictReader(path.open()):
        out.append(
            {
                "finish_ms": read_float(row, "finish_ms"),
                "neighbor_update_ms": read_float(row, "graph_old_update_loop_ms"),
                "prune_ms": read_float(row, "graph_prune_heuristic_ms"),
                "visits": read_float(row, "graph_existing_node_visits"),
            }
        )
    return out


def load_insert_batches(case_dir: Path, batch_size: int = BATCH_SIZE) -> dict[str, list[float]]:
    path = case_dir / "raw_latency" / "insert_wide.csv"
    out = {key: [] for key in ("x", "neighbor_update_ms", "prune_ms", "visits")}
    if not path.exists():
        return out
    for row in csv.DictReader(path.open()):
        idx = int(float(row["sample_idx"]))
        out["x"].append((idx + 1) * batch_size / 1000.0)
        out["neighbor_update_ms"].append(read_float(row, "graph_old_update_loop_ms"))
        out["prune_ms"].append(read_float(row, "graph_prune_heuristic_ms"))
        out["visits"].append(read_float(row, "graph_existing_node_visits"))
    return out


def plot_hnsw_operation_rootcause(root: Path, out: Path) -> None:
    normal = binned_progress_case(root, "baseline")
    skip = binned_progress_case(root, "no_old_rewire")

    fig, axes = plt.subplots(3, 1, figsize=(13.0, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.0, 0.9]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    for data, label, color, style in (
        (normal, "normal HNSW insert", "#2563eb", "-"),
        (skip, "skip neighbor-list update/prune", "#059669", "--"),
    ):
        x, y = finite_pairs(data["x"], data["p99"])
        ax.plot(x, y, color=color, linestyle=style, linewidth=2.3, marker="o", markersize=3.8, label=label)
    ax.set_ylabel("search e2e p99 (ms)")
    ax.set_title("SIFT HNSW no-lock spike: existing-neighbor list update/prune is the trigger", loc="left", fontsize=13)
    ax.grid(True, axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

    ax = axes[1]
    ax.plot(
        normal["x"],
        normal["neighbor_update_prune_ms"],
        color="#0f766e",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="normal: update+prune existing-neighbor lists",
    )
    ax.plot(
        skip["x"],
        skip["neighbor_update_prune_ms"],
        color="#059669",
        linewidth=1.7,
        linestyle="--",
        label="skip: update/prune disabled",
    )
    ax.set_ylabel("HNSW work / 100ms bin (ms)")
    ax.set_title("Same bins: HNSW existing-neighbor adjacency-list update/prune work", loc="left", fontsize=12)
    ax.grid(True, axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

    ax = axes[2]
    ax.plot(normal["x"], normal["l2"], color="#ea580c", linewidth=2.2, label="normal: L2 misses / 100ms")
    ax.plot(skip["x"], skip["l2"], color="#ea580c", linewidth=1.7, linestyle="--", alpha=0.85, label="skip: L2 misses / 100ms")
    ax.set_ylabel("L2 misses / 100ms (M)")
    ax.set_xlabel("index progress: inserted points since 300K SIFT base (K), full run inserts 40K points to 340K")
    ax.set_title("Hardware symptom used as the main counter: L2 miss pressure collapses with the HNSW operation removed", loc="left", fontsize=12)
    ax.grid(True, axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)

    fig.text(
        0.01,
        0.01,
        "Measured with this repo's ./bin/bench on SIFT; batch_size=200, begin_num=300K, max_elements=340K, no external RW lock.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def finite_values(values: list[float]) -> list[float]:
    return [v for v in values if not math.isnan(v)]


def median(values: list[float]) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    return vals[len(vals) // 2]


def plot_hnsw_clean_evidence(root: Path, out: Path) -> None:
    normal = binned_progress_case(root, "baseline")
    skip = binned_progress_case(root, "no_old_rewire")

    normal_p99 = finite_values(normal["p99"])
    skip_p99 = finite_values(skip["p99"])
    normal_hnsw = finite_values(normal["neighbor_update_prune_ms"])
    skip_hnsw = finite_values(skip["neighbor_update_prune_ms"])
    normal_l2 = finite_values(normal["l2"])
    skip_l2 = finite_values(skip["l2"])

    fig = plt.figure(figsize=(13.2, 7.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.25])
    fig.patch.set_facecolor("white")

    bar_specs = [
        (
            "Worst search bin",
            "search e2e p99 (ms)",
            max(normal_p99 or [0.0]),
            max(skip_p99 or [0.0]),
            "#2563eb",
        ),
        (
            "HNSW operation work",
            "update+prune work / 100ms bin (ms)",
            median(normal_hnsw),
            median(skip_hnsw),
            "#0f766e",
        ),
        (
            "Hardware symptom",
            "L2 misses / 100ms bin (M)",
            median(normal_l2),
            median(skip_l2),
            "#ea580c",
        ),
    ]

    for i, (title, ylabel, normal_v, skip_v, color) in enumerate(bar_specs):
        ax = fig.add_subplot(gs[0, i])
        vals = [normal_v, skip_v]
        ax.bar(["normal\nHNSW insert", "disable\nneighbor update/prune"], vals, color=[color, "#059669"], alpha=0.9)
        ymax = max(vals + [1.0])
        for j, v in enumerate(vals):
            ax.text(j, v + ymax * 0.04, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
        if skip_v > 0:
            ax.text(0.5, ymax * 0.88, f"{normal_v / skip_v:.1f}x", ha="center", fontsize=11, color="#334155")
        else:
            ax.text(0.5, ymax * 0.88, "disabled", ha="center", fontsize=11, color="#334155")
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#e5e7eb")
        ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[1, :])
    points = [
        (x, work, lat, l2)
        for x, work, lat, l2 in zip(
            normal["x"],
            normal["neighbor_update_prune_ms"],
            normal["p99"],
            normal["l2"],
        )
        if not math.isnan(work) and not math.isnan(lat)
    ]
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    colors = [p[0] for p in points]
    sizes = [45 + max(0.0, p[3]) * 0.65 for p in points]
    sc = ax.scatter(xs, ys, c=colors, s=sizes, cmap="viridis", alpha=0.86, edgecolors="#0f172a", linewidths=0.4)
    if len(xs) >= 2:
        corr = pearson(xs, ys)
        ax.text(
            0.02,
            0.94,
            f"normal-only bins: corr(HNSW update+prune work, search p99) = {corr:.2f}",
            transform=ax.transAxes,
            fontsize=11,
            color="#334155",
            va="top",
        )
    ax.set_title("Normal run only: remove flat ablation lines, compare bins by actual HNSW work", loc="left", fontsize=12)
    ax.set_xlabel("HNSW existing-neighbor update+prune work in the same 100ms bin (ms, summed across insert workers)")
    ax.set_ylabel("search e2e p99 in the same 100ms bin (ms)")
    ax.grid(True, color="#e5e7eb")
    ax.set_axisbelow(True)
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("index progress since 300K SIFT base (K)")

    fig.suptitle("Cleaner evidence: use ablation for causality, use normal-only scatter for spike strength", fontsize=14, y=0.99)
    fig.text(
        0.01,
        0.01,
        "SIFT 300K->340K, batch_size=200, no external RW lock. The disable case intentionally skips HNSW existing-neighbor adjacency-list update/prune, so its operation line is zero by design.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.965))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def load_preflight_summary(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def normalized_case(row: dict[str, str]) -> str:
    case = row.get("case", "")
    if case.startswith("normal_iw") and not case.endswith("8"):
        return "normal_insert40k"
    if case.startswith("skip_") and case.endswith("insert40k"):
        return "skip_existing_node_repair_insert40k"
    return case


def plot_batch_period_summary(preflight_summary: Path, out: Path) -> None:
    rows = load_preflight_summary(preflight_summary)
    baseline_case = "normal_insert40k"
    ablation_case = "skip_existing_node_repair_insert40k"
    target = [
        r
        for r in rows
        if r.get("query_mode") == "chasing"
        and r.get("write_rate") == "40000"
        and r.get("read_rate") == "40000"
        and normalized_case(r) in {baseline_case, ablation_case}
    ]
    order = {baseline_case: 0, ablation_case: 1}
    target.sort(key=lambda r: order.get(normalized_case(r), 99))

    labels = ["baseline\nHNSW insert", "no existing-node\nupdate/prune"]
    p99 = [read_float(r, "p99_e2e_ms") for r in target]
    mean = [read_float(r, "p99_op_ms") for r in target]
    period = [read_float(r, "period_query_hit_prob_mean") for r in target]
    active = [read_float(r, "active_node_query_hit_prob_mean") * 100.0 for r in target]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.patch.set_facecolor("white")

    axes[0].bar(labels, p99, color=["#2563eb", "#059669"], alpha=0.9)
    axes[0].set_ylabel("search batch e2e p99 (ms)")
    axes[0].set_title("Search batch tail latency", loc="left")
    for i, v in enumerate(p99):
        axes[0].text(i, v + max(p99) * 0.03, f"{v:.1f} ms", ha="center")

    axes[1].bar(labels, period, color=["#64748b", "#94a3b8"], alpha=0.9)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("rewrite-period query hit probability")
    axes[1].set_title("Did search run during rewrite period?", loc="left")
    for i, v in enumerate(period):
        axes[1].text(i, v + 0.04, f"{v:.2f}", ha="center")

    axes[2].bar(labels, active, color=["#7c3aed", "#c4b5fd"], alpha=0.9)
    axes[2].set_ylabel("exact active-node hit probability (%)")
    axes[2].set_title("Did search hit the exact node being rewritten?", loc="left")
    for i, v in enumerate(active):
        axes[2].text(i, v + max(active + [0.001]) * 0.08, f"{v:.4f}%", ha="center")

    for ax in axes:
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("No-lock spike is period-level, not exact-node-hit", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if len(sys.argv) not in {4, 5}:
        print(
            "usage: plot_m3_rewrite_cache_story.py <interval_root> <perf_summary.csv> <out_dir> [period_preflight_summary.csv]",
            file=sys.stderr,
        )
        return 2
    interval_root = Path(sys.argv[1])
    perf_summary = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_alignment(interval_root, out_dir / "m3_rewrite_period_llc_alignment.png")
    plot_binned_latency_cache_alignment(interval_root, out_dir / "m3_rewrite_binned_latency_cache_alignment.png")
    plot_normal_period_cache_latency(interval_root, out_dir / "m3_rewrite_normal_period_cache_latency_progress.png")
    plot_normal_vs_skip_progress(interval_root, out_dir / "m3_rewrite_normal_vs_skip_l1_l2_llc_progress.png")
    plot_hnsw_operation_rootcause(interval_root, out_dir / "m3_hnsw_neighbor_update_prune_rootcause_progress.png")
    plot_hnsw_clean_evidence(interval_root, out_dir / "m3_hnsw_neighbor_update_prune_clean_evidence.png")
    plot_perf_ratios(perf_summary, out_dir / "m3_rewrite_cache_miss_ratios.png")
    if len(sys.argv) == 5:
        plot_batch_period_summary(Path(sys.argv[4]), out_dir / "m3_rewrite_period_hit_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
