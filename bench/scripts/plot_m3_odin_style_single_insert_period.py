#!/usr/bin/env python3
"""Plot a single-run OdinANN-style measured-search figure for insert periods."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


INSERT_PHASE_COLUMNS = [
    "insert_path_search_active_workers",
    "existing_neighbor_load_scan_active_workers",
    "existing_neighbor_prune_active_workers",
    "existing_neighbor_undo_record_active_workers",
    "existing_neighbor_rewrite_active_workers",
]


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def load_config(path: Path) -> tuple[list[tuple[float, float]], float, float, float]:
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    windows = []
    for item in workload.get("insert_burst_schedule") or []:
        start = f(item.get("start_ms")) / 1000.0
        end = start + f(item.get("duration_ms")) / 1000.0
        if end > start:
            windows.append((start, end))
    return (
        windows,
        f(workload.get("schedule_horizon_ms")) / 1000.0,
        f(workload.get("insert_event_rate")) / 1000.0,
        f(workload.get("search_event_rate")) / 1000.0,
    )


def load_batch_size(path: Path) -> int:
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    batch_size = int(f(workload.get("batch_size")) or 1)
    return max(1, batch_size)


def load_num_threads(path: Path) -> int:
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    num_threads = int(f(workload.get("num_threads")) or 0)
    return max(1, num_threads)


def target_offsets_and_counts(path: Path) -> tuple[list[int], list[int]]:
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    data = doc.get("data") or {}
    workload = doc.get("workload") or {}
    offset = int(f(data.get("begin_num")))
    max_elements = int(f(data.get("max_elements")))
    batch_size = max(1, int(f(workload.get("batch_size")) or 1))
    default_rate = f(workload.get("insert_event_rate"))
    targets: list[int] = []
    counts: list[int] = []
    for item in workload.get("insert_burst_schedule") or []:
        duration_s = f(item.get("duration_ms")) / 1000.0
        rate = f(item.get("insert_event_rate")) or default_rate
        if duration_s <= 0.0 or rate <= 0.0:
            targets.append(offset)
            counts.append(0)
            continue
        interval_s = batch_size / rate
        releases = 0
        elapsed = 0.0
        while elapsed < duration_s and offset < max_elements:
            offset = min(offset + batch_size, max_elements)
            releases += 1
            elapsed = releases * interval_s
        targets.append(offset)
        counts.append(releases)
    return targets, counts


def bin_measured_latency(run_dir: Path, horizon: float, bin_s: float, quantile: float = 0.95) -> tuple[list[float], list[float], list[float]]:
    nbins = int(math.ceil(horizon / bin_s)) + 1
    e2e: list[list[float]] = [[] for _ in range(nbins)]
    op: list[list[float]] = [[] for _ in range(nbins)]
    path = run_dir / "raw_latency" / "search_compact.csv"
    if not path.exists():
        path = run_dir / "raw_latency" / "search_wide.csv"
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if "measured" in row and f(row["measured"]) < 0.5:
                continue
            elapsed = f(row["finish_ms"]) / 1000.0
            if elapsed < 0 or elapsed > horizon:
                continue
            idx = min(nbins - 1, int(elapsed / bin_s))
            op_ms = f(row["op_ms"])
            e2e_ms = f(row["e2e_ms"])
            op[idx].append(op_ms)
            e2e[idx].append(e2e_ms)
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    return (
        xs,
        [percentile(bucket, quantile) for bucket in e2e],
        [percentile(bucket, quantile) for bucket in op],
    )


def load_measured_search_rows(run_dir: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    path = run_dir / "raw_latency" / "search_compact.csv"
    if not path.exists():
        path = run_dir / "raw_latency" / "search_wide.csv"
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if "measured" in row and f(row["measured"]) < 0.5:
                continue
            finish_s = f(row.get("finish_ms")) / 1000.0
            e2e_ms = f(row.get("e2e_ms"))
            exec_ms = f(row.get("op_ms"))
            queue_ms = max(0.0, e2e_ms - exec_ms)
            rows.append((finish_s, e2e_ms, exec_ms, queue_ms))
    return rows


def phase_latency_deltas(
    run_dir: Path,
    insert_spans: list[tuple[float, float]],
    pre_window_s: float = 1.5,
) -> list[dict[str, float]]:
    rows = load_measured_search_rows(run_dir)
    out: list[dict[str, float]] = []
    for idx, (start, end) in enumerate(insert_spans, start=1):
        pre_start = max(0.0, start - pre_window_s)
        pre = [row for row in rows if pre_start <= row[0] < start]
        ins = [row for row in rows if start <= row[0] <= end]
        if not pre or not ins:
            continue
        pre_e2e = percentile([row[1] for row in pre], 0.95)
        pre_exec = percentile([row[2] for row in pre], 0.95)
        pre_queue = percentile([row[3] for row in pre], 0.95)
        ins_e2e = percentile([row[1] for row in ins], 0.95)
        ins_exec = percentile([row[2] for row in ins], 0.95)
        ins_queue = percentile([row[3] for row in ins], 0.95)
        out.append(
            {
                "burst": float(idx),
                "delta_e2e": ins_e2e - pre_e2e,
                "delta_exec": ins_exec - pre_exec,
                "delta_queue": ins_queue - pre_queue,
                "pre_e2e": pre_e2e,
                "insert_e2e": ins_e2e,
            }
        )
    return out


def phase_latency_summary(run_dir: Path, insert_spans: list[tuple[float, float]]) -> dict[str, dict[str, float]]:
    rows = load_measured_search_rows(run_dir)
    groups = {"search-only": [], "insert": []}
    for row in rows:
        target = "insert" if any(start <= row[0] <= end for start, end in insert_spans) else "search-only"
        groups[target].append(row)
    summary: dict[str, dict[str, float]] = {}
    for name, values in groups.items():
        summary[name] = {
            "E2E p95": percentile([row[1] for row in values], 0.95),
            "Exec p95": percentile([row[2] for row in values], 0.95),
            "Queue p95": percentile([row[3] for row in values], 0.95),
            "count": float(len(values)),
        }
    return summary


def phase_latency_samples(run_dir: Path, insert_spans: list[tuple[float, float]], column_idx: int) -> dict[str, list[float]]:
    samples = {"search-only": [], "insert": []}
    for row in load_measured_search_rows(run_dir):
        target = "insert" if any(start <= row[0] <= end for start, end in insert_spans) else "search-only"
        samples[target].append(row[column_idx])
    return samples


def search_only_spans(insert_spans: list[tuple[float, float]], horizon: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in insert_spans:
        if start > cursor:
            spans.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < horizon:
        spans.append((cursor, horizon))
    return spans


def span_percentile(rows: list[tuple[float, float, float, float]], start: float, end: float, column_idx: int, q: float) -> float:
    return percentile([row[column_idx] for row in rows if start <= row[0] <= end], q)


def ccdf_points(values: list[float]) -> tuple[list[float], list[float]]:
    ordered = sorted(value for value in values if not math.isnan(value))
    n = len(ordered)
    if n == 0:
        return [], []
    xs: list[float] = []
    ys: list[float] = []
    for idx, value in enumerate(ordered):
        xs.append(value)
        ys.append((n - idx) / n)
    return xs, ys


def cdf_points(values: list[float], max_points: int = 6000) -> tuple[list[float], list[float]]:
    ordered = sorted(value for value in values if not math.isnan(value))
    n = len(ordered)
    if n == 0:
        return [], []
    stride = max(1, math.ceil(n / max_points))
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    return [ordered[idx] for idx in indices], [(idx + 1) / n for idx in indices]


def ccdf_points(values: list[float], max_points: int = 6000) -> tuple[list[float], list[float]]:
    ordered = sorted(value for value in values if not math.isnan(value))
    n = len(ordered)
    if n == 0:
        return [], []
    stride = max(1, math.ceil(n / max_points))
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    return [ordered[idx] for idx in indices], [max((n - idx) / n, 1.0 / n) for idx in indices]


def observed_points(xs: list[float], *series: list[float]) -> tuple[list[float], ...]:
    out: list[list[float]] = [[] for _ in range(len(series) + 1)]
    for idx, x in enumerate(xs):
        values = [s[idx] for s in series]
        if any(math.isnan(value) for value in values):
            continue
        out[0].append(x)
        for out_series, value in zip(out[1:], values):
            out_series.append(value)
    return tuple(out)


def load_bench_start_raw_ns(run_dir: Path) -> float:
    path = run_dir / "raw_latency" / "meta.csv"
    if not path.exists():
        return 0.0
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row.get("key") == "bench_start_raw_ns":
                return f(row.get("value"))
    return 0.0


def load_burst_issue_begins(run_dir: Path, counts: list[int]) -> list[float]:
    path = run_dir / "raw_latency" / "insert_wide.csv"
    if not path.exists():
        return []
    start_raw = load_bench_start_raw_ns(run_dir)
    rows: list[dict[str, str]] = []
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    begins: list[float] = []
    cursor = 0
    for count in counts:
        chunk = rows[cursor : cursor + count]
        cursor += count
        raw_values = [f(row.get("create_raw_ns")) for row in chunk if f(row.get("create_raw_ns")) > 0]
        if raw_values:
            begins.append((min(raw_values) - start_raw) / 1e9)
    return begins


def add_active_interval(buckets: list[float], start_s: float, finish_s: float, bin_s: float, scale: float = 1.0) -> None:
    if finish_s <= 0.0 or finish_s <= start_s:
        return
    i0 = max(0, int(start_s / bin_s))
    i1 = min(len(buckets) - 1, int(finish_s / bin_s))
    for idx in range(i0, i1 + 1):
        lo = idx * bin_s
        hi = lo + bin_s
        overlap = max(0.0, min(finish_s, hi) - max(start_s, lo))
        buckets[idx] += (overlap / bin_s) * scale


def bin_task_workers(run_dir: Path, horizon: float, bin_s: float, batch_size: int) -> tuple[list[float], list[float], list[float], list[float]]:
    nbins = int(math.ceil(horizon / bin_s)) + 1
    insert_workers = [0.0 for _ in range(nbins)]
    start_raw = load_bench_start_raw_ns(run_dir)
    insert_path = run_dir / "raw_latency" / "insert_wide.csv"
    if insert_path.exists():
        with insert_path.open(newline="") as file:
            for row in csv.DictReader(file):
                start_s = (f(row.get("start_raw_ns")) - start_raw) / 1e9
                finish_s = (f(row.get("finish_raw_ns")) - start_raw) / 1e9
                add_active_interval(insert_workers, start_s, finish_s, bin_s)
    period_path = run_dir / "period_groups.csv"
    if period_path.exists():
        _, search_workers = bin_period(run_dir, horizon, bin_s, ["search_op_active_workers"])
    else:
        search_workers = bin_search_workers_from_raw(run_dir, horizon, bin_s, batch_size)
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    total_workers = [s + i for s, i in zip(search_workers, insert_workers)]
    return xs, total_workers, search_workers, insert_workers


def bin_search_workers_from_raw(run_dir: Path, horizon: float, bin_s: float, batch_size: int) -> list[float]:
    nbins = int(math.ceil(horizon / bin_s)) + 1
    search_workers = [0.0 for _ in range(nbins)]
    start_raw = load_bench_start_raw_ns(run_dir)
    path = run_dir / "raw_latency" / "search_compact.csv"
    if not path.exists():
        path = run_dir / "raw_latency" / "search_wide.csv"
    if not path.exists():
        return search_workers
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            start_s = (f(row.get("start_raw_ns")) - start_raw) / 1e9
            finish_s = (f(row.get("finish_raw_ns")) - start_raw) / 1e9
            add_active_interval(search_workers, start_s, finish_s, bin_s, scale=1.0 / max(1, batch_size))
    return search_workers


def bin_period(run_dir: Path, horizon: float, bin_s: float, columns: list[str], scale: float = 1.0) -> tuple[list[float], list[float]]:
    nbins = int(math.ceil(horizon / bin_s)) + 1
    buckets: list[list[float]] = [[] for _ in range(nbins)]
    with (run_dir / "period_groups.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            elapsed = (f(row.get("bin_start_ms")) + f(row.get("bin_end_ms"))) / 2000.0
            if elapsed < 0 or elapsed > horizon:
                continue
            value = sum(f(row.get(column)) for column in columns) * scale
            idx = min(nbins - 1, int(elapsed / bin_s))
            buckets[idx].append(value)
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    ys = [statistics.mean(bucket) if bucket else float("nan") for bucket in buckets]
    return xs, ys


def load_insert_finish_times(run_dir: Path, horizon: float) -> list[float]:
    path = run_dir / "raw_latency" / "insert_wide.csv"
    if not path.exists():
        return []
    finishes: list[float] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            finish_ms = f(row.get("finish_ms"))
            if finish_ms > horizon * 1000.0 * 5.0:
                finish_ms = f(row.get("measured"))
            if 0.0 <= finish_ms <= horizon * 1000.0 * 2.0:
                finishes.append(finish_ms / 1000.0)
    return finishes


def load_commit_times(run_dir: Path, horizon: float) -> list[float]:
    path = run_dir / "raw_latency" / "commit_events.csv"
    if not path.exists():
        return []
    commits: list[float] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            commit_ms = f(row.get("commit_ms"))
            if 0.0 <= commit_ms <= horizon * 1000.0 * 2.0:
                commits.append(commit_ms / 1000.0)
    return commits


def load_commit_events(run_dir: Path) -> list[tuple[float, int]]:
    path = run_dir / "raw_latency" / "commit_events.csv"
    if not path.exists():
        return []
    rows: list[tuple[float, int]] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            rows.append((f(row.get("commit_ms")) / 1000.0, int(f(row.get("committed_offset")))))
    return rows


def last_insert_finish_by_window(finishes: list[float], windows: list[tuple[float, float]], tail_s: float = 3.0) -> list[float]:
    last_finishes: list[float] = []
    previous_end = 0.0
    for start, end in windows:
        lower = max(previous_end, start)
        upper = end + tail_s
        candidates = [value for value in finishes if lower <= value <= upper]
        if candidates:
            last_finishes.append(max(candidates))
        previous_end = upper
    return last_finishes


def commit_times_for_window_targets(commit_events: list[tuple[float, int]], targets: list[int]) -> list[float]:
    commits: list[float] = []
    cursor = 0
    for target in targets:
        while cursor < len(commit_events) and commit_events[cursor][1] < target:
            cursor += 1
        if cursor < len(commit_events):
            commits.append(commit_events[cursor][0])
    return commits


def insert_boundary_spans(starts: list[float], commits: list[float], horizon: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for idx, start in enumerate(starts):
        finish = commits[idx] if idx < len(commits) else start
        spans.append((start, min(max(start, finish), horizon)))
    return spans


def scheduled_rates(
    windows: list[tuple[float, float]],
    horizon: float,
    bin_s: float,
    insert_rate: float,
    measured_search_rate: float,
    background_search_rate: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    nbins = int(math.ceil(horizon / bin_s)) + 1
    xs = [(idx + 0.5) * bin_s for idx in range(nbins)]
    in_insert = [any(start <= x <= end for start, end in windows) for x in xs]
    inserts = [insert_rate if active else 0.0 for x, active in zip(xs, in_insert)]
    measured_searches = [measured_search_rate for _ in xs]
    background_searches = [0.0 if active else background_search_rate for active in in_insert]
    return xs, inserts, measured_searches, background_searches


def scheduled_stack_series(
    windows: list[tuple[float, float]],
    horizon: float,
    insert_rate: float,
    measured_search_rate: float,
    background_search_rate: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    points = {0.0, horizon}
    for start, end in windows:
        points.add(max(0.0, min(horizon, start)))
        points.add(max(0.0, min(horizon, end)))
    xs = sorted(points)
    measured: list[float] = []
    background: list[float] = []
    inserts: list[float] = []
    for idx, x in enumerate(xs):
        if idx + 1 < len(xs):
            probe = (x + xs[idx + 1]) / 2.0
        else:
            probe = x
        active = any(start <= probe < end for start, end in windows)
        measured.append(measured_search_rate)
        background.append(0.0 if active else background_search_rate)
        inserts.append(insert_rate if active else 0.0)
    if len(xs) > 1:
        measured[-1] = measured[-2]
        background[-1] = background[-2]
        inserts[-1] = inserts[-2]
    return xs, measured, background, inserts


def mark_windows(ax: plt.Axes, windows: list[tuple[float, float]], color: str = "#e5e7eb", alpha: float = 0.48) -> None:
    for start, end in windows:
        ax.axvspan(start, end, color=color, alpha=alpha, lw=0)
        ax.axvline(start, color="#6b7280", lw=0.45)
        ax.axvline(end, color="#6b7280", lw=0.45)


def mark_drain_tails(ax: plt.Axes, windows: list[tuple[float, float]], finish_marks: list[float]) -> None:
    for (_, end), finish in zip(windows, finish_marks):
        if finish > end:
            ax.axvspan(end, finish, color="#fee2e2", alpha=0.34, lw=0)
            ax.axvline(finish, color="#b91c1c", lw=0.5, ls=":")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bin-s", type=float, default=0.30)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--title-suffix", default="")
    parser.add_argument("--timeline", action="store_true")
    parser.add_argument("--tail-ccdf", action="store_true")
    parser.add_argument("--measured-search-rate-kpts", type=float, default=None)
    parser.add_argument("--background-search-rate-kpts", type=float, default=0.0)
    args = parser.parse_args()

    windows, horizon, _insert_rate, _search_rate = load_config(args.config)
    batch_size = load_batch_size(args.config)
    num_threads = load_num_threads(args.config)
    commit_events = load_commit_events(args.run_dir)
    target_offsets, burst_counts = target_offsets_and_counts(args.config)
    burst_starts = load_burst_issue_begins(args.run_dir, burst_counts)
    if not burst_starts:
        burst_starts = [start for start, _ in windows]
    commit_times = commit_times_for_window_targets(commit_events, target_offsets)
    boundary_times = commit_times or load_commit_times(args.run_dir, horizon) or load_insert_finish_times(args.run_dir, horizon)
    finish_marks = boundary_times[: len(burst_starts)]
    if finish_marks:
        horizon = max(horizon, max(finish_marks) + 2.0)
    insert_spans = insert_boundary_spans(burst_starts, finish_marks, horizon)
    plt.rcParams.update({"font.size": 7.0, "axes.titlesize": 8.2, "axes.labelsize": 7.4, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.4})
    fig, axes = plt.subplots(3, 1, figsize=(7.9, 5.7), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax_lat, ax_phase, ax_cdf = axes

    measured_rows = load_measured_search_rows(args.run_dir)
    latency_samples = phase_latency_samples(args.run_dir, insert_spans, column_idx=1)
    quantile_label = f"p{int(round(args.quantile * 100))}"
    search_p95 = percentile(latency_samples["search-only"], args.quantile)
    insert_p95 = percentile(latency_samples["insert"], args.quantile)
    period_p95_values: list[float] = []
    if args.timeline:
        xs, e2e_q, _ = bin_measured_latency(args.run_dir, horizon, args.bin_s, args.quantile)
        xs_obs, e2e_obs = observed_points(xs, e2e_q)
        ax_lat.plot(xs_obs, e2e_obs, color="#9ca3af", lw=0.8, alpha=0.7, zorder=1)
        outside_x: list[float] = []
        outside_y: list[float] = []
        inside_x: list[float] = []
        inside_y: list[float] = []
        for x, y in zip(xs_obs, e2e_obs):
            period_p95_values.append(y)
            if any(start <= x <= end for start, end in insert_spans):
                inside_x.append(x)
                inside_y.append(y)
            else:
                outside_x.append(x)
                outside_y.append(y)
        ax_lat.scatter(outside_x, outside_y, color="#2563eb", s=10, label=f"search-only bins {quantile_label}", zorder=3)
        ax_lat.scatter(inside_x, inside_y, color="#dc2626", s=12, label=f"insert bins {quantile_label}", zorder=4)
    else:
        first_search_label = True
        for start, end in search_only_spans(insert_spans, horizon):
            value = span_percentile(measured_rows, start, end, column_idx=1, q=args.quantile)
            if not math.isnan(value):
                period_p95_values.append(value)
            ax_lat.hlines(value, start, end, color="#2563eb", lw=2.2, label=f"search-only {quantile_label}" if first_search_label else None)
            first_search_label = False
        first_insert_label = True
        for start, end in insert_spans:
            value = span_percentile(measured_rows, start, end, column_idx=1, q=args.quantile)
            if not math.isnan(value):
                period_p95_values.append(value)
            ax_lat.hlines(value, start, end, color="#dc2626", lw=2.2, label=f"insert {quantile_label}" if first_insert_label else None)
            first_insert_label = False
    title_suffix = f" ({args.title_suffix})" if args.title_suffix else ""
    ax_lat.set_title(f"(a) Measured-search {quantile_label} over phases{title_suffix}", loc="left", pad=2)
    ax_lat.set_ylabel("Latency\n(ms)")
    if period_p95_values:
        ax_lat.set_ylim(0, max(period_p95_values) * 1.35)
    else:
        ax_lat.set_ylim(bottom=0)
    ax_lat.set_xlim(0, horizon)
    mark_windows(ax_lat, insert_spans, color="#e5e7eb", alpha=0.42)
    ax_lat.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.04, 1.0), borderaxespad=0.0, handlelength=1.6)

    xs_workers, total_workers, search_workers, insert_workers = bin_task_workers(args.run_dir, horizon, args.bin_s, batch_size)
    insert_top = [s + i for s, i in zip(search_workers, insert_workers)]
    ax_phase.fill_between(xs_workers, 0, search_workers, color="#2563eb", alpha=0.18, lw=0, label="search workers")
    ax_phase.fill_between(xs_workers, search_workers, insert_top, color="#dc2626", alpha=0.24, lw=0, label="insert workers")
    ax_phase.plot(xs_workers, total_workers, color="#111827", lw=1.05, label="active workers")
    ax_phase.axhline(num_threads, color="#9ca3af", lw=0.7, ls=":", label=f"{num_threads} threads")
    ax_phase.set_title("(b) Shared worker occupancy", loc="left", pad=2)
    ax_phase.set_ylabel("Active\nworkers")
    ax_phase.set_ylim(0, num_threads * 1.05)
    mark_windows(ax_phase, insert_spans, color="#e5e7eb", alpha=0.42)
    ax_phase.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.08, 1.0), borderaxespad=0.0, handlelength=1.6)

    point_builder = ccdf_points if args.tail_ccdf else cdf_points
    search_x, search_y = point_builder(latency_samples["search-only"])
    insert_x, insert_y = point_builder(latency_samples["insert"])
    ax_cdf.plot(search_x, search_y, color="#2563eb", lw=1.6, label="search-only")
    ax_cdf.plot(insert_x, insert_y, color="#dc2626", lw=1.6, label="insert span")
    if not math.isnan(search_p95):
        ax_cdf.axvline(search_p95, color="#2563eb", lw=0.9, ls=":", alpha=0.8)
    if not math.isnan(insert_p95):
        ax_cdf.axvline(insert_p95, color="#dc2626", lw=0.9, ls=":", alpha=0.8)
    all_cdf_values = latency_samples["search-only"] + latency_samples["insert"]
    p997 = percentile(all_cdf_values, 0.997)
    if not math.isnan(p997) and p997 > 0:
        ax_cdf.set_xlim(0, p997 * 1.05)
    if args.tail_ccdf:
        ax_cdf.set_yscale("log")
        ax_cdf.set_ylim(1e-4, 1.0)
        ax_cdf.set_title("(c) Measured-search E2E tail CCDF", loc="left", pad=2)
        ax_cdf.set_ylabel("Pr(latency > x)")
    else:
        ax_cdf.set_title("(c) Measured-search E2E CDF", loc="left", pad=2)
        ax_cdf.set_ylabel("CDF")
        ax_cdf.set_ylim(0, 1.01)
    ax_cdf.set_xlabel("Measured-search latency (ms, p99.7 view)")
    ax_cdf.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.04, 1.0), borderaxespad=0.0, handlelength=1.8)

    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_lat.set_xlim(0, horizon)
    ax_phase.set_xlim(0, horizon)
    ax_phase.set_xlabel("Elapsed time (s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(f"[m3-single-insert-period] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
