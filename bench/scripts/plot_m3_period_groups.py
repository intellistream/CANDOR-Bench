#!/usr/bin/env python3
"""Join and plot period-aligned insert, perf, and search metrics."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DEFAULT_PERF_EVENTS = [
    "L1-dcache-load-misses",
    "l2_rqsts.miss",
    "LLC-load-misses",
    "cache-misses",
    "dtlb_load_misses.walk_completed",
]

INSERT_PHASES = [
    ("insert_path_search_active_workers", "path search"),
    ("existing_neighbor_load_scan_active_workers", "load/scan"),
    ("existing_neighbor_prune_active_workers", "prune"),
    ("existing_neighbor_rewrite_active_workers", "rewrite"),
    ("existing_neighbor_undo_record_active_workers", "undo"),
]


def read_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def read_int_file(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    return int(float(text))


def read_meta(path: Path | None) -> dict[str, int]:
    out: dict[str, int] = {}
    if path is None or not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                out[row["key"]] = int(float(row["value"]))
            except (KeyError, ValueError):
                continue
    return out


def normalize_event(event: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", event).strip("_").lower()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * p)
    return ordered[max(0, min(idx, len(ordered) - 1))]


def positive_percentile(values: list[float], p: float) -> float:
    return percentile([v for v in values if v > 0], p)


def load_period_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        required = {"bin_idx", "bin_start_ns", "bin_end_ns"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for raw in reader:
            rows.append({k: read_float(raw, k) for k in raw})
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def preferred_cache_signal(rows: list[dict[str, float]], events: list[str], explicit: str | None) -> str:
    if explicit:
        return explicit
    preferred = [
        "perf_l2_rqsts_miss_per_s",
        "perf_llc_load_misses_per_s",
        "perf_cache_misses_per_s",
        "perf_l1_dcache_load_misses_per_s",
    ]
    for key in preferred:
        if any(r.get(key, 0.0) > 0 for r in rows):
            return key
    for event in events:
        key = f"perf_{normalize_event(event)}_per_s"
        if any(r.get(key, 0.0) > 0 for r in rows):
            return key
    return ""


def annotate_three_highs(
    rows: list[dict[str, float]],
    events: list[str],
    insert_pct: float,
    cache_pct: float,
    search_pct: float,
    cache_signal: str | None,
) -> dict[str, float | str]:
    cache_key = preferred_cache_signal(rows, events, cache_signal)
    insert_scores = [r.get("existing_neighbor_update_active_workers", 0.0) for r in rows]
    prune_scores = [r.get("existing_neighbor_prune_candidates", 0.0) for r in rows]
    cache_scores = [r.get(cache_key, 0.0) if cache_key else 0.0 for r in rows]
    search_scores = [r.get("search_finish_e2e_ms_max", 0.0) for r in rows]

    insert_thr = positive_percentile(insert_scores, insert_pct)
    prune_thr = positive_percentile(prune_scores, insert_pct)
    cache_thr = positive_percentile(cache_scores, cache_pct)
    search_thr = positive_percentile(search_scores, search_pct)

    for r in rows:
        insert_score = r.get("existing_neighbor_update_active_workers", 0.0)
        prune_score = r.get("existing_neighbor_prune_candidates", 0.0)
        cache_score = r.get(cache_key, 0.0) if cache_key else 0.0
        search_score = r.get("search_finish_e2e_ms_max", 0.0)

        high_insert = (insert_thr > 0 and insert_score >= insert_thr) or (
            prune_thr > 0 and prune_score >= prune_thr
        )
        high_cache = cache_thr > 0 and cache_score >= cache_thr
        high_search = search_thr > 0 and search_score >= search_thr

        r["three_high_insert_score"] = insert_score
        r["three_high_prune_candidate_score"] = prune_score
        r["three_high_cache_score"] = cache_score
        r["three_high_search_score"] = search_score
        r["three_high_insert_threshold"] = insert_thr
        r["three_high_prune_candidate_threshold"] = prune_thr
        r["three_high_cache_threshold"] = cache_thr
        r["three_high_search_threshold"] = search_thr
        r["high_insert_work"] = 1.0 if high_insert else 0.0
        r["high_cache_miss"] = 1.0 if high_cache else 0.0
        r["high_search_latency"] = 1.0 if high_search else 0.0
        r["triple_high"] = 1.0 if high_insert and high_cache and high_search else 0.0

    return {
        "cache_signal": cache_key,
        "insert_threshold": insert_thr,
        "prune_candidate_threshold": prune_thr,
        "cache_threshold": cache_thr,
        "search_threshold": search_thr,
        "triple_high_count": sum(1 for r in rows if r.get("triple_high", 0.0) > 0),
    }


def perf_start_raw_ns(args: argparse.Namespace) -> int | None:
    explicit = read_int_file(args.perf_start_raw_ns)
    if explicit is not None:
        return explicit
    perf_unix = read_int_file(args.perf_start_unix_ns)
    meta = read_meta(args.meta)
    if perf_unix is None:
        return None
    if "bench_start_raw_ns" not in meta or "bench_start_unix_ns" not in meta:
        return None
    return meta["bench_start_raw_ns"] + (perf_unix - meta["bench_start_unix_ns"])


def parse_perf_intervals(path: Path, events: set[str]) -> list[tuple[str, float, float, float]]:
    intervals: list[tuple[str, float, float, float]] = []
    last_end_s: dict[str, float] = {}
    with path.open(errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.rstrip().split(",")]
            if len(parts) < 4:
                continue
            event = parts[3]
            if events and event not in events:
                continue
            value_text = parts[1]
            if not value_text or value_text.startswith("<"):
                continue
            try:
                end_s = float(parts[0])
                value = float(value_text.replace(",", ""))
            except ValueError:
                continue
            start_s = last_end_s.get(event, 0.0)
            if end_s <= start_s:
                continue
            last_end_s[event] = end_s
            intervals.append((event, start_s, end_s, value))
    return intervals


def add_perf_to_periods(
    rows: list[dict[str, float]],
    perf_path: Path | None,
    perf_start_ns: int | None,
    events: list[str],
) -> None:
    if perf_path is None or not perf_path.exists() or perf_start_ns is None:
        return
    origin_ns = int(rows[0]["bin_start_ns"])
    bin_ns = int(rows[0]["bin_end_ns"] - rows[0]["bin_start_ns"])
    if bin_ns <= 0:
        raise ValueError("period bin width must be positive")
    event_set = set(events)
    for event, start_s, end_s, value in parse_perf_intervals(perf_path, event_set):
        start_ns = perf_start_ns + int(start_s * 1_000_000_000.0)
        end_ns = perf_start_ns + int(end_s * 1_000_000_000.0)
        duration_ns = max(1, end_ns - start_ns)
        first = max(0, (start_ns - origin_ns) // bin_ns)
        last = min(len(rows) - 1, ((end_ns - 1) - origin_ns) // bin_ns)
        key = f"perf_{normalize_event(event)}"
        for idx in range(int(first), int(last) + 1):
            bin_start = int(rows[idx]["bin_start_ns"])
            bin_end = int(rows[idx]["bin_end_ns"])
            overlap = min(end_ns, bin_end) - max(start_ns, bin_start)
            if overlap <= 0:
                continue
            rows[idx][key] = rows[idx].get(key, 0.0) + value * overlap / duration_ns
    for row in rows:
        width_s = max(1.0, row["bin_end_ns"] - row["bin_start_ns"]) / 1_000_000_000.0
        for event in events:
            key = f"perf_{normalize_event(event)}"
            row[f"{key}_per_s"] = row.get(key, 0.0) / width_s


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_groups(path: Path, rows: list[dict[str, float]], events: list[str]) -> None:
    import matplotlib.pyplot as plt

    x = [r["bin_start_ms"] for r in rows]
    bin_ms = rows[0]["bin_end_ms"] - rows[0]["bin_start_ms"]
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.8), sharex=True)
    fig.patch.set_facecolor("white")

    for ax in axes:
        first = True
        for r in rows:
            if r.get("triple_high", 0.0) <= 0:
                continue
            ax.axvspan(
                r["bin_start_ms"],
                r["bin_end_ms"],
                color="#ef4444",
                alpha=0.10,
                label="3H period" if first else None,
                linewidth=0,
            )
            first = False

    ax = axes[0]
    bottoms = [0.0] * len(rows)
    colors = ["#2563eb", "#0f766e", "#dc2626", "#9333ea", "#64748b"]
    for (field, label), color in zip(INSERT_PHASES, colors):
        vals = [r.get(field, 0.0) for r in rows]
        ax.bar(x, vals, bottom=bottoms, width=bin_ms * 0.95,
               color=color, alpha=0.72, label=label)
        bottoms = [a + b for a, b in zip(bottoms, vals)]
    ax2 = ax.twinx()
    ax2.plot(x, [r.get("existing_neighbor_prune_candidates", 0.0) for r in rows],
             color="#991b1b", linewidth=1.15, label="prune candidates")
    ax.set_ylabel("insert active workers")
    ax2.set_ylabel("prune candidates / period")
    ax.set_title("Group 1: insert-side dynamic work per period", loc="left", fontsize=11)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8, ncol=3, frameon=False)

    ax = axes[1]
    perf_colors = ["#ea580c", "#db2777", "#7c2d12", "#4f46e5", "#0891b2"]
    for event, color in zip(events, perf_colors):
        key = f"perf_{normalize_event(event)}_per_s"
        if any(r.get(key, 0.0) for r in rows):
            ax.plot(x, [r.get(key, 0.0) / 1e6 for r in rows], linewidth=1.3, color=color, label=event)
    ax.set_ylabel("events / s (millions)")
    ax.set_title("Group 2: perf/cache counters on the same period bins", loc="left", fontsize=11)
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc="upper right", fontsize=8, ncol=2, frameon=False)

    ax = axes[2]
    ax.plot(x, [r.get("search_finish_e2e_ms_max", 0.0) for r in rows],
            color="#111827", linewidth=1.4, label="search e2e max")
    ax.plot(x, [r.get("search_finish_e2e_ms_mean", 0.0) for r in rows],
            color="#6b7280", linewidth=1.1, label="search e2e mean")
    ax2 = ax.twinx()
    ax2.fill_between(x, [r.get("search_e2e_active_workers", 0.0) for r in rows],
                     color="#22c55e", alpha=0.18, label="search e2e active workers")
    ax.set_ylabel("search latency ms")
    ax2.set_ylabel("search active workers")
    ax.set_title("Group 3: search latency on the same period bins", loc="left", fontsize=11)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8, ncol=3, frameon=False)

    for ax in axes:
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("elapsed time from bench start (ms)")
    triple_count = sum(1 for r in rows if r.get("triple_high", 0.0) > 0)
    fig.suptitle(
        f"Period-aligned insert work, cache pressure, and search latency (3H periods={triple_count})",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build one period-aligned CSV/figure with three metric groups."
    )
    parser.add_argument("case_dir", type=Path, nargs="?", help="case dir containing periods.csv and perf_interval.csv")
    parser.add_argument("--periods", type=Path, help="period CSV from bin_m3_insert_phase_trace.py")
    parser.add_argument("--perf-interval", type=Path, help="perf stat -I CSV")
    parser.add_argument("--perf-start-raw-ns", type=Path, help="file containing perf start CLOCK_MONOTONIC_RAW ns")
    parser.add_argument("--perf-start-unix-ns", type=Path, help="fallback file containing perf start unix ns")
    parser.add_argument("--meta", type=Path, help="raw_latency/meta.csv")
    parser.add_argument("--perf-events", default=",".join(DEFAULT_PERF_EVENTS))
    parser.add_argument("--insert-high-pct", type=float, default=0.90)
    parser.add_argument("--cache-high-pct", type=float, default=0.90)
    parser.add_argument("--search-high-pct", type=float, default=0.90)
    parser.add_argument(
        "--cache-signal",
        help="period_groups CSV column to use for cache high flag; default prefers perf_l2_rqsts_miss_per_s",
    )
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-fig", type=Path)
    args = parser.parse_args()

    case_dir = args.case_dir
    periods = args.periods or (case_dir / "periods.csv" if case_dir else None)
    if periods is None:
        raise ValueError("pass a case_dir or --periods")
    if case_dir:
        args.perf_interval = args.perf_interval or case_dir / "perf_interval.csv"
        args.perf_start_raw_ns = args.perf_start_raw_ns or case_dir / "perf_start_raw_ns.txt"
        args.perf_start_unix_ns = args.perf_start_unix_ns or case_dir / "perf_start_unix_ns.txt"
        args.meta = args.meta or case_dir / "raw_latency" / "meta.csv"
    out_csv = args.out_csv or (case_dir / "period_groups.csv" if case_dir else periods.with_name("period_groups.csv"))
    out_fig = args.out_fig or (case_dir / "period_groups.png" if case_dir else periods.with_name("period_groups.png"))
    events = [e.strip() for e in args.perf_events.split(",") if e.strip()]

    rows = load_period_rows(periods)
    add_perf_to_periods(rows, args.perf_interval, perf_start_raw_ns(args), events)
    summary = annotate_three_highs(
        rows,
        events,
        args.insert_high_pct,
        args.cache_high_pct,
        args.search_high_pct,
        args.cache_signal,
    )
    write_csv(out_csv, rows)
    plot_groups(out_fig, rows, events)
    print(f"[period-groups] wrote {out_csv}")
    print(f"[period-groups] wrote {out_fig}")
    print(
        "[period-groups] 3H "
        f"count={summary['triple_high_count']} "
        f"cache_signal={summary['cache_signal']} "
        f"insert_thr={summary['insert_threshold']:.6g} "
        f"cache_thr={summary['cache_threshold']:.6g} "
        f"search_thr={summary['search_threshold']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
