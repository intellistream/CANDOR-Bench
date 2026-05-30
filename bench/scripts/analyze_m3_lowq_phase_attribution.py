#!/usr/bin/env python3
"""10ms-bin attribution for low-queue M3 measured-search op latency."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


PHASE_COLS = [
    ("path", "insert_path_search_active_workers"),
    ("load_scan", "existing_neighbor_load_scan_active_workers"),
    ("prune", "existing_neighbor_prune_active_workers"),
    ("update", "existing_neighbor_update_active_workers"),
    ("rewrite", "existing_neighbor_rewrite_active_workers"),
]


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    xm = mean([x for x, _ in pairs])
    ym = mean([y for _, y in pairs])
    num = sum((x - xm) * (y - ym) for x, y in pairs)
    xd = math.sqrt(sum((x - xm) ** 2 for x, _ in pairs))
    yd = math.sqrt(sum((y - ym) ** 2 for _, y in pairs))
    if xd == 0.0 or yd == 0.0:
        return float("nan")
    return num / (xd * yd)


def load_windows(config: Path) -> tuple[list[tuple[float, float]], float]:
    with config.open() as file:
        doc = yaml.safe_load(file) or {}
    workload = doc.get("workload") or {}
    windows = []
    for item in workload.get("insert_burst_schedule") or []:
        start = f(item.get("start_ms")) / 1000.0
        end = start + f(item.get("duration_ms")) / 1000.0
        if end > start:
            windows.append((start, end))
    return windows, f(workload.get("schedule_horizon_ms")) / 1000.0


def window_name(t_s: float, windows: list[tuple[float, float]]) -> str:
    last = 0.0
    for idx, (start, end) in enumerate(windows, 1):
        if last <= t_s < start:
            return f"O{idx}"
        if start <= t_s < end:
            return f"I{idx}"
        last = end
    return "post"


def load_latency_bins(run_dir: Path, bin_ms: float) -> dict[int, dict[str, float]]:
    buckets: dict[int, list[float]] = {}
    queue_buckets: dict[int, list[float]] = {}
    with (run_dir / "raw_latency" / "search_compact.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            if f(row.get("measured")) < 0.5:
                continue
            idx = int(f(row.get("finish_ms")) / bin_ms)
            op_ms = f(row.get("op_ms"))
            e2e_ms = f(row.get("e2e_ms"))
            buckets.setdefault(idx, []).append(op_ms)
            queue_buckets.setdefault(idx, []).append(max(0.0, e2e_ms - op_ms))
    out = {}
    for idx, values in buckets.items():
        out[idx] = {
            "sample_count": float(len(values)),
            "op_p50": percentile(values, 0.50),
            "op_p95": percentile(values, 0.95),
            "op_p99": percentile(values, 0.99),
            "queue_p95": percentile(queue_buckets.get(idx, []), 0.95),
        }
    return out


def load_phase_bins(run_dir: Path) -> dict[int, dict[str, float]]:
    rows = {}
    with (run_dir / "periods.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            idx = int(f(row.get("bin_idx")))
            item = {
                "bin_start_ms": f(row.get("bin_start_ms")),
                "bin_end_ms": f(row.get("bin_end_ms")),
                "search_workers": f(row.get("search_e2e_active_workers")),
            }
            for short, col in PHASE_COLS:
                item[f"{short}_workers"] = f(row.get(col))
            rows[idx] = item
    return rows


def build_rows(label: str, run_dir: Path, config: Path, bin_ms: float) -> list[dict[str, object]]:
    windows, horizon = load_windows(config)
    lat = load_latency_bins(run_dir, bin_ms)
    phase = load_phase_bins(run_dir)
    rows: list[dict[str, object]] = []
    for idx, p in phase.items():
        mid_s = (p["bin_start_ms"] + p["bin_end_ms"]) / 2000.0
        if mid_s < 0.0 or mid_s >= horizon:
            continue
        l = lat.get(idx)
        if not l:
            continue
        row: dict[str, object] = {
            "case": label,
            "bin_idx": idx,
            "window": window_name(mid_s, windows),
            "mid_s": mid_s,
            **l,
            **p,
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "bin_idx",
        "window",
        "mid_s",
        "sample_count",
        "op_p50",
        "op_p95",
        "op_p99",
        "queue_p95",
        "search_workers",
    ]
    fields += [f"{short}_workers" for short, _ in PHASE_COLS]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize(rows: list[dict[str, object]], out_path: Path) -> list[dict[str, object]]:
    summary = []
    for case in sorted({str(r["case"]) for r in rows}):
        case_rows = [r for r in rows if r["case"] == case and r["window"] != "post"]
        insert_rows = [r for r in case_rows if str(r["window"]).startswith("I")]
        high_thr = percentile([f(r["op_p99"]) for r in insert_rows], 0.90)
        high_rows = [r for r in insert_rows if f(r["op_p99"]) >= high_thr]
        low_rows = [r for r in insert_rows if f(r["op_p99"]) < high_thr]
        item: dict[str, object] = {
            "case": case,
            "insert_bin_count": len(insert_rows),
            "insert_high_thr_op_p99": high_thr,
            "high_bin_count": len(high_rows),
            "high_op_p99_mean": mean([f(r["op_p99"]) for r in high_rows]),
            "low_op_p99_mean": mean([f(r["op_p99"]) for r in low_rows]),
            "high_queue_p95_mean": mean([f(r["queue_p95"]) for r in high_rows]),
        }
        for short, _ in PHASE_COLS:
            key = f"{short}_workers"
            item[f"corr_op_p99_{short}"] = pearson([f(r["op_p99"]) for r in insert_rows], [f(r[key]) for r in insert_rows])
            item[f"high_{short}_workers_mean"] = mean([f(r[key]) for r in high_rows])
            item[f"low_{short}_workers_mean"] = mean([f(r[key]) for r in low_rows])
        summary.append(item)

    fields = list(summary[0].keys()) if summary else []
    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    return summary


def plot(rows: list[dict[str, object]], out_path: Path) -> None:
    colors = {"bg0": "#2563eb", "bg80k": "#16a34a", "bg130k": "#dc2626"}
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.0), sharex=True, constrained_layout=True)
    for case in ["bg0", "bg80k", "bg130k"]:
        case_rows = [r for r in rows if r["case"] == case and r["window"] != "post"]
        xs = [f(r["mid_s"]) for r in case_rows]
        axes[0].plot(xs, [f(r["op_p99"]) for r in case_rows], color=colors[case], lw=0.9, label=case)
        axes[1].plot(xs, [f(r["path_workers"]) for r in case_rows], color=colors[case], lw=0.9, label=case)
        axes[2].scatter([f(r["path_workers"]) for r in case_rows], [f(r["op_p99"]) for r in case_rows], s=8, color=colors[case], alpha=0.65, label=case)
    axes[0].set_title("10ms bins: measured-search op p99", loc="left")
    axes[0].set_ylabel("op p99 ms")
    axes[1].set_title("10ms bins: insert path-search worker occupancy", loc="left")
    axes[1].set_ylabel("active workers")
    axes[2].set_title("10ms-bin relation: path workers vs op p99", loc="left")
    axes[2].set_xlabel("insert_path_search active workers")
    axes[2].set_ylabel("op p99 ms")
    for ax in axes:
        ax.grid(color="#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncol=3, loc="upper left")
    axes[1].set_xlabel("elapsed time (s)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", nargs=3, metavar=("LABEL", "RUN_DIR", "CONFIG"), required=True)
    parser.add_argument("--bin-ms", type=float, default=10.0)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-fig", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for label, run_dir, config in args.case:
        rows.extend(build_rows(label, Path(run_dir), Path(config), args.bin_ms))
    write_csv(args.out_csv, rows)
    summary = summarize(rows, args.out_summary)
    plot(rows, args.out_fig)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_fig}")
    for item in summary:
        print(
            f"{item['case']}: high_thr={f(item['insert_high_thr_op_p99']):.3f} "
            f"corr_path={f(item['corr_op_p99_path']):+.3f} "
            f"corr_prune={f(item['corr_op_p99_prune']):+.3f} "
            f"corr_update={f(item['corr_op_p99_update']):+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
