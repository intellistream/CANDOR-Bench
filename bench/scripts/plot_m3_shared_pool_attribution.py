#!/usr/bin/env python3
"""Plot shared-pool mixed-workload attribution for M3 burst experiments."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


MODE_READONLY = "readonly_noop"
MODE_CPU = "cpu_dummy"
MODE_REPAIR_DISABLED = "path" + "_only_cpu_delay"
MODE_FULL = "full_insert"

DISPLAY = {
    MODE_READONLY: "schedule-only",
    MODE_CPU: "CPU-hold",
    MODE_REPAIR_DISABLED: "repair-disabled",
    MODE_FULL: "full insert",
}

COLORS = {
    MODE_READONLY: "#9ca3af",
    MODE_CPU: "#64748b",
    MODE_REPAIR_DISABLED: "#2563eb",
    MODE_FULL: "#dc2626",
}


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


def load_windows(path: Path) -> list[tuple[float, float]]:
    with path.open() as file:
        doc = yaml.safe_load(file) or {}
    windows = []
    for item in (doc.get("workload") or {}).get("insert_burst_schedule") or []:
        start = f(item.get("start_ms")) / 1000.0
        end = start + f(item.get("duration_ms")) / 1000.0
        if end > start:
            windows.append((start, end))
    return windows


def in_windows(t: float, windows: list[tuple[float, float]], drain_s: float = 0.0) -> bool:
    return any(start <= t <= end + drain_s for start, end in windows)


def load_raw_latency(path: Path, windows: list[tuple[float, float]]) -> dict[str, float]:
    release_op: list[float] = []
    release_wait: list[float] = []
    release_e2e: list[float] = []
    quiet_op: list[float] = []
    quiet_wait: list[float] = []
    quiet_e2e: list[float] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            elapsed_s = f(row["finish_ms"]) / 1000.0
            op = f(row["op_ms"])
            e2e = f(row["e2e_ms"])
            wait = e2e - op
            if in_windows(elapsed_s, windows):
                release_op.append(op)
                release_wait.append(wait)
                release_e2e.append(e2e)
            elif not in_windows(elapsed_s, windows, drain_s=1.0):
                quiet_op.append(op)
                quiet_wait.append(wait)
                quiet_e2e.append(e2e)
    return {
        "release_n": len(release_op),
        "release_op_mean": statistics.mean(release_op) if release_op else 0.0,
        "release_op_p95": percentile(release_op, 0.95),
        "release_wait_mean": statistics.mean(release_wait) if release_wait else 0.0,
        "release_wait_p95": percentile(release_wait, 0.95),
        "release_e2e_mean": statistics.mean(release_e2e) if release_e2e else 0.0,
        "release_e2e_p95": percentile(release_e2e, 0.95),
        "quiet_op_mean": statistics.mean(quiet_op) if quiet_op else 0.0,
        "quiet_wait_mean": statistics.mean(quiet_wait) if quiet_wait else 0.0,
        "quiet_e2e_mean": statistics.mean(quiet_e2e) if quiet_e2e else 0.0,
    }


def load_benchmark(path: Path) -> dict[str, float]:
    with path.open(newline="") as file:
        row = next(csv.DictReader(file))
    return {
        "insert_qps": f(row.get("insert_qps (per-point)")),
        "search_qps": f(row.get("search_qps (per-point)")),
        "insert_op_mean": f(row.get("insert_mean_op_latency_ms (per-batch)")),
        "insert_op_p95": f(row.get("insert_p95_op_latency_ms (per-batch)")),
    }


def load_period_summary(path: Path, windows: list[tuple[float, float]]) -> dict[str, float]:
    insert_cols = [
        "insert_path_search_active_workers",
        "existing_neighbor_load_scan_active_workers",
        "existing_neighbor_prune_active_workers",
        "existing_neighbor_undo_record_active_workers",
        "existing_neighbor_rewrite_active_workers",
    ]
    insert_active: list[float] = []
    search_op_active: list[float] = []
    search_e2e_active: list[float] = []
    llc_miss: list[float] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            elapsed_s = (f(row.get("bin_start_ms")) + f(row.get("bin_end_ms"))) / 2000.0
            if not in_windows(elapsed_s, windows):
                continue
            insert_active.append(sum(f(row.get(col)) for col in insert_cols))
            search_op_active.append(f(row.get("search_op_active_workers")))
            search_e2e_active.append(f(row.get("search_e2e_active_workers")))
            llc_miss.append(f(row.get("perf_llc_load_misses_per_s")) / 1e6)
    return {
        "insert_active_mean": statistics.mean(insert_active) if insert_active else 0.0,
        "insert_active_p95": percentile(insert_active, 0.95),
        "search_op_active_mean": statistics.mean(search_op_active) if search_op_active else 0.0,
        "search_e2e_active_mean": statistics.mean(search_e2e_active) if search_e2e_active else 0.0,
        "llc_miss_mean": statistics.mean(llc_miss) if llc_miss else 0.0,
    }


def case_dirs(root: Path, mode: str, case_prefix: str, label: str, reps: int) -> list[Path]:
    if reps == 1:
        return [root / mode / case_prefix / label]
    return [root / mode / f"{case_prefix}_rep{rep:02d}" / label for rep in range(1, reps + 1)]


def average_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: statistics.mean(row[key] for row in rows) for key in keys}


def collect(root: Path, label: str, case_prefix: str, reps: int, windows: list[tuple[float, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for mode in [MODE_READONLY, MODE_CPU, MODE_REPAIR_DISABLED, MODE_FULL]:
        raw_rows = []
        bench_rows = []
        period_rows = []
        for case_dir in case_dirs(root, mode, case_prefix, label, reps):
            raw_rows.append(load_raw_latency(case_dir / "raw_latency" / "search_wide.csv", windows))
            bench_rows.append(load_benchmark(case_dir / "benchmark_results.csv"))
            period_rows.append(load_period_summary(case_dir / "period_groups.csv", windows))
        merged = {}
        merged.update(average_dicts(raw_rows))
        merged.update(average_dicts(bench_rows))
        merged.update(average_dicts(period_rows))
        out[mode] = merged
    return out


def write_csv(path: Path, rows: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["mode"] + list(next(iter(rows.values())).keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for mode, values in rows.items():
            writer.writerow({"mode": DISPLAY[mode], **values})


def plot(path: Path, rows: dict[str, dict[str, float]], title_note: str) -> None:
    modes = [MODE_READONLY, MODE_CPU, MODE_REPAIR_DISABLED, MODE_FULL]
    labels = [DISPLAY[mode] for mode in modes]
    x = list(range(len(modes)))

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), gridspec_kw={"width_ratios": [1.2, 1.05, 1.15]})
    fig.patch.set_facecolor("white")

    wait = [rows[mode]["release_wait_mean"] for mode in modes]
    op = [rows[mode]["release_op_mean"] for mode in modes]
    axes[0].bar(x, wait, color="#cbd5e1", label="wait")
    axes[0].bar(x, op, bottom=wait, color=[COLORS[mode] for mode in modes], label="execution")
    axes[0].set_title("(a) Search latency in release windows", loc="left")
    axes[0].set_ylabel("mean latency per query (ms)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].legend(frameon=False)

    insert_mean = [rows[mode]["insert_op_mean"] for mode in modes]
    insert_p95 = [rows[mode]["insert_op_p95"] for mode in modes]
    axes[1].bar(x, insert_mean, color=[COLORS[mode] for mode in modes], alpha=0.86)
    axes[1].scatter(x, insert_p95, color="#111827", marker="_", s=80, label="p95")
    axes[1].set_title("(b) Insert worker hold time", loc="left")
    axes[1].set_ylabel("insert service time per batch (ms)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].legend(frameon=False)

    base = MODE_READONLY
    diffs = [
        ("worker\nhold", MODE_CPU, base),
        ("insert\npath", MODE_REPAIR_DISABLED, MODE_CPU),
        ("graph\nrepair", MODE_FULL, MODE_REPAIR_DISABLED),
    ]
    dx = list(range(len(diffs)))
    wait_diff = [rows[right]["release_wait_mean"] - rows[left]["release_wait_mean"] for _, right, left in diffs]
    op_diff = [rows[right]["release_op_mean"] - rows[left]["release_op_mean"] for _, right, left in diffs]
    axes[2].axhline(0, color="#111827", lw=0.7)
    axes[2].bar([i - 0.18 for i in dx], wait_diff, width=0.36, color="#94a3b8", label="wait delta")
    axes[2].bar([i + 0.18 for i in dx], op_diff, width=0.36, color="#ef4444", label="execution delta")
    axes[2].set_title("(c) Incremental attribution", loc="left")
    axes[2].set_ylabel("mean delta per query (ms)")
    axes[2].set_xticks(dx)
    axes[2].set_xticklabels([label for label, _, _ in diffs])
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title_note, x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--case-prefix", default="shared_worker_pool")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title-note", default="Shared-pool mixed workload attribution")
    args = parser.parse_args()

    rows = collect(args.root, args.label, args.case_prefix, args.reps, load_windows(args.config))
    write_csv(args.out_dir / "m3_shared_pool_attribution.csv", rows)
    plot(args.out_dir / "m3_shared_pool_attribution.png", rows, args.title_note)
    print(f"[m3-shared-pool-attribution] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
