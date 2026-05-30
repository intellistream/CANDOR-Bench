#!/usr/bin/env python3
"""Summarize burst read-latency controls against matched read-only replay."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


LABEL = "sift1m_b200_round_robin_burst_w40000_r5000_nolock"
BASELINE_MODE = "readonly_noop"
FULL_MODE = "full_insert"
PATH_MODE = "path_only_cpu_delay"
REPS = (1, 2, 3)


MODE_LABELS = {
    FULL_MODE: "full insert",
    PATH_MODE: "repair-disabled CPU-delay",
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
    workload = doc.get("workload") or {}
    for item in workload.get("insert_burst_schedule") or []:
        start = f(item.get("start_ms")) / 1000.0
        end = start + f(item.get("duration_ms")) / 1000.0
        if end > start:
            windows.append((start, end))
    return windows


def load_search_batches(path: Path, batch_size: int) -> dict[int, dict[str, float]]:
    batches: dict[int, dict[str, float]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            idx = int(f(row.get("sample_idx"))) // batch_size
            batch = batches.setdefault(
                idx,
                {
                    "op_ms": 0.0,
                    "e2e_ms": 0.0,
                    "finish_ms": 0.0,
                    "count": 0.0,
                },
            )
            batch["op_ms"] += f(row.get("op_ms"))
            batch["e2e_ms"] += f(row.get("e2e_ms"))
            batch["finish_ms"] = max(batch["finish_ms"], f(row.get("finish_ms")))
            batch["count"] += 1.0
    for batch in batches.values():
        if batch["count"] > 0:
            batch["op_ms"] /= batch["count"]
            batch["e2e_ms"] /= batch["count"]
    return batches


def load_benchmark(path: Path) -> dict[str, float]:
    with path.open(newline="") as file:
        row = next(csv.DictReader(file))
    return {
        "insert_qps": f(row.get("insert_qps (per-point)")),
        "search_qps": f(row.get("search_qps (per-point)")),
    }


def in_any_window(t: float, windows: list[tuple[float, float]], drain_s: float = 0.0) -> bool:
    return any(start <= t <= end + drain_s for start, end in windows)


def collect_mode(
    mode: str,
    root: Path,
    readonly_root: Path,
    windows: list[tuple[float, float]],
    batch_size: int,
    case_prefix: str,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    batch_rows: list[dict[str, float | int | str]] = []
    bench_rows: list[dict[str, float | int | str]] = []
    for rep in REPS:
        case = f"{case_prefix}_rep{rep:02d}/{LABEL}"
        run_dir = root / mode / case
        readonly_dir = readonly_root / BASELINE_MODE / case
        full = load_search_batches(run_dir / "raw_latency/search_wide.csv", batch_size)
        readonly = load_search_batches(readonly_dir / "raw_latency/search_wide.csv", batch_size)
        for idx in sorted(set(full) & set(readonly)):
            elapsed_s = full[idx]["finish_ms"] / 1000.0
            op_delta = full[idx]["op_ms"] - readonly[idx]["op_ms"]
            e2e_delta = full[idx]["e2e_ms"] - readonly[idx]["e2e_ms"]
            in_release = in_any_window(elapsed_s, windows, 0.0)
            in_release_plus_drain = in_any_window(elapsed_s, windows, 1.0)
            batch_rows.append(
                {
                    "mode": mode,
                    "rep": rep,
                    "batch_idx": idx,
                    "elapsed_s": elapsed_s,
                    "in_release": int(in_release),
                    "in_release_plus_drain": int(in_release_plus_drain),
                    "is_quiet": int(not in_release_plus_drain),
                    "execution_overhead_ms": op_delta,
                    "e2e_overhead_ms": e2e_delta,
                    "wait_overhead_ms": e2e_delta - op_delta,
                }
            )
        bench = load_benchmark(run_dir / "benchmark_results.csv")
        bench_rows.append({"mode": mode, "rep": rep, **bench})
    return batch_rows, bench_rows


def summarize_batches(batch_rows: list[dict[str, float | int | str]], batch_size: int) -> list[dict[str, float | str]]:
    rows = []
    selectors = {
        "release": lambda row: int(row["in_release"]) == 1,
        "release_plus_drain": lambda row: int(row["in_release_plus_drain"]) == 1,
        "quiet": lambda row: int(row["is_quiet"]) == 1,
    }
    for mode in (FULL_MODE, PATH_MODE):
        mode_rows = [row for row in batch_rows if row["mode"] == mode]
        for window, selector in selectors.items():
            values = [float(row["execution_overhead_ms"]) for row in mode_rows if selector(row)]
            e2e = [float(row["e2e_overhead_ms"]) for row in mode_rows if selector(row)]
            wait = [float(row["wait_overhead_ms"]) for row in mode_rows if selector(row)]
            mean = statistics.mean(values) if values else 0.0
            p95 = percentile(values, 0.95)
            rows.append(
                {
                    "mode": mode,
                    "window": window,
                    "batches": len(values),
                    "execution_overhead_mean_ms": mean,
                    "execution_overhead_p95_ms": p95,
                    "batch_equiv_execution_overhead_mean_ms": mean * batch_size,
                    "batch_equiv_execution_overhead_p95_ms": p95 * batch_size,
                    "e2e_overhead_mean_ms": statistics.mean(e2e) if e2e else 0.0,
                    "wait_overhead_mean_ms": statistics.mean(wait) if wait else 0.0,
                }
            )
    return rows


def summarize_bench(bench_rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    out = []
    for mode in (FULL_MODE, PATH_MODE):
        rows = [row for row in bench_rows if row["mode"] == mode]
        qps = [float(row["insert_qps"]) for row in rows]
        out.append(
            {
                "mode": mode,
                "insert_qps_mean": statistics.mean(qps) if qps else 0.0,
                "unserved_insert_release_kpts_per_s": max(0.0, 40000.0 - (statistics.mean(qps) if qps else 0.0)) / 1000.0,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, summary: list[dict[str, float | str]], bench: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_mode_window = {(row["mode"], row["window"]): row for row in summary}
    bench_by_mode = {row["mode"]: row for row in bench}
    with path.open("w") as file:
        file.write("# M3 Burst Matched Control\n\n")
        file.write("Scope: SIFT1M controlled insert bursts, 3 repeats, matched read-only replay, 48 shared foreground workers. The primary metric is search execution overhead, so queue tail is not the causal metric.\n\n")
        file.write("## Window Summary\n\n")
        file.write("| mode | window | batches | exec mean ms | exec p95 ms | batch-equiv exec mean ms | batch-equiv exec p95 ms | e2e mean ms | wait mean ms |\n")
        file.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            file.write(
                f"| {MODE_LABELS.get(str(row['mode']), str(row['mode']))} | {row['window']} | {int(row['batches'])} | "
                f"{float(row['execution_overhead_mean_ms']):.3f} | {float(row['execution_overhead_p95_ms']):.3f} | "
                f"{float(row['batch_equiv_execution_overhead_mean_ms']):.2f} | {float(row['batch_equiv_execution_overhead_p95_ms']):.2f} | "
                f"{float(row['e2e_overhead_mean_ms']):.3f} | {float(row['wait_overhead_mean_ms']):.3f} |\n"
            )
        file.write("\n## Writer Pressure\n\n")
        file.write("| mode | insert QPS mean | unserved insert release kpts/s |\n")
        file.write("|---|---:|---:|\n")
        for row in bench:
            file.write(
                f"| {MODE_LABELS.get(str(row['mode']), str(row['mode']))} | {float(row['insert_qps_mean']):.0f} | "
                f"{float(row['unserved_insert_release_kpts_per_s']):.1f} |\n"
            )
        full_release = by_mode_window[(FULL_MODE, "release")]
        path_release = by_mode_window[(PATH_MODE, "release")]
        full_quiet = by_mode_window[(FULL_MODE, "quiet")]
        path_quiet = by_mode_window[(PATH_MODE, "quiet")]
        file.write("\n## Mechanism Read\n\n")
        file.write(
            f"Full insert release windows have {float(full_release['execution_overhead_mean_ms']):.3f} ms mean execution overhead, "
            f"or {float(full_release['batch_equiv_execution_overhead_mean_ms']):.2f} ms batch-equivalent overhead, "
            f"while repair-disabled CPU-delay release windows have {float(path_release['execution_overhead_mean_ms']):.3f} ms "
            f"or {float(path_release['batch_equiv_execution_overhead_mean_ms']):.2f} ms batch-equivalent overhead. "
            f"Quiet-window means are {float(full_quiet['execution_overhead_mean_ms']):.3f} ms and "
            f"{float(path_quiet['execution_overhead_mean_ms']):.3f} ms. "
            "Because the repair-disabled control keeps the burst schedule and writer service pressure but disables existing-node repair, "
            "this directly tests whether queue tail, generic backlog, or CPU delay alone explains the local read execution-latency spike.\n"
        )


def plot(out: Path, summary: list[dict[str, float | str]], bench: list[dict[str, float | str]]) -> None:
    labels = ["release", "release+drain", "quiet"]
    windows = ["release", "release_plus_drain", "quiet"]
    by_mode_window = {(row["mode"], row["window"]): row for row in summary}
    bench_by_mode = {row["mode"]: row for row in bench}
    x = list(range(len(windows)))
    width = 0.36
    full = [float(by_mode_window[(FULL_MODE, window)]["batch_equiv_execution_overhead_mean_ms"]) for window in windows]
    path = [float(by_mode_window[(PATH_MODE, window)]["batch_equiv_execution_overhead_mean_ms"]) for window in windows]
    full_p95 = [float(by_mode_window[(FULL_MODE, window)]["batch_equiv_execution_overhead_p95_ms"]) for window in windows]
    path_p95 = [float(by_mode_window[(PATH_MODE, window)]["batch_equiv_execution_overhead_p95_ms"]) for window in windows]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.patch.set_facecolor("white")
    axes[0].bar([i - width / 2 for i in x], full, width=width, color="#1d4ed8", label="full insert")
    axes[0].bar([i + width / 2 for i in x], path, width=width, color="#64748b", label="repair-disabled CPU delay")
    axes[0].scatter([i - width / 2 for i in x], full_p95, color="#111827", s=22, marker="_", label="p95")
    axes[0].scatter([i + width / 2 for i in x], path_p95, color="#111827", s=22, marker="_")
    axes[0].set_title("(a) Local execution overhead requires full mutation", loc="left")
    axes[0].set_ylabel("matched search execution overhead\n(batch-equivalent ms)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=16, ha="right")
    axes[0].legend(frameon=False)

    modes = [FULL_MODE, PATH_MODE]
    pressure = [float(bench_by_mode[mode]["unserved_insert_release_kpts_per_s"]) for mode in modes]
    axes[1].bar([0, 1], pressure, color=["#1d4ed8", "#64748b"])
    axes[1].set_title("(b) Writer pressure is not lower in the control", loc="left")
    axes[1].set_ylabel("unserved insert release (k points/s)")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["full insert", "repair-disabled\nCPU delay"])
    for i, value in enumerate(pressure):
        axes[1].text(i, value + max(pressure) * 0.03, f"{value:.1f}", ha="center", fontsize=8)

    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Burst matched control removes queue/backlog-only explanations", x=0.02, y=1.03, ha="left", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--readonly-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--case-prefix", default="shared_worker_pool")
    args = parser.parse_args()

    windows = load_windows(args.config)
    full_batches, full_bench = collect_mode(FULL_MODE, args.full_root, args.readonly_root, windows, args.batch_size, args.case_prefix)
    path_batches, path_bench = collect_mode(PATH_MODE, args.path_root, args.readonly_root, windows, args.batch_size, args.case_prefix)
    batch_rows = full_batches + path_batches
    bench_rows = full_bench + path_bench
    summary = summarize_batches(batch_rows, args.batch_size)
    bench = summarize_bench(bench_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "m3_sift1m_burst_matched_control_batches.csv", batch_rows)
    write_csv(args.out_dir / "m3_sift1m_burst_matched_control_summary.csv", summary)
    write_csv(args.out_dir / "m3_sift1m_burst_matched_control_pressure.csv", bench)
    write_md(args.out_dir / "m3_sift1m_burst_matched_control.md", summary, bench)
    plot(args.out_dir / "m3_sift1m_burst_matched_control.png", summary, bench)
    print(f"[m3-burst-matched-control] wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
