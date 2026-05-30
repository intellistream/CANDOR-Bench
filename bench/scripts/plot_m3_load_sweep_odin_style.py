#!/usr/bin/env python3
"""Plot measured-search impact across background-load periods."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_m3_odin_style_single_insert_period as base


def span_values(
    rows: list[tuple[float, float, float, float]],
    start: float,
    end: float,
    column_idx: int,
) -> list[float]:
    return [row[column_idx] for row in rows if start <= row[0] <= end]


def matched_pre_spans(insert_spans: list[tuple[float, float]], pre_window_s: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    previous_end = 0.0
    for start, _ in insert_spans:
        pre_start = max(previous_end, start - pre_window_s)
        spans.append((pre_start, start))
        previous_end = start
    return spans


def parse_rates(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bin-s", type=float, default=0.30)
    parser.add_argument("--pre-window-s", type=float, default=2.0)
    parser.add_argument("--load-rates", default="240000,560000,880000,1100000")
    args = parser.parse_args()

    windows, horizon, _insert_rate, _search_rate = base.load_config(args.config)
    batch_size = base.load_batch_size(args.config)
    target_offsets, burst_counts = base.target_offsets_and_counts(args.config)
    burst_starts = base.load_burst_issue_begins(args.run_dir, burst_counts) or [start for start, _ in windows]
    commit_times = base.commit_times_for_window_targets(base.load_commit_events(args.run_dir), target_offsets)
    finish_marks = commit_times[: len(burst_starts)]
    if finish_marks:
        horizon = max(horizon, max(finish_marks) + 2.0)
    insert_spans = base.insert_boundary_spans(burst_starts, finish_marks, horizon)
    pre_spans = matched_pre_spans(insert_spans, args.pre_window_s)
    load_rates = parse_rates(args.load_rates)

    rows = base.load_measured_search_rows(args.run_dir)
    pre_samples: list[float] = []
    insert_samples: list[float] = []
    pre_p95: list[float] = []
    insert_p95: list[float] = []
    pre_p99: list[float] = []
    insert_p99: list[float] = []

    for pre_span, insert_span in zip(pre_spans, insert_spans):
        pre = span_values(rows, pre_span[0], pre_span[1], 1)
        ins = span_values(rows, insert_span[0], insert_span[1], 1)
        pre_samples.extend(pre)
        insert_samples.extend(ins)
        pre_p95.append(base.percentile(pre, 0.95))
        insert_p95.append(base.percentile(ins, 0.95))
        pre_p99.append(base.percentile(pre, 0.99))
        insert_p99.append(base.percentile(ins, 0.99))

    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.4,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.9, 5.9), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax_lat, ax_workers, ax_cdf = axes

    period_max = 0.0
    for idx, ((pre_start, pre_end), (ins_start, ins_end)) in enumerate(zip(pre_spans, insert_spans)):
        rate = load_rates[idx] if idx < len(load_rates) else load_rates[-1]
        label = f"{rate / 1000:.0f}k pts/s"
        if not math.isnan(pre_p95[idx]):
            ax_lat.hlines(pre_p95[idx], pre_start, pre_end, color="#2563eb", lw=2.2, label="pre-insert search p95" if idx == 0 else None)
            period_max = max(period_max, pre_p95[idx])
        if not math.isnan(insert_p95[idx]):
            ax_lat.hlines(insert_p95[idx], ins_start, ins_end, color="#dc2626", lw=2.2, label="insert-span search p95" if idx == 0 else None)
            period_max = max(period_max, insert_p95[idx])
        delta = insert_p95[idx] - pre_p95[idx] if not math.isnan(pre_p95[idx]) and not math.isnan(insert_p95[idx]) else float("nan")
        if not math.isnan(delta):
            ax_lat.text((ins_start + ins_end) / 2.0, insert_p95[idx] * 1.04, f"+{delta:.2f} ms", ha="center", va="bottom", color="#991b1b", fontsize=6.2)
        ax_lat.text((pre_start + ins_end) / 2.0, 0.02, label, ha="center", va="bottom", color="#374151", fontsize=6.2)

    base.mark_windows(ax_lat, insert_spans, color="#e5e7eb", alpha=0.42)
    ax_lat.set_title("(a) Measured-search tail by load period", loc="left", pad=2)
    ax_lat.set_ylabel("E2E p95\n(ms)")
    ax_lat.set_xlim(0, horizon)
    ax_lat.set_ylim(0, max(0.25, period_max * 1.38))
    ax_lat.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.04, 1.0), borderaxespad=0.0, handlelength=1.6)

    xs_workers, total_workers, search_workers, insert_workers = base.bin_task_workers(args.run_dir, horizon, args.bin_s, batch_size)
    insert_top = [s + i for s, i in zip(search_workers, insert_workers)]
    ax_workers.fill_between(xs_workers, 0, search_workers, color="#2563eb", alpha=0.18, lw=0, label="search workers")
    ax_workers.fill_between(xs_workers, search_workers, insert_top, color="#dc2626", alpha=0.24, lw=0, label="insert workers")
    ax_workers.plot(xs_workers, total_workers, color="#111827", lw=1.05, label="active workers")
    ax_workers.axhline(96, color="#9ca3af", lw=0.7, ls=":", label="96 threads")
    base.mark_windows(ax_workers, insert_spans, color="#e5e7eb", alpha=0.42)
    ax_workers.set_title("(b) Shared worker occupancy", loc="left", pad=2)
    ax_workers.set_ylabel("Active\nworkers")
    ax_workers.set_xlim(0, horizon)
    ax_workers.set_ylim(0, 100)
    ax_workers.set_xlabel("Elapsed time (s)")
    ax_workers.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.08, 1.0), borderaxespad=0.0, handlelength=1.6)

    pre_x, pre_y = base.cdf_points(pre_samples)
    ins_x, ins_y = base.cdf_points(insert_samples)
    ax_cdf.plot(pre_x, pre_y, color="#2563eb", lw=1.6, label="matched pre-insert")
    ax_cdf.plot(ins_x, ins_y, color="#dc2626", lw=1.6, label="insert span")
    pre_cdf_p95 = base.percentile(pre_samples, 0.95)
    ins_cdf_p95 = base.percentile(insert_samples, 0.95)
    if not math.isnan(pre_cdf_p95):
        ax_cdf.axvline(pre_cdf_p95, color="#2563eb", lw=0.9, ls=":", alpha=0.8)
    if not math.isnan(ins_cdf_p95):
        ax_cdf.axvline(ins_cdf_p95, color="#dc2626", lw=0.9, ls=":", alpha=0.8)
    p997 = base.percentile(pre_samples + insert_samples, 0.997)
    if not math.isnan(p997) and p997 > 0:
        ax_cdf.set_xlim(0, p997 * 1.05)
    ax_cdf.set_title("(c) Matched measured-search E2E CDF", loc="left", pad=2)
    ax_cdf.set_xlabel("Measured-search latency (ms, p99.7 view)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_ylim(0, 1.01)
    ax_cdf.legend(frameon=False, loc="lower right", handlelength=1.8)

    for ax in axes:
        ax.grid(axis="y", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(f"[m3-load-sweep] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
