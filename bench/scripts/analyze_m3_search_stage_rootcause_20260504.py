#!/usr/bin/env python3
"""Analyze M3 measured-search tail attribution for the 2026-05-04 runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_COLS = [
    "search_index_ms",
    "search_post_index_ms",
    "search_results_lock_wait_ms",
    "search_results_lock_hold_ms",
    "search_record_lock_wait_ms",
    "search_unattributed_ms",
]

BASE_USECOLS = [
    "measured",
    "finish_ms",
    "create_raw_ns",
    "start_raw_ns",
    "finish_raw_ns",
    "op_ms",
    "e2e_ms",
    "search_index_ms",
    "search_post_index_ms",
    "search_results_lock_wait_ms",
    "search_results_lock_hold_ms",
    "search_record_lock_wait_ms",
    "diag_dist_comps",
    "diag_hops",
    "diag_batch_search_arena_ms",
    "diag_batch_search_knn_ms",
    "diag_batch_search_copy_ms",
    "diag_phase_existing_update_samples",
    "diag_phase_load_scan_samples",
    "diag_phase_prune_samples",
    "diag_phase_rewrite_samples",
]


@dataclass
class RunSpec:
    label: str
    root: Path

    @property
    def case_dir(self) -> Path:
        preferred = self.root / "closed_loop_odin_efc35"
        if preferred.exists():
            return preferred
        matches = sorted(self.root.glob("closed_loop_odin_efc*"))
        if matches:
            return matches[0]
        return preferred


def pctl(series: pd.Series, pct: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(np.percentile(vals.to_numpy(), pct))


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> tuple[float, int]:
    frame = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 4 or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
        return float("nan"), len(frame)
    return float(frame["a"].corr(frame["b"], method=method)), len(frame)


def load_origin(case_dir: Path) -> int:
    meta = pd.read_csv(case_dir / "raw_latency" / "meta.csv").set_index("key")["value"]
    return int(meta["bench_start_raw_ns"])


def load_boundaries(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "closed_loop_odin_tail_insert_boundaries.csv")


def mark_windows(values: pd.Series, windows: list[tuple[float, float]]) -> pd.Series:
    mask = pd.Series(False, index=values.index)
    for start, end in windows:
        mask |= (values >= start) & (values < end)
    return mask


def load_search(case_dir: Path, origin_raw_ns: int) -> pd.DataFrame:
    header = pd.read_csv(case_dir / "raw_latency" / "search_wide.csv", nrows=0)
    usecols = [col for col in BASE_USECOLS if col in header.columns]
    search = pd.read_csv(case_dir / "raw_latency" / "search_wide.csv", usecols=usecols)
    for col in BASE_USECOLS:
        if col not in search.columns:
            search[col] = 0.0
    search = search[search["measured"] == 1].copy()
    search["queue_ms"] = search["e2e_ms"] - search["op_ms"]
    search["start_ms"] = (search["start_raw_ns"] - origin_raw_ns) / 1_000_000.0
    search["finish_raw_ms"] = (search["finish_raw_ns"] - origin_raw_ns) / 1_000_000.0
    search["mid_ms"] = (search["start_ms"] + search["finish_raw_ms"]) / 2.0
    recorded = (
        search["search_index_ms"]
        + search["search_post_index_ms"]
        + search["search_results_lock_wait_ms"]
        + search["search_results_lock_hold_ms"]
        + search["search_record_lock_wait_ms"]
    )
    search["search_unattributed_ms"] = (search["op_ms"] - recorded).clip(lower=0.0)
    return search


def load_periods(case_dir: Path) -> pd.DataFrame:
    path = case_dir / "period_groups.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def phase_summary(label: str, lowq: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for phase_name, phase_value in [("Outside", False), ("Insert Active", True)]:
        part = lowq[lowq["insert_active"] == phase_value]
        row: dict[str, object] = {
            "run": label,
            "phase": phase_name,
            "lowq_measured_searches": int(len(part)),
            "op_mean_ms": float(part["op_ms"].mean()) if len(part) else float("nan"),
            "op_p95_ms": pctl(part["op_ms"], 95),
            "op_p99_ms": pctl(part["op_ms"], 99),
            "queue_p95_ms": pctl(part["queue_ms"], 95),
        }
        for col in STAGE_COLS:
            row[f"{col}_mean"] = float(part[col].mean()) if len(part) else float("nan")
            row[f"{col}_p99"] = pctl(part[col], 99)
        rows.append(row)
    return rows


def top_tail_summary(label: str, lowq: pd.DataFrame) -> list[dict[str, object]]:
    active = lowq[lowq["insert_active"]].copy()
    if active.empty:
        return []
    threshold = pctl(active["op_ms"], 99)
    active["tail_group"] = np.where(active["op_ms"] >= threshold, "Top 1% Insert-Active", "Other Insert-Active")
    rows: list[dict[str, object]] = []
    for group in ["Top 1% Insert-Active", "Other Insert-Active"]:
        part = active[active["tail_group"] == group]
        row: dict[str, object] = {
            "run": label,
            "group": group,
            "threshold_op_ms": threshold,
            "samples": int(len(part)),
            "op_mean_ms": float(part["op_ms"].mean()) if len(part) else float("nan"),
            "op_p50_ms": pctl(part["op_ms"], 50),
            "op_p95_ms": pctl(part["op_ms"], 95),
            "op_p99_ms": pctl(part["op_ms"], 99),
        }
        for col in STAGE_COLS:
            row[f"{col}_mean"] = float(part[col].mean()) if len(part) else float("nan")
            row[f"{col}_p95"] = pctl(part[col], 95)
            row[f"{col}_p99"] = pctl(part[col], 99)
        rows.append(row)
    return rows


def window_stats(label: str, lowq: pd.DataFrame, periods: pd.DataFrame, windows: list[tuple[float, float]], horizon_ms: float, bin_ms: float, window_ms: float) -> pd.DataFrame:
    lowq = lowq.sort_values("mid_ms").reset_index(drop=True)
    times = lowq["mid_ms"].to_numpy()
    bin_starts = np.arange(0.0, horizon_ms, bin_ms)
    half = window_ms / 2.0
    rows: list[dict[str, object]] = []
    perf_cols: list[str] = []
    period_cols: list[str] = []
    if not periods.empty:
        perf_cols = [col for col in periods.columns if col.startswith("perf_") and not col.endswith("_per_s")]
        period_cols = [
            col
            for col in [
                "insert_path_search_active_workers",
                "existing_neighbor_update_active_workers",
                "existing_neighbor_load_scan_active_workers",
                "existing_neighbor_prune_active_workers",
                "existing_neighbor_rewrite_active_workers",
                "existing_neighbor_undo_record_active_workers",
            ]
            if col in periods.columns
        ]
        periods = periods.set_index("bin_start_ms", drop=False)

    for bin_start in bin_starts:
        center = bin_start + bin_ms / 2.0
        lo = np.searchsorted(times, center - half, side="left")
        hi = np.searchsorted(times, center + half, side="left")
        part = lowq.iloc[lo:hi]
        blo = np.searchsorted(times, bin_start, side="left")
        bhi = np.searchsorted(times, bin_start + bin_ms, side="left")
        bin_count = int(bhi - blo)
        row: dict[str, object] = {
            "run": label,
            "bin_ms": bin_start,
            "insert_active": any(start <= center < end for start, end in windows),
            "lowq_search_count_window": int(len(part)),
            "lowq_search_count_bin": bin_count,
            "lowq_op_p95_ms": pctl(part["op_ms"], 95),
            "lowq_op_p99_ms": pctl(part["op_ms"], 99),
            "lowq_op_mean_ms": float(part["op_ms"].mean()) if len(part) else float("nan"),
            "lowq_queue_p95_ms": pctl(part["queue_ms"], 95),
        }
        for col in STAGE_COLS:
            row[f"lowq_{col}_p99"] = pctl(part[col], 99)
            row[f"lowq_{col}_mean"] = float(part[col].mean()) if len(part) else float("nan")
        for diag_col in [
            "diag_dist_comps",
            "diag_hops",
            "diag_batch_search_arena_ms",
            "diag_batch_search_knn_ms",
            "diag_batch_search_copy_ms",
            "diag_phase_existing_update_samples",
            "diag_phase_load_scan_samples",
            "diag_phase_prune_samples",
            "diag_phase_rewrite_samples",
        ]:
            if diag_col in part:
                row[f"lowq_{diag_col}_mean"] = float(part[diag_col].mean()) if len(part) else float("nan")
                row[f"lowq_{diag_col}_p99"] = pctl(part[diag_col], 99)
        if not periods.empty and bin_start in periods.index:
            prow = periods.loc[bin_start]
            for col in period_cols:
                row[col] = float(prow[col])
            for col in perf_cols:
                val = float(prow[col])
                row[col] = val
                row[f"{col}_per_measured_search"] = val / bin_count if bin_count > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def correlations(bin_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, group in bin_df.groupby("run"):
        candidates = [
            col
            for col in group.columns
            if (
                (col.startswith("lowq_") and col not in {"lowq_op_p99_ms", "lowq_op_p95_ms", "lowq_op_mean_ms"})
                or col.endswith("_workers")
                or col.endswith("_per_measured_search")
            )
        ]
        for col in candidates:
            pearson, n = safe_corr(group["lowq_op_p99_ms"], group[col], "pearson")
            spearman, _ = safe_corr(group["lowq_op_p99_ms"], group[col], "spearman")
            rows.append(
                {
                    "run": label,
                    "metric": col,
                    "pearson_vs_lowq_op_p99": pearson,
                    "spearman_vs_lowq_op_p99": spearman,
                    "bins": n,
                }
            )
    return pd.DataFrame(rows).sort_values(["run", "pearson_vs_lowq_op_p99"], ascending=[True, False])


def phase_delta(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, group in summary.groupby("run"):
        outside = group[group["phase"] == "Outside"]
        active = group[group["phase"] == "Insert Active"]
        if outside.empty or active.empty:
            continue
        out = outside.iloc[0]
        ins = active.iloc[0]
        row: dict[str, object] = {
            "run": label,
            "outside_op_p99_ms": out["op_p99_ms"],
            "insert_active_op_p99_ms": ins["op_p99_ms"],
            "delta_op_p99_ms": ins["op_p99_ms"] - out["op_p99_ms"],
            "ratio_op_p99": ins["op_p99_ms"] / out["op_p99_ms"] if out["op_p99_ms"] else float("nan"),
            "outside_queue_p95_ms": out["queue_p95_ms"],
            "insert_active_queue_p95_ms": ins["queue_p95_ms"],
        }
        for col in STAGE_COLS:
            row[f"delta_{col}_p99"] = ins[f"{col}_p99"] - out[f"{col}_p99"]
        rows.append(row)
    return pd.DataFrame(rows)


def hardware_phase_summary(bin_df: pd.DataFrame) -> pd.DataFrame:
    perf_cols = [col for col in bin_df.columns if col.endswith("_per_measured_search")]
    rows: list[dict[str, object]] = []
    for (label, active), group in bin_df.groupby(["run", "insert_active"]):
        if not perf_cols:
            continue
        row: dict[str, object] = {
            "run": label,
            "phase": "Insert Active" if active else "Outside",
            "bins": int(len(group)),
        }
        for col in perf_cols:
            row[f"{col}_mean"] = float(group[col].replace([np.inf, -np.inf], np.nan).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def plot_timeline(out: Path, bin_df: pd.DataFrame, boundaries: pd.DataFrame, diag_off_label: str, batch_label: str) -> None:
    off = bin_df[bin_df["run"] == diag_off_label]
    batch = bin_df[bin_df["run"] == batch_label]
    if off.empty or batch.empty:
        return
    fig, axes = plt.subplots(5, 1, figsize=(10, 9.5), sharex=True, constrained_layout=True)
    windows = list(boundaries[["active_start_ms", "active_end_ms"]].itertuples(index=False, name=None))
    for ax in axes:
        for start, end in windows:
            ax.axvspan(start / 1000.0, end / 1000.0, color="#d95f5f", alpha=0.18, lw=0)
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)

    axes[0].plot(batch["bin_ms"] / 1000.0, batch["lowq_op_p99_ms"], color="#d62728", lw=1.8, label="M3 Batch Diagnostics On")
    axes[0].set_ylabel("Search Latency (ms)")
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].plot(off["bin_ms"] / 1000.0, off["lowq_op_p99_ms"], color="#1f77b4", lw=1.8, label="Diagnostics Off")
    axes[1].set_ylabel("Search Latency (ms)")
    axes[1].legend(loc="upper right", frameon=False)

    axes[2].plot(batch["bin_ms"] / 1000.0, batch["lowq_search_index_ms_p99"], color="#2ca02c", lw=1.6, label="Search Index Time")
    axes[2].plot(batch["bin_ms"] / 1000.0, batch["lowq_search_post_index_ms_p99"], color="#9467bd", lw=1.6, label="Post-Index Time")
    axes[2].plot(batch["bin_ms"] / 1000.0, batch["lowq_search_record_lock_wait_ms_p99"], color="#8c564b", lw=1.4, label="Result Recording Wait")
    axes[2].set_ylabel("Stage Time (ms)")
    axes[2].legend(loc="upper right", frameon=False, ncol=3)

    axes[3].plot(batch["bin_ms"] / 1000.0, batch["lowq_search_unattributed_ms_p99"], color="#e377c2", lw=1.8, label="Op Minus Recorded Stages")
    axes[3].set_ylabel("Stage Gap (ms)")
    axes[3].legend(loc="upper right", frameon=False)

    axes[4].plot(batch["bin_ms"] / 1000.0, batch.get("insert_path_search_active_workers", 0), color="#444444", lw=1.5, label="Insert Path Workers")
    axes[4].plot(batch["bin_ms"] / 1000.0, batch.get("existing_neighbor_update_active_workers", 0), color="#ff7f0e", lw=1.5, label="Existing Update Workers")
    axes[4].set_ylabel("Threads")
    axes[4].set_xlabel("Elapsed Time (s)")
    axes[4].legend(loc="upper right", frameon=False, ncol=2)

    axes[0].set_title("Measured Search P99 With Real Insert Active Periods")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def parse_run(value: str) -> RunSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be label=/path/to/result/root")
    label, root = value.split("=", 1)
    return RunSpec(label=label, root=Path(root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--bin-ms", type=float, default=50.0)
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--horizon-ms", type=float, default=12000.0)
    parser.add_argument("--diag-off-label", default="diag_off_r1")
    parser.add_argument("--batch-label", default="batch_diag_only_r1")
    args = parser.parse_args()

    tables = args.out_root / "tables"
    figures = args.out_root / "figures"
    all_phase: list[dict[str, object]] = []
    all_tail: list[dict[str, object]] = []
    all_bins: list[pd.DataFrame] = []
    boundaries_for_plot: pd.DataFrame | None = None

    for spec in args.run:
        origin = load_origin(spec.case_dir)
        boundaries = load_boundaries(spec.root)
        if spec.label == args.batch_label:
            boundaries_for_plot = boundaries
        windows = list(boundaries[["active_start_ms", "active_end_ms"]].itertuples(index=False, name=None))
        search = load_search(spec.case_dir, origin)
        search["insert_active"] = mark_windows(search["mid_ms"], windows)
        lowq = search[search["queue_ms"] < 0.1].copy()
        periods = load_periods(spec.case_dir)
        all_phase.extend(phase_summary(spec.label, lowq))
        all_tail.extend(top_tail_summary(spec.label, lowq))
        all_bins.append(window_stats(spec.label, lowq, periods, windows, args.horizon_ms, args.bin_ms, args.window_ms))

    phase = pd.DataFrame(all_phase)
    tail = pd.DataFrame(all_tail)
    bins = pd.concat(all_bins, ignore_index=True) if all_bins else pd.DataFrame()
    deltas = phase_delta(phase)
    corr = correlations(bins)
    hw = hardware_phase_summary(bins)

    write_csv(tables / "stage_phase_summary.csv", phase)
    write_csv(tables / "stage_phase_delta_summary.csv", deltas)
    write_csv(tables / "top_tail_attribution.csv", tail)
    write_csv(tables / "bin_stage_metrics.csv", bins)
    write_csv(tables / "bin_stage_correlations.csv", corr)
    write_csv(tables / "hardware_phase_summary.csv", hw)

    if boundaries_for_plot is None and args.run:
        boundaries_for_plot = load_boundaries(args.run[0].root)
    if boundaries_for_plot is not None:
        plot_timeline(
            figures / "m3_diagnostic_tail_timeline.png",
            bins,
            boundaries_for_plot,
            args.diag_off_label,
            args.batch_label,
        )


if __name__ == "__main__":
    main()
