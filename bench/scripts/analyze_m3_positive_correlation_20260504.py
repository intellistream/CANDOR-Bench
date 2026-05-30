#!/usr/bin/env python3
"""Analyze clean M3 positive-correlation probes for 2026-05-04."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRUCTURAL_METRICS = [
    "insert_path_search_active_workers",
    "existing_neighbor_update_active_workers",
    "existing_neighbor_load_scan_active_workers",
    "existing_neighbor_prune_active_workers",
    "existing_neighbor_rewrite_active_workers",
    "existing_neighbor_undo_record_active_workers",
    "existing_neighbor_edges_written",
    "existing_neighbor_prune_dist_comps",
    "existing_neighbor_edges_loaded",
    "insert_path_dist_comps",
]

PERF_PREFIX = "perf_"


@dataclass(frozen=True)
class RunSpec:
    label: str
    root: Path
    co_workers: int
    efc: int
    batch: int
    perf: bool

    @property
    def case_dir(self) -> Path:
        matches = sorted(self.root.glob("closed_loop_odin_efc*"))
        if not matches:
            raise FileNotFoundError(f"{self.root} has no closed_loop_odin_efc* directory")
        return matches[0]


def parse_run(path: Path) -> RunSpec | None:
    if not path.is_dir():
        return None
    match = re.match(r"clean_b(?P<batch>\d+)_(?P<perf>perf_)?cw(?P<cw>\d+)_efc(?P<efc>\d+)$", path.name)
    if not match:
        return None
    return RunSpec(
        label=path.name,
        root=path,
        co_workers=int(match.group("cw")),
        efc=int(match.group("efc")),
        batch=int(match.group("batch")),
        perf=bool(match.group("perf")),
    )


def pctl(values: pd.Series, pct: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(np.percentile(vals.to_numpy(), pct))


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> tuple[float, int]:
    frame = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
        return float("nan"), len(frame)
    return float(frame["a"].corr(frame["b"], method=method)), int(len(frame))


def read_origin(case_dir: Path) -> int:
    meta = pd.read_csv(case_dir / "raw_latency" / "meta.csv").set_index("key")["value"]
    return int(float(meta["bench_start_raw_ns"]))


def mark_windows(values: pd.Series, windows: list[tuple[float, float]]) -> pd.Series:
    mask = pd.Series(False, index=values.index)
    for start, end in windows:
        mask |= (values >= start) & (values < end)
    return mask


def load_search(case_dir: Path, origin_raw_ns: int) -> pd.DataFrame:
    path = case_dir / "raw_latency" / "search_wide.csv"
    header = pd.read_csv(path, nrows=0).columns
    base_cols = [
        "measured",
        "start_raw_ns",
        "finish_raw_ns",
        "op_ms",
        "e2e_ms",
        "search_index_ms",
        "search_post_index_ms",
        "search_results_lock_wait_ms",
        "search_results_lock_hold_ms",
        "search_record_lock_wait_ms",
    ]
    diag_cols = [col for col in header if col.startswith("diag_")]
    usecols = [col for col in base_cols + diag_cols if col in header]
    search = pd.read_csv(path, usecols=usecols)
    for col in base_cols:
        if col not in search.columns:
            search[col] = 0.0
    search = search[search["measured"] == 1].copy()
    search["queue_ms"] = search["e2e_ms"] - search["op_ms"]
    search["mid_ms"] = (
        (search["start_raw_ns"] - origin_raw_ns) / 1_000_000.0
        + (search["finish_raw_ns"] - origin_raw_ns) / 1_000_000.0
    ) / 2.0
    search["diag_sum"] = search[diag_cols].sum(axis=1) if diag_cols else 0.0
    return search


def load_windows(run_root: Path) -> list[tuple[float, float]]:
    bounds = pd.read_csv(run_root / "closed_loop_odin_tail_insert_boundaries.csv")
    return list(bounds[["active_start_ms", "active_end_ms"]].itertuples(index=False, name=None))


def sample_phase_summary(spec: RunSpec, search: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for active, name in [(False, "outside"), (True, "insert_active")]:
        part = search[search["insert_active"] == active]
        row: dict[str, object] = {
            "run": spec.label,
            "co_workers": spec.co_workers,
            "efc": spec.efc,
            "batch": spec.batch,
            "perf": int(spec.perf),
            "phase": name,
            "measured_searches": int(len(part)),
            "op_mean_ms": float(part["op_ms"].mean()) if len(part) else float("nan"),
            "op_p95_ms": pctl(part["op_ms"], 95),
            "op_p99_ms": pctl(part["op_ms"], 99),
            "search_index_mean_ms": float(part["search_index_ms"].mean()) if len(part) else float("nan"),
            "search_index_p95_ms": pctl(part["search_index_ms"], 95),
            "search_index_p99_ms": pctl(part["search_index_ms"], 99),
            "search_post_index_p99_ms": pctl(part["search_post_index_ms"], 99),
            "record_wait_p99_ms": pctl(part["search_record_lock_wait_ms"], 99),
            "queue_p95_ms": pctl(part["queue_ms"], 95),
            "diag_sum": float(part["diag_sum"].sum()) if len(part) else 0.0,
        }
        rows.append(row)
    return rows


def bin_search_metrics(spec: RunSpec, search: pd.DataFrame, periods: pd.DataFrame, windows: list[tuple[float, float]]) -> pd.DataFrame:
    search = search.sort_values("mid_ms").reset_index(drop=True)
    times = search["mid_ms"].to_numpy()
    perf_cols = [col for col in periods.columns if col.startswith(PERF_PREFIX) and not col.endswith("_per_s")]
    rows: list[dict[str, object]] = []
    for period in periods.itertuples(index=False):
        row_raw = period._asdict()
        start = float(row_raw["bin_start_ms"])
        end = float(row_raw["bin_end_ms"])
        center = (start + end) / 2.0
        lo = np.searchsorted(times, start, side="left")
        hi = np.searchsorted(times, end, side="left")
        part = search.iloc[lo:hi]
        count = int(len(part))
        row: dict[str, object] = {
            "run": spec.label,
            "co_workers": spec.co_workers,
            "efc": spec.efc,
            "batch": spec.batch,
            "perf": int(spec.perf),
            "bin_start_ms": start,
            "insert_active": any(wstart <= center < wend for wstart, wend in windows),
            "measured_search_count": count,
            "op_mean_ms": float(part["op_ms"].mean()) if count else float("nan"),
            "op_p95_ms": pctl(part["op_ms"], 95),
            "op_p99_ms": pctl(part["op_ms"], 99),
            "search_index_mean_ms": float(part["search_index_ms"].mean()) if count else float("nan"),
            "search_index_p95_ms": pctl(part["search_index_ms"], 95),
            "search_index_p99_ms": pctl(part["search_index_ms"], 99),
            "search_post_index_p99_ms": pctl(part["search_post_index_ms"], 99),
            "record_wait_p99_ms": pctl(part["search_record_lock_wait_ms"], 99),
            "queue_p95_ms": pctl(part["queue_ms"], 95),
        }
        for col in STRUCTURAL_METRICS:
            if col in row_raw:
                row[col] = float(row_raw[col])
        for col in perf_cols:
            value = float(row_raw[col])
            row[col] = value
            row[f"{col}_per_measured_search"] = value / count if count else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def phase_delta(phase: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, group in phase.groupby("run"):
        outside = group[group["phase"] == "outside"]
        active = group[group["phase"] == "insert_active"]
        if outside.empty or active.empty:
            continue
        out = outside.iloc[0]
        ins = active.iloc[0]
        row = {
            "run": label,
            "co_workers": int(ins["co_workers"]),
            "efc": int(ins["efc"]),
            "batch": int(ins["batch"]),
            "perf": int(ins["perf"]),
            "delta_op_p99_ms": float(ins["op_p99_ms"] - out["op_p99_ms"]),
            "delta_search_index_p99_ms": float(ins["search_index_p99_ms"] - out["search_index_p99_ms"]),
            "outside_op_p99_ms": float(out["op_p99_ms"]),
            "insert_active_op_p99_ms": float(ins["op_p99_ms"]),
            "outside_search_index_p99_ms": float(out["search_index_p99_ms"]),
            "insert_active_search_index_p99_ms": float(ins["search_index_p99_ms"]),
            "outside_queue_p95_ms": float(out["queue_p95_ms"]),
            "insert_active_queue_p95_ms": float(ins["queue_p95_ms"]),
            "diag_sum": float(out["diag_sum"] + ins["diag_sum"]),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["perf", "co_workers", "efc", "run"])


def structural_phase_summary(bins: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    targets = ["op_p99_ms", "op_mean_ms", "search_index_p99_ms", "search_index_mean_ms", "queue_p95_ms"]
    for keys, group in bins.groupby(["run", "co_workers", "efc", "batch", "perf", "insert_active"]):
        run, co_workers, efc, batch, perf, active = keys
        row: dict[str, object] = {
            "run": run,
            "co_workers": co_workers,
            "efc": efc,
            "batch": batch,
            "perf": perf,
            "phase": "insert_active" if active else "outside",
            "bins": int(len(group)),
            "measured_search_count": int(group["measured_search_count"].sum()),
        }
        for target in targets:
            row[f"{target}_bin_mean"] = float(group[target].mean())
        for metric in STRUCTURAL_METRICS:
            if metric in group:
                row[f"{metric}_mean"] = float(group[metric].mean())
        perf_norm_cols = [col for col in group.columns if col.endswith("_per_measured_search")]
        for col in perf_norm_cols:
            row[f"{col}_mean"] = float(group[col].replace([np.inf, -np.inf], np.nan).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["perf", "co_workers", "efc", "phase", "run"])


def correlations(bins: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    targets = ["op_p99_ms", "op_mean_ms", "search_index_p99_ms", "search_index_mean_ms"]
    metrics = [m for m in STRUCTURAL_METRICS if m in bins.columns]
    metrics += [col for col in bins.columns if col.endswith("_per_measured_search")]
    rows: list[dict[str, object]] = []
    for target in targets:
        for metric in metrics:
            pearson, n = safe_corr(bins[target], bins[metric], "pearson")
            spearman, _ = safe_corr(bins[target], bins[metric], "spearman")
            rows.append(
                {
                    "subset": subset_name,
                    "target": target,
                    "metric": metric,
                    "pearson": pearson,
                    "spearman": spearman,
                    "bins": n,
                }
            )
    return pd.DataFrame(rows).sort_values(["subset", "target", "pearson"], ascending=[True, True, False])


def per_run_correlations(bins: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (label, active), group in bins.groupby(["run", "insert_active"]):
        for target in ["op_p99_ms", "search_index_p99_ms", "search_index_mean_ms"]:
            for metric in ["insert_path_search_active_workers", "existing_neighbor_update_active_workers", "existing_neighbor_edges_written"]:
                if metric not in group:
                    continue
                pearson, n = safe_corr(group[target], group[metric], "pearson")
                spearman, _ = safe_corr(group[target], group[metric], "spearman")
                rows.append(
                    {
                        "run": label,
                        "subset": "insert_active" if active else "outside",
                        "target": target,
                        "metric": metric,
                        "pearson": pearson,
                        "spearman": spearman,
                        "bins": n,
                    }
                )
    return pd.DataFrame(rows).sort_values(["run", "subset", "target", "metric"])


def intervention_correlations(case_summary: pd.DataFrame) -> pd.DataFrame:
    active = case_summary[(case_summary["phase"] == "insert_active") & (case_summary["perf"] == 0)].copy()
    metrics = [
        "insert_path_search_active_workers_mean",
        "existing_neighbor_update_active_workers_mean",
        "existing_neighbor_edges_written_mean",
    ]
    targets = ["search_index_p99_ms_bin_mean", "search_index_mean_ms_bin_mean", "op_p99_ms_bin_mean"]
    rows: list[dict[str, object]] = []
    for target in targets:
        for metric in metrics:
            pearson, n = safe_corr(active[target], active[metric], "pearson")
            spearman, _ = safe_corr(active[target], active[metric], "spearman")
            rows.append(
                {
                    "scope": "insert_active_case_means",
                    "target": target,
                    "metric": metric,
                    "pearson": pearson,
                    "spearman": spearman,
                    "cases": n,
                }
            )
    for co_workers, group in active.groupby("co_workers"):
        group = group.sort_values("efc")
        for target in targets:
            for metric in metrics:
                pearson, n = safe_corr(group[target], group[metric], "pearson")
                spearman, _ = safe_corr(group[target], group[metric], "spearman")
                rows.append(
                    {
                        "scope": f"insert_active_efc_sweep_cw{co_workers}",
                        "target": target,
                        "metric": metric,
                        "pearson": pearson,
                        "spearman": spearman,
                        "cases": n,
                    }
                )
    return pd.DataFrame(rows)


def efc_intervention_deltas(case_summary: pd.DataFrame) -> pd.DataFrame:
    active = case_summary[(case_summary["phase"] == "insert_active") & (case_summary["perf"] == 0)].copy()
    rows: list[dict[str, object]] = []
    for co_workers, group in active.groupby("co_workers"):
        by_efc = group.set_index("efc")
        for low, high in [(35, 100), (100, 200), (35, 200)]:
            if low not in by_efc.index or high not in by_efc.index:
                continue
            lo = by_efc.loc[low]
            hi = by_efc.loc[high]
            rows.append(
                {
                    "co_workers": co_workers,
                    "efc_low": low,
                    "efc_high": high,
                    "delta_insert_path_active_workers_mean": hi["insert_path_search_active_workers_mean"] - lo["insert_path_search_active_workers_mean"],
                    "delta_existing_update_active_workers_mean": hi["existing_neighbor_update_active_workers_mean"] - lo["existing_neighbor_update_active_workers_mean"],
                    "delta_search_index_p99_ms_bin_mean": hi["search_index_p99_ms_bin_mean"] - lo["search_index_p99_ms_bin_mean"],
                    "delta_search_index_mean_ms_bin_mean": hi["search_index_mean_ms_bin_mean"] - lo["search_index_mean_ms_bin_mean"],
                    "delta_op_p99_ms_bin_mean": hi["op_p99_ms_bin_mean"] - lo["op_p99_ms_bin_mean"],
                }
            )
    return pd.DataFrame(rows)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def plot_timeline(out: Path, bins: pd.DataFrame, run_root: Path, label: str) -> None:
    run_bins = bins[bins["run"] == label].copy()
    if run_bins.empty:
        return
    windows = load_windows(run_root)
    x = run_bins["bin_start_ms"] / 1000.0
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.8), sharex=True, constrained_layout=True)
    for ax in axes:
        for start, end in windows:
            ax.axvspan(start / 1000.0, end / 1000.0, color="#d95f5f", alpha=0.18, lw=0)
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    axes[0].plot(x, run_bins["search_index_p99_ms"], color="#1f77b4", lw=1.6, label="Search Index P99")
    axes[0].set_ylabel("ms")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].plot(x, run_bins["insert_path_search_active_workers"], color="#444444", lw=1.6, label="Insert Active Workers")
    axes[1].plot(x, run_bins["existing_neighbor_update_active_workers"], color="#ff7f0e", lw=1.4, label="Existing Update Workers")
    axes[1].set_ylabel("workers")
    axes[1].legend(frameon=False, loc="upper right", ncol=2)
    axes[2].plot(x, run_bins["op_p99_ms"], color="#2ca02c", lw=1.5, label="Search Op P99")
    axes[2].set_ylabel("ms")
    axes[2].legend(frameon=False, loc="upper right")
    axes[3].plot(x, run_bins["record_wait_p99_ms"], color="#8c564b", lw=1.4, label="Record Wait")
    axes[3].plot(x, run_bins["queue_p95_ms"], color="#9467bd", lw=1.2, label="Queue P95")
    axes[3].set_ylabel("ms")
    axes[3].set_xlabel("Elapsed Time (s)")
    axes[3].legend(frameon=False, loc="upper right", ncol=2)
    axes[0].set_title("Clean Real Insert Timeline")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def collect(root: Path, perf: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = [spec for p in sorted(root.iterdir()) if (spec := parse_run(p)) is not None]
    specs = [spec for spec in specs if spec.perf == perf]
    phase_rows: list[dict[str, object]] = []
    bin_frames: list[pd.DataFrame] = []
    sanity_rows: list[dict[str, object]] = []
    for spec in specs:
        case_dir = spec.case_dir
        origin = read_origin(case_dir)
        windows = load_windows(spec.root)
        search = load_search(case_dir, origin)
        search["insert_active"] = mark_windows(search["mid_ms"], windows)
        search = search[search["queue_ms"] < 0.1].copy()
        periods = pd.read_csv(case_dir / "period_groups.csv")
        bins = bin_search_metrics(spec, search, periods, windows)
        phase_rows.extend(sample_phase_summary(spec, search))
        bin_frames.append(bins)
        sanity_rows.append(
            {
                "run": spec.label,
                "co_workers": spec.co_workers,
                "efc": spec.efc,
                "batch": spec.batch,
                "perf": int(spec.perf),
                "measured_searches_lowq": int(len(search)),
                "diag_sum": float(search["diag_sum"].sum()),
                "max_queue_ms": float(search["queue_ms"].max()) if len(search) else float("nan"),
            }
        )
    phase = pd.DataFrame(phase_rows)
    bins = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
    structural = structural_phase_summary(bins) if not bins.empty else pd.DataFrame()
    sanity = pd.DataFrame(sanity_rows).sort_values(["perf", "co_workers", "efc"]) if sanity_rows else pd.DataFrame()
    return phase, bins, structural, sanity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--perf-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--timeline-label", default="clean_b20_cw16_efc100")
    args = parser.parse_args()

    tables = args.out_root / "tables"
    figures = args.out_root / "figures"

    clean_phase, clean_bins, clean_structural, clean_sanity = collect(args.clean_root, perf=False)
    perf_phase, perf_bins, perf_structural, perf_sanity = collect(args.perf_root, perf=True)

    clean_delta = phase_delta(clean_phase)
    perf_delta = phase_delta(perf_phase)
    clean_corr = pd.concat(
        [
            correlations(clean_bins, "all_clean_bins"),
            correlations(clean_bins[clean_bins["insert_active"]], "insert_active_clean_bins"),
            correlations(clean_bins[~clean_bins["insert_active"]], "outside_clean_bins"),
        ],
        ignore_index=True,
    )
    perf_corr = pd.concat(
        [
            correlations(perf_bins, "all_perf_bins"),
            correlations(perf_bins[perf_bins["insert_active"]], "insert_active_perf_bins"),
        ],
        ignore_index=True,
    )
    run_corr = per_run_correlations(clean_bins)
    intervention_corr = intervention_correlations(clean_structural)
    intervention_delta = efc_intervention_deltas(clean_structural)

    write_csv(tables / "diagnostic_sanity.csv", pd.concat([clean_sanity, perf_sanity], ignore_index=True))
    write_csv(tables / "clean_sample_phase_summary.csv", clean_phase)
    write_csv(tables / "clean_sample_phase_delta.csv", clean_delta)
    write_csv(tables / "clean_bin_metrics.csv", clean_bins)
    write_csv(tables / "clean_structural_phase_summary.csv", clean_structural)
    write_csv(tables / "clean_structural_correlations.csv", clean_corr)
    write_csv(tables / "clean_per_run_correlations.csv", run_corr)
    write_csv(tables / "intervention_case_correlations.csv", intervention_corr)
    write_csv(tables / "intervention_efc_deltas.csv", intervention_delta)
    write_csv(tables / "perf_sample_phase_summary.csv", perf_phase)
    write_csv(tables / "perf_sample_phase_delta.csv", perf_delta)
    write_csv(tables / "perf_bin_metrics.csv", perf_bins)
    write_csv(tables / "perf_structural_phase_summary.csv", perf_structural)
    write_csv(tables / "perf_structural_correlations.csv", perf_corr)

    copy_if_exists(args.clean_root / "commands.log", tables / "clean_matrix_commands.txt")
    copy_if_exists(args.perf_root / "commands.log", tables / "perf_control_commands.txt")
    copy_if_exists(args.clean_root.parent / f"{args.clean_root.name}.commands.log", tables / "clean_matrix_commands.txt")
    copy_if_exists(args.perf_root.parent / f"{args.perf_root.name}.commands.log", tables / "perf_control_commands.txt")

    plot_timeline(
        figures / "search_index_insert_active_timeline_cw16_efc100.png",
        clean_bins,
        args.clean_root / args.timeline_label,
        args.timeline_label,
    )


if __name__ == "__main__":
    main()
