#!/usr/bin/env python3
"""Analyze ef_construction=200 hardware counters for clean M3 real-insert runs."""

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
    "existing_neighbor_prune_active_workers",
    "existing_neighbor_rewrite_active_workers",
    "existing_neighbor_load_scan_active_workers",
    "existing_neighbor_undo_record_active_workers",
    "existing_neighbor_edges_written",
    "existing_neighbor_prune_dist_comps",
    "existing_neighbor_edges_loaded",
    "insert_path_dist_comps",
]

HARDWARE_EVENTS = [
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "LLC-loads",
    "LLC-load-misses",
    "dTLB-load-misses",
    "dtlb_load_misses.walk_completed",
    "l2_rqsts.all_rfo",
    "l2_rqsts.rfo_miss",
    "offcore_requests.demand_data_rd",
    "mem_load_retired.l3_miss",
    "mem_load_l3_miss_retired.remote_hitm",
    "ocr.demand_rfo.l3_miss",
    "ocr.demand_rfo.any_response",
    "offcore_requests.l3_miss_demand_data_rd",
    "offcore_requests_outstanding.l3_miss_demand_data_rd",
]


def normalize_event(event: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", event).strip("_").lower()


EVENT_TO_COLUMN = {event: f"perf_{normalize_event(event)}" for event in HARDWARE_EVENTS}

RATIO_DEFS = {
    "cycles_per_instruction": ("perf_cycles", "perf_instructions"),
    "cache_misses_per_instruction": ("perf_cache_misses", "perf_instructions"),
    "llc_load_misses_per_instruction": ("perf_llc_load_misses", "perf_instructions"),
    "dtlb_load_misses_per_instruction": ("perf_dtlb_load_misses", "perf_instructions"),
    "dtlb_walks_per_instruction": ("perf_dtlb_load_misses_walk_completed", "perf_instructions"),
    "rfo_misses_per_instruction": ("perf_l2_rqsts_rfo_miss", "perf_instructions"),
    "remote_hitm_per_instruction": ("perf_mem_load_l3_miss_retired_remote_hitm", "perf_instructions"),
    "ocr_rfo_l3_miss_per_instruction": ("perf_ocr_demand_rfo_l3_miss", "perf_instructions"),
    "offcore_l3_miss_data_rd_per_instruction": ("perf_offcore_requests_l3_miss_demand_data_rd", "perf_instructions"),
    "llc_load_miss_ratio": ("perf_llc_load_misses", "perf_llc_loads"),
    "cache_miss_ratio": ("perf_cache_misses", "perf_cache_references"),
    "rfo_l2_miss_ratio": ("perf_l2_rqsts_rfo_miss", "perf_l2_rqsts_all_rfo"),
    "ocr_rfo_l3_miss_ratio": ("perf_ocr_demand_rfo_l3_miss", "perf_ocr_demand_rfo_any_response"),
    "remote_hitm_per_llc_miss_load": ("perf_mem_load_l3_miss_retired_remote_hitm", "perf_mem_load_retired_l3_miss"),
    "llc_load_misses_per_cycle": ("perf_llc_load_misses", "perf_cycles"),
    "rfo_misses_per_cycle": ("perf_l2_rqsts_rfo_miss", "perf_cycles"),
    "ocr_rfo_l3_miss_per_cycle": ("perf_ocr_demand_rfo_l3_miss", "perf_cycles"),
}


@dataclass(frozen=True)
class RunSpec:
    label: str
    root: Path
    co_workers: int
    efc: int
    batch: int

    @property
    def case_dir(self) -> Path:
        matches = sorted(self.root.glob("closed_loop_odin_efc*"))
        if not matches:
            raise FileNotFoundError(f"{self.root} has no closed_loop_odin_efc* directory")
        return matches[0]


def parse_run(path: Path) -> RunSpec | None:
    match = re.match(r"clean_b(?P<batch>\d+)_perf_cw(?P<cw>\d+)_efc(?P<efc>\d+)$", path.name)
    if not path.is_dir() or not match:
        return None
    return RunSpec(
        label=path.name,
        root=path,
        co_workers=int(match.group("cw")),
        efc=int(match.group("efc")),
        batch=int(match.group("batch")),
    )


def pctl(values: pd.Series, pct: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float("nan")
    return float(np.percentile(vals.to_numpy(), pct))


def safe_corr(a: pd.Series, b: pd.Series, method: str) -> tuple[float, int]:
    frame = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["a"].nunique() < 2 or frame["b"].nunique() < 2:
        return float("nan"), int(len(frame))
    return float(frame["a"].corr(frame["b"], method=method)), int(len(frame))


def read_origin(case_dir: Path) -> int:
    meta = pd.read_csv(case_dir / "raw_latency" / "meta.csv").set_index("key")["value"]
    return int(float(meta["bench_start_raw_ns"]))


def load_windows(run_root: Path) -> list[tuple[float, float]]:
    bounds = pd.read_csv(run_root / "closed_loop_odin_tail_insert_boundaries.csv")
    return list(bounds[["active_start_ms", "active_end_ms"]].itertuples(index=False, name=None))


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


def sample_phase_summary(spec: RunSpec, search: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for active, name in [(False, "outside"), (True, "insert_active")]:
        part = search[search["insert_active"] == active]
        row: dict[str, object] = {
            "run": spec.label,
            "co_workers": spec.co_workers,
            "efc": spec.efc,
            "batch": spec.batch,
            "phase": name,
            "measured_searches": int(len(part)),
            "queue_mean_ms": float(part["queue_ms"].mean()) if len(part) else float("nan"),
            "queue_p95_ms": pctl(part["queue_ms"], 95),
            "queue_p99_ms": pctl(part["queue_ms"], 99),
            "op_mean_ms": float(part["op_ms"].mean()) if len(part) else float("nan"),
            "op_p99_ms": pctl(part["op_ms"], 99),
            "search_index_mean_ms": float(part["search_index_ms"].mean()) if len(part) else float("nan"),
            "search_index_p95_ms": pctl(part["search_index_ms"], 95),
            "search_index_p99_ms": pctl(part["search_index_ms"], 99),
            "search_post_index_p99_ms": pctl(part["search_post_index_ms"], 99),
            "record_wait_p99_ms": pctl(part["search_record_lock_wait_ms"], 99),
            "diag_sum": float(part["diag_sum"].sum()) if len(part) else 0.0,
        }
        rows.append(row)
    return rows


def add_hardware_normalizations(row: dict[str, object], count: int) -> None:
    for event, raw_col in EVENT_TO_COLUMN.items():
        if raw_col not in row:
            continue
        value = float(row[raw_col])
        row[f"{raw_col}_per_measured_search"] = value / count if count else float("nan")
    for ratio_name, (numerator, denominator) in RATIO_DEFS.items():
        if numerator not in row or denominator not in row:
            continue
        den = float(row[denominator])
        row[ratio_name] = float(row[numerator]) / den if den > 0 else float("nan")


def bin_metrics(spec: RunSpec, search: pd.DataFrame, periods: pd.DataFrame, windows: list[tuple[float, float]]) -> pd.DataFrame:
    search = search.sort_values("mid_ms").reset_index(drop=True)
    times = search["mid_ms"].to_numpy()
    period_cols = set(periods.columns)
    rows: list[dict[str, object]] = []
    for period in periods.itertuples(index=False):
        raw = period._asdict()
        start = float(raw["bin_start_ms"])
        end = float(raw["bin_end_ms"])
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
            "bin_start_ms": start,
            "bin_end_ms": end,
            "insert_active": any(wstart <= center < wend for wstart, wend in windows),
            "measured_search_count": count,
            "op_mean_ms": float(part["op_ms"].mean()) if count else float("nan"),
            "op_p99_ms": pctl(part["op_ms"], 99),
            "search_index_mean_ms": float(part["search_index_ms"].mean()) if count else float("nan"),
            "search_index_p95_ms": pctl(part["search_index_ms"], 95),
            "search_index_p99_ms": pctl(part["search_index_ms"], 99),
            "queue_p95_ms": pctl(part["queue_ms"], 95),
            "record_wait_p99_ms": pctl(part["search_record_lock_wait_ms"], 99),
        }
        for metric in STRUCTURAL_METRICS:
            if metric in period_cols:
                row[metric] = float(raw[metric])
        for raw_col in EVENT_TO_COLUMN.values():
            if raw_col in period_cols:
                row[raw_col] = float(raw[raw_col])
        add_hardware_normalizations(row, count)
        rows.append(row)
    return pd.DataFrame(rows)


def phase_from_bins(bins: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_targets = [
        "op_mean_ms",
        "op_p99_ms",
        "search_index_mean_ms",
        "search_index_p95_ms",
        "search_index_p99_ms",
        "queue_p95_ms",
    ]
    metric_cols = [col for col in bins.columns if col in STRUCTURAL_METRICS]
    metric_cols += [col for col in bins.columns if col.endswith("_per_measured_search")]
    metric_cols += [col for col in RATIO_DEFS if col in bins.columns]
    for keys, group in bins.groupby(["run", "co_workers", "efc", "batch", "insert_active"]):
        run, co_workers, efc, batch, active = keys
        row: dict[str, object] = {
            "run": run,
            "co_workers": co_workers,
            "efc": efc,
            "batch": batch,
            "phase": "insert_active" if active else "outside",
            "bins": int(len(group)),
            "measured_search_count": int(group["measured_search_count"].sum()),
        }
        for col in base_targets + metric_cols:
            if col in group.columns:
                row[f"{col}_bin_mean"] = float(group[col].replace([np.inf, -np.inf], np.nan).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["co_workers", "phase"])


def phase_delta(phase: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run, group in phase.groupby("run"):
        outside = group[group["phase"] == "outside"]
        active = group[group["phase"] == "insert_active"]
        if outside.empty or active.empty:
            continue
        out = outside.iloc[0]
        ins = active.iloc[0]
        row: dict[str, object] = {
            "run": run,
            "co_workers": int(ins["co_workers"]),
            "efc": int(ins["efc"]),
            "batch": int(ins["batch"]),
        }
        for col in phase.columns:
            if not col.endswith("_bin_mean"):
                continue
            row[f"outside_{col}"] = float(out[col])
            row[f"insert_active_{col}"] = float(ins[col])
            row[f"delta_{col}"] = float(ins[col] - out[col])
            if abs(float(out[col])) > 0:
                row[f"ratio_insert_vs_outside_{col}"] = float(ins[col] / out[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["co_workers", "run"])


def correlations(bins: pd.DataFrame, metrics: list[str], subset: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    targets = ["search_index_mean_ms", "search_index_p99_ms"]
    frame = bins.replace([np.inf, -np.inf], np.nan)
    for target in targets:
        for metric in metrics:
            if metric not in frame.columns:
                continue
            pearson, n = safe_corr(frame[target], frame[metric], "pearson")
            spearman, _ = safe_corr(frame[target], frame[metric], "spearman")
            rows.append(
                {
                    "subset": subset,
                    "target": target,
                    "metric": metric,
                    "pearson": pearson,
                    "spearman": spearman,
                    "bins": n,
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "pearson"], ascending=[True, False])


def per_run_correlations(bins: pd.DataFrame, metrics: list[str], kind: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    active = bins[bins["insert_active"]].copy()
    for run, group in active.groupby("run"):
        for target in ["search_index_mean_ms", "search_index_p99_ms"]:
            for metric in metrics:
                if metric not in group.columns:
                    continue
                pearson, n = safe_corr(group[target], group[metric], "pearson")
                spearman, _ = safe_corr(group[target], group[metric], "spearman")
                rows.append(
                    {
                        "kind": kind,
                        "run": run,
                        "target": target,
                        "metric": metric,
                        "pearson": pearson,
                        "spearman": spearman,
                        "bins": n,
                    }
                )
    return pd.DataFrame(rows).sort_values(["kind", "run", "target", "pearson"], ascending=[True, True, True, False])


def metric_score(corr: pd.DataFrame) -> pd.DataFrame:
    active = corr[corr["subset"] == "insert_active"].copy()
    rows: list[dict[str, object]] = []
    for metric, group in active.groupby("metric"):
        vals = group.set_index("target")
        mean_p = vals.loc["search_index_mean_ms", "pearson"] if "search_index_mean_ms" in vals.index else float("nan")
        mean_s = vals.loc["search_index_mean_ms", "spearman"] if "search_index_mean_ms" in vals.index else float("nan")
        p99_p = vals.loc["search_index_p99_ms", "pearson"] if "search_index_p99_ms" in vals.index else float("nan")
        p99_s = vals.loc["search_index_p99_ms", "spearman"] if "search_index_p99_ms" in vals.index else float("nan")
        positives = sum(1 for v in [mean_p, mean_s, p99_p, p99_s] if pd.notna(v) and v > 0)
        avg = float(np.nanmean([mean_p, mean_s, p99_p, p99_s]))
        rows.append(
            {
                "metric": metric,
                "search_index_mean_pearson": mean_p,
                "search_index_mean_spearman": mean_s,
                "search_index_p99_pearson": p99_p,
                "search_index_p99_spearman": p99_s,
                "positive_count_of_4": positives,
                "average_corr": avg,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["positive_count_of_4", "average_corr"], ascending=[False, False]
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def plot_timeline(out: Path, bins: pd.DataFrame, run_label: str, hardware_metric: str) -> None:
    run = bins[bins["run"] == run_label].copy()
    if run.empty or hardware_metric not in run.columns:
        return
    x = run["bin_start_ms"] / 1000.0
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        for _, row in run[run["insert_active"]].iterrows():
            ax.axvspan(row["bin_start_ms"] / 1000.0, row["bin_end_ms"] / 1000.0, color="#d95f5f", alpha=0.17, lw=0)
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    axes[0].plot(x, run["search_index_mean_ms"], color="#1f77b4", lw=1.6)
    axes[0].set_ylabel("Search Index Latency (ms)")
    axes[1].plot(x, run.get("insert_path_search_active_workers", 0.0), color="#374151", lw=1.5, label="insert path search")
    axes[1].plot(x, run.get("existing_neighbor_update_active_workers", 0.0), color="#e67e22", lw=1.4, label="existing update")
    axes[1].set_ylabel("Insert Active Workers")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    label_map = {
        "remote_hitm_per_instruction": "Remote HITM Loads Per Instruction",
        "remote_hitm_per_llc_miss_load": "Remote HITM Share Of L3-Miss Loads",
        "rfo_misses_per_instruction": "RFO Misses Per Instruction",
        "rfo_misses_per_cycle": "RFO Misses Per Cycle",
        "perf_l2_rqsts_rfo_miss_per_measured_search": "RFO Misses Per Search",
        "perf_llc_load_misses_per_measured_search": "LLC Misses Per Search",
    }
    label = label_map.get(hardware_metric)
    if label is None and hardware_metric.startswith("perf_"):
        label = hardware_metric.replace("perf_", "").replace("_per_measured_search", " per search").replace("_", " ").title()
    if label is None:
        label = hardware_metric.replace("_", " ").title()
    axes[2].plot(x, run[hardware_metric], color="#0f766e", lw=1.5)
    axes[2].set_ylabel(label)
    axes[2].set_xlabel("Elapsed Time (s)")
    axes[0].set_title(f"efc=200 Clean Real Insert, co-workers={int(run['co_workers'].iloc[0])}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def collect(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = [spec for p in sorted(root.iterdir()) if (spec := parse_run(p)) is not None]
    if not specs:
        raise FileNotFoundError(f"no clean_b*_perf_cw*_efc* runs found under {root}")
    phase_rows: list[dict[str, object]] = []
    bin_frames: list[pd.DataFrame] = []
    sanity_rows: list[dict[str, object]] = []
    for spec in specs:
        case_dir = spec.case_dir
        origin = read_origin(case_dir)
        windows = load_windows(spec.root)
        search = load_search(case_dir, origin)
        search["insert_active"] = mark_windows(search["mid_ms"], windows)
        periods = pd.read_csv(case_dir / "period_groups.csv")
        bins = bin_metrics(spec, search, periods, windows)
        phase_rows.extend(sample_phase_summary(spec, search))
        bin_frames.append(bins)
        sanity_rows.append(
            {
                "run": spec.label,
                "co_workers": spec.co_workers,
                "efc": spec.efc,
                "batch": spec.batch,
                "measured_searches": int(len(search)),
                "diag_sum": float(search["diag_sum"].sum()),
                "queue_mean_ms": float(search["queue_ms"].mean()) if len(search) else float("nan"),
                "queue_p95_ms": pctl(search["queue_ms"], 95),
                "queue_p99_ms": pctl(search["queue_ms"], 99),
                "queue_max_ms": float(search["queue_ms"].max()) if len(search) else float("nan"),
            }
        )
    phase = pd.DataFrame(phase_rows).sort_values(["co_workers", "phase"])
    bins = pd.concat(bin_frames, ignore_index=True)
    sanity = pd.DataFrame(sanity_rows).sort_values(["co_workers", "run"])
    return phase, bins, sanity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--timeline-run", default="clean_b20_perf_cw32_efc200")
    args = parser.parse_args()

    tables = args.out_root / "tables"
    figures = args.out_root / "figures"
    phase, bins, sanity = collect(args.run_root)
    phase_bins = phase_from_bins(bins)
    delta = phase_delta(phase_bins)

    hardware_metrics = [col for col in bins.columns if col.endswith("_per_measured_search")]
    hardware_metrics += [col for col in RATIO_DEFS if col in bins.columns]
    structural_metrics = [col for col in STRUCTURAL_METRICS if col in bins.columns]

    active = bins[bins["insert_active"] & (bins["measured_search_count"] > 0)].copy()
    outside = bins[(~bins["insert_active"]) & (bins["measured_search_count"] > 0)].copy()
    hw_corr = pd.concat(
        [
            correlations(active, hardware_metrics, "insert_active"),
            correlations(outside, hardware_metrics, "outside"),
            correlations(bins[bins["measured_search_count"] > 0], hardware_metrics, "all_bins"),
        ],
        ignore_index=True,
    )
    structural_corr = pd.concat(
        [
            correlations(active, structural_metrics, "insert_active"),
            correlations(outside, structural_metrics, "outside"),
            correlations(bins[bins["measured_search_count"] > 0], structural_metrics, "all_bins"),
        ],
        ignore_index=True,
    )
    hw_score = metric_score(hw_corr)
    structural_score = metric_score(structural_corr)
    per_run_hw = per_run_correlations(bins, hardware_metrics, "hardware")
    per_run_structural = per_run_correlations(bins, structural_metrics, "structural")

    write_csv(tables / "diagnostic_sanity.csv", sanity)
    write_csv(tables / "sample_phase_summary.csv", phase)
    write_csv(tables / "bin_metrics.csv", bins)
    write_csv(tables / "phase_bin_summary.csv", phase_bins)
    write_csv(tables / "phase_bin_delta.csv", delta)
    write_csv(tables / "hardware_correlations.csv", hw_corr)
    write_csv(tables / "hardware_metric_scores.csv", hw_score)
    write_csv(tables / "structural_correlations.csv", structural_corr)
    write_csv(tables / "structural_metric_scores.csv", structural_score)
    write_csv(tables / "per_run_hardware_correlations.csv", per_run_hw)
    write_csv(tables / "per_run_structural_correlations.csv", per_run_structural)
    copy_if_exists(args.run_root.parent / f"{args.run_root.name}.commands.log", tables / "perf_matrix_commands.txt")

    top_metric = "perf_l2_rqsts_rfo_miss_per_measured_search"
    if not hw_score.empty:
        top_metric = str(hw_score.iloc[0]["metric"])
    plot_timeline(figures / "efc200_hardware_timeline_cw32.png", bins, args.timeline_run, top_metric)


if __name__ == "__main__":
    main()
