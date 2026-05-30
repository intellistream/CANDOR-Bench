#!/usr/bin/env python3
"""Analyze ef_construction=200 M3 hardware-chain intervention runs."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

STRUCTURAL_METRICS = [
    "insert_path_search_active_workers",
    "existing_neighbor_update_active_workers",
    "existing_neighbor_load_scan_active_workers",
    "existing_neighbor_prune_active_workers",
    "existing_neighbor_rewrite_active_workers",
    "existing_neighbor_undo_record_active_workers",
    "insert_path_dist_comps",
    "existing_neighbor_edges_loaded",
    "existing_neighbor_prune_candidates",
    "existing_neighbor_prune_dist_comps",
    "existing_neighbor_edges_written",
    "existing_neighbor_edges_pruned",
    "existing_neighbor_undo_edges_recorded",
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
    "remote_hitm_per_instruction": ("perf_mem_load_l3_miss_retired_remote_hitm", "perf_instructions"),
    "remote_hitm_per_cycle": ("perf_mem_load_l3_miss_retired_remote_hitm", "perf_cycles"),
    "remote_hitm_per_llc_miss_load": ("perf_mem_load_l3_miss_retired_remote_hitm", "perf_mem_load_retired_l3_miss"),
    "rfo_requests_per_instruction": ("perf_l2_rqsts_all_rfo", "perf_instructions"),
    "rfo_misses_per_instruction": ("perf_l2_rqsts_rfo_miss", "perf_instructions"),
    "rfo_misses_per_cycle": ("perf_l2_rqsts_rfo_miss", "perf_cycles"),
    "rfo_l2_miss_ratio": ("perf_l2_rqsts_rfo_miss", "perf_l2_rqsts_all_rfo"),
    "ocr_rfo_l3_miss_per_instruction": ("perf_ocr_demand_rfo_l3_miss", "perf_instructions"),
    "ocr_rfo_l3_miss_per_cycle": ("perf_ocr_demand_rfo_l3_miss", "perf_cycles"),
    "ocr_rfo_any_response_per_instruction": ("perf_ocr_demand_rfo_any_response", "perf_instructions"),
    "ocr_rfo_l3_miss_ratio": ("perf_ocr_demand_rfo_l3_miss", "perf_ocr_demand_rfo_any_response"),
    "offcore_demand_data_rd_per_instruction": ("perf_offcore_requests_demand_data_rd", "perf_instructions"),
    "offcore_l3_miss_data_rd_per_instruction": ("perf_offcore_requests_l3_miss_demand_data_rd", "perf_instructions"),
    "outstanding_l3_miss_data_rd_per_cycle": (
        "perf_offcore_requests_outstanding_l3_miss_demand_data_rd",
        "perf_cycles",
    ),
}

KEY_CASES = {
    "neighbor_writes_on": {
        "label": "Neighbor Writes On",
        "order": 0,
        "flags": "",
        "semantics": "Production insert semantics: inserted-node links and existing-neighbor append/prune/rewrite/undo record remain enabled.",
    },
    "neighbor_read_scan_no_writes": {
        "label": "Neighbor Read Scan, No Writes",
        "order": 1,
        "flags": "EXISTING_UPDATE_READ_SCAN_ONLY=1",
        "semantics": "Intervention/control: existing-neighbor read scan and prune computation remain, but append/rewrite/undo writes are suppressed.",
    },
    "undo_record_off": {
        "label": "Undo Record Off",
        "order": 2,
        "flags": "DISABLE_EXISTING_UNDO_RECORD=1",
        "semantics": "Intervention/control: existing-neighbor append/rewrite remain, but MVCC undo-record writes for pruned edges are suppressed.",
    },
}

KEY_PHASE_COLUMNS = [
    "search_index_mean_ms",
    "search_index_p99_ms",
    "insert_path_search_active_workers",
    "existing_neighbor_update_active_workers",
    "existing_neighbor_prune_active_workers",
    "existing_neighbor_rewrite_active_workers",
    "existing_neighbor_undo_record_active_workers",
    "existing_neighbor_edges_written",
    "remote_hitm_per_instruction",
    "rfo_misses_per_instruction",
    "rfo_misses_per_cycle",
    "ocr_rfo_l3_miss_per_instruction",
    "ocr_rfo_l3_miss_per_cycle",
    "ocr_rfo_l3_miss_ratio",
]


@dataclass(frozen=True)
class RunSpec:
    root: Path
    co_workers: int
    case_key: str
    label: str
    order: int

    @property
    def case_dir(self) -> Path:
        matches = sorted(self.root.glob("closed_loop_odin_efc*"))
        if not matches:
            raise FileNotFoundError(f"{self.root} has no closed_loop_odin_efc* directory")
        return matches[0]


def parse_run(path: Path) -> RunSpec | None:
    if not path.is_dir():
        return None
    match = re.match(r"cw(?P<cw>\d+)_(?P<case>.+)$", path.name)
    if not match:
        return None
    case_key = match.group("case")
    if case_key not in KEY_CASES:
        return None
    meta = KEY_CASES[case_key]
    return RunSpec(
        root=path,
        co_workers=int(match.group("cw")),
        case_key=case_key,
        label=str(meta["label"]),
        order=int(meta["order"]),
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


def read_recall(case_dir: Path) -> float:
    path = case_dir / "benchmark_results.csv"
    if not path.exists():
        return float("nan")
    frame = pd.read_csv(path)
    if frame.empty or "recall" not in frame.columns:
        return float("nan")
    return float(frame["recall"].iloc[0])


def aggregate_ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> float:
    if numerator not in frame.columns or denominator not in frame.columns:
        return float("nan")
    num = pd.to_numeric(frame[numerator], errors="coerce").replace([np.inf, -np.inf], np.nan).sum()
    den = pd.to_numeric(frame[denominator], errors="coerce").replace([np.inf, -np.inf], np.nan).sum()
    return float(num / den) if den > 0 else float("nan")


def add_bin_normalizations(row: dict[str, object]) -> None:
    count = int(row.get("measured_search_count", 0) or 0)
    for raw_col in EVENT_TO_COLUMN.values():
        if raw_col in row:
            value = float(row[raw_col])
            row[f"{raw_col}_per_measured_search"] = value / count if count else float("nan")
    for ratio_name, (numerator, denominator) in RATIO_DEFS.items():
        if numerator not in row or denominator not in row:
            continue
        den = float(row[denominator])
        row[ratio_name] = float(row[numerator]) / den if den > 0 else float("nan")


def build_bin_metrics(spec: RunSpec, search: pd.DataFrame, periods: pd.DataFrame, windows: list[tuple[float, float]]) -> pd.DataFrame:
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
            "run": spec.root.name,
            "co_workers": spec.co_workers,
            "case_key": spec.case_key,
            "case_label": spec.label,
            "case_order": spec.order,
            "bin_start_ms": start,
            "bin_end_ms": end,
            "insert_active": any(wstart <= center < wend for wstart, wend in windows),
            "measured_search_count": count,
            "search_index_mean_ms": float(part["search_index_ms"].mean()) if count else float("nan"),
            "search_index_p95_ms": pctl(part["search_index_ms"], 95),
            "search_index_p99_ms": pctl(part["search_index_ms"], 99),
            "op_mean_ms": float(part["op_ms"].mean()) if count else float("nan"),
            "op_p99_ms": pctl(part["op_ms"], 99),
            "queue_p95_ms": pctl(part["queue_ms"], 95),
            "queue_p99_ms": pctl(part["queue_ms"], 99),
        }
        for metric in STRUCTURAL_METRICS:
            if metric in period_cols:
                row[metric] = float(raw[metric])
        for raw_col in EVENT_TO_COLUMN.values():
            if raw_col in period_cols:
                row[raw_col] = float(raw[raw_col])
        add_bin_normalizations(row)
        rows.append(row)
    return pd.DataFrame(rows)


def phase_rows(spec: RunSpec, search: pd.DataFrame, bins: pd.DataFrame, windows: list[tuple[float, float]], recall: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for active, phase in [(False, "outside"), (True, "insert_active")]:
        part = search[search["insert_active"] == active]
        bpart = bins[bins["insert_active"] == active]
        row: dict[str, object] = {
            "run": spec.root.name,
            "co_workers": spec.co_workers,
            "case_key": spec.case_key,
            "case_label": spec.label,
            "case_order": spec.order,
            "phase": phase,
            "recall_at_10": recall,
            "measured_searches": int(len(part)),
            "bins": int(len(bpart)),
            "queue_mean_ms": float(part["queue_ms"].mean()) if len(part) else float("nan"),
            "queue_p95_ms": pctl(part["queue_ms"], 95),
            "queue_p99_ms": pctl(part["queue_ms"], 99),
            "queue_max_ms": float(part["queue_ms"].max()) if len(part) else float("nan"),
            "op_mean_ms": float(part["op_ms"].mean()) if len(part) else float("nan"),
            "op_p99_ms": pctl(part["op_ms"], 99),
            "search_index_mean_ms": float(part["search_index_ms"].mean()) if len(part) else float("nan"),
            "search_index_p95_ms": pctl(part["search_index_ms"], 95),
            "search_index_p99_ms": pctl(part["search_index_ms"], 99),
            "search_post_index_mean_ms": float(part["search_post_index_ms"].mean()) if len(part) else float("nan"),
            "record_wait_p99_ms": pctl(part["search_record_lock_wait_ms"], 99),
            "diag_sum": float(part["diag_sum"].sum()) if len(part) else 0.0,
            "active_windows": len(windows),
        }
        for metric in STRUCTURAL_METRICS:
            if metric in bpart.columns:
                row[metric] = float(bpart[metric].mean())
        for raw_col in EVENT_TO_COLUMN.values():
            if raw_col in bpart.columns:
                value_sum = float(pd.to_numeric(bpart[raw_col], errors="coerce").sum())
                row[raw_col] = value_sum
                row[f"{raw_col}_per_measured_search"] = (
                    value_sum / int(bpart["measured_search_count"].sum())
                    if int(bpart["measured_search_count"].sum()) > 0
                    else float("nan")
                )
        for ratio_name, (numerator, denominator) in RATIO_DEFS.items():
            row[ratio_name] = aggregate_ratio(bpart, numerator, denominator)
        rows.append(row)
    return rows


def collect(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = [spec for p in sorted(run_root.iterdir()) if (spec := parse_run(p)) is not None]
    if not specs:
        raise FileNotFoundError(f"no cw*_case intervention runs found under {run_root}")

    phase: list[dict[str, object]] = []
    sanity: list[dict[str, object]] = []
    bin_frames: list[pd.DataFrame] = []
    semantics: list[dict[str, object]] = []
    for spec in sorted(specs, key=lambda s: (s.co_workers, s.order)):
        case_dir = spec.case_dir
        origin = read_origin(case_dir)
        windows = load_windows(spec.root)
        search = load_search(case_dir, origin)
        search["insert_active"] = mark_windows(search["mid_ms"], windows)
        periods = pd.read_csv(case_dir / "period_groups.csv")
        bins = build_bin_metrics(spec, search, periods, windows)
        recall = read_recall(case_dir)
        phase.extend(phase_rows(spec, search, bins, windows, recall))
        bin_frames.append(bins)
        semantics.append(
            {
                "run": spec.root.name,
                "co_workers": spec.co_workers,
                "case_key": spec.case_key,
                "case_label": spec.label,
                "env_flags": KEY_CASES[spec.case_key]["flags"],
                "semantics": KEY_CASES[spec.case_key]["semantics"],
                "result_root": str(spec.root),
                "case_dir": str(case_dir),
            }
        )
        sanity.append(
            {
                "run": spec.root.name,
                "co_workers": spec.co_workers,
                "case_key": spec.case_key,
                "case_label": spec.label,
                "recall_at_10": recall,
                "measured_searches": int(len(search)),
                "diag_sum": float(search["diag_sum"].sum()),
                "queue_mean_ms": float(search["queue_ms"].mean()) if len(search) else float("nan"),
                "queue_p95_ms": pctl(search["queue_ms"], 95),
                "queue_p99_ms": pctl(search["queue_ms"], 99),
                "queue_max_ms": float(search["queue_ms"].max()) if len(search) else float("nan"),
            }
        )
    phase_frame = pd.DataFrame(phase).sort_values(["co_workers", "case_order", "phase"])
    sanity_frame = pd.DataFrame(sanity).sort_values(["co_workers", "case_key"])
    bins_frame = pd.concat(bin_frames, ignore_index=True).sort_values(["co_workers", "case_order", "bin_start_ms"])
    semantics_frame = pd.DataFrame(semantics).sort_values(["co_workers", "case_key"])
    return phase_frame, bins_frame, sanity_frame, semantics_frame


def deltas(case_means: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for co_workers, group in case_means.groupby("co_workers"):
        base_rows = group[group["case_key"] == "neighbor_writes_on"]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        for _, row in group.iterrows():
            out: dict[str, object] = {
                "co_workers": co_workers,
                "case_key": row["case_key"],
                "case_label": row["case_label"],
                "base_case_label": base["case_label"],
            }
            for col in KEY_PHASE_COLUMNS:
                if col in group.columns:
                    out[f"{col}"] = row[col]
                    out[f"delta_{col}"] = row[col] - base[col]
                    out[f"ratio_vs_base_{col}"] = row[col] / base[col] if abs(base[col]) > 0 else float("nan")
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["co_workers", "case_key"])


def correlation_table(frame: pd.DataFrame, metrics: list[str], subset: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    targets = ["search_index_mean_ms", "search_index_p99_ms"]
    clean = frame.replace([np.inf, -np.inf], np.nan)
    for target in targets:
        for metric in metrics:
            if metric not in clean.columns or target not in clean.columns:
                continue
            pearson, n = safe_corr(clean[target], clean[metric], "pearson")
            spearman, _ = safe_corr(clean[target], clean[metric], "spearman")
            rows.append(
                {
                    "subset": subset,
                    "target": target,
                    "metric": metric,
                    "pearson": pearson,
                    "spearman": spearman,
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def all_correlations(case_means: pd.DataFrame, bins: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "insert_path_search_active_workers",
        "existing_neighbor_update_active_workers",
        "existing_neighbor_prune_active_workers",
        "existing_neighbor_rewrite_active_workers",
        "existing_neighbor_undo_record_active_workers",
        "existing_neighbor_edges_written",
        "remote_hitm_per_instruction",
        "rfo_misses_per_instruction",
        "rfo_misses_per_cycle",
        "ocr_rfo_l3_miss_per_instruction",
        "ocr_rfo_l3_miss_per_cycle",
        "ocr_rfo_l3_miss_ratio",
        "perf_mem_load_l3_miss_retired_remote_hitm_per_measured_search",
        "perf_l2_rqsts_rfo_miss_per_measured_search",
        "perf_ocr_demand_rfo_l3_miss_per_measured_search",
    ]
    case_frames = [correlation_table(case_means, metrics, "all_case_means")]
    for co_workers, group in case_means.groupby("co_workers"):
        case_frames.append(correlation_table(group, metrics, f"cw{co_workers}_case_means"))

    active_bins = bins[bins["insert_active"] & (bins["measured_search_count"] > 0)].copy()
    bin_frames = [correlation_table(active_bins, metrics, "all_insert_active_bins")]
    for co_workers, group in active_bins.groupby("co_workers"):
        bin_frames.append(correlation_table(group, metrics, f"cw{co_workers}_insert_active_bins"))

    return (
        pd.concat(case_frames, ignore_index=True).sort_values(["subset", "target", "metric"]),
        pd.concat(bin_frames, ignore_index=True).sort_values(["subset", "target", "metric"]),
    )


def fmt_case_label(label: str) -> str:
    return label.replace(", ", ",\n")


def plot_case_means(path: Path, case_means: pd.DataFrame) -> None:
    frame = case_means.sort_values(["co_workers", "case_order"]).copy()
    labels = [f"cw={int(r.co_workers)}\n{fmt_case_label(r.case_label)}" for r in frame.itertuples()]
    x = np.arange(len(frame))
    colors = {
        "neighbor_writes_on": "#4b5563",
        "neighbor_writes_off": "#2563eb",
        "neighbor_read_scan_no_writes": "#0f766e",
        "undo_record_off": "#7c3aed",
    }
    bar_colors = [colors[str(k)] for k in frame["case_key"]]

    fig, axes = plt.subplots(3, 1, figsize=(13.6, 9.0), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].bar(x, frame["existing_neighbor_edges_written"], color=bar_colors, alpha=0.86)
    axes[0].set_ylabel("Neighbor Edges Written")

    axes[1].plot(x, frame["remote_hitm_per_instruction"], marker="o", color="#dc2626", lw=1.8, label="Remote HITM / Instruction")
    axes[1].plot(x, frame["rfo_misses_per_instruction"], marker="s", color="#f59e0b", lw=1.8, label="RFO Misses / Instruction")
    axes[1].plot(x, frame["ocr_rfo_l3_miss_per_instruction"], marker="^", color="#0f766e", lw=1.6, label="OCR Demand RFO L3 Miss / Instruction")
    axes[1].set_ylabel("Normalized Hardware")
    axes[1].legend(frameon=False, ncol=3, loc="upper left")

    axes[2].bar(x, frame["search_index_mean_ms"], color=bar_colors, alpha=0.78, label="Mean")
    axes[2].plot(x, frame["search_index_p99_ms"], marker="o", color="#111827", lw=1.7, label="p99")
    axes[2].set_ylabel("Search Index Latency (ms)")
    axes[2].legend(frameon=False, ncol=2, loc="upper left")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    fig.suptitle("efc=200 Intervention Case Means", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def contiguous_windows(run: pd.DataFrame) -> list[tuple[float, float]]:
    active = run[run["insert_active"]].sort_values("bin_start_ms")
    if active.empty:
        return []
    windows: list[tuple[float, float]] = []
    start = float(active.iloc[0]["bin_start_ms"])
    last_end = float(active.iloc[0]["bin_end_ms"])
    for _, row in active.iloc[1:].iterrows():
        bstart = float(row["bin_start_ms"])
        bend = float(row["bin_end_ms"])
        if abs(bstart - last_end) <= 1e-6:
            last_end = bend
        else:
            windows.append((start, last_end))
            start, last_end = bstart, bend
    windows.append((start, last_end))
    return windows


def plot_timeline_grid(path: Path, bins: pd.DataFrame, co_workers: int) -> None:
    frame = bins[bins["co_workers"] == co_workers].copy()
    if frame.empty:
        return
    cases = sorted(frame["case_key"].unique(), key=lambda key: int(KEY_CASES[key]["order"]))
    fig, axes = plt.subplots(len(cases), 3, figsize=(14.5, 8.4), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    if len(cases) == 1:
        axes = np.array([axes])
    for row_idx, case_key in enumerate(cases):
        run = frame[frame["case_key"] == case_key].sort_values("bin_start_ms")
        x = run["bin_start_ms"] / 1000.0
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            for start, end in contiguous_windows(run):
                ax.axvspan(start / 1000.0, end / 1000.0, color="#d95f5f", alpha=0.16, lw=0)
            ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[row_idx, 0].plot(x, run["search_index_mean_ms"], color="#1f77b4", lw=1.45)
        axes[row_idx, 1].plot(x, run["existing_neighbor_edges_written"], color="#4b5563", lw=1.35)
        axes[row_idx, 2].plot(x, run["remote_hitm_per_instruction"], color="#dc2626", lw=1.35)
        axes[row_idx, 0].set_ylabel(str(KEY_CASES[case_key]["label"]), fontsize=9)
    axes[0, 0].set_title("Search Index Mean (ms)")
    axes[0, 1].set_title("Neighbor Edges Written")
    axes[0, 2].set_title("Remote HITM / Instruction")
    for ax in axes[-1, :]:
        ax.set_xlabel("Elapsed Time (s)")
    fig.suptitle(f"efc=200 Intervention Timeline, co-running workers={co_workers}", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_commands(path: Path, run_root: Path) -> None:
    events = ",".join(HARDWARE_EVENTS)
    lines = [
        "# efc=200 chain intervention commands",
        "cd /home/junyao/code/ANN-CC-bench/bench",
        "cmake --build ./build --target index -j 8",
        "LD_LIBRARY_PATH=./build/lib:${LD_LIBRARY_PATH:-} go build -o ./bin/bench .",
        "",
    ]
    for cw in [32, 40]:
        for case_key in ["neighbor_writes_on", "neighbor_writes_off", "neighbor_read_scan_no_writes", "undo_record_off"]:
            case = KEY_CASES[case_key]
            extra = f" {case['flags']}" if case["flags"] else ""
            run_name = f"cw{cw}_{case_key}"
            cmd = (
                f"STAMP=20260504_m3_efc200_chain_{run_name} "
                f"ROOT=./result/{run_root.name}/{run_name} "
                f"CONFIG_ROOT=./config/{run_root.name}/{run_name} "
                f"PAPER_ROOT=../paper-progress/m3/search-tail-rootcause-2026-05-01/efc200_chain_20260504/runs/{run_name} "
                f"MEASURED_WORKERS=8 COMPETING_WORKERS={cw} EF_SEARCH=35 EF_CONSTRUCTION=200 "
                f"BATCH_SIZE=20 DURATION_MS=12000 RUN_PERF=1 INTERVAL_MS=100 "
                f"PERF_EVENTS={events} DISABLE_BUILD_STEP=1 M3_BATCH_DIAG=0 "
                f"PHASE_OVERLAP_DIAG=0 REWRITE_ACTIVE_DIAG=0{extra} "
                "bash scripts/run_m3_closed_loop_odin_style.sh"
            )
            lines.append(cmd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    tables = args.out_root / "tables"
    figures = args.out_root / "figures"
    phase, bins, sanity, semantics = collect(args.run_root)
    case_means = phase[phase["phase"] == "insert_active"].copy()
    delta = deltas(case_means)
    case_corr, bin_corr = all_correlations(case_means, bins)

    write_csv(tables / "queue_diagnostic_sanity.csv", sanity)
    write_csv(tables / "intervention_semantics.csv", semantics)
    write_csv(tables / "phase_sample_summary.csv", phase)
    write_csv(tables / "case_insert_active_means.csv", case_means)
    write_csv(tables / "case_delta_vs_neighbor_writes_on.csv", delta)
    write_csv(tables / "case_mean_correlations.csv", case_corr)
    write_csv(tables / "bin_correlations_insert_active.csv", bin_corr)
    write_csv(tables / "bin_metrics.csv", bins)
    write_commands(tables / "run_commands.txt", args.run_root)

    plot_case_means(figures / "efc200_chain_case_means.png", case_means)
    for co_workers in sorted(case_means["co_workers"].unique()):
        plot_timeline_grid(figures / f"efc200_chain_timeline_cw{int(co_workers)}.png", bins, int(co_workers))

    print(f"wrote {args.out_root}")


if __name__ == "__main__":
    main()
