#!/usr/bin/env python3
"""Summarize graph write probe runs with active-window unit costs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


PERF_EVENTS = {
    "cycles": "perf_cycles",
    "instructions": "perf_instructions",
    "cache_misses": "perf_cache_misses",
    "llc_load_misses": "perf_llc_load_misses",
    "dtlb_load_misses": "perf_dtlb_load_misses",
    "l3_miss": "perf_mem_load_retired_l3_miss",
    "remote_hitm": "perf_mem_load_l3_miss_retired_remote_hitm",
    "rfo_all": "perf_l2_rqsts_all_rfo",
    "rfo_miss": "perf_l2_rqsts_rfo_miss",
    "offcore_l3_miss_rd": "perf_offcore_requests_l3_miss_demand_data_rd",
    "offcore_l3_miss_rd_outstanding": "perf_offcore_requests_outstanding_l3_miss_demand_data_rd",
}


def percentile(values: pd.Series, pct: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return math.nan
    return float(np.percentile(vals.to_numpy(), pct))


def case_dir(run_root: Path) -> Path:
    matches = sorted(run_root.glob("closed_loop_odin_efc*"))
    if not matches:
        raise FileNotFoundError(f"missing closed_loop_odin_efc* under {run_root}")
    return matches[0]


def resolve_run_root(raw: object) -> Path:
    run_root = Path(str(raw))
    if not run_root.exists() and (Path("bench") / run_root).exists():
        run_root = Path("bench") / run_root
    return run_root


def load_search(root: Path) -> pd.DataFrame:
    path = root / "raw_latency" / "search_wide.csv"
    header = pd.read_csv(path, nrows=0).columns
    cols = [
        "measured",
        "finish_ms",
        "start_raw_ns",
        "finish_raw_ns",
        "search_index_ms",
        "op_ms",
        "work_searchknn_ms",
        "work_base_search_ms",
        "work_level0_edges_scanned",
        "work_distance_computations",
    ]
    usecols = [c for c in cols if c in header]
    frame = pd.read_csv(path, usecols=usecols)
    frame = frame[frame["measured"] == 1].copy()
    frame["mid_raw_ns"] = (frame["start_raw_ns"] + frame["finish_raw_ns"]) / 2.0
    if "finish_ms" in frame:
        frame["mid_ms"] = frame["finish_ms"] - (frame["finish_raw_ns"] - frame["mid_raw_ns"]) / 1_000_000.0
    else:
        frame["mid_ms"] = (frame["mid_raw_ns"] - frame["mid_raw_ns"].min()) / 1_000_000.0
    if "search_index_ms" not in frame:
        frame["search_index_ms"] = frame["op_ms"]
    return frame


def load_insert_windows(root: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]], pd.DataFrame]:
    path = root / "raw_latency" / "insert_wide.csv"
    if not path.exists():
        return [], [], pd.DataFrame()
    inserts = pd.read_csv(path)
    if inserts.empty or "start_raw_ns" not in inserts or "finish_raw_ns" not in inserts:
        return [], [], inserts

    raw_windows: list[tuple[float, float]] = []
    ms_windows: list[tuple[float, float]] = []
    for item in inserts.sort_values("start_raw_ns")[["start_raw_ns", "finish_raw_ns", "finish_ms"]].itertuples(index=False):
        start, end, finish_ms = item
        start = float(start)
        end = float(end)
        finish_ms = float(finish_ms)
        start_ms = finish_ms - (end - start) / 1_000_000.0
        if not raw_windows or start > raw_windows[-1][1] + 10_000_000.0:
            raw_windows.append((start, end))
            ms_windows.append((start_ms, finish_ms))
        else:
            old_start, old_end = raw_windows[-1]
            raw_windows[-1] = (old_start, max(old_end, end))
            old_start_ms, old_end_ms = ms_windows[-1]
            ms_windows[-1] = (old_start_ms, max(old_end_ms, finish_ms))
    return raw_windows, ms_windows, inserts


def select_active(frame: pd.DataFrame, windows: list[tuple[float, float]], time_col: str) -> pd.DataFrame:
    if not windows:
        return frame
    mask = pd.Series(False, index=frame.index)
    for start, end in windows:
        mask |= (frame[time_col] >= start) & (frame[time_col] < end)
    active = frame[mask].copy()
    return active if not active.empty else frame


def summarize_perf(root: Path, windows: list[tuple[float, float]], l0_edges: float, time_col: str) -> dict[str, float]:
    path = root / "period_groups.csv"
    out: dict[str, float] = {}
    if not path.exists() or l0_edges <= 0:
        return out

    groups = pd.read_csv(path)
    if time_col == "mid_ms":
        groups["mid_ms"] = (groups["bin_start_ms"] + groups["bin_end_ms"]) / 2.0
    else:
        groups["mid_raw_ns"] = (groups["bin_start_ns"] + groups["bin_end_ns"]) / 2.0
    active = select_active(groups, windows, time_col)
    out["perf_active_bins"] = float(len(active))
    for name, col in PERF_EVENTS.items():
        if col not in active:
            continue
        total = float(pd.to_numeric(active[col], errors="coerce").fillna(0.0).sum())
        out[f"{name}_total"] = total
        out[f"{name}_per_l0_edge"] = total / l0_edges
    return out


def summarize_case(row: pd.Series) -> dict[str, object]:
    root = case_dir(resolve_run_root(row["run_root"]))
    search = load_search(root)
    raw_windows, _, inserts = load_insert_windows(root)
    active = select_active(search, raw_windows, "mid_raw_ns")

    l0_edges = float(active["work_level0_edges_scanned"].sum())
    base_ms = float(active["work_base_search_ms"].sum())
    searchknn_ms = float(active["work_searchknn_ms"].sum())
    search_index_ms = float(active["search_index_ms"].sum())
    distance_comps = float(active["work_distance_computations"].sum())
    out: dict[str, object] = {
        "dataset": row["dataset"],
        "case_key": row["case_key"],
        "case_label": row["case_label"],
        "competing_workers": int(row["competing_workers"]),
        "active_searches": int(len(active)),
        "active_insert_batches": int(len(inserts)),
        "active_windows": int(len(raw_windows)),
        "l0_edges_total": l0_edges,
        "distance_computations_total": distance_comps,
        "search_index_ms_mean": float(active["search_index_ms"].mean()),
        "search_index_ms_p99": percentile(active["search_index_ms"], 99),
        "work_base_search_ms_mean": float(active["work_base_search_ms"].mean()),
        "base_us_per_l0_edge": base_ms * 1000.0 / l0_edges if l0_edges else math.nan,
        "searchknn_us_per_l0_edge": searchknn_ms * 1000.0 / l0_edges if l0_edges else math.nan,
        "search_index_us_per_l0_edge": search_index_ms * 1000.0 / l0_edges if l0_edges else math.nan,
    }
    if "graph_existing_neighbor_edges_written" in inserts and len(inserts):
        out["graph_edges_written_total"] = float(inserts["graph_existing_neighbor_edges_written"].sum())
        out["graph_edges_written_mean"] = float(inserts["graph_existing_neighbor_edges_written"].mean())
    else:
        out["graph_edges_written_total"] = 0.0
        out["graph_edges_written_mean"] = 0.0
    out.update(summarize_perf(root, raw_windows, l0_edges, "mid_raw_ns"))
    return out


def summarize_elapsed_window(row: pd.Series, windows_ms: list[tuple[float, float]]) -> dict[str, float]:
    root = case_dir(resolve_run_root(row["run_root"]))
    search = load_search(root)
    active = select_active(search, windows_ms, "mid_ms")

    l0_edges = float(active["work_level0_edges_scanned"].sum())
    base_ms = float(active["work_base_search_ms"].sum())
    searchknn_ms = float(active["work_searchknn_ms"].sum())
    search_index_ms = float(active["search_index_ms"].sum())
    out: dict[str, float] = {
        "active_searches": float(len(active)),
        "l0_edges_total": l0_edges,
        "search_index_ms_mean": float(active["search_index_ms"].mean()),
        "search_index_ms_p99": percentile(active["search_index_ms"], 99),
        "base_us_per_l0_edge": base_ms * 1000.0 / l0_edges if l0_edges else math.nan,
        "searchknn_us_per_l0_edge": searchknn_ms * 1000.0 / l0_edges if l0_edges else math.nan,
        "search_index_us_per_l0_edge": search_index_ms * 1000.0 / l0_edges if l0_edges else math.nan,
    }
    out.update(summarize_perf(root, windows_ms, l0_edges, "mid_ms"))
    return out


def make_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "search_index_ms_mean",
        "search_index_ms_p99",
        "base_us_per_l0_edge",
        "searchknn_us_per_l0_edge",
        "search_index_us_per_l0_edge",
        "cache_misses_per_l0_edge",
        "llc_load_misses_per_l0_edge",
        "dtlb_load_misses_per_l0_edge",
        "l3_miss_per_l0_edge",
        "remote_hitm_per_l0_edge",
        "rfo_all_per_l0_edge",
        "rfo_miss_per_l0_edge",
        "offcore_l3_miss_rd_per_l0_edge",
    ]
    rows: list[dict[str, object]] = []
    for (dataset, workers), group in summary.groupby(["dataset", "competing_workers"]):
        base = group[group["case_key"] == "background_search"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, row in group.iterrows():
            if row["case_key"] == "background_search":
                continue
            out: dict[str, object] = {
                "dataset": dataset,
                "competing_workers": int(workers),
                "case_key": row["case_key"],
                "case_label": row["case_label"],
            }
            for metric in metrics:
                if metric not in group:
                    continue
                base_val = float(base_row[metric])
                case_val = float(row[metric])
                out[f"baseline_{metric}"] = base_val
                out[f"case_{metric}"] = case_val
                out[f"delta_{metric}"] = case_val - base_val
                out[f"ratio_{metric}"] = case_val / base_val if base_val else math.nan
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["dataset", "competing_workers", "case_key"])


def make_matched_deltas(params: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "search_index_ms_mean",
        "search_index_ms_p99",
        "base_us_per_l0_edge",
        "searchknn_us_per_l0_edge",
        "search_index_us_per_l0_edge",
        "cache_misses_per_l0_edge",
        "llc_load_misses_per_l0_edge",
        "dtlb_load_misses_per_l0_edge",
        "l3_miss_per_l0_edge",
        "remote_hitm_per_l0_edge",
        "rfo_all_per_l0_edge",
        "rfo_miss_per_l0_edge",
        "offcore_l3_miss_rd_per_l0_edge",
    ]
    rows: list[dict[str, object]] = []
    for _, row in params.iterrows():
        if row["case_key"] == "background_search":
            continue
        base_params = params[
            (params["dataset"] == row["dataset"])
            & (params["competing_workers"] == row["competing_workers"])
            & (params["case_key"] == "background_search")
        ]
        case_summary = summary[
            (summary["dataset"] == row["dataset"])
            & (summary["competing_workers"] == int(row["competing_workers"]))
            & (summary["case_key"] == row["case_key"])
        ]
        if base_params.empty or case_summary.empty:
            continue
        root = case_dir(resolve_run_root(row["run_root"]))
        _, windows_ms, _ = load_insert_windows(root)
        base = summarize_elapsed_window(base_params.iloc[0], windows_ms)
        case = case_summary.iloc[0]
        out: dict[str, object] = {
            "dataset": row["dataset"],
            "competing_workers": int(row["competing_workers"]),
            "case_key": row["case_key"],
            "case_label": row["case_label"],
            "matched_background_active_searches": base.get("active_searches", math.nan),
            "case_active_searches": float(case["active_searches"]),
        }
        for metric in metrics:
            if metric not in case or metric not in base:
                continue
            base_val = float(base[metric])
            case_val = float(case[metric])
            out[f"matched_background_{metric}"] = base_val
            out[f"case_{metric}"] = case_val
            out[f"delta_{metric}"] = case_val - base_val
            out[f"ratio_{metric}"] = case_val / base_val if base_val else math.nan
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["dataset", "competing_workers", "case_key"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    table_root = out_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)

    params = pd.read_csv(args.params)
    summary = pd.DataFrame([summarize_case(row) for _, row in params.iterrows()])
    summary.to_csv(table_root / "graph_write_probe_unit_cost.csv", index=False)
    make_deltas(summary).to_csv(table_root / "graph_write_probe_delta_vs_background.csv", index=False)
    make_matched_deltas(params, summary).to_csv(
        table_root / "graph_write_probe_matched_delta_vs_background.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
