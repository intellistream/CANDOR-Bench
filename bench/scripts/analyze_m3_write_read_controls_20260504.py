#!/usr/bin/env python3
"""Analyze high-recall controls separating write-request work from read concurrency."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


CASE_ORDER = {
    "background_search": 0,
    "insert_ann_search": 1,
    "insert_cpu": 2,
    "real_insert": 3,
}


def percentile(values: pd.Series, pct: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return math.nan
    return float(np.percentile(vals.to_numpy(), pct))


def read_recall(run_dir: Path) -> float:
    path = run_dir / "benchmark_results.csv"
    if not path.exists():
        return math.nan
    frame = pd.read_csv(path)
    if "recall" not in frame or frame.empty:
        return math.nan
    return float(frame["recall"].iloc[0])


def find_case_dir(run_root: Path) -> Path:
    if not run_root.exists() and (Path("bench") / run_root).exists():
        run_root = Path("bench") / run_root
    matches = sorted(run_root.glob("closed_loop_odin_efc*"))
    if not matches:
        raise FileNotFoundError(f"no closed_loop_odin_efc* under {run_root}")
    return matches[0]


def load_search(case_dir: Path) -> pd.DataFrame:
    path = case_dir / "raw_latency" / "search_wide.csv"
    header = pd.read_csv(path, nrows=0).columns
    cols = [
        "measured",
        "create_raw_ns",
        "start_raw_ns",
        "finish_raw_ns",
        "op_ms",
        "e2e_ms",
        "search_index_ms",
        "search_post_index_ms",
        "work_searchknn_ms",
        "work_searchknn_thread_cpu_ms",
        "work_base_search_ms",
        "work_level0_edges_scanned",
        "work_distance_computations",
    ]
    usecols = [c for c in cols if c in header]
    search = pd.read_csv(path, usecols=usecols)
    search = search[search["measured"] == 1].copy()
    search["queue_ms"] = (search["start_raw_ns"] - search["create_raw_ns"]) / 1_000_000.0
    search["mid_raw_ns"] = (search["start_raw_ns"] + search["finish_raw_ns"]) / 2.0
    if "search_index_ms" not in search:
        search["search_index_ms"] = search["op_ms"]
    return search


def load_insert_windows(case_dir: Path) -> tuple[list[tuple[float, float]], pd.DataFrame]:
    path = case_dir / "raw_latency" / "insert_wide.csv"
    if not path.exists():
        return [], pd.DataFrame()
    inserts = pd.read_csv(path)
    if inserts.empty or "start_raw_ns" not in inserts or "finish_raw_ns" not in inserts:
        return [], inserts
    windows: list[tuple[float, float]] = []
    for start, end in inserts.sort_values("start_raw_ns")[["start_raw_ns", "finish_raw_ns"]].itertuples(index=False):
        start = float(start)
        end = float(end)
        if not windows or start > windows[-1][1] + 10_000_000.0:
            windows.append((start, end))
        else:
            old_start, old_end = windows[-1]
            windows[-1] = (old_start, max(old_end, end))
    return windows, inserts


def active_search(search: pd.DataFrame, windows: list[tuple[float, float]]) -> pd.DataFrame:
    if not windows:
        return search
    mask = pd.Series(False, index=search.index)
    for start, end in windows:
        mask |= (search["mid_raw_ns"] >= start) & (search["mid_raw_ns"] < end)
    part = search[mask].copy()
    return part if not part.empty else search


def summarize_insert(inserts: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {"insert_batches": float(len(inserts))}
    wanted = [
        "graph_insert_path_ms",
        "graph_insert_path_edges_scanned",
        "graph_insert_path_dist_comps",
        "graph_existing_neighbor_update_loop_ms",
        "graph_existing_neighbor_prune_ms",
        "graph_existing_neighbor_visits",
        "graph_existing_neighbor_loaded_edges",
        "graph_existing_neighbor_prune_candidates",
        "graph_existing_neighbor_prune_dist_comps",
        "graph_existing_neighbor_edges_written",
        "graph_existing_neighbor_edges_pruned",
        "graph_existing_neighbor_undo_edges_recorded",
    ]
    for col in wanted:
        if col in inserts and len(inserts):
            out[f"{col}_mean"] = float(inserts[col].mean())
        else:
            out[f"{col}_mean"] = 0.0
    if "start_raw_ns" in inserts and "finish_raw_ns" in inserts and len(inserts):
        out["insert_active_span_ms"] = float((inserts["finish_raw_ns"].max() - inserts["start_raw_ns"].min()) / 1_000_000.0)
    else:
        out["insert_active_span_ms"] = 0.0
    return out


def summarize_tail(run_root: Path) -> dict[str, float]:
    path = run_root / "closed_loop_odin_tail.csv"
    out: dict[str, float] = {}
    if not path.exists():
        return out
    tail = pd.read_csv(path)
    if "lowq_search_count" in tail:
        tail = tail[tail["lowq_search_count"] > 0].copy()
    cols = [
        "background_search_active_workers",
        "insert_path_search_active_workers",
        "existing_neighbor_update_active_workers",
        "existing_neighbor_load_scan_active_workers",
        "existing_neighbor_prune_active_workers",
        "existing_neighbor_rewrite_active_workers",
        "existing_neighbor_undo_record_active_workers",
    ]
    for col in cols:
        out[f"{col}_mean"] = float(tail[col].mean()) if col in tail and len(tail) else 0.0
    return out


def summarize_case(run_root: Path, params: pd.Series) -> dict[str, object]:
    case_dir = find_case_dir(run_root)
    search = load_search(case_dir)
    windows, inserts = load_insert_windows(case_dir)
    search_active = active_search(search, windows)
    row: dict[str, object] = {
        "dataset": params["dataset"],
        "case_key": params["case_key"],
        "case_label": params["case_label"],
        "case_order": CASE_ORDER.get(str(params["case_key"]), 99),
        "competing_workers": int(params["competing_workers"]),
        "m": int(params["m"]),
        "efc": int(params["efc"]),
        "efs": int(params["efs"]),
        "ann_search_loops": int(params["ann_search_loops"]),
        "recall": read_recall(case_dir),
        "measured_searches_all": int(len(search)),
        "measured_searches_used": int(len(search_active)),
    }
    for metric in [
        "queue_ms",
        "op_ms",
        "search_index_ms",
        "search_post_index_ms",
        "work_searchknn_ms",
        "work_searchknn_thread_cpu_ms",
        "work_base_search_ms",
        "work_level0_edges_scanned",
        "work_distance_computations",
    ]:
        if metric not in search_active:
            continue
        row[f"{metric}_mean"] = float(search_active[metric].mean())
        row[f"{metric}_p95"] = percentile(search_active[metric], 95)
        row[f"{metric}_p99"] = percentile(search_active[metric], 99)
    row.update(summarize_insert(inserts))
    row.update(summarize_tail(run_root))
    return row


def make_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = [
        "op_ms_mean",
        "op_ms_p99",
        "search_index_ms_mean",
        "search_index_ms_p95",
        "search_index_ms_p99",
        "work_searchknn_ms_mean",
        "work_searchknn_ms_p99",
        "work_base_search_ms_mean",
        "work_base_search_ms_p99",
        "queue_ms_p99",
    ]
    for (dataset, cw), group in summary.groupby(["dataset", "competing_workers"]):
        base = group[group["case_key"] == "background_search"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, row in group.iterrows():
            if row["case_key"] == "background_search":
                continue
            out: dict[str, object] = {
                "dataset": dataset,
                "competing_workers": int(cw),
                "case_key": row["case_key"],
                "case_label": row["case_label"],
                "recall": row["recall"],
                "baseline_recall": base_row["recall"],
            }
            for metric in metrics:
                if metric in group:
                    out[f"baseline_{metric}"] = float(base_row[metric])
                    out[f"case_{metric}"] = float(row[metric])
                    out[f"delta_{metric}"] = float(row[metric] - base_row[metric])
                    out[f"ratio_{metric}"] = float(row[metric] / base_row[metric]) if float(base_row[metric]) else math.nan
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["dataset", "competing_workers", "case_key"])


def correlations(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "background_search_active_workers_mean",
        "insert_path_search_active_workers_mean",
        "existing_neighbor_update_active_workers_mean",
        "existing_neighbor_prune_active_workers_mean",
        "existing_neighbor_rewrite_active_workers_mean",
        "graph_insert_path_ms_mean",
        "graph_insert_path_edges_scanned_mean",
        "graph_insert_path_dist_comps_mean",
        "graph_existing_neighbor_update_loop_ms_mean",
        "graph_existing_neighbor_edges_written_mean",
    ]
    targets = ["search_index_ms_mean", "search_index_ms_p99", "work_searchknn_ms_mean"]
    rows: list[dict[str, object]] = []
    for dataset, group in summary.groupby("dataset"):
        frame = group.replace([np.inf, -np.inf], np.nan)
        for target in targets:
            if target not in frame:
                continue
            for metric in metrics:
                if metric not in frame:
                    continue
                part = frame[[target, metric]].dropna()
                if len(part) < 3 or part[target].nunique() < 2 or part[metric].nunique() < 2:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "metric": metric,
                        "n": int(len(part)),
                        "pearson": float(part[target].corr(part[metric], method="pearson")),
                        "spearman": float(part[target].corr(part[metric], method="spearman")),
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset", "target", "pearson"], ascending=[True, True, False])


def write_readme(out_root: Path, summary: pd.DataFrame, deltas: pd.DataFrame, corr: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# High-Recall Write-vs-Read Controls")
    lines.append("")
    lines.append("Question: under high-recall parameters, is measured-search latency tied to write-request work rather than pure read concurrency?")
    lines.append("")
    lines.append("## Case Means")
    lines.append("")
    cols = [
        "dataset",
        "competing_workers",
        "case_label",
        "recall",
        "op_ms_mean",
        "op_ms_p99",
        "search_index_ms_mean",
        "search_index_ms_p99",
        "work_searchknn_ms_mean",
        "work_searchknn_ms_p99",
        "work_base_search_ms_mean",
        "work_base_search_ms_p99",
        "queue_ms_p99",
    ]
    present = [c for c in cols if c in summary]
    lines.append(summary[present].to_markdown(index=False, floatfmt=".5f"))
    lines.append("")
    lines.append("## Delta Vs Background Search")
    lines.append("")
    dcols = [
        "dataset",
        "competing_workers",
        "case_label",
        "delta_op_ms_mean",
        "delta_op_ms_p99",
        "delta_search_index_ms_mean",
        "delta_search_index_ms_p99",
        "delta_work_searchknn_ms_mean",
        "delta_work_searchknn_ms_p99",
        "delta_work_base_search_ms_mean",
        "delta_work_base_search_ms_p99",
        "delta_queue_ms_p99",
    ]
    present = [c for c in dcols if c in deltas]
    lines.append(deltas[present].to_markdown(index=False, floatfmt=".5f") if present and len(deltas) else "No deltas.")
    lines.append("")
    lines.append("## Strongest Structural Correlations")
    lines.append("")
    if len(corr):
        show = corr[corr["target"] == "search_index_ms_mean"].copy()
        show["abs_pearson"] = show["pearson"].abs()
        show = show.sort_values(["dataset", "abs_pearson"], ascending=[True, False]).drop(columns=["abs_pearson"]).groupby("dataset").head(8)
        lines.append(show.to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No non-constant structural correlations.")
    lines.append("")
    lines.append("## Interpretation Rule")
    lines.append("")
    lines.append("A write-specific claim needs Real Insert to exceed Background Search and Insert Request ANN Search at the same co-running worker count. If Real Insert does not exceed those controls, the result should not be described as write-mutation-driven.")
    lines.append("")
    lines.append("Artifacts:")
    lines.append("")
    lines.append("- `tables/case_summary.csv`")
    lines.append("- `tables/delta_vs_background_search.csv`")
    lines.append("- `tables/structural_correlations.csv`")
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    args = parser.parse_args()

    params = pd.read_csv(args.params)
    rows: list[dict[str, object]] = []
    for _, param in params.iterrows():
        rows.append(summarize_case(Path(param["run_root"]), param))
    summary = pd.DataFrame(rows).sort_values(["dataset", "competing_workers", "case_order"])
    deltas = make_deltas(summary)
    corr = correlations(summary)

    tables = args.out_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "case_summary.csv", index=False)
    deltas.to_csv(tables / "delta_vs_background_search.csv", index=False)
    corr.to_csv(tables / "structural_correlations.csv", index=False)
    write_readme(args.out_root, summary, deltas, corr)


if __name__ == "__main__":
    main()
