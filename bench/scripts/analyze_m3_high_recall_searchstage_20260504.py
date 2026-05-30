#!/usr/bin/env python3
"""Summarize high-recall real-insert SearchKnn stage evidence."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORK_COLS = [
    "work_searchknn_ms",
    "work_searchknn_thread_cpu_ms",
    "work_base_search_ms",
    "work_upper_search_ms",
    "work_result_materialize_ms",
    "work_distance_computations",
    "work_level0_distance_computations",
    "work_level0_edges_scanned",
    "work_level0_expansions",
    "work_candidate_pops",
    "work_candidate_pushes",
    "work_visited_nodes",
]


def percentile(series: pd.Series, q: float) -> float:
    if len(series) == 0:
        return math.nan
    return float(series.quantile(q / 100.0))


def load_recall(run_dir: Path) -> float:
    path = run_dir / "benchmark_results.csv"
    if not path.exists():
        return math.nan
    results = pd.read_csv(path)
    if "recall" in results.columns and len(results):
        return float(results["recall"].iloc[0])
    return math.nan


def load_meta(run_dir: Path) -> dict[str, float]:
    meta = pd.read_csv(run_dir / "raw_latency" / "meta.csv")
    return {str(row.key): float(row.value) for row in meta.itertuples()}


def load_boundaries(case_dir: Path) -> pd.DataFrame:
    bounds = pd.read_csv(case_dir / "closed_loop_odin_tail_insert_boundaries.csv")
    return bounds[["active_start_ms", "active_end_ms"]].copy()


def mark_active(search: pd.DataFrame, bounds: pd.DataFrame) -> pd.Series:
    active = pd.Series(False, index=search.index)
    for row in bounds.itertuples():
        active |= (search["mid_rel_ms"] >= row.active_start_ms) & (
            search["mid_rel_ms"] < row.active_end_ms
        )
    return active


def find_run_dir(case_dir: Path, efc: int) -> Path:
    exact = case_dir / f"closed_loop_odin_efc{efc}"
    if exact.exists():
        return exact
    matches = sorted(case_dir.glob("closed_loop_odin_efc*"))
    if not matches:
        raise FileNotFoundError(f"no closed_loop_odin_efc* under {case_dir}")
    return matches[0]


def load_search(case_dir: Path, run_dir: Path, dataset: str, bin_ms: int) -> pd.DataFrame:
    header = pd.read_csv(run_dir / "raw_latency" / "search_wide.csv", nrows=0)
    present_work = [c for c in WORK_COLS if c in header.columns]
    usecols = [
        "measured",
        "create_raw_ns",
        "start_raw_ns",
        "finish_raw_ns",
        "op_ms",
        "e2e_ms",
        "search_index_ms",
        "search_post_index_ms",
        *present_work,
    ]
    search = pd.read_csv(run_dir / "raw_latency" / "search_wide.csv", usecols=usecols)
    search = search[search["measured"] == 1].copy()
    meta = load_meta(run_dir)
    origin = meta["bench_start_raw_ns"]
    search["mid_rel_ms"] = (
        ((search["start_raw_ns"] + search["finish_raw_ns"]) / 2.0) - origin
    ) / 1e6
    search["queue_ms"] = (search["start_raw_ns"] - search["create_raw_ns"]) / 1e6
    search["bin_ms"] = (np.floor(search["mid_rel_ms"] / bin_ms) * bin_ms).astype(int)
    search["insert_active"] = mark_active(search, load_boundaries(case_dir))
    search["dataset"] = dataset
    if "work_searchknn_ms" in search and "work_searchknn_thread_cpu_ms" in search:
        search["work_searchknn_wall_minus_cpu_ms"] = (
            search["work_searchknn_ms"] - search["work_searchknn_thread_cpu_ms"]
        )
    if "work_base_search_ms" in search and "work_level0_edges_scanned" in search:
        denom = search["work_level0_edges_scanned"].replace(0, np.nan)
        search["work_base_us_per_l0_edge"] = search["work_base_search_ms"] * 1000.0 / denom
    if "work_base_search_ms" in search and "work_level0_distance_computations" in search:
        denom = search["work_level0_distance_computations"].replace(0, np.nan)
        search["work_base_us_per_l0_distance"] = search["work_base_search_ms"] * 1000.0 / denom
    return search


def summarize_phase(
    search: pd.DataFrame,
    dataset: str,
    m: int,
    efc: int,
    efs: int,
    recall_note: float,
    recall_actual: float,
) -> pd.DataFrame:
    metrics = [
        "queue_ms",
        "search_index_ms",
        "search_post_index_ms",
        *[c for c in WORK_COLS if c in search.columns],
        "work_searchknn_wall_minus_cpu_ms",
        "work_base_us_per_l0_edge",
        "work_base_us_per_l0_distance",
    ]
    rows = []
    for phase, part in [
        ("Insert Active", search[search["insert_active"]]),
        ("Outside Insert", search[~search["insert_active"]]),
    ]:
        row: dict[str, float | int | str] = {
            "dataset": dataset,
            "m": m,
            "efc": efc,
            "efs": efs,
            "recall_note": recall_note,
            "recall_actual": recall_actual,
            "phase": phase,
            "measured_searches": int(len(part)),
        }
        for metric in metrics:
            if metric not in part:
                continue
            row[f"{metric}_mean"] = float(part[metric].mean()) if len(part) else math.nan
            row[f"{metric}_p95"] = percentile(part[metric], 95)
            row[f"{metric}_p99"] = percentile(part[metric], 99)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_bins(search: pd.DataFrame, dataset: str) -> pd.DataFrame:
    metrics = [
        "queue_ms",
        "search_index_ms",
        "search_post_index_ms",
        *[c for c in WORK_COLS if c in search.columns],
        "work_searchknn_wall_minus_cpu_ms",
        "work_base_us_per_l0_edge",
        "work_base_us_per_l0_distance",
    ]
    rows = []
    for (bin_ms, active), part in search.groupby(["bin_ms", "insert_active"]):
        row: dict[str, float | int | str | bool] = {
            "dataset": dataset,
            "bin_ms": int(bin_ms),
            "insert_active": bool(active),
            "measured_searches": int(len(part)),
        }
        for metric in metrics:
            if metric not in part:
                continue
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_p95"] = percentile(part[metric], 95)
            row[f"{metric}_p99"] = percentile(part[metric], 99)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "bin_ms", "insert_active"])


def load_period_perf(run_dir: Path, dataset: str) -> pd.DataFrame:
    path = run_dir / "period_groups.csv"
    if not path.exists():
        return pd.DataFrame()
    periods = pd.read_csv(path)
    perf_cols = [c for c in periods.columns if c.startswith("perf_")]
    if not perf_cols:
        return pd.DataFrame()
    out = periods[["bin_start_ms", *perf_cols]].copy()
    out["dataset"] = dataset
    out["bin_ms"] = out["bin_start_ms"].round().astype(int)
    return out.drop(columns=["bin_start_ms"])


def add_derived_hardware(bins: pd.DataFrame) -> pd.DataFrame:
    out = bins.copy()

    def div(name: str, num: str, den: str) -> None:
        if num not in out.columns or den not in out.columns:
            return
        denom = out[den].replace(0, np.nan)
        out[name] = out[num] / denom

    for col in [c for c in out.columns if c.startswith("perf_") and not c.endswith("_per_s")]:
        out[f"{col}_per_measured_search"] = out[col] / out["measured_searches"].replace(0, np.nan)

    div("cycles_per_instruction", "perf_cycles", "perf_instructions")
    div("cache_misses_per_instruction", "perf_cache_misses", "perf_instructions")
    div("llc_load_misses_per_instruction", "perf_llc_load_misses", "perf_instructions")
    div("dtlb_load_misses_per_instruction", "perf_dtlb_load_misses", "perf_instructions")
    div("l3_misses_per_instruction", "perf_mem_load_retired_l3_miss", "perf_instructions")
    div("remote_hitm_per_instruction", "perf_mem_load_l3_miss_retired_remote_hitm", "perf_instructions")
    div("remote_hitm_per_l3_miss", "perf_mem_load_l3_miss_retired_remote_hitm", "perf_mem_load_retired_l3_miss")
    div("l2_rfo_miss_ratio", "perf_l2_rqsts_rfo_miss", "perf_l2_rqsts_all_rfo")
    div("offcore_l3_miss_data_rd_per_instruction", "perf_offcore_requests_l3_miss_demand_data_rd", "perf_instructions")
    return out


def make_deltas(phase: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in phase.groupby("dataset"):
        active = group[group["phase"] == "Insert Active"].iloc[0]
        outside = group[group["phase"] == "Outside Insert"].iloc[0]
        row: dict[str, float | int | str] = {
            "dataset": dataset,
            "m": int(active["m"]),
            "efc": int(active["efc"]),
            "efs": int(active["efs"]),
            "recall_note": float(active["recall_note"]),
            "recall_actual": float(active["recall_actual"]),
            "insert_active_searches": int(active["measured_searches"]),
            "outside_insert_searches": int(outside["measured_searches"]),
        }
        for col in phase.columns:
            if not col.endswith("_mean") and not col.endswith("_p95") and not col.endswith("_p99"):
                continue
            a = float(active[col])
            b = float(outside[col])
            row[f"insert_active_{col}"] = a
            row[f"outside_insert_{col}"] = b
            row[f"delta_{col}"] = a - b
            row[f"ratio_{col}"] = a / b if b and not math.isnan(b) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def correlations(bins: pd.DataFrame) -> pd.DataFrame:
    targets = ["search_index_ms_mean", "search_index_ms_p95", "search_index_ms_p99"]
    metrics = [
        "work_searchknn_thread_cpu_ms_mean",
        "work_searchknn_ms_mean",
        "work_base_search_ms_mean",
        "work_level0_edges_scanned_mean",
        "work_level0_distance_computations_mean",
        "work_distance_computations_mean",
        "work_base_us_per_l0_edge_mean",
        "work_base_us_per_l0_distance_mean",
        "queue_ms_p99",
    ]
    rows = []
    active = bins[bins["insert_active"] & (bins["measured_searches"] > 0)].copy()
    for dataset, group in active.groupby("dataset"):
        for target in targets:
            for metric in metrics:
                if target not in group or metric not in group:
                    continue
                pair = group[[target, metric]].dropna()
                if len(pair) < 3:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "scope": "Insert Active 50ms Bins",
                        "target": target,
                        "metric": metric,
                        "n": int(len(pair)),
                        "pearson": float(pair[target].corr(pair[metric], method="pearson")),
                        "spearman": float(pair[target].corr(pair[metric], method="spearman")),
                    }
                )
    return pd.DataFrame(rows)


def hardware_correlations(bins: pd.DataFrame) -> pd.DataFrame:
    targets = ["search_index_ms_mean", "search_index_ms_p95", "search_index_ms_p99"]
    prefixes = (
        "perf_",
        "cycles_per_instruction",
        "cache_misses_per_instruction",
        "llc_load_misses_per_instruction",
        "dtlb_load_misses_per_instruction",
        "l3_misses_per_instruction",
        "remote_hitm_per_instruction",
        "remote_hitm_per_l3_miss",
        "l2_rfo_miss_ratio",
        "offcore_l3_miss_data_rd_per_instruction",
    )
    metrics = [
        c
        for c in bins.columns
        if c.startswith(prefixes)
        and not c.endswith("_mean")
        and not c.endswith("_p95")
        and not c.endswith("_p99")
    ]
    rows = []
    active = bins[bins["insert_active"] & (bins["measured_searches"] > 0)].copy()
    for dataset, group in active.groupby("dataset"):
        for target in targets:
            for metric in metrics:
                if target not in group or metric not in group:
                    continue
                pair = group[[target, metric]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(pair) < 3 or pair[target].nunique() < 2 or pair[metric].nunique() < 2:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "scope": "Insert Active 50ms Bins",
                        "target": target,
                        "metric": metric,
                        "n": int(len(pair)),
                        "pearson": float(pair[target].corr(pair[metric], method="pearson")),
                        "spearman": float(pair[target].corr(pair[metric], method="spearman")),
                    }
                )
    return pd.DataFrame(rows)


def plot_odin(bins: pd.DataFrame, bounds_by_dataset: dict[str, pd.DataFrame], out: Path) -> None:
    datasets = list(dict.fromkeys(bins["dataset"].tolist()))
    fig, axes = plt.subplots(len(datasets), 1, figsize=(11, max(3.2, 2.8 * len(datasets))), sharex=True)
    if len(datasets) == 1:
        axes = [axes]
    colors = {
        "Search P95": "#4f7cac",
        "SearchKnn CPU": "#6aa58f",
        "Level-0 Base": "#d08c60",
    }
    handles = None
    labels = None
    for ax, dataset in zip(axes, datasets):
        part = bins[bins["dataset"] == dataset].copy()
        merged = (
            part.groupby("bin_ms")
            .agg(
                measured_searches=("measured_searches", "sum"),
                search_p95=("search_index_ms_p95", "mean"),
                searchknn_cpu=("work_searchknn_thread_cpu_ms_mean", "mean"),
                base_search=("work_base_search_ms_mean", "mean"),
            )
            .reset_index()
            .set_index("bin_ms")
        )
        full_index = np.arange(int(part["bin_ms"].min()), int(part["bin_ms"].max()) + 50, 50)
        merged = merged.reindex(full_index)
        merged.index.name = "bin_ms"
        x = merged.index.to_numpy()
        for row in bounds_by_dataset[dataset].itertuples():
            ax.axvspan(row.active_start_ms, row.active_end_ms, color="#f4b6b0", alpha=0.35, lw=0)
        ax.plot(x, merged["search_p95"], color=colors["Search P95"], lw=1.8, label="Search P95")
        ax.plot(x, merged["searchknn_cpu"], color=colors["SearchKnn CPU"], lw=1.5, label="SearchKnn CPU")
        ax.plot(x, merged["base_search"], color=colors["Level-0 Base"], lw=1.5, label="Level-0 Base")
        display_dataset = {"sift": "SIFT", "deep1M": "Deep1M", "gist": "GIST"}.get(dataset, dataset)
        ax.set_ylabel(f"{display_dataset}\nLatency (ms)")
        ax.grid(axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    if handles is not None:
        fig.legend(handles, labels, loc="upper center", frameon=False, fontsize=9, ncol=3, bbox_to_anchor=(0.55, 1.01))
    axes[-1].set_xlabel("Elapsed Time (ms)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_readme(out_root: Path, deltas: pd.DataFrame, corr: pd.DataFrame, hw_corr: pd.DataFrame) -> None:
    any_positive = bool((deltas["delta_search_index_ms_mean"] > 0).any())
    lines = [
        "# High-Recall Real-Insert Search Stage Evidence",
        "",
        "This run uses the high-recall ANNS parameters from the existing experiment scripts.",
        "The comparison is within the same fixed-worker run: red insert-active windows versus the surrounding background-search windows.",
        "",
        "## Conclusion",
        "",
        (
            "These runs support an insert-active latency increase."
            if any_positive
            else "These runs do not support an insert-active latency increase; measured search is lower inside the real-insert windows than outside them."
        ),
        "When measured-search latency moves within insert-active windows, it moves with SearchKnn thread CPU and level-0 base search, not with queue time.",
        "",
        "## Main Table",
        "",
        "| Dataset | M | efc | efs | Recall | Delta Search Mean (ms) | Delta Search P95 (ms) | Delta SearchKnn CPU Mean (ms) | Delta Level-0 Base Mean (ms) | Delta Queue P99 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in deltas.itertuples(index=False):
        lines.append(
            "| {dataset} | {m} | {efc} | {efs} | {recall:.3f} | {d_mean:.4f} | {d_p95:.4f} | {d_cpu:.4f} | {d_base:.4f} | {d_queue:.4f} |".format(
                dataset=row.dataset,
                m=row.m,
                efc=row.efc,
                efs=row.efs,
                recall=row.recall_actual,
                d_mean=getattr(row, "delta_search_index_ms_mean", math.nan),
                d_p95=getattr(row, "delta_search_index_ms_p95", math.nan),
                d_cpu=getattr(row, "delta_work_searchknn_thread_cpu_ms_mean", math.nan),
                d_base=getattr(row, "delta_work_base_search_ms_mean", math.nan),
                d_queue=getattr(row, "delta_queue_ms_p99", math.nan),
            )
        )
    lines.extend(["", "## Strongest Insert-Active Bin Correlations", ""])
    if len(corr):
        top = corr.sort_values("pearson", ascending=False).head(12)
        lines.extend(
            [
                "| Dataset | Target | Metric | Pearson | Spearman | N |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in top.itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.target} | {row.metric} | {row.pearson:.3f} | {row.spearman:.3f} | {row.n} |"
            )
    else:
        lines.append("No correlation table was produced.")
    lines.extend(["", "## Hardware Correlations", ""])
    if len(hw_corr):
        preferred = hw_corr[hw_corr["target"] == "search_index_ms_mean"].copy()
        top = preferred.sort_values("pearson", ascending=False).head(12)
        lines.extend(
            [
                "| Dataset | Metric | Pearson | Spearman | N |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in top.itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.metric} | {row.pearson:.3f} | {row.spearman:.3f} | {row.n} |"
            )
    else:
        lines.append("No hardware counter table was produced.")
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `tables/phase_summary.csv`",
            "- `tables/phase_delta.csv`",
            "- `tables/bin_metrics.csv`",
            "- `tables/insert_active_correlations.csv`",
            "- `tables/hardware_correlations.csv`",
            "- `figures/high_recall_odin_search_stage.png`",
        ]
    )
    (out_root / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--bin-ms", type=int, default=50)
    args = parser.parse_args()

    tables = args.out_root / "tables"
    figures = args.out_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    params = pd.read_csv(args.params)
    phase_parts = []
    bin_parts = []
    bounds_by_dataset: dict[str, pd.DataFrame] = {}

    for row in params.itertuples(index=False):
        dataset = str(row.dataset)
        case_dir = Path(row.run_root)
        run_dir = find_run_dir(case_dir, int(row.efc))
        search = load_search(case_dir, run_dir, dataset, args.bin_ms)
        recall = load_recall(run_dir)
        phase_parts.append(
            summarize_phase(
                search,
                dataset,
                int(row.m),
                int(row.efc),
                int(row.efs),
                float(row.recall_note),
                recall,
            )
        )
        run_bins = summarize_bins(search, dataset)
        perf_bins = load_period_perf(run_dir, dataset)
        if len(perf_bins):
            run_bins = run_bins.merge(perf_bins, on=["dataset", "bin_ms"], how="left")
        bin_parts.append(run_bins)
        bounds_by_dataset[dataset] = load_boundaries(case_dir)

    phase = pd.concat(phase_parts, ignore_index=True)
    bins = add_derived_hardware(pd.concat(bin_parts, ignore_index=True))
    deltas = make_deltas(phase)
    corr = correlations(bins)
    hw_corr = hardware_correlations(bins)

    phase.to_csv(tables / "phase_summary.csv", index=False)
    bins.to_csv(tables / "bin_metrics.csv", index=False)
    deltas.to_csv(tables / "phase_delta.csv", index=False)
    corr.to_csv(tables / "insert_active_correlations.csv", index=False)
    hw_corr.to_csv(tables / "hardware_correlations.csv", index=False)

    plot_odin(bins, bounds_by_dataset, figures / "high_recall_odin_search_stage.png")
    write_readme(args.out_root, deltas, corr, hw_corr)
    print(f"wrote {args.out_root}")


if __name__ == "__main__":
    main()
