#!/usr/bin/env python3
"""Analyze efc=200 direct search-work counter runs."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CASE_ORDER = [
    "neighbor_writes_on",
    "neighbor_writes_off",
    "neighbor_read_scan_no_writes",
    "undo_record_off",
]

CASE_LABEL = {
    "neighbor_writes_on": "Neighbor Writes On",
    "neighbor_writes_off": "Neighbor Writes Off",
    "neighbor_read_scan_no_writes": "Neighbor Read Scan, No Writes",
    "undo_record_off": "Undo Record Off",
}

SEMANTICS = {
    "neighbor_writes_on": "production insert behavior for this benchmark path",
    "neighbor_writes_off": "existing-neighbor update loop bypassed; graph insert semantics and recall change",
    "neighbor_read_scan_no_writes": "scan/prune computation remains but append/rewrite/undo writes are suppressed; graph insert semantics and recall change",
    "undo_record_off": "neighbor append/rewrite remain; MVCC undo-record writes for pruned edges are suppressed",
}

WORK_METRICS = [
    "work_distance_computations",
    "work_upper_distance_computations",
    "work_level0_distance_computations",
    "work_upper_hops",
    "work_upper_edges_scanned",
    "work_level0_expansions",
    "work_level0_edges_scanned",
    "work_candidate_pops",
    "work_candidate_pushes",
    "work_visited_nodes",
    "work_result_pushes",
    "work_invisible_expansions",
    "work_invisible_edges",
    "work_invisible_candidate_dist_comps",
    "work_invisible_candidate_enqueues",
    "work_future_skip_hops",
    "work_searchknn_ms",
    "work_searchknn_thread_cpu_ms",
    "work_result_copy_ms",
    "work_entry_ms",
    "work_upper_search_ms",
    "work_base_search_ms",
    "work_result_materialize_ms",
    "work_snapshot_guard_ms",
    "work_visited_list_get_ms",
    "work_visited_list_release_ms",
    "work_upper_lock_wait_ms",
    "work_level0_lock_wait_ms",
]

CORE_WORK = [
    "work_distance_computations",
    "work_level0_distance_computations",
    "work_level0_expansions",
    "work_level0_edges_scanned",
    "work_candidate_pops",
    "work_candidate_pushes",
    "work_visited_nodes",
]

STAGE_WORK = [
    "work_entry_ms",
    "work_upper_search_ms",
    "work_base_search_ms",
    "work_result_materialize_ms",
    "work_snapshot_guard_ms",
    "work_visited_list_get_ms",
    "work_visited_list_release_ms",
    "work_upper_lock_wait_ms",
    "work_level0_lock_wait_ms",
]


@dataclass(frozen=True)
class CaseSpec:
    case_dir: Path
    cw: int
    case_key: str

    @property
    def run_dir(self) -> Path:
        return self.case_dir / "closed_loop_odin_efc200"

    @property
    def label(self) -> str:
        return CASE_LABEL[self.case_key]


def percentile(series: pd.Series, q: float) -> float:
    if len(series) == 0:
        return math.nan
    return float(series.quantile(q / 100.0))


def parse_case(path: Path) -> CaseSpec | None:
    name = path.name
    if not name.startswith("cw"):
        return None
    parts = name.split("_", 1)
    if len(parts) != 2:
        return None
    try:
        cw = int(parts[0][2:])
    except ValueError:
        return None
    key = parts[1]
    if key not in CASE_LABEL:
        return None
    return CaseSpec(path, cw, key)


def load_meta(run_dir: Path) -> dict[str, float]:
    meta = pd.read_csv(run_dir / "raw_latency" / "meta.csv")
    return {str(row.key): float(row.value) for row in meta.itertuples()}


def load_recall(run_dir: Path) -> float:
    results = pd.read_csv(run_dir / "benchmark_results.csv")
    if "recall" not in results:
        return math.nan
    return float(results["recall"].iloc[0])


def load_boundaries(case_dir: Path) -> pd.DataFrame:
    path = case_dir / "closed_loop_odin_tail_insert_boundaries.csv"
    bounds = pd.read_csv(path)
    return bounds[["active_start_ms", "active_end_ms"]].copy()


def mark_insert_active(search: pd.DataFrame, bounds: pd.DataFrame) -> pd.Series:
    active = pd.Series(False, index=search.index)
    for row in bounds.itertuples():
        active |= (search["mid_rel_ms"] >= row.active_start_ms) & (
            search["mid_rel_ms"] < row.active_end_ms
        )
    return active


def load_search(spec: CaseSpec, bin_ms: int) -> pd.DataFrame:
    header = pd.read_csv(spec.run_dir / "raw_latency" / "search_wide.csv", nrows=0)
    diag_cols = [c for c in header.columns if c.startswith("diag_")]
    usecols = [
        "measured",
        "finish_ms",
        "create_raw_ns",
        "start_raw_ns",
        "finish_raw_ns",
        "op_ms",
        "e2e_ms",
        "search_index_ms",
        "search_post_index_ms",
        *diag_cols,
        *[c for c in WORK_METRICS if c in header.columns],
    ]
    search = pd.read_csv(spec.run_dir / "raw_latency" / "search_wide.csv", usecols=usecols)
    search = search[search["measured"] == 1].copy()
    meta = load_meta(spec.run_dir)
    origin = meta["bench_start_raw_ns"]
    search["mid_rel_ms"] = (
        ((search["start_raw_ns"] + search["finish_raw_ns"]) / 2.0) - origin
    ) / 1e6
    search["queue_ms"] = (search["start_raw_ns"] - search["create_raw_ns"]) / 1e6
    search["queue_from_e2e_ms"] = search["e2e_ms"] - search["op_ms"]
    search["bin_ms"] = (np.floor(search["mid_rel_ms"] / bin_ms) * bin_ms).astype(int)
    search["insert_active"] = mark_insert_active(search, load_boundaries(spec.case_dir))
    search["cw"] = spec.cw
    search["case"] = spec.case_key
    search["case_label"] = spec.label
    search["run"] = spec.case_dir.name
    return search


def summarize_phase(spec: CaseSpec, search: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    diag_cols = [c for c in search.columns if c.startswith("diag_")]
    recall = load_recall(spec.run_dir)
    for phase_name, part in [
        ("all", search),
        ("insert_active", search[search["insert_active"]]),
        ("outside_insert", search[~search["insert_active"]]),
    ]:
        row: dict[str, float | int | str] = {
            "cw": spec.cw,
            "case": spec.case_key,
            "case_label": spec.label,
            "phase": phase_name,
            "measured_searches": int(len(part)),
            "recall": recall,
            "queue_p95_ms": percentile(part["queue_ms"], 95),
            "queue_p99_ms": percentile(part["queue_ms"], 99),
            "queue_max_ms": float(part["queue_ms"].max()) if len(part) else math.nan,
            "diag_sum": float(part[diag_cols].sum().sum()) if diag_cols else 0.0,
            "search_index_mean_ms": float(part["search_index_ms"].mean()) if len(part) else math.nan,
            "search_index_p95_ms": percentile(part["search_index_ms"], 95),
            "search_index_p99_ms": percentile(part["search_index_ms"], 99),
            "search_post_index_mean_ms": float(part["search_post_index_ms"].mean())
            if len(part)
            else math.nan,
        }
        for metric in WORK_METRICS:
            if metric in part:
                row[f"{metric}_mean"] = float(part[metric].mean()) if len(part) else math.nan
                row[f"{metric}_p99"] = percentile(part[metric], 99)
        rows.append(row)
    return rows


def bin_search(search: pd.DataFrame) -> pd.DataFrame:
    rows = []
    diag_cols = [c for c in search.columns if c.startswith("diag_")]
    group_cols = ["cw", "case", "case_label", "run", "bin_ms", "insert_active"]
    for keys, part in search.groupby(group_cols):
        cw, case, label, run, bin_ms, active = keys
        row: dict[str, float | int | str | bool] = {
            "cw": int(cw),
            "case": case,
            "case_label": label,
            "run": run,
            "bin_ms": int(bin_ms),
            "insert_active": bool(active),
            "measured_search_count": int(len(part)),
            "queue_p99_ms": percentile(part["queue_ms"], 99),
            "search_index_mean_ms": float(part["search_index_ms"].mean()),
            "search_index_p95_ms": percentile(part["search_index_ms"], 95),
            "search_index_p99_ms": percentile(part["search_index_ms"], 99),
            "diag_sum": float(part[diag_cols].sum().sum()) if diag_cols else 0.0,
        }
        for metric in WORK_METRICS:
            if metric in part:
                row[f"{metric}_mean"] = float(part[metric].mean())
                row[f"{metric}_p99"] = percentile(part[metric], 99)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cw", "case", "bin_ms", "insert_active"])


def correlations(df: pd.DataFrame, metrics: list[str], scope: str, target: str) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in df or len(df[[target, metric]].dropna()) < 3:
            continue
        pair = df[[target, metric]].dropna()
        rows.append(
            {
                "scope": scope,
                "target": target,
                "metric": metric,
                "n": int(len(pair)),
                "pearson": float(pair[target].corr(pair[metric], method="pearson")),
                "spearman": float(pair[target].corr(pair[metric], method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def case_deltas(active_means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cw, group in active_means.groupby("cw"):
        base = group[group["case"] == "neighbor_writes_on"].iloc[0]
        for row in group.itertuples(index=False):
            out = {
                "cw": int(cw),
                "case": row.case,
                "case_label": row.case_label,
                "recall": row.recall,
                "delta_recall": float(row.recall - base.recall),
                "search_index_mean_ms": row.search_index_mean_ms,
                "delta_search_index_mean_ms": float(
                    row.search_index_mean_ms - base.search_index_mean_ms
                ),
                "search_index_p99_ms": row.search_index_p99_ms,
                "delta_search_index_p99_ms": float(
                    row.search_index_p99_ms - base.search_index_p99_ms
                ),
            }
            for metric in CORE_WORK + ["work_searchknn_ms", "work_searchknn_thread_cpu_ms", "work_result_copy_ms"] + STAGE_WORK:
                col = f"{metric}_mean"
                if col in active_means.columns:
                    out[col] = getattr(row, col)
                    out[f"delta_{col}"] = float(getattr(row, col) - getattr(base, col))
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["cw", "case"])


def overhead_validation(out_root: Path, instrumented: pd.DataFrame) -> pd.DataFrame:
    off_dir = Path("bench/result/20260504_m3_efc200_searchwork_overhead/cw32_work_off")
    if not off_dir.exists():
        return pd.DataFrame()
    spec = CaseSpec(off_dir, 32, "neighbor_writes_on")
    off_search = load_search(spec, 50)
    off_phase = pd.DataFrame(summarize_phase(spec, off_search))
    off_active = off_phase[off_phase["phase"] == "insert_active"].iloc[0]
    on_active = instrumented[
        (instrumented["cw"] == 32)
        & (instrumented["case"] == "neighbor_writes_on")
        & (instrumented["phase"] == "insert_active")
    ]
    if len(on_active) == 0:
        return pd.DataFrame()
    on_active = on_active.iloc[0]
    rows = []
    for label, row in [("search_work_off", off_active), ("search_work_on", on_active)]:
        rows.append(
            {
                "case": label,
                "measured_searches": int(row.measured_searches),
                "recall": float(row.recall),
                "queue_p99_ms": float(row.queue_p99_ms),
                "diag_sum": float(row.diag_sum),
                "search_index_mean_ms": float(row.search_index_mean_ms),
                "search_index_p99_ms": float(row.search_index_p99_ms),
                "work_distance_computations_mean": float(
                    getattr(row, "work_distance_computations_mean", 0.0)
                ),
            }
        )
    base = rows[0]
    on = rows[1]
    rows.append(
        {
            "case": "on_minus_off",
            "measured_searches": on["measured_searches"] - base["measured_searches"],
            "recall": on["recall"] - base["recall"],
            "queue_p99_ms": on["queue_p99_ms"] - base["queue_p99_ms"],
            "diag_sum": on["diag_sum"] - base["diag_sum"],
            "search_index_mean_ms": on["search_index_mean_ms"] - base["search_index_mean_ms"],
            "search_index_p99_ms": on["search_index_p99_ms"] - base["search_index_p99_ms"],
            "work_distance_computations_mean": on["work_distance_computations_mean"]
            - base["work_distance_computations_mean"],
        }
    )
    return pd.DataFrame(rows)


def plot_case_means(active_means: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    xlabels = []
    xpos = np.arange(len(active_means))
    colors = active_means["cw"].map({32: "#2563eb", 40: "#dc2626"}).to_numpy()
    for row in active_means.itertuples():
        label = CASE_LABEL[row.case].replace(", ", ",\n")
        xlabels.append(f"cw{row.cw}\n{label}")
    axes[0].bar(xpos, active_means["search_index_mean_ms"], color=colors, alpha=0.78)
    axes[0].set_ylabel("search_index_ms mean")
    axes[0].set_title("Insert-active measured-search latency by intervention", loc="left")
    ax2 = axes[0].twinx()
    ax2.plot(xpos, active_means["recall"], color="#111827", marker="o", lw=1.4)
    ax2.set_ylabel("recall@10")
    axes[1].bar(
        xpos - 0.2,
        active_means["work_distance_computations_mean"],
        width=0.4,
        color="#0f766e",
        label="distance comps",
    )
    axes[1].bar(
        xpos + 0.2,
        active_means["work_level0_edges_scanned_mean"],
        width=0.4,
        color="#f97316",
        label="L0 edges scanned",
    )
    axes[1].set_ylabel("mean per measured query")
    axes[1].set_title("Direct search-work counters stay nearly flat", loc="left")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(xlabels, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_bin_scatter(bins: pd.DataFrame, out: Path) -> None:
    active = bins[(bins["insert_active"]) & (bins["measured_search_count"] > 0)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for cw, group in active.groupby("cw"):
        axes[0].scatter(
            group["work_distance_computations_mean"],
            group["search_index_mean_ms"],
            s=18,
            alpha=0.55,
            label=f"cw={cw}",
        )
        axes[1].scatter(
            group["work_level0_edges_scanned_mean"],
            group["search_index_mean_ms"],
            s=18,
            alpha=0.55,
            label=f"cw={cw}",
        )
    axes[0].set_xlabel("distance computations / query")
    axes[1].set_xlabel("L0 edges scanned / query")
    for ax in axes:
        ax.set_ylabel("search_index_ms mean")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    axes[0].set_title("Insert-active bins: distance comps vs latency", loc="left")
    axes[1].set_title("Insert-active bins: scanned edges vs latency", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("bench/result/20260504_m3_efc200_searchwork_b20_clean"))
    parser.add_argument("--out-root", type=Path, default=Path("paper-progress/m3/search-tail-rootcause-2026-05-01/efc200_searchwork_20260504"))
    parser.add_argument("--bin-ms", type=int, default=50)
    args = parser.parse_args()

    tables = args.out_root / "tables"
    figures = args.out_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    specs = [parse_case(p) for p in args.run_root.glob("cw*")]
    specs = [s for s in specs if s is not None]
    specs.sort(key=lambda s: (s.cw, CASE_ORDER.index(s.case_key)))
    if not specs:
        raise SystemExit(f"no cases under {args.run_root}")

    searches = []
    phase_rows = []
    for spec in specs:
        search = load_search(spec, args.bin_ms)
        searches.append(search)
        phase_rows.extend(summarize_phase(spec, search))

    search_all = pd.concat(searches, ignore_index=True)
    phase = pd.DataFrame(phase_rows).sort_values(["cw", "case", "phase"])
    active_means = phase[phase["phase"] == "insert_active"].copy()
    bins = bin_search(search_all)
    active_bins = bins[(bins["insert_active"]) & (bins["measured_search_count"] > 0)].copy()
    active_samples = search_all[search_all["insert_active"]].copy()

    phase.to_csv(tables / "phase_sample_summary.csv", index=False)
    active_means.to_csv(tables / "case_insert_active_means.csv", index=False)
    bins.to_csv(tables / "bin_metrics.csv", index=False)
    case_deltas(active_means).to_csv(tables / "case_delta_vs_neighbor_writes_on.csv", index=False)

    semantics = pd.DataFrame(
        [
            {"case": key, "case_label": CASE_LABEL[key], "semantics": SEMANTICS[key]}
            for key in CASE_ORDER
        ]
    )
    semantics.to_csv(tables / "intervention_semantics.csv", index=False)

    sample_corr = [correlations(active_samples, CORE_WORK, "all_insert_active_samples", "search_index_ms")]
    for (cw, case), group in active_samples.groupby(["cw", "case"]):
        sample_corr.append(
            correlations(group, CORE_WORK, f"cw{cw}_{case}_insert_active_samples", "search_index_ms")
        )
    pd.concat(sample_corr, ignore_index=True).to_csv(
        tables / "sample_correlations_insert_active.csv", index=False
    )

    bin_metrics = [f"{m}_mean" for m in CORE_WORK]
    bin_corr = [correlations(active_bins, bin_metrics, "all_insert_active_50ms_bins", "search_index_mean_ms")]
    for (cw, case), group in active_bins.groupby(["cw", "case"]):
        bin_corr.append(
            correlations(group, bin_metrics, f"cw{cw}_{case}_insert_active_50ms_bins", "search_index_mean_ms")
        )
    pd.concat(bin_corr, ignore_index=True).to_csv(
        tables / "bin_correlations_insert_active.csv", index=False
    )

    stage_sample_corr = [
        correlations(active_samples, STAGE_WORK, "all_insert_active_samples", "search_index_ms")
    ]
    for (cw, case), group in active_samples.groupby(["cw", "case"]):
        stage_sample_corr.append(
            correlations(group, STAGE_WORK, f"cw{cw}_{case}_insert_active_samples", "search_index_ms")
        )
    pd.concat(stage_sample_corr, ignore_index=True).to_csv(
        tables / "stage_sample_correlations_insert_active.csv", index=False
    )

    stage_bin_metrics = [f"{m}_mean" for m in STAGE_WORK]
    stage_bin_corr = [
        correlations(active_bins, stage_bin_metrics, "all_insert_active_50ms_bins", "search_index_mean_ms")
    ]
    for (cw, case), group in active_bins.groupby(["cw", "case"]):
        stage_bin_corr.append(
            correlations(group, stage_bin_metrics, f"cw{cw}_{case}_insert_active_50ms_bins", "search_index_mean_ms")
        )
    pd.concat(stage_bin_corr, ignore_index=True).to_csv(
        tables / "stage_bin_correlations_insert_active.csv", index=False
    )

    case_mean_metrics = [f"{m}_mean" for m in CORE_WORK + ["work_searchknn_ms", "work_searchknn_thread_cpu_ms", "work_result_copy_ms"] + STAGE_WORK]
    correlations(active_means, case_mean_metrics, "intervention_case_means", "search_index_mean_ms").to_csv(
        tables / "case_mean_correlations.csv", index=False
    )

    overhead = overhead_validation(args.out_root, phase)
    if len(overhead):
        overhead.to_csv(tables / "overhead_validation.csv", index=False)

    plot_case_means(active_means, figures / "efc200_searchwork_case_means.png")
    plot_bin_scatter(bins, figures / "efc200_searchwork_bin_scatter.png")

    print(f"wrote {args.out_root}")


if __name__ == "__main__":
    main()
