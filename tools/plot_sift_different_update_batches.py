from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path("../results/sift/streamseed_hybrid")
OUT_PREFIX = "sift_different_update_batches"

SERIES = {
    "off": {
        "label": "HNSW",
        "glob": "test*-5w-updates/*_final_results.csv",
        "color": "#0072B2",
        "marker": "o",
    },
    "on": {
        "label": "HNSW-BriskSeed",
        "glob": "test*-5w-updates-on/*_final_results.csv",
        "color": "#D55E00",
        "marker": "s",
    },
}


def qps_tick_label(value: float, _pos=None) -> str:
    if abs(value) >= 1000:
        k_value = value / 1000.0
        if abs(k_value - round(k_value)) < 1e-9:
            return f"{int(round(k_value))}k"
        return f"{k_value:g}k"
    return f"{value:g}"


def test_id_from_path(path: Path) -> int:
    match = re.search(r"test(\d+)-5w-updates", path.as_posix())
    if not match:
        return 0
    return int(match.group(1))


def result_paths(pattern: str) -> list[Path]:
    return sorted(ROOT.glob(pattern), key=test_id_from_path)


def load_series(mode: str, spec: dict[str, str]) -> pd.DataFrame:
    paths = result_paths(spec["glob"])
    if not paths:
        raise RuntimeError(f"No final_results CSV found for {spec['label']} with glob {spec['glob']}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        required = {"batch_idx", "recall", "query_qps"}
        missing = required.difference(df.columns)
        if missing:
            raise RuntimeError(f"Missing columns {sorted(missing)} in {path}")
        part = df[["batch_idx", "recall", "query_qps"]].copy()
        part["test_id"] = test_id_from_path(path)
        part["mode"] = mode
        part["series"] = spec["label"]
        part["file"] = str(path)
        frames.append(part)

    return pd.concat(frames, ignore_index=True)


def load_mean_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.concat(
        [load_series(mode, spec) for mode, spec in SERIES.items()],
        ignore_index=True,
    )
    summary = (
        raw.groupby(["mode", "series", "batch_idx"], as_index=False)
        .agg(
            mean_query_qps=("query_qps", "mean"),
            mean_recall=("recall", "mean"),
            runs=("test_id", "nunique"),
        )
        .sort_values(["mode", "batch_idx"])
        .reset_index(drop=True)
    )
    return raw, summary


def build_overall_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["mode", "series"], as_index=False)
        .agg(
            overall_mean_query_qps=("mean_query_qps", "mean"),
            overall_mean_recall=("mean_recall", "mean"),
            batches=("batch_idx", "nunique"),
            runs=("runs", "max"),
        )
        .sort_values("mode")
        .reset_index(drop=True)
    )


def style_axis(ax) -> None:
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linestyle="-", linewidth=1.2, alpha=1.0)
    ax.tick_params(axis="both", labelsize=16)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharex=True)
    qps_ax, recall_ax = axes

    max_batch = int(summary["batch_idx"].max())
    xticks = sorted(summary["batch_idx"].unique())

    for mode, spec in SERIES.items():
        rows = summary[summary["mode"] == mode].sort_values("batch_idx")
        if rows.empty:
            continue
        common = {
            "color": spec["color"],
            "marker": spec["marker"],
            "markersize": 7.5,
            "markeredgecolor": "white",
            "markeredgewidth": 0.9,
            "linewidth": 2.2,
            "label": spec["label"],
        }
        qps_ax.plot(rows["batch_idx"], rows["mean_query_qps"], **common)
        recall_ax.plot(rows["batch_idx"], rows["mean_recall"], **common)

    for ax in axes:
        style_axis(ax)
        ax.set_xlabel("Batch ID", fontsize=16)
        ax.set_xticks(xticks)
        ax.set_xlim(-0.5, max_batch + 0.5)

    qps_ax.yaxis.set_major_formatter(FuncFormatter(qps_tick_label))
    qps_ax.set_title("Query QPS", fontsize=17, fontweight="semibold")
    recall_ax.set_title("Recall", fontsize=17, fontweight="semibold")
    qps_ax.set_ylabel("Query QPS", fontsize=16)
    recall_ax.set_ylabel("Recall", fontsize=16)

    handles, labels = qps_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        frameon=True,
        facecolor="#f0f0f0",
        edgecolor="#bdbdbd",
        framealpha=1.0,
        fontsize=17,
        handlelength=2.2,
        columnspacing=1.4,
    )

    plt.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.82, wspace=0.22)
    plt.savefig(f"{OUT_PREFIX}.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {OUT_PREFIX}.png and {OUT_PREFIX}.pdf")


def print_summary(summary: pd.DataFrame) -> None:
    print("\nMean Query QPS by Batch ID:")
    for mode, spec in SERIES.items():
        rows = summary[summary["mode"] == mode].sort_values("batch_idx")
        values = [f"batch{int(r.batch_idx)}={r.mean_query_qps:.2f}" for r in rows.itertuples(index=False)]
        print(f"  {spec['label']}: " + ", ".join(values))

    print("\nMean Recall by Batch ID:")
    for mode, spec in SERIES.items():
        rows = summary[summary["mode"] == mode].sort_values("batch_idx")
        values = [f"batch{int(r.batch_idx)}={r.mean_recall:.5f}" for r in rows.itertuples(index=False)]
        print(f"  {spec['label']}: " + ", ".join(values))


def print_overall_summary(overall: pd.DataFrame) -> None:
    print("\nOverall Mean by Line:")
    for mode, spec in SERIES.items():
        rows = overall[overall["mode"] == mode]
        if rows.empty:
            continue
        row = rows.iloc[0]
        print(
            f"  {spec['label']}: "
            f"mean_query_qps={row.overall_mean_query_qps:.2f}, "
            f"mean_recall={row.overall_mean_recall:.5f}"
        )


def main() -> None:
    raw, summary = load_mean_results()
    overall = build_overall_summary(summary)
    raw.to_csv(f"{OUT_PREFIX}_raw.csv", index=False)
    summary.to_csv(f"{OUT_PREFIX}_summary.csv", index=False)
    overall.to_csv(f"{OUT_PREFIX}_overall_summary.csv", index=False)
    plot(summary)
    print(f"Saved {OUT_PREFIX}_raw.csv")
    print(f"Saved {OUT_PREFIX}_summary.csv")
    print(f"Saved {OUT_PREFIX}_overall_summary.csv")
    print_summary(summary)
    print_overall_summary(overall)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
