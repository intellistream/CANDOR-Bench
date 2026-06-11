from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path("../results/sift10M/streamseed_hybrid")
OUT_PREFIX = "sift10M_streamseed_hybrid_qps_recall"

SERIES = {
    "briskseed": {
        "label": "HNSW-BriskSeed",
        "runs": [
            "test1/test1_final_results.csv",
            "test2/test2_final_results.csv",
        ],
        "color": "#D55E00",
    },
    "hnsw": {
        "label": "HNSW",
        "runs": [
            "test1-off/test1-off_final_results.csv",
            "test2-off/test2-off_final_results.csv",
        ],
        "color": "#0072B2",
    },
}


def qps_tick_label(value: float, _pos=None) -> str:
    if abs(value) >= 1000:
        k_value = value / 1000.0
        if abs(k_value - round(k_value)) < 1e-9:
            return f"{int(round(k_value))}k"
        return f"{k_value:g}k"
    return f"{value:g}"


def load_run(path: Path, mode: str, label: str, run_idx: int) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Final results CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"batch_idx", "recall", "query_qps"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns {sorted(missing)} in {path}")

    part = df[["batch_idx", "recall", "query_qps"]].copy()
    part["mode"] = mode
    part["series"] = label
    part["run_idx"] = run_idx
    part["file"] = str(path)
    return part


def load_series(mode: str, spec: dict[str, object]) -> pd.DataFrame:
    label = str(spec["label"])
    runs = spec.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise RuntimeError(f"No runs configured for {label}")

    raw = pd.concat(
        [load_run(ROOT / str(run), mode, label, idx + 1) for idx, run in enumerate(runs)],
        ignore_index=True,
    )
    averaged = (
        raw.groupby(["mode", "series", "batch_idx"], as_index=False)
        .agg(
            recall=("recall", "mean"),
            query_qps=("query_qps", "mean"),
            runs=("run_idx", "nunique"),
        )
        .sort_values("batch_idx")
        .reset_index(drop=True)
    )
    averaged["file"] = ";".join(str(ROOT / str(run)) for run in runs)
    return averaged


def load_results() -> pd.DataFrame:
    data = pd.concat(
        [load_series(mode, spec) for mode, spec in SERIES.items()],
        ignore_index=True,
    )
    return data.sort_values(["mode", "batch_idx"]).reset_index(drop=True)


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["mode", "series"], as_index=False)
        .agg(
            mean_query_qps=("query_qps", "mean"),
            mean_recall=("recall", "mean"),
            batches=("batch_idx", "nunique"),
        )
        .sort_values("mode")
        .reset_index(drop=True)
    )


def style_axis(ax) -> None:
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linestyle="-", linewidth=1.2, alpha=1.0)
    ax.tick_params(axis="both", labelsize=16)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def plot(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharex=True)
    qps_ax, recall_ax = axes

    max_batch = int(data["batch_idx"].max())
    xticks = list(range(0, max_batch + 1, 10))
    if max_batch not in xticks:
        xticks.append(max_batch)

    for mode, spec in SERIES.items():
        rows = data[data["mode"] == mode].sort_values("batch_idx")
        if rows.empty:
            continue

        common = {
            "color": spec["color"],
            "linewidth": 2.6,
            "label": spec["label"],
        }
        qps_ax.plot(rows["batch_idx"], rows["query_qps"], **common)
        recall_ax.plot(rows["batch_idx"], rows["recall"], **common)

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
    handles, labels = handles[::-1], labels[::-1]
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
    print("\nOverall Mean by Line:")
    for mode, spec in SERIES.items():
        rows = summary[summary["mode"] == mode]
        if rows.empty:
            continue
        row = rows.iloc[0]
        print(
            f"  {spec['label']}: "
            f"mean_query_qps={row.mean_query_qps:.2f}, "
            f"mean_recall={row.mean_recall:.5f}"
        )


def main() -> None:
    data = load_results()
    summary = build_summary(data)
    data.to_csv(f"{OUT_PREFIX}_averaged_raw.csv", index=False)
    summary.to_csv(f"{OUT_PREFIX}_summary.csv", index=False)
    plot(data)
    print(f"Saved {OUT_PREFIX}_averaged_raw.csv")
    print(f"Saved {OUT_PREFIX}_summary.csv")
    print_summary(summary)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
