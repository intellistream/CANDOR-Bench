from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


THREADS = [1, 2, 4, 8, 16]
ALGORITHMS = [
    {
        "key": "streamseed_hybrid",
        "title": "HNSW",
        "root": Path("../results/sift/streamseed_hybrid"),
        "csv_glob": "*batch_query_qps.csv",
    },
    {
        "key": "streamseed_hybrid_freshdiskann",
        "title": "FreshDiskANN",
        "root": Path("../results/sift/streamseed_hybrid_freshdiskann"),
        "csv_glob": "*batch_query_qps.csv",
    },
]
OUT_PREFIX = "sift_different_threads_qps"

STYLE_BY_ALGO_MODE = {
    ("streamseed_hybrid", "off"): {
        "label": "HNSW",
        "color": "#0072B2",
        "marker": "o",
    },
    ("streamseed_hybrid", "on"): {
        "label": "HNSW-BriskSeed",
        "color": "#D55E00",
        "marker": "s",
    },
    ("streamseed_hybrid_freshdiskann", "off"): {
        "label": "FreshDiskANN",
        "color": "#CC79A7",
        "marker": "o",
    },
    ("streamseed_hybrid_freshdiskann", "on"): {
        "label": "FreshDiskANN-BriskSeed",
        "color": "#E69F00",
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


def qps_csv_path(exp_dir: Path, csv_glob: str) -> Path | None:
    matches = sorted(exp_dir.glob(csv_glob))
    if not matches:
        return None
    return matches[0]


def load_mean_qps() -> pd.DataFrame:
    rows = []
    for algo in ALGORITHMS:
        root = algo["root"]
        for thread in THREADS:
            for mode in ["off", "on"]:
                exp_dir = root / f"thread{thread}-{mode}"
                if not exp_dir.exists():
                    print(f"Warning: missing result directory: {exp_dir}")
                    continue
                path = qps_csv_path(exp_dir, algo["csv_glob"])
                if path is None:
                    print(f"Warning: no batch_query_qps CSV found under {exp_dir}")
                    continue
                df = pd.read_csv(path)
                if "query_qps" not in df.columns:
                    print(f"Warning: missing query_qps column in {path}")
                    continue
                qps = pd.to_numeric(df["query_qps"], errors="coerce")
                rows.append(
                    {
                        "algorithm": algo["key"],
                        "title": algo["title"],
                        "mode": mode,
                        "thread": thread,
                        "mean_query_qps": qps.mean(),
                        "file": str(path),
                    }
                )
    if not rows:
        raise RuntimeError("No valid thread QPS results found")
    return pd.DataFrame(rows)


def style_axis(ax) -> None:
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linestyle="-", linewidth=1.2, alpha=1.0)
    ax.tick_params(axis="both", labelsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(qps_tick_label))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xticks(THREADS)
    ax.set_xlim(0, max(THREADS) + 1)


def plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharex=True)

    for ax, algo in zip(axes, ALGORITHMS):
        algo_df = summary[summary["algorithm"] == algo["key"]]
        for mode in ["off", "on"]:
            mode_df = algo_df[algo_df["mode"] == mode].sort_values("thread")
            if mode_df.empty:
                continue
            series_style = STYLE_BY_ALGO_MODE[(algo["key"], mode)]
            ax.plot(
                mode_df["thread"],
                mode_df["mean_query_qps"],
                color=series_style["color"],
                marker=series_style["marker"],
                markersize=7.5,
                markeredgecolor="white",
                markeredgewidth=0.9,
                linewidth=2.2,
                label=series_style["label"],
            )
        style_axis(ax)
        ax.set_title(algo["title"], fontsize=17, fontweight="semibold")
        ax.set_xlabel("Threads", fontsize=16)

    axes[0].set_ylabel("Query QPS", fontsize=16)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    # Keep legend order compact: off/on for each subplot.
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=4,
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


def print_qps_summary(summary: pd.DataFrame) -> None:
    print("\nMean Query QPS by thread:")
    for algo in ALGORITHMS:
        for mode in ["off", "on"]:
            style = STYLE_BY_ALGO_MODE[(algo["key"], mode)]
            rows = summary[(summary["algorithm"] == algo["key"]) & (summary["mode"] == mode)].sort_values("thread")
            if rows.empty:
                continue
            values = [f"thread{int(row.thread)}={row.mean_query_qps:.2f}" for row in rows.itertuples(index=False)]
            print(f"  {style['label']}: " + ", ".join(values))


def main() -> None:
    summary = load_mean_qps().sort_values(["algorithm", "mode", "thread"]).reset_index(drop=True)
    summary.to_csv(f"{OUT_PREFIX}_summary.csv", index=False)
    plot(summary)
    print(f"Saved {OUT_PREFIX}_summary.csv")
    print_qps_summary(summary)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
