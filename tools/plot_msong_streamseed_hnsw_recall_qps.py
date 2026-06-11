from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


RESULT_SPECS = [
    {
        "series": "HNSW off",
        "family": "HNSW",
        "mode": "off",
        "root": Path("../results/msong/streamseed_hybrid_grid_off"),
        "pattern": re.compile(r"indexkey(?P<param>\d+)-ef(?P<ef>\d+)$"),
        "param_name": "k",
    },
    {
        "series": "HNSW on",
        "family": "HNSW",
        "mode": "on",
        "root": Path("../results/msong/streamseed_hybrid_grid_on"),
        "pattern": re.compile(r"indexkey(?P<param>\d+)-ef(?P<ef>\d+)-on$"),
        "param_name": "k",
    },
    {
        "series": "SymphonyQG off",
        "family": "SymphonyQG",
        "mode": "off",
        "root": Path("../results/msong/streamseed_hybrid_symphonyqg_grid_off"),
        "pattern": re.compile(r"ef-(?P<ef>\d+)-degree-(?P<param>\d+)$"),
        "param_name": "d",
    },
    {
        "series": "SymphonyQG on",
        "family": "SymphonyQG",
        "mode": "on",
        "root": Path("../results/msong/streamseed_hybrid_symphonyqg_grid_on"),
        "pattern": re.compile(r"ef-(?P<ef>\d+)-degree-(?P<param>\d+)$"),
        "param_name": "d",
    },
    {
        "series": "FreshDiskANN off",
        "family": "FreshDiskANN",
        "mode": "off",
        "root": Path("../results/msong/streamseed_hybrid_freshdiskann_grid_off"),
        "pattern": re.compile(r"R(?P<param>\d+)-ef(?P<ef>\d+)$"),
        "param_name": "R",
    },
    {
        "series": "FreshDiskANN on",
        "family": "FreshDiskANN",
        "mode": "on",
        "root": Path("../results/msong/streamseed_hybrid_freshdiskann_grid_on"),
        "pattern": re.compile(r"R(?P<param>\d+)-ef(?P<ef>\d+)$"),
        "param_name": "R",
    },
]
OUT_PREFIX = "msong_streamseed_hnsw_recall_qps"
PLOT_EFS = {40, 120}
PLOT_PARAMS = {
    "HNSW": {16, 24, 32},
    "SymphonyQG": {32, 64, 96},
    "FreshDiskANN": {32, 40, 48},
}


# Compress visually empty bands so all measured clusters occupy more of the
# plotting area while tick labels stay in the original recall/QPS units.
COMPRESSED_RECALL_BANDS = (
    (0.690, 0.755, 0.16),
    (0.795, 0.865, 0.18),
)
COMPRESSED_QPS_BANDS = (
    (3000.0, 5000.0, 1.75),
    (12500.0, 17500.0, 0.16),
)


PARAM_RANK = {
    family: {param: rank for rank, param in enumerate(sorted(params))}
    for family, params in PLOT_PARAMS.items()
}

MARKER_MAP = {
    (0, 40): "o",
    (0, 120): "s",
    (1, 40): "^",
    (1, 120): "v",
    (2, 40): "D",
    (2, 120): "P",
}


SERIES_STYLE = {
    "HNSW off": {"color": "#0072B2", "linestyle": "-"},
    "HNSW on": {"color": "#D55E00", "linestyle": "-"},
    "SymphonyQG off": {"color": "#6A51A3", "linestyle": "-"},
    "SymphonyQG on": {"color": "#009E73", "linestyle": "-"},
    "FreshDiskANN off": {"color": "#CC79A7", "linestyle": "-"},
    "FreshDiskANN on": {"color": "#E69F00", "linestyle": "-"},
}


def final_results_path(exp_dir: Path) -> Path | None:
    preferred = exp_dir / f"{exp_dir.name}_final_results.csv"
    if preferred.exists():
        return preferred
    matches = sorted(exp_dir.glob("*final_results.csv"))
    return matches[0] if matches else None


def load_summary() -> pd.DataFrame:
    rows = []
    for spec in RESULT_SPECS:
        root = spec["root"]
        if not root.exists():
            print(f"Warning: result directory does not exist: {root}")
            continue
        for exp_dir in sorted(root.iterdir()):
            if not exp_dir.is_dir():
                continue
            match = spec["pattern"].fullmatch(exp_dir.name)
            if match is None:
                continue

            param = int(match.group("param"))
            ef = int(match.group("ef"))
            if ef not in PLOT_EFS or param not in PLOT_PARAMS[spec["family"]]:
                continue

            path = final_results_path(exp_dir)
            if path is None:
                print(f"Warning: no final_results CSV found under {exp_dir}")
                continue

            df = pd.read_csv(path)
            if "recall" not in df.columns or "query_qps" not in df.columns:
                print(f"Warning: missing recall/query_qps in {path}")
                continue

            recall = pd.to_numeric(df["recall"], errors="coerce")
            query_qps = pd.to_numeric(df["query_qps"], errors="coerce")
            rows.append(
                {
                    "series": spec["series"],
                    "family": spec["family"],
                    "mode": spec["mode"],
                    "param_name": spec["param_name"],
                    "param": param,
                    "ef": ef,
                    "config_label": f"{spec['family']} {spec['param_name']}{param}/ef{ef}",
                    "experiment": exp_dir.name,
                    "mean_recall": recall.mean(),
                    "mean_query_qps": query_qps.mean(),
                    "file": str(path),
                }
            )

    if not rows:
        roots = ", ".join(str(spec["root"]) for spec in RESULT_SPECS)
        raise RuntimeError(f"No valid final_results CSV files found under: {roots}")
    return pd.DataFrame(rows)


def compress_value(value: float, bands: tuple[tuple[float, float, float], ...]) -> float:
    value = float(value)
    shifted = value
    for start, end, factor in bands:
        saving = (end - start) * (1.0 - factor)
        if value <= start:
            continue
        if value >= end:
            shifted -= saving
        else:
            shifted -= (value - start) * (1.0 - factor)
    return shifted


def compress_recall(value: float) -> float:
    return compress_value(value, COMPRESSED_RECALL_BANDS)


def compress_qps(value: float) -> float:
    return compress_value(value, COMPRESSED_QPS_BANDS)


def qps_tick_label(value: float) -> str:
    if abs(value) >= 1000:
        k_value = value / 1000.0
        if abs(k_value - round(k_value)) < 1e-9:
            return f"{int(round(k_value))}k"
        return f"{k_value:g}k"
    return f"{value:g}"


def add_compression_marks(ax) -> None:
    for start, end, _ in COMPRESSED_RECALL_BANDS:
        ax.text(
            compress_recall((start + end) / 2.0),
            0.0,
            "//",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#444444",
            clip_on=False,
            zorder=20,
        )
    for start, end, factor in COMPRESSED_QPS_BANDS:
        if factor >= 1.0:
            continue
        ax.text(
            0.0,
            compress_qps((start + end) / 2.0),
            "//",
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#444444",
            clip_on=False,
            zorder=20,
        )


def marker_size(marker: str) -> int:
    return 112




def main() -> None:
    summary = load_summary().sort_values(["family", "mode", "param", "ef"]).reset_index(drop=True)
    summary.to_csv(f"{OUT_PREFIX}_summary.csv", index=False)

    summary["plot_recall"] = summary["mean_recall"].map(compress_recall)
    summary["plot_query_qps"] = summary["mean_query_qps"].map(compress_qps)

    fig, ax = plt.subplots(figsize=(8.4, 5.1))

    for series in ["SymphonyQG off", "SymphonyQG on", "HNSW off", "HNSW on", "FreshDiskANN off", "FreshDiskANN on"]:
        series_df = summary[summary["series"] == series].sort_values(["mean_recall", "mean_query_qps"])
        if series_df.empty:
            continue
        style = SERIES_STYLE[series]
        ax.plot(
            series_df["plot_recall"],
            series_df["plot_query_qps"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.1,
            alpha=0.95,
            zorder=2,
        )
        for _, row in series_df.iterrows():
            param_rank = PARAM_RANK[row["family"]][int(row["param"])]
            marker = MARKER_MAP.get((param_rank, int(row["ef"])), "o")
            ax.scatter(
                row["plot_recall"],
                row["plot_query_qps"],
                color=style["color"],
                marker=marker,
                s=marker_size(marker),
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )

    ax.set_xlabel("Recall (%)", fontsize=16)
    ax.set_ylabel("Query QPS", fontsize=16)
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linestyle="-", linewidth=1.2, alpha=1.0)
    ax.tick_params(axis="both", labelsize=14)

    recall_ticks = [0.66, 0.68, 0.78, 0.87, 0.90, 0.95, 1.00]
    qps_ticks = [3000, 4000, 5000, 9000, 12500, 18000, 21000]
    ax.set_xticks([compress_recall(value) for value in recall_ticks])
    ax.set_xticklabels([f"{int(round(value * 100))}" for value in recall_ticks])
    ax.set_yticks([compress_qps(value) for value in qps_ticks])
    ax.set_yticklabels([qps_tick_label(value) for value in qps_ticks])

    x_pad = 0.008
    y_pad = 700
    ax.set_xlim(
        compress_recall(summary["mean_recall"].min() - x_pad),
        compress_recall(summary["mean_recall"].max() + x_pad),
    )
    ax.set_ylim(
        compress_qps(summary["mean_query_qps"].min() - y_pad),
        compress_qps(summary["mean_query_qps"].max() + y_pad),
    )
    add_compression_marks(ax)

    series_handles = [
        Line2D(
            [0],
            [0],
            color=SERIES_STYLE[series]["color"],
            linestyle="-",
            linewidth=2.5,
            marker="o",
            markersize=6.5,
            markerfacecolor=SERIES_STYLE[series]["color"],
            markeredgecolor="white",
            label=series.replace(" off", "").replace(" on", "-BriskSeed"),
        )
        for series in ["HNSW off", "HNSW on", "SymphonyQG off", "SymphonyQG on", "FreshDiskANN off", "FreshDiskANN on"]
    ]
    ax.legend(
        handles=series_handles,
        loc="lower center",
        bbox_to_anchor=(0.46, 1.015),
        ncol=3,
        frameon=True,
        facecolor="#f0f0f0",
        edgecolor="#bdbdbd",
        framealpha=1.0,
        fontsize=12.8,
        handlelength=2.2,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    plt.subplots_adjust(left=0.105, right=0.985, bottom=0.085, top=0.78)
    plt.savefig(f"{OUT_PREFIX}.png", dpi=400, bbox_inches="tight", pad_inches=0.05)
    plt.savefig(f"{OUT_PREFIX}.pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {OUT_PREFIX}.png and {OUT_PREFIX}.pdf")
    print(f"Saved {OUT_PREFIX}_summary.csv")
    print(
        summary[["series", "param_name", "param", "ef", "mean_recall", "mean_query_qps"]]
        .sort_values(["series", "param", "ef"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
