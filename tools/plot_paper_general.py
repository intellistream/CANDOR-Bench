#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from export_results import export_results


COLORS = {
    "faiss_HNSW": "#1f4e79",
    "gammafresh": "#1b9e77",
    "diskann": "#d95f02",
}

LABELS = {
    "faiss_HNSW": "Faiss-HNSW",
    "gammafresh": "GammaFresh",
    "diskann": "FreshDiskANN",
}


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def _label(algo: str) -> str:
    return LABELS.get(algo, algo)


def _color(algo: str) -> str:
    return COLORS.get(algo, "#555555")


def _load_final_results(dataset: str, runbook: str, algorithms: list[str], results_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for algo in algorithms:
        algo_dir = results_dir / dataset / algo
        final_csvs = sorted(algo_dir.glob("*/*_final_results.csv"))
        if final_csvs:
            frames[algo] = pd.read_csv(final_csvs[0])
            continue

        param_dirs = sorted(p.name for p in algo_dir.iterdir() if p.is_dir()) if algo_dir.exists() else []
        param_folder = param_dirs[0] if param_dirs else None
        df, _summary = export_results(
            dataset,
            algo,
            runbook,
            output_dir=str(results_dir),
            param_folder=param_folder,
        )
        frames[algo] = df
    return frames


def _plot_step_recall(frames: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for algo, df in frames.items():
        ax.plot(df["batch_idx"], df["recall"], marker="o", linewidth=2.2, markersize=4, label=_label(algo), color=_color(algo))
    ax.set_xlabel("Continuous Query Checkpoint")
    ax.set_ylabel("Recall@10")
    ax.set_title("Stepwise Recall")
    ax.legend(frameon=False)
    out = output_dir / "paper_step_recall.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_latency_cdf(frames: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for algo, df in frames.items():
        vals = np.sort(df["query_latency_ms"].dropna().to_numpy(dtype=float))
        if vals.size == 0:
            continue
        ys = np.arange(1, vals.size + 1, dtype=float) / float(vals.size)
        ax.plot(vals, ys, linewidth=2.2, label=_label(algo), color=_color(algo))
    ax.set_xlabel("Batch Query Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Batch Latency CDF")
    ax.legend(frameon=False)
    out = output_dir / "paper_batch_latency_cdf.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_query_tp(frames: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for algo, df in frames.items():
        vals = df["query_qps"].dropna()
        ax.plot(df.loc[vals.index, "batch_idx"], vals, marker="o", linewidth=2.2, markersize=4, label=_label(algo), color=_color(algo))
    ax.set_xlabel("Continuous Query Checkpoint")
    ax.set_ylabel("Queries/s")
    ax.set_title("Stepwise Query Throughput")
    ax.legend(frameon=False)
    out = output_dir / "paper_query_throughput.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_insert_tp(frames: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for algo, df in frames.items():
        vals = df["insert_qps"].dropna()
        ax.plot(df.loc[vals.index, "batch_idx"], vals, marker="o", linewidth=2.2, markersize=4, label=_label(algo), color=_color(algo))
    ax.set_xlabel("Continuous Query Checkpoint")
    ax.set_ylabel("Vectors/s")
    ax.set_title("Insertion Throughput")
    ax.legend(frameon=False)
    out = output_dir / "paper_insert_throughput.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _write_summary(frames: dict[str, pd.DataFrame], output_dir: Path) -> Path:
    rows = []
    for algo, df in frames.items():
        rows.append(
            {
                "algorithm": algo,
                "final_recall": float(df["recall"].dropna().iloc[-1]) if not df["recall"].dropna().empty else np.nan,
                "mean_query_qps": float(df["query_qps"].dropna().mean()) if "query_qps" in df else np.nan,
                "mean_insert_qps": float(df["insert_qps"].dropna().mean()) if "insert_qps" in df else np.nan,
                "mean_query_latency_ms": float(df["query_latency_ms"].dropna().mean()) if "query_latency_ms" in df else np.nan,
            }
        )
    out = output_dir / "paper_summary.csv"
    pd.DataFrame(rows).sort_values("final_recall", ascending=False).to_csv(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-style general-experiment plots from root benchmark results")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runbook", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--algorithms", nargs="+", default=["faiss_HNSW", "gammafresh"])
    args = parser.parse_args()

    _style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_final_results(args.dataset, args.runbook, args.algorithms, Path(args.results_dir))
    outputs = [
        _write_summary(frames, output_dir),
        _plot_step_recall(frames, output_dir),
        _plot_latency_cdf(frames, output_dir),
        _plot_query_tp(frames, output_dir),
        _plot_insert_tp(frames, output_dir),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
