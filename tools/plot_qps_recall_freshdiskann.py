import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def get_batch_col(df):
    col_lower = {col.lower(): col for col in df.columns}
    if "batch_idx" in col_lower:
        return col_lower["batch_idx"]
    if "batch_id" in col_lower:
        return col_lower["batch_id"]
    if "batch" in col_lower:
        return col_lower["batch"]
    return df.columns[0]


def read_qps_csv(path):
    df = pd.read_csv(path)
    col_lower = {col.lower(): col for col in df.columns}
    batch_col = get_batch_col(df)

    if "query_qps" in col_lower:
        qps_col = col_lower["query_qps"]
    elif "qps" in col_lower:
        qps_col = col_lower["qps"]
    elif "throughput" in col_lower:
        qps_col = col_lower["throughput"]
    else:
        qps_col = df.columns[-1]

    df = df[[batch_col, qps_col]].rename(
        columns={batch_col: "batch_idx", qps_col: "query_qps"}
    )
    df["batch_idx"] = pd.to_numeric(df["batch_idx"], errors="coerce")
    df["query_qps"] = pd.to_numeric(df["query_qps"], errors="coerce")
    return df.dropna(subset=["batch_idx", "query_qps"])


def read_recall_csv(path):
    df = pd.read_csv(path)
    col_lower = {col.lower(): col for col in df.columns}
    batch_col = get_batch_col(df)

    if "recall" not in col_lower:
        raise ValueError("recall column not found")
    recall_col = col_lower["recall"]

    df = df[[batch_col, recall_col]].rename(
        columns={batch_col: "batch_idx", recall_col: "recall"}
    )
    df["batch_idx"] = pd.to_numeric(df["batch_idx"], errors="coerce")
    df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
    return df.dropna(subset=["batch_idx", "recall"])


def find_one(pattern, base_dir):
    matches = sorted(base_dir.glob(pattern))
    return matches[0] if matches else None


def load_qps_data(dataset, algo, tests):
    rows = []
    for test in tests:
        test_dir = Path("..") / "results" / dataset / algo / test
        path = find_one("*_batch_query_qps.csv", test_dir)
        if path is None:
            print(f"Warning: no *_batch_query_qps.csv found under {test_dir}.")
            continue
        try:
            df = read_qps_csv(path)
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}")
            continue
        if df.empty:
            print(f"Warning: {path} has no valid qps rows.")
            continue
        df["dataset"] = dataset
        df["algo"] = algo
        df["test"] = test
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_recall_data(dataset, algo, tests):
    rows = []
    for test in tests:
        test_dir = Path("..") / "results" / dataset / algo / test
        preferred = test_dir / f"{test}_final_results.csv"
        path = preferred if preferred.exists() else find_one("*final_results.csv", test_dir)
        if path is None:
            print(f"Warning: no *final_results.csv found under {test_dir}.")
            continue
        try:
            df = read_recall_csv(path)
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}")
            continue
        if df.empty:
            print(f"Warning: {path} has no valid recall rows.")
            continue
        df["dataset"] = dataset
        df["algo"] = algo
        df["test"] = test
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def trim_extreme_qps(df, drop_low=3, drop_high=3):
    group_cols = ["dataset", "algo", "batch_idx"]

    def trim_group(group):
        group = group.sort_values("query_qps")
        if len(group) <= drop_low + drop_high:
            print(
                f"Warning: keep all rows for {group.name}; "
                f"need more than {drop_low + drop_high} rows to trim extremes."
            )
            return group
        return group.iloc[drop_low : len(group) - drop_high]

    return (
        df.groupby(group_cols, group_keys=False)
        .apply(trim_group)
        .reset_index(drop=True)
    )


def remove_axis_legend(ax):
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def plot_qps(ax, data, dataset, algo_labels, palette, title, y_limits):
    dataset_df = data[data["dataset"] == dataset]
    if dataset_df.empty:
        ax.set_visible(False)
        return

    sns.lineplot(
        data=dataset_df,
        x="batch_idx",
        y="query_qps",
        hue="algo_label",
        hue_order=algo_labels,
        estimator="mean",
        errorbar=("ci", 95),
        err_kws={"alpha": 0.08},
        linewidth=1.8,
        palette=palette,
        ax=ax,
    )
    ax.set_title(title, fontsize=18)
    if dataset in y_limits:
        ax.set_ylim(*y_limits[dataset])
    ax.set_xlabel("Batch ID")
    ax.set_ylabel("Query QPS")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x / 1000:g}k" if abs(x) >= 1000 else f"{x:g}")
    )
    ax.grid(True, linestyle="--", alpha=0.45)
    remove_axis_legend(ax)


def plot_recall(ax, data, dataset, algo_labels, palette, title):
    dataset_df = data[data["dataset"] == dataset]
    if dataset_df.empty:
        ax.set_visible(False)
        return

    sns.lineplot(
        data=dataset_df,
        x="batch_idx",
        y="recall",
        hue="algo_label",
        hue_order=algo_labels,
        estimator="mean",
        errorbar=None,
        linewidth=1.8,
        palette=palette,
        ax=ax,
    )
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Batch ID")
    ax.set_ylabel("Recall")
    ax.grid(True, linestyle="--", alpha=0.45)
    remove_axis_legend(ax)


def main():
    datasets = ["sift", "msong", "glove"]
    algos = [
        "streamseed_hybrid_freshdiskann",
        "streamseed_hybrid_freshdiskann_on",
    ]
    tests = [f"test{i}" for i in range(1, 11)]
    briskseed_recall_tests_map = {
        "sift": ["test11"],
        "msong": ["test11"],
        "glove": tests,
    }
    title_map = {
        "sift": "SIFT",
        "msong": "Msong",
        "glove": "Glove",
    }
    label_map = {
        "streamseed_hybrid_freshdiskann": "FreshDiskANN",
        "streamseed_hybrid_freshdiskann_on": "FreshDiskANN-BriskSeed",
    }
    algo_labels = [label_map[algo] for algo in algos]
    palette = {
        "FreshDiskANN": "#009E73",
        "FreshDiskANN-BriskSeed": "#6A51A3",
    }
    qps_y_limits = {
        "sift": (10000, 80000),
        "msong": (1000, 9000),
        "glove": (2000, 10000),
    }

    qps_rows = []
    recall_rows = []
    for dataset in datasets:
        for algo in algos:
            qps_df = load_qps_data(dataset, algo, tests)
            if not qps_df.empty:
                qps_rows.append(qps_df)
            recall_tests = tests
            if algo == "streamseed_hybrid_freshdiskann_on":
                recall_tests = briskseed_recall_tests_map[dataset]
            recall_df = load_recall_data(dataset, algo, recall_tests)
            if not recall_df.empty:
                recall_rows.append(recall_df)

    if not qps_rows:
        raise RuntimeError("No valid qps csv files found.")
    if not recall_rows:
        raise RuntimeError("No valid final result CSV files found.")

    qps_df = trim_extreme_qps(pd.concat(qps_rows, ignore_index=True), drop_low=3, drop_high=3)
    recall_df = pd.concat(recall_rows, ignore_index=True)
    qps_df["algo_label"] = qps_df["algo"].map(label_map)
    recall_df["algo_label"] = recall_df["algo"].map(label_map)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 6, figsize=(24, 4), sharex=False)

    for ax in axes:
        ax.set_box_aspect(1)

    for i, dataset in enumerate(datasets):
        name = title_map.get(dataset, dataset.upper())
        plot_qps(
            axes[i],
            qps_df,
            dataset,
            algo_labels,
            palette,
            name,
            qps_y_limits,
        )
        plot_recall(
            axes[i + 3],
            recall_df,
            dataset,
            algo_labels,
            palette,
            name,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            bbox_to_anchor=(0.5, 1.08),
        )

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    out_file = "qps_recall_freshdiskann.png"
    plt.savefig(out_file, dpi=400, bbox_inches="tight", pad_inches=0.1)
    plt.savefig("qps_recall_freshdiskann.pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Plot saved to {out_file}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
