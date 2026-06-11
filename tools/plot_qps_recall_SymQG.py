import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter, LinearLocator


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


def get_recall_path(dataset, algo, test):
    test_dir = f"../results/{dataset}/{algo}/{test}"
    candidates = [
        os.path.join(test_dir, "ef-120_final_results.csv"),
        os.path.join(test_dir, f"{test}_final_results.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    matches = sorted(glob.glob(os.path.join(test_dir, "*_final_results.csv")))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Warning: multiple final result CSV files found in {test_dir}; use {matches[0]}.")
        return matches[0]
    return candidates[0]


def load_qps_data(dataset, algo, tests):
    rows = []
    for test in tests:
        path = f"../results/{dataset}/{algo}/{test}/ef-120_batch_query_qps.csv"
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
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
        path = get_recall_path(dataset, algo, test)
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
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
        return group.iloc[drop_low:len(group) - drop_high]

    return (
        df.groupby(group_cols, group_keys=False)
        .apply(trim_group)
        .reset_index(drop=True)
    )


def remove_axis_legend(ax):
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def get_mean_curve(data):
    return (
        data.groupby(["algo_label", "batch_idx"], as_index=False)["recall"]
        .mean()
        .sort_values(["algo_label", "batch_idx"])
    )


def add_recall_inset(ax, dataset_df, algo_labels, palette):
    mean_df = get_mean_curve(dataset_df)
    if mean_df.empty:
        return

    diff_df = mean_df.pivot(index="batch_idx", columns="algo_label", values="recall")
    if len(algo_labels) >= 2 and all(label in diff_df.columns for label in algo_labels[:2]):
        diff_df["abs_diff"] = (diff_df[algo_labels[0]] - diff_df[algo_labels[1]]).abs()
        diff_df = diff_df.dropna(subset=["abs_diff"]).sort_index()
        batch_ids = list(diff_df.index)
        if len(batch_ids) > 2:
            best_start = max(
                range(len(batch_ids) - 1),
                key=lambda i: diff_df.loc[batch_ids[i:i + 2], "abs_diff"].sum(),
            )
            zoom_batches = batch_ids[best_start:best_start + 2]
        else:
            zoom_batches = batch_ids
    else:
        zoom_batches = sorted(mean_df["batch_idx"].dropna().unique())[-2:]

    zoom_df = mean_df[mean_df["batch_idx"].isin(zoom_batches)]
    if zoom_df.empty:
        zoom_df = mean_df

    inset = inset_axes(ax, width="38%", height="34%", loc="upper right", borderpad=0.6)
    for label in algo_labels:
        line_df = zoom_df[zoom_df["algo_label"] == label]
        if line_df.empty:
            continue
        inset.plot(
            line_df["batch_idx"],
            line_df["recall"],
            color=palette[label],
            linewidth=1.4,
        )

    y_min = zoom_df["recall"].min()
    y_max = zoom_df["recall"].max()
    y_pad = max((y_max - y_min) * 0.25, 0.002)
    inset.set_xlim(zoom_df["batch_idx"].min(), zoom_df["batch_idx"].max())
    inset.set_ylim(y_min - y_pad, y_max + y_pad)
    inset.set_xticks(zoom_batches)
    inset.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x))}"))
    inset.yaxis.set_major_locator(LinearLocator(3))
    inset.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    inset.grid(True, linestyle="--", alpha=0.35)
    inset.tick_params(axis="both", labelsize=11, pad=1)
    inset.set_xlabel("")
    inset.set_ylabel("")
    remove_axis_legend(inset)


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
    add_recall_inset(ax, dataset_df, algo_labels, palette)


def main():
    datasets = ["sift", "msong", "glove"]
    algos = ["streamseed_hybrid_symphonyqg_off", "streamseed_hybrid_symphonyqg"]
    tests = [f"test{i}" for i in range(1, 11)]
    title_map = {
        "sift": "SIFT",
        "msong": "Msong",
        "glove": "Glove",
    }
    label_map = {
        "streamseed_hybrid_symphonyqg_off": "SymphonyQG",
        "streamseed_hybrid_symphonyqg": "SymphonyQG-BriskSeed",
    }
    algo_labels = [label_map[algo] for algo in algos]
    palette = {
        "SymphonyQG": "#FF9933",
        "SymphonyQG-BriskSeed": "#7B1FA2",
    }
    qps_y_limits = {
        "sift": (0, 60000),
        "msong": (2000, 9000),
        "glove": (3000, 11000),
    }

    qps_rows = []
    recall_rows = []
    for dataset in datasets:
        for algo in algos:
            qps_df = load_qps_data(dataset, algo, tests)
            if not qps_df.empty:
                qps_rows.append(qps_df)
            recall_df = load_recall_data(dataset, algo, tests)
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
    out_file = "qps_recall_SymQG.png"
    plt.savefig(out_file, dpi=400, bbox_inches="tight", pad_inches=0.1)
    plt.savefig("qps_recall_SymQG.pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Plot saved to {out_file}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
