import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


def read_qps_csv(path):
    df = pd.read_csv(path)
    col_lower = {col.lower(): col for col in df.columns}

    if "batch_idx" in col_lower:
        batch_col = col_lower["batch_idx"]
    elif "batch_id" in col_lower:
        batch_col = col_lower["batch_id"]
    elif "batch" in col_lower:
        batch_col = col_lower["batch"]
    else:
        batch_col = df.columns[0]

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


def load_data(dataset, algo, tests):
    rows = []
    for test in tests:
        path = f"../results/{dataset}/{algo}/{test}/ef-120_batch_query_qps.csv"
        if os.path.exists(path):
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
        else:
            print(f"Warning: {path} not found.")

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


def main():
    datasets = ["sift", "msong", "glove"]
    algos = ["streamseed_hybrid", "streamseed_hybrid_on"]
    tests = [f"test{i}" for i in range(1, 11)]

    all_rows = []
    for dataset in datasets:
        for algo in algos:
            df = load_data(dataset, algo, tests)
            if not df.empty:
                all_rows.append(df)

    if not all_rows:
        raise RuntimeError("No valid qps csv files found.")

    plot_df = pd.concat(all_rows, ignore_index=True)
    plot_df = trim_extreme_qps(plot_df, drop_low=3, drop_high=3)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    palette = {
        "streamseed_hybrid": "#D62828",
        "streamseed_hybrid_on": "#0057B7",
    }
    title_map = {
        "sift": "SIFT",
        "msong": "MSONG",
        "glove": "GLOVE",
    }
    y_limits = {
        "sift": (10000, 80000),
        "msong": (6000, 12000),
        "glove": (7500, 25000),
    }

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        dataset_df = plot_df[plot_df["dataset"] == dataset]
        if dataset_df.empty:
            ax.set_visible(False)
            continue

        sns.lineplot(
            data=dataset_df,
            x="batch_idx",
            y="query_qps",
            hue="algo",
            hue_order=algos,
            estimator="mean",
            errorbar=("ci", 95),
            err_kws={"alpha": 0.12},
            linewidth=2.2,
            palette=palette,
            ax=ax,
        )

        ax.set_title(title_map.get(dataset, dataset.upper()), fontsize=20)
        if dataset in y_limits:
            ax.set_ylim(*y_limits[dataset])
        ax.set_xlabel("Batch ID")
        if i == 0:
            ax.set_ylabel("Query QPS")
        else:
            ax.set_ylabel("")

        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x / 1000:g}k" if abs(x) >= 1000 else f"{x:g}")
        )
        ax.grid(True, linestyle="--", alpha=0.45)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            bbox_to_anchor=(0.5, 1.03),
        )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_file = "qps_datasets_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.savefig("qps_datasets_comparison.pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Plot saved to {out_file}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
