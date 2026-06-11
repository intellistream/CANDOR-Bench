import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_recall_csv(path):
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

    if "recall" in col_lower:
        recall_col = col_lower["recall"]
    else:
        raise ValueError("recall column not found")

    df = df[[batch_col, recall_col]].rename(
        columns={batch_col: "batch_idx", recall_col: "recall"}
    )
    df["batch_idx"] = pd.to_numeric(df["batch_idx"], errors="coerce")
    df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
    return df.dropna(subset=["batch_idx", "recall"])


def load_data(dataset, algo, tests):
    rows = []
    for test in tests:
        path = f"../results/{dataset}/{algo}/{test}/{test}_final_results.csv"
        if os.path.exists(path):
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
        else:
            print(f"Warning: {path} not found.")

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


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
        raise RuntimeError("No valid final result CSV files found.")

    plot_df = pd.concat(all_rows, ignore_index=True)

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

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        dataset_df = plot_df[plot_df["dataset"] == dataset]
        if dataset_df.empty:
            ax.set_visible(False)
            continue

        sns.lineplot(
            data=dataset_df,
            x="batch_idx",
            y="recall",
            hue="algo",
            hue_order=algos,
            estimator="mean",
            errorbar=None,
            linewidth=2.2,
            palette=palette,
            ax=ax,
        )

        ax.set_title(title_map.get(dataset, dataset.upper()), fontsize=20)
        ax.set_xlabel("Batch ID")
        if i == 0:
            ax.set_ylabel("Recall")
        else:
            ax.set_ylabel("")

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
    out_file = "recall_datasets_comparison_HNSW.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.savefig("recall_datasets_comparison_HNSW.pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Plot saved to {out_file}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
