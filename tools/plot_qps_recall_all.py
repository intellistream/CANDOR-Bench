import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, LinearLocator, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


DATASETS = ["sift", "msong", "glove"]
TESTS = [f"test{i}" for i in range(1, 11)]
TITLE_MAP = {
    "sift": "SIFT",
    "msong": "Msong",
    "glove": "Glove",
}


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


def first_existing_qps(test_dir, fixed_name=None):
    if fixed_name is not None:
        path = test_dir / fixed_name
        if path.exists():
            return path
    matches = sorted(test_dir.glob("*_batch_query_qps.csv"))
    return matches[0] if matches else None


def first_existing_recall(test_dir, test):
    candidates = [
        test_dir / "ef-120_final_results.csv",
        test_dir / f"{test}_final_results.csv",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = sorted(test_dir.glob("*_final_results.csv"))
    return matches[0] if matches else None


def load_qps_data(dataset, algo, tests, fixed_qps_name=None):
    rows = []
    for test in tests:
        test_dir = Path("..") / "results" / dataset / algo / test
        path = first_existing_qps(test_dir, fixed_name=fixed_qps_name)
        if path is None:
            print(f"Warning: no qps CSV found under {test_dir}.")
            continue
        try:
            df = read_qps_csv(path)
        except Exception as exc:
            print(f"Warning: failed to read {path}: {exc}")
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
        path = first_existing_recall(test_dir, test)
        if path is None:
            print(f"Warning: no final_results CSV found under {test_dir}.")
            continue
        try:
            df = read_recall_csv(path)
        except Exception as exc:
            print(f"Warning: failed to read {path}: {exc}")
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


def get_mean_recall_curve(data):
    return (
        data.groupby(["algo_label", "batch_idx"], as_index=False)["recall"]
        .mean()
        .sort_values(["algo_label", "batch_idx"])
    )


def add_recall_inset(ax, dataset_df, algo_labels, palette):
    mean_df = get_mean_recall_curve(dataset_df)
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
                key=lambda i: diff_df.loc[batch_ids[i : i + 2], "abs_diff"].sum(),
            )
            zoom_batches = batch_ids[best_start : best_start + 2]
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
    inset.set_facecolor("#f0f0f0")
    inset.grid(True, color="white", linestyle="-", linewidth=1.0, alpha=0.35)
    inset.tick_params(axis="both", labelsize=13, pad=1)
    inset.set_xlabel("")
    inset.set_ylabel("")
    remove_axis_legend(inset)


def format_k_ticks(x, _):
    if abs(x) < 1000:
        return f"{x:g}"
    value = x / 1000
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}k"
    return f"{value:g}k"


def style_axis(ax):
    ax.set_facecolor("#f0f0f0")
    ax.grid(True, color="white", linestyle="-", linewidth=1.2, alpha=1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def plot_qps(ax, data, dataset, algo_labels, palette, title, y_limits, show_ylabel, y_ticks=None):
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
        err_kws={"alpha": 0.18},
        linewidth=1.8,
        palette=palette,
        ax=ax,
    )
    ax.set_title(title, fontsize=18, fontweight="semibold")
    if dataset in y_limits:
        ax.set_ylim(*y_limits[dataset])
    ax.set_xlabel("Batch ID", fontsize=16)
    ax.set_ylabel("Query QPS" if show_ylabel else "", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(format_k_ticks))
    style_axis(ax)
    if y_ticks is not None and dataset in y_ticks:
        ax.set_yticks(y_ticks[dataset])
    remove_axis_legend(ax)


def plot_recall(ax, data, dataset, algo_labels, palette, title, show_ylabel, add_inset=False):
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
    ax.set_title(title, fontsize=18, fontweight="semibold")
    ax.set_xlabel("Batch ID", fontsize=16)
    ax.set_ylabel("Recall" if show_ylabel else "", fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    style_axis(ax)
    remove_axis_legend(ax)
    if add_inset:
        add_recall_inset(ax, dataset_df, algo_labels, palette)


def dataset_system_means(qps_df, recall_df, family):
    qps_means = (
        qps_df.groupby(["dataset", "algo_label"], as_index=False)["query_qps"]
        .mean()
        .rename(columns={"query_qps": "mean_query_qps"})
    )
    recall_means = (
        recall_df.groupby(["dataset", "algo_label"], as_index=False)["recall"]
        .mean()
        .rename(columns={"recall": "mean_recall"})
    )
    means = qps_means.merge(recall_means, on=["dataset", "algo_label"], how="outer")
    means["family"] = family["name"]
    return means


def summarize_system_metrics(dataset_means):
    system_summary = (
        dataset_means.groupby(["family", "algo_label"], as_index=False)
        .agg(
            avg_query_qps=("mean_query_qps", "mean"),
            avg_recall=("mean_recall", "mean"),
            datasets=("dataset", "nunique"),
        )
    )

    pair_rows = []
    for family, group in dataset_means.groupby("family"):
        labels = list(group["algo_label"].dropna().unique())
        base_labels = [label for label in labels if not label.endswith("-BriskSeed")]
        brisk_labels = [label for label in labels if label.endswith("-BriskSeed")]
        if not base_labels or not brisk_labels:
            continue
        base_label = base_labels[0]
        brisk_label = brisk_labels[0]
        wide = group.pivot(index="dataset", columns="algo_label", values=["mean_query_qps", "mean_recall"])
        for metric, out_name in [("mean_query_qps", "query_qps"), ("mean_recall", "recall")]:
            if (metric, base_label) not in wide.columns or (metric, brisk_label) not in wide.columns:
                continue
            diff = wide[(metric, brisk_label)] - wide[(metric, base_label)]
            diff = diff.dropna()
            if diff.empty:
                continue
            max_dataset = diff.abs().idxmax()
            pair_rows.append(
                {
                    "family": family,
                    "metric": out_name,
                    "baseline": base_label,
                    "briskseed": brisk_label,
                    "max_abs_diff": float(diff.loc[max_dataset]),
                    "max_abs_diff_abs": float(abs(diff.loc[max_dataset])),
                    "dataset": max_dataset,
                }
            )
    pair_summary = pd.DataFrame(pair_rows)
    return system_summary, pair_summary


def load_family_data(family):
    qps_rows = []
    recall_rows = []
    for dataset in DATASETS:
        for algo in family["algos"]:
            qps_df = load_qps_data(
                dataset,
                algo,
                TESTS,
                fixed_qps_name=family.get("fixed_qps_name"),
            )
            if not qps_df.empty:
                qps_rows.append(qps_df)

            recall_tests = TESTS
            recall_tests_by_algo = family.get("recall_tests_by_algo", {})
            recall_tests_by_dataset_algo = family.get("recall_tests_by_dataset_algo", {})
            if algo in recall_tests_by_algo:
                recall_tests = recall_tests_by_algo[algo]
            if (dataset, algo) in recall_tests_by_dataset_algo:
                recall_tests = recall_tests_by_dataset_algo[(dataset, algo)]

            recall_df = load_recall_data(dataset, algo, recall_tests)
            if not recall_df.empty:
                recall_rows.append(recall_df)

    if not qps_rows:
        raise RuntimeError(f"No valid qps CSV files found for {family['name']}.")
    if not recall_rows:
        raise RuntimeError(f"No valid final result CSV files found for {family['name']}.")

    qps_df = trim_extreme_qps(pd.concat(qps_rows, ignore_index=True), drop_low=3, drop_high=3)
    recall_df = pd.concat(recall_rows, ignore_index=True)
    qps_df["algo_label"] = qps_df["algo"].map(family["label_map"])
    recall_df["algo_label"] = recall_df["algo"].map(family["label_map"])
    return qps_df, recall_df


def main():
    families = [
        {
            "name": "HNSW",
            "algos": ["streamseed_hybrid", "streamseed_hybrid_on"],
            "label_map": {
                "streamseed_hybrid": "HNSW",
                "streamseed_hybrid_on": "HNSW-BriskSeed",
            },
            "palette": {
                "HNSW": "#D62828",
                "HNSW-BriskSeed": "#0057B7",
            },
            "fixed_qps_name": "ef-120_batch_query_qps.csv",
            "qps_y_limits": {
                "sift": (10000, 80000),
                "msong": (6000, 12000),
                "glove": (7500, 25000),
            },
            "qps_y_ticks": {
                "msong": [6000, 8000, 10000, 12000],
            },
        },
        {
            "name": "SymphonyQG",
            "algos": ["streamseed_hybrid_symphonyqg_off", "streamseed_hybrid_symphonyqg"],
            "label_map": {
                "streamseed_hybrid_symphonyqg_off": "SymphonyQG",
                "streamseed_hybrid_symphonyqg": "SymphonyQG-BriskSeed",
            },
            "palette": {
                "SymphonyQG": "#FF9933",
                "SymphonyQG-BriskSeed": "#4B0082",
            },
            "fixed_qps_name": "ef-120_batch_query_qps.csv",
            "qps_y_limits": {
                "sift": (0, 60000),
                "msong": (2000, 9000),
                "glove": (3000, 11000),
            },
            "recall_inset": True,
        },
        {
            "name": "FreshDiskANN",
            "algos": ["streamseed_hybrid_freshdiskann", "streamseed_hybrid_freshdiskann_on"],
            "label_map": {
                "streamseed_hybrid_freshdiskann": "FreshDiskANN",
                "streamseed_hybrid_freshdiskann_on": "FreshDiskANN-BriskSeed",
            },
            "palette": {
                "FreshDiskANN": "#009E73",
                "FreshDiskANN-BriskSeed": "#6A51A3",
            },
            "qps_y_limits": {
                "sift": (10000, 80000),
                "msong": (1000, 9000),
                "glove": (2000, 10000),
            },
            "recall_tests_by_dataset_algo": {
                ("sift", "streamseed_hybrid_freshdiskann_on"): ["test11"],
                ("msong", "streamseed_hybrid_freshdiskann_on"): ["test11"],
            },
        },
    ]

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(3, 6, figsize=(24, 10), sharex=False)

    dataset_mean_rows = []
    for row, family in enumerate(families):
        qps_df, recall_df = load_family_data(family)
        dataset_mean_rows.append(dataset_system_means(qps_df, recall_df, family))
        algo_labels = [family["label_map"][algo] for algo in family["algos"]]

        for col, dataset in enumerate(DATASETS):
            title_prefix = f"{family['name']} - " if col == 0 else ""
            title = title_prefix + TITLE_MAP.get(dataset, dataset.upper())
            plot_qps(
                axes[row, col],
                qps_df,
                dataset,
                algo_labels,
                family["palette"],
                title,
                family["qps_y_limits"],
                show_ylabel=(col == 0),
                y_ticks=family.get("qps_y_ticks"),
            )
            plot_recall(
                axes[row, col + 3],
                recall_df,
                dataset,
                algo_labels,
                family["palette"],
                title,
                show_ylabel=(col == 0),
                add_inset=family.get("recall_inset", False),
            )

    legend_handles = []
    legend_labels = []
    seen_labels = set()
    for row in range(len(families)):
        handles, labels = axes[row, 0].get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in seen_labels:
                legend_handles.append(handle)
                legend_labels.append(label)
                seen_labels.add(label)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=6,
            frameon=True,
            fancybox=False,
            facecolor="#f0f0f0",
            edgecolor="#bdbdbd",
            framealpha=1.0,
            bbox_to_anchor=(0.5, 0.990),
            fontsize=15,
            columnspacing=1.25,
            handlelength=2.0,
        )

    fig.text(0.275, 0.905, "Query QPS", ha="center", va="top", fontsize=22, fontweight="semibold")
    fig.text(0.78, 0.905, "Recall", ha="center", va="top", fontsize=22, fontweight="semibold")
    plt.tight_layout(rect=(0, 0, 1, 0.885), h_pad=0.8, w_pad=0.8)

    # Add a small visual gutter between the QPS block and the Recall block.
    for row in range(axes.shape[0]):
        for col in range(3, axes.shape[1]):
            pos = axes[row, col].get_position()
            axes[row, col].set_position([pos.x0 + 0.016, pos.y0, pos.width, pos.height])

    out_file = "qps_recall_all.png"
    plt.savefig(out_file, dpi=400, bbox_inches="tight", pad_inches=0.08)
    plt.savefig("qps_recall_all.pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Plot saved to {out_file}")

    dataset_means = pd.concat(dataset_mean_rows, ignore_index=True)
    system_summary, pair_summary = summarize_system_metrics(dataset_means)
    dataset_means.to_csv("qps_recall_all_dataset_means.csv", index=False)
    system_summary.to_csv("qps_recall_all_system_summary.csv", index=False)
    pair_summary.to_csv("qps_recall_all_pair_diffs.csv", index=False)

    print("\nPer-system averages across the three datasets (dataset means averaged equally):")
    print(
        system_summary.sort_values(["family", "algo_label"])[
            ["family", "algo_label", "datasets", "avg_query_qps", "avg_recall"]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )

    if not pair_summary.empty:
        print("\nMax BriskSeed-vs-baseline differences across datasets:")
        print(
            pair_summary.sort_values(["family", "metric"])[
                ["family", "metric", "dataset", "baseline", "briskseed", "max_abs_diff", "max_abs_diff_abs"]
            ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
        )


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
