import argparse
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, gridspec
from matplotlib.colors import (LinearSegmentedColormap, Normalize, SymLogNorm,
                               TwoSlopeNorm, to_hex)
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


def load_and_clean(
    path: str, diff_round: Optional[int]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
    df = pd.read_csv(path)

    diff_col = None
    for c in df.columns:
        if "diff" in c.lower():
            diff_col = c
            break
    if diff_col is None:
        raise ValueError(
            "Missing diff column (need 'diff' or 'diff (baseline - partial)')."
        )

    if "stage_offset" not in df.columns:
        raise ValueError("Missing 'stage_offset' column.")
    if "count" not in df.columns:
        for alt in ["n", "cnt", "count_all"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "count"})
                break
    if "count" not in df.columns:
        raise ValueError("Missing 'count' column.")

    df = df.rename(columns={diff_col: "diff"})
    df["stage_offset"] = pd.to_numeric(df["stage_offset"], errors="coerce")
    df["diff"] = pd.to_numeric(df["diff"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["stage_offset", "diff"]).copy()

    # Overall recall (before filtering diff!=0)
    overall_recall_wlock = 0.0
    overall_recall_wolock = 0.0
    if "avg_recall_base" in df.columns and "avg_recall_partial" in df.columns:
        base_all = pd.to_numeric(df["avg_recall_base"], errors="coerce").dropna()
        part_all = pd.to_numeric(df["avg_recall_partial"], errors="coerce").dropna()
        if len(base_all) > 0:
            overall_recall_wlock = float(base_all.mean())
        if len(part_all) > 0:
            overall_recall_wolock = float(part_all.mean())

    df = df[df["diff"] != 0].copy()

    if df.empty:
        print("Warning: No non-zero diffs found. Plotting empty diff distribution.")
        dummy = pd.DataFrame(
            {
                "stage_offset": [0],
                "diff": [0],
                "count": [0],
                "diff_bin": [0],
                "pct_within_stage": [0],
                "sign": ["zero"],
            }
        )
        dummy_summary = pd.DataFrame(
            {
                "stage_offset": [0],
                "stage_count": [0],
                "count_pos": [0],
                "count_neg": [0],
            }
        )
        dummy_pivot = pd.DataFrame({0: [0]}, index=[0])
        return (dummy, dummy_summary, dummy_pivot, 0.0, 0.0, 0.0, 0.0)

    if diff_round is None:
        df["diff_bin"] = df["diff"]
    else:
        df["diff_bin"] = df["diff"].round(diff_round)

    stage_totals = df.groupby("stage_offset")["count"].sum().replace(0, np.nan)
    df["pct_within_stage"] = df.apply(
        lambda r: (
            (r["count"] / stage_totals.loc[r["stage_offset"]] * 100.0)
            if pd.notna(stage_totals.loc[r["stage_offset"]])
            else 0.0
        ),
        axis=1,
    )

    stage_volume = stage_totals.rename("stage_count").reset_index()
    conditions = [df["diff"] > 0, df["diff"] < 0]
    choices = ["pos", "neg"]
    df["sign"] = np.select(conditions, choices, default="zero")

    volume_split = (
        df.groupby(["stage_offset", "sign"])["count"].sum().unstack(fill_value=0)
    )

    for col in ["pos", "neg"]:
        if col not in volume_split.columns:
            volume_split[col] = 0

    volume_split = volume_split.rename(
        columns={"pos": "count_pos", "neg": "count_neg"}
    ).reset_index()
    summary = stage_volume.merge(volume_split, on="stage_offset", how="left").fillna(0)

    pivot = (
        df.groupby(["stage_offset", "diff_bin"])["pct_within_stage"].sum().reset_index()
    )
    pivot = pivot.pivot(
        index="stage_offset", columns="diff_bin", values="pct_within_stage"
    ).fillna(0.0)
    # Ensure sorted columns (x axis)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    avg_pos_diff = df[df["diff"] > 0]["diff"].mean() if not df.empty else 0.0
    avg_neg_diff = df[df["diff"] < 0][
        "diff"
    ].mean()  # Calculate weighted average recall for diff != 0
    avg_recall_wlock = 0.0
    avg_recall_wolock = 0.0

    if "avg_recall_base" in df.columns and "avg_recall_partial" in df.columns:
        total_count = df["count"].sum()
        if total_count > 0:
            avg_recall_wlock = (df["avg_recall_base"] * df["count"]).sum() / total_count
            avg_recall_wolock = (
                df["avg_recall_partial"] * df["count"]
            ).sum() / total_count

    # Handle NaN
    if pd.isna(avg_pos_diff):
        avg_pos_diff = 0.0
    if pd.isna(avg_neg_diff):
        avg_neg_diff = 0.0
    if pd.isna(avg_recall_wlock):
        avg_recall_wlock = 0.0
    if pd.isna(avg_recall_wolock):
        avg_recall_wolock = 0.0

    return (
        df,
        summary,
        pivot,
        avg_pos_diff,
        avg_neg_diff,
        avg_recall_wlock,
        avg_recall_wolock,
        overall_recall_wlock,
        overall_recall_wolock,
    )


def downsample_rows(
    summary: pd.DataFrame,
    pivot: pd.DataFrame,
    max_rows: int = None,
    height: float = 12.0,
    dpi: int = 300,
):
    n_rows = summary.shape[0]
    if max_rows is None:
        max_rows = max(400, int(height * dpi * 0.4))
    if n_rows <= max_rows:
        return summary, pivot

    stride = int(np.ceil(n_rows / max_rows))
    idx = list(range(0, n_rows, stride))
    if idx[-1] != n_rows - 1:
        idx.append(n_rows - 1)

    sampled_summary = summary.iloc[idx].reset_index(drop=True)
    sampled_pivot = pivot.iloc[idx].copy()
    sampled_pivot.index = sampled_summary["stage_offset"].to_numpy()

    return sampled_summary, sampled_pivot


def select_scale(values: np.ndarray) -> Tuple[float, str]:
    if values.size == 0:
        return 1.0, ""
    max_abs = float(np.max(np.abs(values)))
    scale_options = [
        (1e9, 1e9, "×1e9"),
        (1e8, 1e8, "×1e8"),
        (1e7, 1e7, "×1e7"),
        (1e6, 1e6, "×1e6"),
        (1e5, 1e5, "×1e5"),
        (1e4, 1e4, "×1e4"),
        (1e3, 1e3, "×1e3"),
    ]
    for threshold, factor, label in scale_options:
        if max_abs >= threshold:
            return factor, label
    return 1.0, ""


def format_stage_label(value: float) -> str:
    value_abs = abs(value)
    if value_abs >= 100.0:
        text = f"{value:.0f}"
    elif value_abs >= 10.0:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def build_single_figure(
    summary: pd.DataFrame,
    pivot: pd.DataFrame,
    avg_pos_diff: float,
    avg_neg_diff: float,
    avg_recall_wlock: float,
    avg_recall_wolock: float,
    overall_recall_wlock: float,
    overall_recall_wolock: float,
    sort_by: str = "stage",
    height: float = 12.0,
    dpi: int = 300,
    out: Optional[str] = None,
):
    if sort_by == "volume":
        order = summary.sort_values("stage_count")["stage_offset"].tolist()
    else:  # "stage": numeric ascending
        order = summary.sort_values("stage_offset")["stage_offset"].tolist()

    pivot_ord = pivot.reindex(order)
    summary_ord = summary.set_index("stage_offset").loc[order].reset_index()

    summary_ord, pivot_ord = downsample_rows(summary_ord, pivot_ord)
    order = summary_ord["stage_offset"].tolist()

    n_rows = pivot_ord.shape[0]
    n_cols = pivot_ord.shape[1]

    if n_rows == 0 or n_cols == 0:
        raise ValueError("No data to plot after cleaning; check your CSV.")

    stage_values = summary_ord["stage_offset"].to_numpy(dtype=float)
    vol_pos = summary_ord.get("count_pos", 0.0).to_numpy(dtype=float)
    vol_neg = summary_ord.get("count_neg", 0.0).to_numpy(dtype=float)
    diff_values = np.array(pivot_ord.columns, dtype=float)

    def _tick_positions(length: int, max_ticks: int) -> np.ndarray:
        if length == 0:
            return np.array([])
        step = max(1, int(np.ceil(length / max_ticks)))
        ticks = list(range(0, length, step))
        if ticks[-1] != length - 1:
            ticks.append(length - 1)
        return np.array(sorted(set(ticks)))

    def _select_scale(values: np.ndarray) -> Tuple[float, str]:
        if values.size == 0:
            return 1.0, ""
        max_abs = float(np.max(np.abs(values)))
        scale_options = [
            (1e9, 1e9, "×1e9"),
            (1e8, 1e8, "×1e8"),
            (1e7, 1e7, "×1e7"),
            (1e6, 1e6, "×1e6"),
            (1e5, 1e5, "×1e5"),
            (1e4, 1e4, "×1e4"),
            (1e3, 1e3, "×1e3"),
        ]
        for threshold, factor, label in scale_options:
            if max_abs >= threshold:
                return factor, label
        return 1.0, ""

    def _format_stage_label(value: float) -> str:
        value_abs = abs(value)
        if value_abs >= 100.0:
            text = f"{value:.0f}"
        elif value_abs >= 10.0:
            text = f"{value:.1f}"
        else:
            text = f"{value:.2f}"
        return text.rstrip("0").rstrip(".")

    scale_factor, scale_suffix = _select_scale(stage_values)
    stage_positions = stage_values / scale_factor if scale_factor != 0 else stage_values
    stage_labels = [_format_stage_label(v) for v in stage_positions]

    width = max(10.0, min(18.0, 7.0 + n_rows * 0.12))
    style_overrides = {
        "axes.facecolor": "#f0f2f5",
        "axes.edgecolor": "#6b6f75",
        "axes.labelsize": 26,
        "axes.titlesize": 24,
        "figure.facecolor": "#ffffff",
        "font.size": 20,
        "text.color": "#1f1f1f",
        "xtick.color": "#2f2f2f",
        "ytick.color": "#2f2f2f",
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
    }
    diff_min = float(np.min(diff_values)) if diff_values.size > 0 else -1.0
    diff_max = float(np.max(diff_values)) if diff_values.size > 0 else 1.0
    norm_min, norm_max = diff_min, diff_max
    if np.isclose(norm_min, norm_max):
        spread = abs(norm_min) * 0.1
        if spread < 1e-6:
            spread = 1.0
        norm_min -= spread
        norm_max += spread

    if norm_min < 0.0 and norm_max > 0.0:
        linthresh = max(1e-5, 0.01 * max(abs(norm_min), abs(norm_max)))
        norm: Normalize = SymLogNorm(
            linthresh=linthresh,
            linscale=0.3,
            vmin=norm_min,
            vmax=norm_max,
            base=10,
        )
    else:
        norm = Normalize(vmin=norm_min, vmax=norm_max)

    cmap = LinearSegmentedColormap.from_list(
        "diff_contrast",
        [
            "#0b1d2b",
            "#184e77",
            "#e1f7ee",
            "#f9dfc7",
            "#c92c20",
        ],
    )
    pos_color = "#d14900"
    neg_color = "#005f99"

    with plt.rc_context(style_overrides):
        dpi_effective = max(1, int(dpi * 4))
        fig = plt.figure(figsize=(width, height), dpi=dpi_effective)
        gs = gridspec.GridSpec(
            nrows=2,
            ncols=1,
            height_ratios=[3.5, 5.0],
            hspace=0.08,
        )
        ax_top = fig.add_subplot(gs[0])
        ax_mid = fig.add_subplot(gs[1], sharex=ax_top)

        x = stage_positions
        vol_pos = np.where(np.isnan(vol_pos), 0.0, vol_pos)
        vol_neg = np.where(np.isnan(vol_neg), 0.0, vol_neg)

        if stage_positions.size > 1:
            spacing = np.diff(stage_positions)
            positive_spacing = spacing[spacing > 0]
            min_spacing = (
                float(np.min(positive_spacing))
                if positive_spacing.size > 0
                else float(spacing[0])
            )
        else:
            min_spacing = 1.0
        if min_spacing <= 0 or not np.isfinite(min_spacing):
            min_spacing = 1.0
        bar_width = min_spacing * 0.75

        total_pos = float(np.nansum(vol_pos))
        total_neg = float(np.nansum(vol_neg))
        total_all = total_pos + total_neg

        pct_pos = total_pos / total_all * 100.0 if total_all > 0 else 0.0
        pct_neg = total_neg / total_all * 100.0 if total_all > 0 else 0.0

        bar_core_width = float(
            np.clip(bar_width * 0.45, min_spacing * 0.15, min_spacing * 0.4)
        )

        pos_bars = ax_top.bar(
            x,
            vol_pos,
            width=bar_core_width,
            align="center",
            color=pos_color,
            edgecolor=pos_color,
            linewidth=1.2,
            alpha=0.9,
            label=f"wLock Higher: {pct_pos:.1f}%",
        )
        neg_bars = ax_top.bar(
            x,
            -vol_neg,
            width=bar_core_width,
            align="center",
            color=neg_color,
            edgecolor=neg_color,
            linewidth=1.2,
            alpha=0.9,
            label=f"woLock Higher: {pct_neg:.1f}%",
        )
        ymax = float(
            max(vol_pos.max(initial=0.0), vol_neg.max(initial=0.0))
            if vol_pos.size
            else 0.0
        )
        if ymax <= 0.0:
            ymax = 1.0
        ax_top.set_ylim(-ymax * 1.2, ymax * 1.2)
        ax_top.axhline(0.0, color="#7f7f7f", linewidth=1.3, linestyle="--", alpha=0.6)
        ax_top.set_ylabel("Recall Diff Count")
        ax_top.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5, color="#cbd2dc")
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.tick_params(axis="y", labelsize=28)
        ax_top.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{abs(val):g}"))
        ax_top.legend(
            loc="lower right",
            fontsize=20,
            frameon=False,
            ncol=2,
            columnspacing=0.8,
            handlelength=1.2,
        )

        bin_counts_per_stage = pivot_ord.astype(bool).sum(axis=1)
        max_bins_per_stage = (
            int(bin_counts_per_stage.max()) if not bin_counts_per_stage.empty else 1
        )
        if max_bins_per_stage < 1:
            max_bins_per_stage = 1
        bin_width = bar_width / max(1, max_bins_per_stage)

        ax_mid.axhline(0.0, color="#7f7f7f", linewidth=1.0, linestyle="--", alpha=0.7)

        for idx, stage in enumerate(order):
            row = pivot_ord.loc[stage]
            nonzero = row[row != 0.0]
            if nonzero.empty:
                continue
            m = len(nonzero)
            if m == 1:
                offsets = np.array([0.0])
            else:
                offsets = np.linspace(-(m - 1) / 2.0, (m - 1) / 2.0, m) * bin_width
            for offset, (diff_val, pct) in zip(offsets, nonzero.items()):
                bar_width_local = float(
                    np.clip(bin_width * 0.7, min_spacing * 0.1, min_spacing * 0.75)
                )
                if pd.isna(pct):
                    pct = 0.0
                alpha = float(np.clip(0.55 + (pct / 100.0) * 0.45, 0.55, 1.0))
                color = to_hex(cmap(norm(diff_val)))
                ax_mid.bar(
                    x[idx] + offset,
                    diff_val,
                    width=bar_width_local,
                    color=color,
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=alpha,
                    align="center",
                )

        if diff_min < 0.0 and diff_max > 0.0:
            max_abs = max(abs(diff_min), abs(diff_max))
            ax_mid.set_ylim(-max_abs * 1.1, max_abs * 1.1)
        else:
            lower = diff_min * 1.05 if diff_min < 0 else diff_min * 0.95
            upper = diff_max * 1.05 if diff_max > 0 else diff_max * 0.95
            if np.isclose(lower, upper):
                upper = lower + 1.0
            ax_mid.set_ylim(lower, upper)

        if stage_positions.size > 1:
            span = stage_positions[-1] - stage_positions[0]
            if span > 0:
                tick_positions = np.linspace(
                    stage_positions[0], stage_positions[-1], num=10
                )
            else:
                tick_positions = stage_positions
        else:
            tick_positions = stage_positions
        ax_mid.set_xticks(tick_positions)
        ax_mid.set_xticklabels(
            [_format_stage_label(v) for v in tick_positions],
            rotation=0,
            ha="center",
        )
        ax_mid.set_ylabel("Recall Diff (wLock - woLock)")
        legend_elements = [
            Patch(
                facecolor="#c92c20",
                edgecolor="none",
                label=f"wLock Higher (Avg diff-set: {avg_recall_wlock:.4f}, overall: {overall_recall_wlock:.4f})",
            ),
            Patch(
                facecolor="#005f99",
                edgecolor="none",
                label=f"woLock Higher (Avg diff-set: {avg_recall_wolock:.4f}, overall: {overall_recall_wolock:.4f})",
            ),
        ]
        ax_mid.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=20,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
        )
        if scale_suffix:
            ax_mid.set_xlabel(f"Insert Offset ({scale_suffix})")
        else:
            ax_mid.set_xlabel("Insert Offset")
        ax_mid.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5, color="#cbd2dc")
        ax_mid.tick_params(axis="x", labelsize=26)
        ax_mid.tick_params(axis="y", labelsize=26)
        label_coord = -0.08
        ax_top.yaxis.set_label_coords(label_coord, 0.5)
        ax_mid.yaxis.set_label_coords(label_coord, 0.5)

        left_pad = (
            stage_positions[0] - min_spacing * 0.5 if stage_positions.size > 0 else -0.5
        )
        right_pad = (
            stage_positions[-1] + min_spacing * 0.5
            if stage_positions.size > 0
            else n_rows - 0.5
        )
        ax_mid.set_xlim(left_pad, right_pad)
        ax_top.set_xlim(left_pad, right_pad)

        for ax in (ax_top, ax_mid):
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#1a1a1a")
                spine.set_linewidth(2.2)

        fig.subplots_adjust(left=0.09, right=0.96, bottom=0.18, top=0.95, hspace=0.12)

        if out:
            fig.savefig(out, bbox_inches="tight")
            print(f"saved figure -> {out}")
        else:
            plt.show()
        plt.close(fig)


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0 or weights.size == 0:
        return np.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cum_weights = np.cumsum(weights)
    cutoff = q * weights.sum()
    idx = np.searchsorted(cum_weights, cutoff, side="left")
    idx = min(idx, len(values) - 1)
    return float(values[idx])


def auto_bin_size(values: np.ndarray, max_bins: int = 150) -> float:
    if values.size == 0:
        return 0.0
    vmin, vmax = float(values.min()), float(values.max())
    span = vmax - vmin
    if span <= 0:
        return 0.0
    bin_size = span / max_bins
    mag = 10 ** np.floor(np.log10(bin_size))
    for m in [1, 2, 5, 10]:
        candidate = m * mag
        if candidate >= bin_size:
            return candidate
    return bin_size


def derive_out_path(base: Optional[str], suffix: str) -> Optional[str]:
    if base is None:
        return None
    lower = base.lower()
    if lower.endswith(".png") or lower.endswith(".pdf"):
        return base[:-4] + suffix + base[-4:]
    return base + suffix


def plot_wo_future_diff(pure_df: pd.DataFrame, out_path: Optional[str]) -> None:
    if pure_df is None or pure_df.empty or out_path is None:
        return

    summary = (
        pure_df.groupby("stage_offset")
        .apply(
            lambda x: pd.Series(
                {"avg_diff": (x["diff"] * x["count"]).sum() / x["count"].sum()}
            )
        )
        .reset_index()
    )
    if summary.empty:
        print("wo_future diff: no data to plot.")
        return

    total_count = pure_df["count"].sum()
    avg_recall_wlock = (
        pure_df["avg_recall_base"] * pure_df["count"]
    ).sum() / total_count
    avg_recall_wolock = (
        pure_df["avg_recall_partial"] * pure_df["count"]
    ).sum() / total_count

    style_overrides = {
        "axes.facecolor": "#f0f2f5",
        "axes.edgecolor": "#6b6f75",
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "figure.facecolor": "#ffffff",
        "font.size": 17,
        "text.color": "#1f1f1f",
        "xtick.color": "#2f2f2f",
        "ytick.color": "#2f2f2f",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }

    width = 12.0
    height = 6.0

    with plt.rc_context(style_overrides):
        fig, ax = plt.subplots(figsize=(width, height))

        stages = summary["stage_offset"].to_numpy(dtype=float)
        diffs = summary["avg_diff"].to_numpy()

        scale_factor, scale_suffix = select_scale(stages)
        stage_positions = stages / scale_factor if scale_factor != 0 else stages

        if len(stage_positions) > 1:
            min_step = np.min(np.diff(np.sort(np.unique(stage_positions))))
            bar_width = min_step * 0.8
        else:
            bar_width = 100.0 / scale_factor if scale_factor != 0 else 100.0

        colors = np.where(diffs > 0, "#c92c20", "#005f99")

        ax.bar(stage_positions, diffs, color=colors, alpha=0.7, width=bar_width)

        legend_elements = [
            Patch(
                facecolor="#c92c20", label=f"wLock Higher (Avg: {avg_recall_wlock:.4f})"
            ),
            Patch(
                facecolor="#005f99",
                label=f"woLock Higher (Avg: {avg_recall_wolock:.4f})",
            ),
        ]
        ax.legend(handles=legend_elements, fontsize=18)

        if scale_suffix:
            ax.set_xlabel(f"Insert Offset ({scale_suffix})")
        else:
            ax.set_xlabel("Insert Offset")

        ax.set_ylabel("Recall Diff (wLock - woLock)")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8, color="#cbd2dc")

        if len(stage_positions) > 0:
            span = stage_positions[-1] - stage_positions[0]
            if span > 0:
                tick_positions = np.linspace(
                    stage_positions[0], stage_positions[-1], num=10
                )
            else:
                tick_positions = stage_positions

            ax.set_xticks(tick_positions)
            ax.set_xticklabels([format_stage_label(v) for v in tick_positions])

        fig.savefig(out_path, bbox_inches="tight", dpi=100)
        print(f"saved wo_future diff -> {out_path}")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a combined stage volume + diff distribution figure."
    )
    parser.add_argument(
        "--csv", required=True, help="Input CSV with stage/diff/count data."
    )
    parser.add_argument(
        "--round",
        dest="diff_round",
        type=int,
        default=None,
        help="Round diffs to this many decimals (default: no rounding).",
    )
    parser.add_argument(
        "--sort-by",
        choices=["stage", "volume"],
        default="stage",
        help="Row ordering strategy.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=12.0,
        help="Figure height in inches (default: 12).",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Figure DPI (default: 300)."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path for the PNG (otherwise shows interactively).",
    )
    parser.add_argument(
        "--no-main",
        action="store_true",
        help="Disable main diff plot generation.",
    )
    parser.add_argument(
        "--no-wo-future",
        action="store_true",
        help="Disable wo_future diff plot generation (requires avg_future_tags column).",
    )
    parser.add_argument(
        "--out-wo-future",
        default=None,
        help="Output path for wo_future diff plot (default: <out> with _wo_future suffix).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    (
        df,
        summary,
        pivot,
        avg_pos_diff,
        avg_neg_diff,
        avg_recall_wlock,
        avg_recall_wolock,
        overall_recall_wlock,
        overall_recall_wolock,
    ) = load_and_clean(args.csv, args.diff_round)
    base_out = args.out
    if base_out is None:
        if args.csv.lower().endswith(".csv"):
            base_out = args.csv[:-4] + ".png"
        else:
            base_out = args.csv + ".png"

    if not args.no_main:
        build_single_figure(
            summary=summary,
            pivot=pivot,
            avg_pos_diff=avg_pos_diff,
            avg_neg_diff=avg_neg_diff,
            avg_recall_wlock=avg_recall_wlock,
            avg_recall_wolock=avg_recall_wolock,
            overall_recall_wlock=overall_recall_wlock,
            overall_recall_wolock=overall_recall_wolock,
            sort_by=args.sort_by,
            height=args.height,
            dpi=args.dpi,
            out=base_out,
        )

    if not args.no_wo_future and "avg_future_tags" in df.columns:
        pure_df = df[(df["avg_future_tags"] == 0) & (df["diff"] != 0)].copy()
        if pure_df.empty:
            print("wo_future: no rows (avg_future_tags==0 and diff!=0); skipping.")
        else:
            out_wo = args.out_wo_future or derive_out_path(base_out, "_wo_future")
            plot_wo_future_diff(pure_df, out_wo)
    elif not args.no_wo_future:
        print("wo_future: missing avg_future_tags column; skipping.")


if __name__ == "__main__":
    main()
