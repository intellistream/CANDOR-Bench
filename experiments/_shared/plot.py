"""Generate paper-style figures from output_*.json across experiments.

Figures produced (PNG + PDF in experiments/figures/):
- fig_e15_main.png         : main matrix (datasets × patterns) bar of relative Δtime
- fig_e16_cross_dataset.png: gamma vs hnswlib on each dataset/pattern
- fig_e17_churn_curve.png  : Δtime% vs churn ratio per pattern
- fig_e18_buffer_curve.png : total time vs buf_mult per pattern
- fig_e19_param_grid.png   : gamma vs hnswlib param-grid Pareto

Run: python experiments/_shared/plot.py
"""
import os, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
EXP_DIR = REPO / "experiments"
FIG_DIR = EXP_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fig_e15_main():
    """Bar chart: gamma vs hnswlib direct relative Δtime per pattern, 200K and 1M."""
    rows_200K = []
    p_200K = EXP_DIR / "e15_delete_pattern_matrix" / "output_200K.json"
    if p_200K.exists():
        rows_200K = load_json(p_200K)
    rows_1M = []
    for p in (EXP_DIR / "e15_delete_pattern_matrix").glob("output_1M*.json"):
        rows_1M.extend(load_json(p))
    if not (rows_200K or rows_1M):
        return

    patterns = ["sequential", "random", "cluster", "partial_reset"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, scale, rows in [(axes[0], "200K", rows_200K), (axes[1], "1M", rows_1M)]:
        deltas = []
        for pat in patterns:
            g = next((r for r in rows if r.get("pattern") == pat
                      and r.get("name") == "gamma_v2+hnswlib"), None)
            d = next((r for r in rows if r.get("pattern") == pat
                      and r.get("name") == "hnswlib direct"), None)
            if g and d and d["total_s"] > 0:
                pct = (g["total_s"] - d["total_s"]) / d["total_s"] * 100
                deltas.append(pct)
            else:
                deltas.append(None)
        xs = np.arange(len(patterns))
        ys = [d if d is not None else 0 for d in deltas]
        colors = ["#5b8def" if d is not None and d < 0 else "#e07b5b" for d in deltas]
        ax.bar(xs, ys, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(patterns, rotation=20)
        ax.set_title(f"SIFT {scale}")
        ax.set_ylabel("Δ time gamma vs hnswlib direct (%)")
        for x, y in zip(xs, ys):
            if y != 0:
                ax.text(x, y, f"{y:+.0f}%", ha="center",
                        va="bottom" if y > 0 else "top")
    fig.suptitle("e15 — gamma_v2+hnswlib vs hnswlib direct (lower = gamma faster)")
    fig.tight_layout()
    out = FIG_DIR / "fig_e15_main.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def fig_e16_cross_dataset():
    files = list((EXP_DIR / "e16_cross_dataset").glob("output_*.json"))
    if not files:
        return
    by_ds = defaultdict(list)
    for p in files:
        for r in load_json(p):
            by_ds[(r.get("dataset"), r.get("scale"))].append(r)
    if not by_ds:
        return
    patterns = ["sequential", "random", "cluster", "partial_reset"]
    keys = sorted(by_ds, key=lambda k: (k[0], k[1]))
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        rows = by_ds[k]
        deltas = []
        for pat in patterns:
            g = next((r for r in rows if r.get("pattern") == pat
                      and r.get("name") == "gamma_v2+hnswlib"), None)
            d = next((r for r in rows if r.get("pattern") == pat
                      and r.get("name") == "hnswlib direct"), None)
            if g and d and d["total_s"] > 0:
                deltas.append((g["total_s"] - d["total_s"]) / d["total_s"] * 100)
            else:
                deltas.append(None)
        xs = np.arange(len(patterns))
        ys = [d if d is not None else 0 for d in deltas]
        colors = ["#5b8def" if d is not None and d < 0 else "#e07b5b" for d in deltas]
        ax.bar(xs, ys, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(patterns, rotation=20, fontsize=8)
        ax.set_title(f"{k[0]} {k[1]}")
        for x, y in zip(xs, ys):
            if y != 0:
                ax.text(x, y, f"{y:+.0f}%", ha="center",
                        va="bottom" if y > 0 else "top", fontsize=8)
    axes[0].set_ylabel("Δ time vs hnswlib direct (%)")
    fig.suptitle("e16 — cross-dataset gamma_v2+hnswlib vs hnswlib direct")
    fig.tight_layout()
    out = FIG_DIR / "fig_e16_cross_dataset.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def fig_e17_churn():
    files = list((EXP_DIR / "e17_churn_rate_sweep").glob("output_*.json"))
    if not files:
        return
    by_pat = defaultdict(list)
    for p in files:
        for r in load_json(p):
            by_pat[r.get("pattern")].append(r)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = {"sequential": "#888", "random": "#5b8def", "cluster": "#e07b5b",
               "partial_reset": "#7bbe70"}
    for pat in ["sequential", "random", "cluster", "partial_reset"]:
        rows = by_pat.get(pat, [])
        ratios = sorted(set(r.get("ratio") for r in rows
                            if r.get("ratio") is not None and r.get("ratio") != ""))
        deltas = []
        for ratio in ratios:
            g = next((r for r in rows if r.get("ratio") == ratio
                      and r.get("name") == "gamma_v2+hnswlib"), None)
            d = next((r for r in rows if r.get("ratio") == ratio
                      and r.get("name") == "hnswlib direct"), None)
            if g and d and d["total_s"] > 0:
                deltas.append((g["total_s"] - d["total_s"]) / d["total_s"] * 100)
            else:
                deltas.append(None)
        xs = [r for r, d in zip(ratios, deltas) if d is not None]
        ys = [d for d in deltas if d is not None]
        if xs:
            ax.plot(xs, ys, marker="o", label=pat, color=palette[pat])
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("delete-per-batch / insert-batch ratio")
    ax.set_ylabel("Δ time gamma vs hnswlib direct (%)")
    ax.set_title("e17 — churn-rate sweep (SIFT 200K, hnswlib backend)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_e17_churn_curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def fig_e18_buffer():
    files = list((EXP_DIR / "e18_buffer_capacity").glob("output_*.json"))
    if not files:
        return
    by_pat = defaultdict(list)
    for p in files:
        for r in load_json(p):
            by_pat[r.get("pattern")].append(r)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    for ax, pat in zip(axes, ["sequential", "random", "cluster", "partial_reset"]):
        rows = by_pat.get(pat, [])
        baseline = next((r["total_s"] for r in rows
                         if r.get("buf_mult") in (None, "")
                         and r.get("name") == "hnswlib direct"), None)
        gamma = sorted([r for r in rows if isinstance(r.get("buf_mult"), int)],
                       key=lambda r: r["buf_mult"])
        if gamma:
            xs = [r["buf_mult"] for r in gamma]
            ys = [r["total_s"] for r in gamma]
            ax.plot(xs, ys, marker="o", label="gamma_v2+hnswlib", color="#5b8def")
            ax.set_xscale("log")
        if baseline is not None:
            ax.axhline(baseline, color="#e07b5b", linestyle="--",
                       label="hnswlib direct")
        ax.set_xlabel("buf capacity / batch")
        ax.set_title(pat)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("total time (s)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("e18 — buffer-capacity ablation (SIFT 200K)")
    fig.tight_layout()
    out = FIG_DIR / "fig_e18_buffer_curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def fig_e19_pareto():
    files = list((EXP_DIR / "e19_hnswlib_param_sensitivity").glob("output_*.json"))
    if not files:
        return
    by_pat = defaultdict(list)
    for p in files:
        for r in load_json(p):
            by_pat[r.get("pattern")].append(r)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False)
    for ax, pat in zip(axes, ["sequential", "random", "cluster", "partial_reset"]):
        rows = by_pat.get(pat, [])
        gamma = [r for r in rows if r.get("is_gamma")]
        direct = [r for r in rows if not r.get("is_gamma")]
        if direct:
            ax.scatter([r["recall"] for r in direct],
                       [r["total_s"] for r in direct],
                       color="#e07b5b", alpha=0.8, s=40,
                       label=f"hnswlib direct ({len(direct)} configs)")
        if gamma:
            ax.scatter([r["recall"] for r in gamma],
                       [r["total_s"] for r in gamma],
                       color="#5b8def", marker="*", s=200,
                       label="gamma_v2+hnswlib (default)")
        ax.set_xlabel("recall@10")
        ax.set_title(pat)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("total time (s)")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("e19 — gamma vs hnswlib param-grid Pareto (SIFT 200K)")
    fig.tight_layout()
    out = FIG_DIR / "fig_e19_param_grid.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def fig_e20_ablation():
    """Bar chart per pattern: baseline vs each variant time."""
    files = list((EXP_DIR / "e20_router_component_ablation").glob("output_sift_*_200K.json"))
    if not files:
        return
    by_pat = {}
    for p in files:
        rows = load_json(p)
        if rows:
            by_pat[rows[0]["pattern"]] = rows
    if not by_pat:
        return
    patterns = ["sequential", "random", "cluster", "partial_reset"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=False)
    variant_order = ["(hnswlib direct)", "full", "no_buffer_scan",
                      "no_buffer_inserts", "no_absorb_deletes", "eager_maint"]
    colors = {"(hnswlib direct)": "#888",
              "full": "#5b8def",
              "no_buffer_scan": "#7bbe70",
              "no_buffer_inserts": "#e0a85b",
              "no_absorb_deletes": "#e07b5b",
              "eager_maint": "#a05bbe"}
    for ax, pat in zip(axes, patterns):
        rows = by_pat.get(pat, [])
        ts, recs, labels = [], [], []
        for v in variant_order:
            r = next((r for r in rows if r.get("variant") == v), None)
            if r:
                ts.append(r["total_s"])
                recs.append(r["recall"])
                labels.append(v)
        if not ts:
            ax.set_title(f"{pat} (no data)")
            continue
        xs = np.arange(len(ts))
        ax.bar(xs, ts, color=[colors.get(l, "#999") for l in labels])
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, fontsize=8, ha="right")
        ax.set_title(f"{pat}")
        for x, y, r in zip(xs, ts, recs):
            ax.text(x, y, f"{y:.0f}s\n{r:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("total time (s)")
    fig.suptitle("e20 — router-component ablation (SIFT 200K, hnswlib backend)")
    fig.tight_layout()
    out = FIG_DIR / "fig_e20_ablation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def fig_e21_cpp_ablation():
    files = list((EXP_DIR / "e21_cpp_component_ablation").glob("output_sift_*_200K.json"))
    if not files:
        return
    by_pat = {}
    for p in files:
        rows = load_json(p)
        if rows:
            by_pat[rows[0]["pattern"]] = rows
    if not by_pat:
        return
    patterns = ["sequential", "random", "cluster", "partial_reset"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=False)
    for ax, pat in zip(axes, patterns):
        rows = by_pat.get(pat, [])
        if not rows:
            ax.set_title(f"{pat} (no data)")
            continue
        labels = [r["variant"] for r in rows]
        ts = [r["total_s"] for r in rows]
        recs = [r["recall"] for r in rows]
        xs = np.arange(len(ts))
        ax.bar(xs, ts, color="#5b8def")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, fontsize=7, ha="right")
        ax.set_title(f"{pat}")
        for x, y, r in zip(xs, ts, recs):
            ax.text(x, y, f"{y:.0f}s\nr={r:.3f}", ha="center", va="bottom", fontsize=6)
    axes[0].set_ylabel("total time (s)")
    fig.suptitle("e21 — C++ component ablation (gamma_cpp[variant], SIFT 200K)")
    fig.tight_layout()
    out = FIG_DIR / "fig_e21_cpp_ablation.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


def main():
    fig_e15_main()
    fig_e16_cross_dataset()
    fig_e17_churn()
    fig_e18_buffer()
    fig_e19_pareto()
    fig_e20_ablation()
    fig_e21_cpp_ablation()


if __name__ == "__main__":
    main()
