#!/usr/bin/env python3
"""Plot fixed-worker closed-loop M3 tail results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def f(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def label(group: str, efc: str, mutation: str) -> str:
    if group == "bg_search":
        return "bg search"
    if mutation == "skip_existing_update":
        return "insert efc=35 skip-update"
    return f"insert efc={efc}"


def color(name: str) -> str:
    return {
        "bg search": "#475569",
        "insert efc=200": "#ea580c",
        "insert efc=35": "#dc2626",
        "insert efc=35 skip-update": "#16a34a",
    }.get(name, "#334155")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load(args.summary)
    pairs = load(args.pairs)
    series: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        group = row["group"]
        if group == "baseline":
            continue
        name = label(group, row.get("ef_construction", ""), row.get("mutation", ""))
        series[name].append((f(row["workers"]), f(row["op_p95_lowq_ms"]), f(row["op_p99_lowq_ms"])))
    for vals in series.values():
        vals.sort()

    plt.rcParams.update({
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.2,
    })
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    for name in ["bg search", "insert efc=200", "insert efc=35", "insert efc=35 skip-update"]:
        vals = series.get(name, [])
        if not vals:
            continue
        xs = [v[0] for v in vals]
        p99 = [v[2] for v in vals]
        ax.plot(xs, p99, marker="o", lw=1.25, color=color(name), label=name)
    ax.set_title("(a) Closed-loop fixed-worker measured-search p99", loc="left", pad=2)
    ax.set_xlabel("competing workers (fixed, no input rate)")
    ax.set_ylabel("low-queue op p99 (ms)")
    ax.grid(axis="both", color="#e5e7eb", lw=0.8)
    ax.legend(frameon=False, loc="upper left", ncol=2)

    ax = axes[1]
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in pairs:
        case = row["insert_case"]
        if "skip" in case:
            name = "insert efc=35 skip-update"
        elif "efc200" in case:
            name = "insert efc=200"
        else:
            name = "insert efc=35"
        grouped[name].append((f(row["workers"]), f(row["delta_p99_ms"])))
    width = 0.22
    workers = sorted({w for vals in grouped.values() for w, _ in vals})
    offsets = {
        "insert efc=200": -width,
        "insert efc=35": 0.0,
        "insert efc=35 skip-update": width,
    }
    ax.axhline(0.0, color="#94a3b8", lw=0.8)
    for name in ["insert efc=200", "insert efc=35", "insert efc=35 skip-update"]:
        vals = dict(grouped.get(name, []))
        ax.bar([w + offsets[name] for w in workers], [vals.get(w, 0.0) for w in workers],
               width=width, color=color(name), label=name)
    ax.set_xticks(workers, [str(int(w)) for w in workers])
    ax.set_title("(b) Insert p99 residual vs same-worker bg-search", loc="left", pad=2)
    ax.set_xlabel("competing workers")
    ax.set_ylabel("p99 delta (ms)")
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.legend(frameon=False, loc="upper left", ncol=2)

    ax = axes[2]
    throughput: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row["group"] == "baseline":
            continue
        name = label(row["group"], row.get("ef_construction", ""), row.get("mutation", ""))
        qps = f(row["insert_qps_per_point"]) if row["group"].startswith("insert") else f(row["search_qps_per_point"])
        throughput[name].append((f(row["workers"]), qps))
    for name in ["bg search", "insert efc=200", "insert efc=35", "insert efc=35 skip-update"]:
        vals = sorted(throughput.get(name, []))
        if vals:
            ax.plot([v[0] for v in vals], [v[1] for v in vals], marker="o", lw=1.1,
                    color=color(name), label=name)
    ax.set_title("(c) Throughput is an outcome, not a controlled rate", loc="left", pad=2)
    ax.set_xlabel("competing workers")
    ax.set_ylabel("throughput (points/s)")
    ax.grid(axis="both", color="#e5e7eb", lw=0.8)
    ax.legend(frameon=False, loc="upper left", ncol=2)

    for ax in axes:
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
