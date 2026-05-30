#!/usr/bin/env python3
"""Plot M3 bandwidth audit:

Three panels packed for fig3(c) replacement:
  (a) per-thread LLC-load-miss bandwidth vs N for {insert, spin, search} +
      reader baseline. Y log scale. Directly answers "is writer per-thread
      bandwidth low?".
  (b) socket DRAM bandwidth (read+write) vs N, faceted by mode. Shows that
      total DRAM stays in the 5-20 GB/s band well below socket ceiling -
      contradicts the cache-saturation framing.
  (c) reader P99 ms vs N by mode. Connects bandwidth to outcome metric.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODE_STYLE = {
    "insert": ("o-", "C3", "Writer (real HNSW insert)"),
    "spin":   ("s--", "C7", "Spin (no memory access)"),
    "search": ("^-.", "C0", "Reader competing (background search)"),
    "idle":   ("*", "k", "Reader-only baseline"),
}


def load_summary(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                r["n_competing"] = int(r["n_competing"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows


def fnum(r: dict, key: str) -> float:
    try:
        v = float(r.get(key, "nan"))
        return v
    except (TypeError, ValueError):
        return float("nan")


def panel_per_thread_bw(ax, rows: list[dict]) -> None:
    for mode in ("insert", "spin", "search"):
        m_rows = [r for r in rows if r.get("mode") == mode and r.get("phase") == "phase1_bandwidth"]
        m_rows = [r for r in m_rows if r["n_competing"] > 0]
        m_rows.sort(key=lambda r: r["n_competing"])
        xs = [r["n_competing"] for r in m_rows]
        ys = [fnum(r, "competing_per_thread_LLC_loadmiss_bytes_per_s") / 1e6 for r in m_rows]
        style, color, label = MODE_STYLE[mode]
        ax.plot(xs, ys, style, color=color, label=label, markersize=6)
    # Reader baseline as a horizontal band (range across phase1 cells)
    reader_y = []
    for r in rows:
        if r.get("phase") != "phase1_bandwidth": continue
        v = fnum(r, "reader_per_thread_LLC_loadmiss_bytes_per_s") / 1e6
        if not math.isnan(v): reader_y.append(v)
    if reader_y:
        lo, hi = min(reader_y), max(reader_y)
        ax.axhspan(lo, hi, alpha=0.10, color="C0", label=f"Reader per-thread band ({lo:.0f}-{hi:.0f} MB/s)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of competing workers $N$")
    ax.set_ylabel("Per-thread LLC-load-miss bandwidth (MB/s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("(a) Per-thread competing-worker DRAM-bound bandwidth")


def panel_socket_dram(ax, rows: list[dict]) -> None:
    for mode in ("idle", "insert", "spin", "search"):
        m_rows = [r for r in rows if r.get("mode") == mode and r.get("phase") == "phase1_bandwidth"]
        m_rows.sort(key=lambda r: r["n_competing"])
        if mode == "idle":
            v = fnum(m_rows[0], "socket_DRAM_total_bytes_per_s") / 1e9 if m_rows else float("nan")
            if not math.isnan(v):
                ax.axhline(v, linestyle=":", color="k", label=f"{MODE_STYLE['idle'][2]} ({v:.1f} GB/s)")
            continue
        xs = [r["n_competing"] for r in m_rows]
        ys = [fnum(r, "socket_DRAM_total_bytes_per_s") / 1e9 for r in m_rows]
        style, color, label = MODE_STYLE[mode]
        ax.plot(xs, ys, style, color=color, label=label, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Number of competing workers $N$")
    ax.set_ylabel("Socket DRAM bandwidth (GB/s)")
    ax.set_title("(b) System DRAM throughput (uncore IMC)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def panel_reader_p99(ax, rows: list[dict]) -> None:
    for mode in ("idle", "insert", "spin", "search"):
        m_rows = [r for r in rows if r.get("mode") == mode and r.get("phase") == "phase1_bandwidth"]
        m_rows.sort(key=lambda r: r["n_competing"])
        if mode == "idle":
            v = fnum(m_rows[0], "bench_search_p99_ms") if m_rows else float("nan")
            if not math.isnan(v):
                ax.axhline(v, linestyle=":", color="k", label=f"{MODE_STYLE['idle'][2]} ({v:.3f} ms)")
            continue
        xs = [r["n_competing"] for r in m_rows if r["n_competing"] > 0]
        ys = [fnum(r, "bench_search_p99_ms") for r in m_rows if r["n_competing"] > 0]
        style, color, label = MODE_STYLE[mode]
        ax.plot(xs, ys, style, color=color, label=label, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Number of competing workers $N$")
    ax.set_ylabel("Reader search P99 (ms)")
    ax.set_title("(c) Outcome: measured reader P99")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def plot_hotcold(rows: list[dict], out_path: Path) -> None:
    p3 = [r for r in rows if r.get("phase") == "phase3_hotcold"]
    if not p3:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ax_bw, ax_p99 = axes
    # Bar: hot vs cold per-thread store-miss bytes/s + reader P99
    hot_bw = [fnum(r, "competing_per_thread_LLC_storemiss_bytes_per_s") / 1e6 for r in p3 if r.get("mode") == "hot"]
    cold_bw = [fnum(r, "competing_per_thread_LLC_storemiss_bytes_per_s") / 1e6 for r in p3 if r.get("mode") == "cold"]
    hot_p99 = [fnum(r, "bench_search_p99_ms") for r in p3 if r.get("mode") == "hot"]
    cold_p99 = [fnum(r, "bench_search_p99_ms") for r in p3 if r.get("mode") == "cold"]

    def avg(xs): xs = [x for x in xs if not math.isnan(x)]; return sum(xs)/len(xs) if xs else float("nan")

    ax_bw.bar(["cold writer", "hot writer"],
              [avg(cold_bw), avg(hot_bw)],
              yerr=[max(cold_bw)-avg(cold_bw) if cold_bw else 0,
                    max(hot_bw)-avg(hot_bw) if hot_bw else 0],
              color=["C0", "C3"])
    ax_bw.set_ylabel("Writer per-thread\nLLC-store-miss bandwidth (MB/s)")
    ax_bw.set_title("Writer-side cache traffic: hot vs cold")
    ax_bw.grid(True, alpha=0.3)
    ax_p99.bar(["cold writer", "hot writer"],
               [avg(cold_p99), avg(hot_p99)],
               yerr=[max(cold_p99)-avg(cold_p99) if cold_p99 else 0,
                     max(hot_p99)-avg(hot_p99) if hot_p99 else 0],
               color=["C0", "C3"])
    ax_p99.set_ylabel("Reader search P99 (ms)")
    ax_p99.set_title("Reader tail: hot vs cold")
    ax_p99.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hotcold-out", type=Path)
    args = ap.parse_args()
    rows = load_summary(args.summary)
    if not rows:
        print("no rows"); return 1
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    panel_per_thread_bw(axes[0], rows)
    panel_socket_dram(axes[1], rows)
    panel_reader_p99(axes[2], rows)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    if args.hotcold_out:
        plot_hotcold(rows, args.hotcold_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
