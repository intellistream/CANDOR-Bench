#!/usr/bin/env python3
"""Matched-window latency deltas for M3 read-write screens."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "op_ms",
    "e2e_ms",
    "search_index_ms",
    "work_searchknn_ms",
    "work_searchknn_thread_cpu_ms",
    "work_base_search_ms",
    "work_level0_edges_scanned",
    "work_distance_computations",
    "queue_ms",
]


def percentile(values: pd.Series, pct: float) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return math.nan
    return float(np.percentile(vals.to_numpy(), pct))


def resolve_run_root(raw: object) -> Path:
    run_root = Path(str(raw))
    if not run_root.exists() and (Path("bench") / run_root).exists():
        run_root = Path("bench") / run_root
    return run_root


def case_dir(run_root: Path) -> Path:
    matches = sorted(run_root.glob("closed_loop_odin_efc*"))
    if not matches:
        raise FileNotFoundError(f"missing closed_loop_odin_efc* under {run_root}")
    return matches[0]


def load_search(root: Path) -> pd.DataFrame:
    path = root / "raw_latency" / "search_wide.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["measured"] == 1].copy()
    if "search_index_ms" not in frame:
        frame["search_index_ms"] = frame["op_ms"]
    frame["queue_ms"] = (frame["start_raw_ns"] - frame["create_raw_ns"]) / 1_000_000.0
    frame["mid_raw_ns"] = (frame["start_raw_ns"] + frame["finish_raw_ns"]) / 2.0
    frame["mid_ms"] = frame["finish_ms"] - (frame["finish_raw_ns"] - frame["mid_raw_ns"]) / 1_000_000.0
    return frame


def load_windows(root: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]], int]:
    path = root / "raw_latency" / "insert_wide.csv"
    if not path.exists():
        return [], [], 0
    inserts = pd.read_csv(path)
    if inserts.empty or "start_raw_ns" not in inserts or "finish_raw_ns" not in inserts:
        return [], [], len(inserts)

    raw_windows: list[tuple[float, float]] = []
    elapsed_windows: list[tuple[float, float]] = []
    for start, end, finish_ms in inserts.sort_values("start_raw_ns")[
        ["start_raw_ns", "finish_raw_ns", "finish_ms"]
    ].itertuples(index=False):
        start = float(start)
        end = float(end)
        finish_ms = float(finish_ms)
        start_ms = finish_ms - (end - start) / 1_000_000.0
        if not raw_windows or start > raw_windows[-1][1] + 10_000_000.0:
            raw_windows.append((start, end))
            elapsed_windows.append((start_ms, finish_ms))
        else:
            raw_windows[-1] = (raw_windows[-1][0], max(raw_windows[-1][1], end))
            elapsed_windows[-1] = (
                elapsed_windows[-1][0],
                max(elapsed_windows[-1][1], finish_ms),
            )
    return raw_windows, elapsed_windows, len(inserts)


def select_window(frame: pd.DataFrame, windows: list[tuple[float, float]], col: str) -> pd.DataFrame:
    if not windows:
        return frame
    mask = pd.Series(False, index=frame.index)
    for start, end in windows:
        mask |= (frame[col] >= start) & (frame[col] < end)
    out = frame[mask].copy()
    return out if not out.empty else frame


def summarize(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {f"{prefix}_rows": float(len(frame))}
    for metric in METRICS:
        if metric not in frame:
            continue
        out[f"{prefix}_{metric}_mean"] = float(pd.to_numeric(frame[metric], errors="coerce").mean())
        out[f"{prefix}_{metric}_p95"] = percentile(frame[metric], 95)
        out[f"{prefix}_{metric}_p99"] = percentile(frame[metric], 99)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    params = pd.read_csv(args.params)
    rows: list[dict[str, object]] = []
    for (dataset, cw), group in params.groupby(["dataset", "competing_workers"], sort=False):
        bg = group[group["case_key"] == "background_search"]
        if bg.empty:
            continue
        bg_row = bg.iloc[0]
        bg_root = case_dir(resolve_run_root(bg_row["run_root"]))
        bg_search = load_search(bg_root)
        for _, row in group.iterrows():
            if row["case_key"] == "background_search":
                continue
            root = case_dir(resolve_run_root(row["run_root"]))
            raw_windows, elapsed_windows, insert_batches = load_windows(root)
            case_search = select_window(load_search(root), raw_windows, "mid_raw_ns")
            bg_matched = select_window(bg_search, elapsed_windows, "mid_ms")
            out: dict[str, object] = {
                "dataset": dataset,
                "competing_workers": int(cw),
                "case_key": row["case_key"],
                "case_label": row["case_label"],
                "m": int(row["m"]),
                "efc": int(row["efc"]),
                "efs": int(row["efs"]),
                "insert_batches": int(insert_batches),
                "case_rows": int(len(case_search)),
                "matched_background_rows": int(len(bg_matched)),
            }
            out.update(summarize(bg_matched, "matched_background"))
            out.update(summarize(case_search, "case"))
            for metric in METRICS:
                for stat in ["mean", "p95", "p99"]:
                    bg_col = f"matched_background_{metric}_{stat}"
                    case_col = f"case_{metric}_{stat}"
                    if bg_col in out and case_col in out:
                        bg_val = float(out[bg_col])
                        case_val = float(out[case_col])
                        out[f"delta_{metric}_{stat}"] = case_val - bg_val
                        out[f"ratio_{metric}_{stat}"] = case_val / bg_val if bg_val else math.nan
            rows.append(out)

    out_root = args.out_root
    tables = out_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(tables / "matched_latency_delta.csv", index=False)

    lines = [
        "# Matched Read-Write Latency Screen",
        "",
        "Each case is compared against the pure-search background in the same elapsed active windows.",
        "",
    ]
    if len(result):
        show_cols = [
            "dataset",
            "competing_workers",
            "case_label",
            "delta_op_ms_p99",
            "delta_work_searchknn_ms_p99",
            "delta_work_base_search_ms_p99",
            "delta_search_index_ms_p99",
            "delta_queue_ms_p99",
        ]
        present = [c for c in show_cols if c in result]
        show = result[present].sort_values(
            [c for c in ["delta_work_searchknn_ms_p99", "delta_op_ms_p99"] if c in result],
            ascending=False,
        )
        lines.append(show.to_markdown(index=False, floatfmt=".5f"))
    else:
        lines.append("No non-background cases found.")
    lines.append("")
    lines.append("Primary ranking metric: `delta_work_searchknn_ms_p99`, with `delta_op_ms_p99` as user-visible tail.")
    (out_root / "matched_latency_README.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
