#!/usr/bin/env python3
"""Align slow search batches with insert-side graph-connection phase bursts."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


INSERT_METRICS = [
    "op_ms",
    "graph_connect_ms",
    "graph_link_critical_ms",
    "graph_unique_lock_wait_ms",
    "graph_link_updates",
    "graph_connect_calls",
    "graph_new_node_link_ms",
    "graph_old_update_loop_ms",
    "graph_prune_heuristic_ms",
    "graph_undo_record_ms",
    "graph_existing_node_visits",
    "graph_existing_node_appends",
    "graph_existing_node_prunes",
    "graph_pruned_edges",
]


def batch_size_from_label(label: str) -> int:
    match = re.search(r"_b(\d+)_", label)
    if not match:
        raise ValueError(f"cannot infer batch size from {label}")
    return int(match.group(1))


def parse_label(label: str) -> dict[str, str]:
    out = {"label": label}
    parts = label.split("_")
    for i, part in enumerate(parts):
        if part.startswith("b") and part[1:].isdigit():
            out["batch"] = part[1:]
        elif part == "chasing":
            out["query_mode"] = "chasing"
        elif part == "round" and i + 1 < len(parts) and parts[i + 1] == "robin":
            out["query_mode"] = "round_robin"
        elif part.startswith("w") and part[1:].isdigit():
            out["write_rate"] = part[1:]
        elif part.startswith("r") and part[1:].isdigit():
            out["read_rate"] = part[1:]
        elif part in {"nolock", "nodelock", "rwlock"}:
            out["sync_mode"] = part
    return out


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def val(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def aggregate_search_batches(rows: list[dict[str, str]], batch_size: int) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        if len(chunk) < batch_size:
            break
        out.append(
            {
                "batch_idx": len(out),
                "finish_ms": max(val(r, "finish_ms") for r in chunk),
                "op_ms": sum(val(r, "op_ms") for r in chunk),
                "e2e_ms": sum(val(r, "e2e_ms") for r in chunk),
            }
        )
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    return ordered[int((len(ordered) - 1) * p)]


def summarize_run(run_dir: Path, window_ms: float) -> tuple[dict[str, object], list[dict[str, object]]]:
    label = run_dir.name
    batch_size = batch_size_from_label(label)
    search = aggregate_search_batches(
        read_csv(run_dir / "raw_latency" / "search_wide.csv"), batch_size
    )
    inserts = read_csv(run_dir / "raw_latency" / "insert_wide.csv")
    if not search or not inserts:
        raise ValueError("missing search or insert rows")

    top_n = max(10, min(50, max(1, len(search) // 100)))
    top = sorted(search, key=lambda r: r["op_ms"], reverse=True)[:top_n]
    near: list[dict[str, str]] = []
    for srow in top:
        t = srow["finish_ms"]
        near.extend([irow for irow in inserts if abs(val(irow, "finish_ms") - t) <= window_ms])
    seen: set[tuple[str, str]] = set()
    near_unique: list[dict[str, str]] = []
    for row in near:
        key = (row.get("sample_idx", ""), row.get("finish_ms", ""))
        if key in seen:
            continue
        seen.add(key)
        near_unique.append(row)

    base: dict[str, list[float]] = {}
    nearby: dict[str, list[float]] = {}
    for metric in INSERT_METRICS:
        base[metric] = [val(r, metric) for r in inserts]
        nearby[metric] = [val(r, metric) for r in near_unique]

    summary: dict[str, object] = {
        **parse_label(label),
        "num_search_batches": len(search),
        "num_insert_batches": len(inserts),
        "top_n": top_n,
        "window_ms": window_ms,
        "near_insert_batches": len(near_unique),
        "mean_search_op_ms": mean([r["op_ms"] for r in search]),
        "p99_search_op_ms": percentile([r["op_ms"] for r in search], 0.99),
        "max_search_op_ms": max(r["op_ms"] for r in search),
        "top_mean_search_op_ms": mean([r["op_ms"] for r in top]),
    }
    for metric in INSERT_METRICS:
        b = mean(base[metric])
        n = mean(nearby[metric])
        summary[f"{metric}_mean"] = b
        summary[f"{metric}_near_mean"] = n
        summary[f"{metric}_near_enrich"] = n / b if b > 0 else 0.0

    examples: list[dict[str, object]] = []
    for rank, srow in enumerate(top[:10], start=1):
        nearest = sorted(inserts, key=lambda r: abs(val(r, "finish_ms") - srow["finish_ms"]))[:3]
        for n_rank, irow in enumerate(nearest, start=1):
            examples.append(
                {
                    **parse_label(label),
                    "rank": rank,
                    "nearest_rank": n_rank,
                    "search_batch_idx": int(srow["batch_idx"]),
                    "search_finish_ms": srow["finish_ms"],
                    "search_op_ms": srow["op_ms"],
                    "search_e2e_ms": srow["e2e_ms"],
                    "insert_dt_ms": val(irow, "finish_ms") - srow["finish_ms"],
                    **{f"insert_{m}": val(irow, m) for m in INSERT_METRICS},
                }
            )
    return summary, examples


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        print("usage: analyze_m3_insert_burst_alignment.py <result_dir> [window_ms]", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    window_ms = float(sys.argv[2]) if len(sys.argv) == 3 else 20.0

    summaries: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    for wide in sorted(root.glob("*/raw_latency/search_wide.csv")):
        run_dir = wide.parents[1]
        if not (run_dir / "raw_latency" / "insert_wide.csv").exists():
            continue
        try:
            summary, ex = summarize_run(run_dir, window_ms)
        except Exception as exc:
            print(f"[insert-burst] skip {run_dir}: {exc}", file=sys.stderr)
            continue
        summaries.append(summary)
        examples.extend(ex)

    write_csv(root / "m3_insert_burst_alignment.csv", summaries)
    write_csv(root / "m3_insert_burst_examples.csv", examples)

    report = root / "m3_insert_burst_alignment.md"
    with report.open("w") as f:
        f.write("# M3 Insert Burst Alignment\n\n")
        f.write(f"Window: +/- {window_ms:.1f} ms around top search batches.\n\n")
        for row in sorted(summaries, key=lambda r: float(r["p99_search_op_ms"]), reverse=True):
            f.write(
                "## {label}\n\n"
                "- search op: mean={mean_search_op_ms:.2f}, p99={p99_search_op_ms:.2f}, max={max_search_op_ms:.2f}, top_mean={top_mean_search_op_ms:.2f}\n"
                "- nearby insert batches={near_insert_batches}\n"
                "- graph_connect enrich={graph_connect_ms_near_enrich:.2f}, old_update_loop enrich={graph_old_update_loop_ms_near_enrich:.2f}\n"
                "- prune_heuristic enrich={graph_prune_heuristic_ms_near_enrich:.2f}, undo_record enrich={graph_undo_record_ms_near_enrich:.2f}\n"
                "- old_prunes enrich={graph_existing_node_prunes_near_enrich:.2f}, pruned_edges enrich={graph_pruned_edges_near_enrich:.2f}\n"
                "- link_updates enrich={graph_link_updates_near_enrich:.2f}, lock_wait enrich={graph_unique_lock_wait_ms_near_enrich:.2f}\n\n".format(**row)
            )

    print(f"[insert-burst] wrote {root / 'm3_insert_burst_alignment.csv'}")
    print(f"[insert-burst] wrote {root / 'm3_insert_burst_examples.csv'}")
    print(f"[insert-burst] wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
