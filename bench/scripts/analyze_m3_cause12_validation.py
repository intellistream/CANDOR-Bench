#!/usr/bin/env python3
"""Cause-specific analysis for M3 search batch-E2E fluctuation experiments."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path


LABEL_RE = re.compile(
    r"(?P<dataset>.+)_b(?P<batch>\d+)_(?P<qmode>round_robin|chasing)"
    r"_w(?P<write>\d+)_r(?P<read>\d+)_(?P<sync>nodelock|nolock|rwlock)$"
)


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[int((len(s) - 1) * p)]


def parse_label(label: str) -> dict[str, object] | None:
    match = LABEL_RE.match(label)
    if not match:
        return None
    out: dict[str, object] = match.groupdict()
    out["batch"] = int(out["batch"])
    out["write"] = int(out["write"])
    out["read"] = int(out["read"])
    return out


def load_summary_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def true_batch_from_raw(wide: Path, batch: int) -> tuple[list[float], list[float], list[float], list[float]]:
    e2e_batches: list[float] = []
    op_batches: list[float] = []
    active_batches: list[float] = []
    missed_batches: list[float] = []
    cur_e2e: list[float] = []
    cur_op: list[float] = []
    cur_active: list[float] = []
    cur_missed: list[float] = []
    with wide.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cur_e2e.append(float(row["e2e_ms"]))
            cur_op.append(float(row["op_ms"]))
            cur_active.append(float(row.get("freshness_active_front_pts", 0.0)))
            cur_missed.append(float(row.get("freshness_missed_e2e_pts", 0.0)))
            if len(cur_e2e) == batch:
                e2e_batches.append(sum(cur_e2e))
                op_batches.append(sum(cur_op))
                active_batches.append(sum(cur_active) / batch)
                missed_batches.append(sum(cur_missed) / batch)
                cur_e2e.clear()
                cur_op.clear()
                cur_active.clear()
                cur_missed.clear()
    return e2e_batches, op_batches, active_batches, missed_batches


def worst_window(values: list[float], width: int = 100) -> tuple[float, int]:
    best = 0.0
    best_start = 0
    for i in range(0, len(values), width):
        chunk = values[i : i + width]
        if len(chunk) < max(10, width // 5):
            break
        avg = sum(chunk) / len(chunk)
        if avg > best:
            best = avg
            best_start = i
    return best, best_start


def load_cases(result_dir: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for child in sorted(result_dir.iterdir()):
        if not child.is_dir():
            continue
        meta = parse_label(child.name)
        if not meta:
            continue
        summary = load_summary_row(child / "benchmark_results.csv")
        wide = child / "raw_latency" / "search_wide.csv"
        if not summary or not wide.exists():
            cases.append({**meta, "label": child.name, "complete": False})
            continue
        batch = int(meta["batch"])
        e2e, op, active, missed = true_batch_from_raw(wide, batch)
        if not e2e:
            cases.append({**meta, "label": child.name, "complete": False})
            continue
        worst, worst_start = worst_window(e2e)
        worst_op, _ = worst_window(op)
        mean_e2e = sum(e2e) / len(e2e)
        cases.append(
            {
                **meta,
                "label": child.name,
                "complete": True,
                "num_batches": len(e2e),
                "mean_batch_e2e_ms": mean_e2e,
                "p95_batch_e2e_ms": pct(e2e, 0.95),
                "p99_batch_e2e_ms": pct(e2e, 0.99),
                "max_batch_e2e_ms": max(e2e),
                "worst100_batch_e2e_ms": worst,
                "worst100_over_mean": worst / mean_e2e if mean_e2e else 0.0,
                "worst100_batch_op_ms": worst_op,
                "worst100_queue_est_ms": worst - worst_op,
                "mean_active_front_pts": sum(active) / len(active),
                "p99_active_front_pts": pct(active, 0.99),
                "max_active_front_pts": max(active),
                "mean_missed_e2e_pts": sum(missed) / len(missed),
                "p99_missed_e2e_pts": pct(missed, 0.99),
                "search_qps": float(summary.get("search_qps (per-point)", 0.0) or 0.0),
                "recall": float(summary.get("recall", 0.0) or 0.0),
                "stats": summary.get("stats", ""),
                "worst100_start_batch": worst_start,
            }
        )
    return cases


def write_cases_csv(result_dir: Path, cases: list[dict[str, object]]) -> Path:
    out = result_dir / "cause12_cases.csv"
    fields = [
        "label",
        "complete",
        "sync",
        "batch",
        "qmode",
        "write",
        "read",
        "num_batches",
        "mean_batch_e2e_ms",
        "p95_batch_e2e_ms",
        "p99_batch_e2e_ms",
        "max_batch_e2e_ms",
        "worst100_batch_e2e_ms",
        "worst100_over_mean",
        "worst100_batch_op_ms",
        "worst100_queue_est_ms",
        "mean_active_front_pts",
        "p99_active_front_pts",
        "max_active_front_pts",
        "mean_missed_e2e_pts",
        "p99_missed_e2e_pts",
        "search_qps",
        "recall",
        "worst100_start_batch",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases)
    return out


def cause1(cases: list[dict[str, object]]) -> list[str]:
    lines = ["Cause 1: writer mutation burst validation"]
    complete = [c for c in cases if c.get("complete") and c.get("sync") == "nolock"]
    if not complete:
        return lines + ["NO COMPLETE nolock CASES"]

    lines.append("\nExperiment 1A: write-pressure gradient under nolock")
    groups: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for c in complete:
        groups[(int(c["batch"]), str(c["qmode"]))].append(c)
    for key, group in sorted(groups.items()):
        group = sorted(group, key=lambda c: int(c["write"]))
        if len(group) < 2:
            continue
        vals = [
            f"w{int(c['write'])}/r{int(c['read'])}:p99={float(c['p99_batch_e2e_ms']):.1f},"
            f"active99={float(c['p99_active_front_pts']):.0f},worst100={float(c['worst100_batch_e2e_ms']):.1f}"
            for c in group
        ]
        lines.append(f"  b{key[0]} {key[1]} -> " + " | ".join(vals))

    lines.append("\nExperiment 1B: batch-size/burst amplification under nolock")
    groups2: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for c in complete:
        groups2[(str(c["qmode"]), int(c["write"]), int(c["read"]))].append(c)
    for key, group in sorted(groups2.items()):
        group = sorted(group, key=lambda c: int(c["batch"]))
        if len(group) < 2:
            continue
        vals = [
            f"b{int(c['batch'])}:mean={float(c['mean_batch_e2e_ms']):.1f},"
            f"p99={float(c['p99_batch_e2e_ms']):.1f},worst100={float(c['worst100_batch_e2e_ms']):.1f}"
            for c in group
        ]
        lines.append(f"  {key[0]} w{key[1]}/r{key[2]} -> " + " | ".join(vals))

    lines.append("\nExperiment 1C: queue-artifact rejection")
    top = sorted(complete, key=lambda c: float(c["p99_batch_e2e_ms"]), reverse=True)[:8]
    for c in top:
        lines.append(
            "  {label}: p99={p99_batch_e2e_ms:.1f}, worst100={worst100_batch_e2e_ms:.1f}, "
            "op100={worst100_batch_op_ms:.1f}, queue_est={worst100_queue_est_ms:.1f}".format(**c)
        )
    lines.append("  Decision: if queue_est is small vs worst100, spikes are inside search execution.")
    return lines


def cause2(cases: list[dict[str, object]]) -> list[str]:
    lines = ["Cause 2: node-lock convoy validation"]
    complete = [c for c in cases if c.get("complete") and c.get("sync") in {"nodelock", "nolock"}]
    if not complete:
        return lines + ["NO COMPLETE nodelock/nolock CASES"]

    paired: dict[tuple[int, str, int, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for c in complete:
        paired[(int(c["batch"]), str(c["qmode"]), int(c["write"]), int(c["read"]))][str(c["sync"])] = c

    lines.append("\nExperiment 2A: direct paired nodelock vs nolock")
    for key, pair in sorted(paired.items()):
        if "nodelock" not in pair or "nolock" not in pair:
            continue
        n = pair["nodelock"]
        l = pair["nolock"]
        n99 = float(n["p99_batch_e2e_ms"])
        l99 = float(l["p99_batch_e2e_ms"])
        delta = n99 - l99
        ratio = n99 / l99 if l99 > 0 else 0.0
        lines.append(
            f"  b{key[0]} {key[1]} w{key[2]}/r{key[3]}: "
            f"nodelock_p99={n99:.1f}, nolock_p99={l99:.1f}, "
            f"delta={delta:.1f}, ratio={ratio:.2f}x"
        )

    lines.append("\nExperiment 2B: lock effect grows with write-active front")
    deltas = []
    for key, pair in paired.items():
        if "nodelock" not in pair or "nolock" not in pair:
            continue
        n = pair["nodelock"]
        l = pair["nolock"]
        deltas.append(
            (
                float(n["p99_active_front_pts"]),
                float(n["p99_batch_e2e_ms"]) - float(l["p99_batch_e2e_ms"]),
                key,
            )
        )
    for active, delta, key in sorted(deltas, reverse=True)[:10]:
        lines.append(f"  b{key[0]} {key[1]} w{key[2]}/r{key[3]}: active99={active:.0f}, lock_delta_p99={delta:.1f}")

    lines.append("\nExperiment 2C: negative/control cases")
    lines.append("  If nodelock <= nolock for a workload, node locks are not the dominant cause there.")
    for key, pair in sorted(paired.items()):
        if "nodelock" not in pair or "nolock" not in pair:
            continue
        n99 = float(pair["nodelock"]["p99_batch_e2e_ms"])
        l99 = float(pair["nolock"]["p99_batch_e2e_ms"])
        if n99 <= l99:
            lines.append(f"  control: b{key[0]} {key[1]} w{key[2]}/r{key[3]} nodelock={n99:.1f} <= nolock={l99:.1f}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--cause", choices=["cause1", "cause2", "both"], default="both")
    args = parser.parse_args()

    cases = load_cases(args.result_dir)
    out = write_cases_csv(args.result_dir, cases)
    print(f"[m3-cause12] wrote {out}")
    incomplete = [c["label"] for c in cases if not c.get("complete")]
    if incomplete:
        print("[m3-cause12] incomplete cases:")
        for label in incomplete:
            print(f"  {label}")
    if args.cause in {"cause1", "both"}:
        print()
        print("\n".join(cause1(cases)))
    if args.cause in {"cause2", "both"}:
        print()
        print("\n".join(cause2(cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
