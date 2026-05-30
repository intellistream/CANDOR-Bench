#!/usr/bin/env python3
"""Validate whether search E2E spikes align with insert graph-mutation bursts.

This is intentionally stricter than a raw correlation dump.  It asks whether
the top search-E2E buckets are enriched for top insert-mutation buckets within
a small lag window, and compares the observed hit rate against circular-shift
null baselines.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


Metric = tuple[str, str]


METRICS: list[Metric] = [
    ("connect_ms", "graph_connect_ms"),
    ("critical_ms", "graph_link_critical_ms"),
    ("updates", "graph_link_updates"),
    ("lock_wait_ms", "graph_unique_lock_wait_ms"),
]


def infer_batch(label: str) -> int:
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


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * p)]


def top_indices(values: list[float], pct: float, min_count: int) -> set[int]:
    if not values:
        return set()
    count = max(min_count, int(math.ceil(len(values) * pct)))
    count = min(count, len(values))
    ranked = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return set(ranked[:count])


def load_search_batches(path: Path, batch: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    cur: list[dict[str, str]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"finish_ms", "e2e_ms", "op_ms"}
        if not required.issubset(fields):
            raise ValueError(f"{path} missing {sorted(required - fields)}")
        for row in reader:
            cur.append(row)
            if len(cur) == batch:
                # The bench writes one amortized row per query when
                # per_query_latency is enabled.  Sum rows back to the original
                # task-level batch latency.
                rows.append(
                    {
                        "finish_ms": float(cur[-1]["finish_ms"]),
                        "e2e_ms": sum(float(r["e2e_ms"]) for r in cur),
                        "op_ms": sum(float(r["op_ms"]) for r in cur),
                        "active_front": sum(float(r.get("freshness_active_front_pts", 0.0)) for r in cur)
                        / len(cur),
                    }
                )
                cur.clear()
    return rows


def load_insert_rows(path: Path) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"finish_ms"} | {raw for _, raw in METRICS}
        if not required.issubset(fields):
            raise ValueError(f"{path} missing {sorted(required - fields)}")
        for row in reader:
            x = {"finish_ms": float(row["finish_ms"])}
            for name, raw in METRICS:
                x[name] = float(row.get(raw, 0.0))
            out.append(x)
    return out


def build_buckets(
    search: list[dict[str, float]],
    inserts: list[dict[str, float]],
    bucket_ms: float,
) -> list[dict[str, float]]:
    buckets: dict[int, dict[str, float]] = {}
    for s in search:
        b = int(s["finish_ms"] // bucket_ms)
        x = buckets.setdefault(
            b,
            {
                "bucket": float(b),
                "time_ms": b * bucket_ms,
                "search_count": 0.0,
                "e2e_sum": 0.0,
                "op_sum": 0.0,
                "active_sum": 0.0,
                **{name: 0.0 for name, _ in METRICS},
            },
        )
        x["search_count"] += 1.0
        x["e2e_sum"] += s["e2e_ms"]
        x["op_sum"] += s["op_ms"]
        x["active_sum"] += s["active_front"]

    for ins in inserts:
        b = int(ins["finish_ms"] // bucket_ms)
        x = buckets.setdefault(
            b,
            {
                "bucket": float(b),
                "time_ms": b * bucket_ms,
                "search_count": 0.0,
                "e2e_sum": 0.0,
                "op_sum": 0.0,
                "active_sum": 0.0,
                **{name: 0.0 for name, _ in METRICS},
            },
        )
        for name, _ in METRICS:
            x[name] += ins[name]

    rows: list[dict[str, float]] = []
    for b in sorted(buckets):
        x = buckets[b]
        if x["search_count"] <= 0:
            continue
        rows.append(
            {
                "bucket": x["bucket"],
                "time_ms": x["time_ms"],
                "e2e_ms": x["e2e_sum"] / x["search_count"],
                "op_ms": x["op_sum"] / x["search_count"],
                "active_front": x["active_sum"] / x["search_count"],
                **{name: x[name] for name, _ in METRICS},
            }
        )
    return rows


def lag_hit_count(spikes: set[int], bursts: set[int], lag: int, n: int) -> int:
    hits = 0
    for i in spikes:
        j = i + lag
        if 0 <= j < n and j in bursts:
            hits += 1
    return hits


def null_stats(spikes: set[int], bursts: set[int], lag: int, n: int, observed: int) -> tuple[float, float, float]:
    if not spikes or not bursts or n <= 1:
        return 0.0, 0.0, 1.0
    counts: list[int] = []
    burst_list = sorted(bursts)
    for shift in range(1, n):
        shifted = {(j + shift) % n for j in burst_list}
        counts.append(lag_hit_count(spikes, shifted, lag, n))
    expected = sum(counts) / len(counts) if counts else 0.0
    p_value = (1 + sum(1 for c in counts if c >= observed)) / (1 + len(counts))
    enrichment = observed / expected if expected > 0 else (float("inf") if observed > 0 else 0.0)
    return expected, enrichment, p_value


def evaluate_metric(
    rows: list[dict[str, float]],
    metric: str,
    search_top_pct: float,
    mutation_top_pct: float,
    min_spikes: int,
    min_bursts: int,
    max_abs_lag: int,
) -> dict[str, object]:
    n = len(rows)
    e2e = [r["e2e_ms"] for r in rows]
    vals = [r[metric] for r in rows]
    spikes = top_indices(e2e, search_top_pct, min_spikes)
    bursts = top_indices(vals, mutation_top_pct, min_bursts)
    best: dict[str, object] | None = None
    near_best: dict[str, object] | None = None

    for lag in range(-max_abs_lag, max_abs_lag + 1):
        observed = lag_hit_count(spikes, bursts, lag, n)
        expected, enrichment, p_value = null_stats(spikes, bursts, lag, n, observed)
        item: dict[str, object] = {
            "lag": lag,
            "hit_count": observed,
            "hit_rate": observed / len(spikes) if spikes else 0.0,
            "null_expected_hits": expected,
            "enrichment": enrichment,
            "p_value": p_value,
        }
        if best is None or (float(item["p_value"]), -float(item["enrichment"])) < (
            float(best["p_value"]),
            -float(best["enrichment"]),
        ):
            best = item
        if abs(lag) <= 1 and (
            near_best is None
            or (float(item["p_value"]), -float(item["enrichment"])) < (
                float(near_best["p_value"]),
                -float(near_best["enrichment"]),
            )
        ):
            near_best = item

    spike_vals = [vals[i] for i in sorted(spikes)]
    non_spike_vals = [v for i, v in enumerate(vals) if i not in spikes]
    return {
        "metric": metric,
        "num_buckets": n,
        "num_spikes": len(spikes),
        "num_bursts": len(bursts),
        "search_p99": percentile(e2e, 0.99),
        "search_max": max(e2e) if e2e else 0.0,
        "mutation_p90": percentile(vals, 0.90),
        "mutation_p99": percentile(vals, 0.99),
        "spike_mutation_mean": sum(spike_vals) / len(spike_vals) if spike_vals else 0.0,
        "nonspike_mutation_mean": sum(non_spike_vals) / len(non_spike_vals) if non_spike_vals else 0.0,
        "near_lag": near_best or {},
        "best_lag": best or {},
    }


def classify(result: dict[str, object]) -> str:
    near = result["near_lag"]
    if not isinstance(near, dict):
        return "unsupported"
    hit_rate = float(near.get("hit_rate", 0.0))
    enrichment = float(near.get("enrichment", 0.0))
    p_value = float(near.get("p_value", 1.0))
    if hit_rate >= 0.50 and enrichment >= 2.0 and p_value <= 0.05:
        return "strong"
    if hit_rate >= 0.35 and enrichment >= 1.5 and p_value <= 0.10:
        return "weak"
    return "unsupported"


def analyze_case(case_dir: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    label = case_dir.name
    batch = infer_batch(label)
    search_path = case_dir / "raw_latency" / "search_wide.csv"
    insert_path = case_dir / "raw_latency" / "insert_wide.csv"
    if not search_path.exists() or not insert_path.exists():
        return []
    rows = build_buckets(
        load_search_batches(search_path, batch),
        load_insert_rows(insert_path),
        args.bucket_ms,
    )
    out: list[dict[str, object]] = []
    for metric, _ in METRICS:
        r = evaluate_metric(
            rows,
            metric,
            args.search_top_pct,
            args.mutation_top_pct,
            args.min_spikes,
            args.min_bursts,
            args.max_abs_lag,
        )
        near = r["near_lag"]
        best = r["best_lag"]
        assert isinstance(near, dict)
        assert isinstance(best, dict)
        out.append(
            {
                **parse_label(label),
                "metric": metric,
                "num_buckets": r["num_buckets"],
                "num_spikes": r["num_spikes"],
                "num_bursts": r["num_bursts"],
                "search_p99_ms": r["search_p99"],
                "search_max_ms": r["search_max"],
                "mutation_p90": r["mutation_p90"],
                "mutation_p99": r["mutation_p99"],
                "spike_mutation_mean": r["spike_mutation_mean"],
                "nonspike_mutation_mean": r["nonspike_mutation_mean"],
                "near_lag": near.get("lag", ""),
                "near_hit_rate": near.get("hit_rate", 0.0),
                "near_hits": near.get("hit_count", 0),
                "near_expected_hits": near.get("null_expected_hits", 0.0),
                "near_enrichment": near.get("enrichment", 0.0),
                "near_p_value": near.get("p_value", 1.0),
                "best_lag": best.get("lag", ""),
                "best_hit_rate": best.get("hit_rate", 0.0),
                "best_enrichment": best.get("enrichment", 0.0),
                "best_p_value": best.get("p_value", 1.0),
                "support": classify(r),
            }
        )
    return out


def write_markdown(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    by_metric: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_metric.setdefault(str(row["metric"]), []).append(row)

    lines: list[str] = [
        "# M3 Mutation Spike Alignment",
        "",
        f"Result dir: `{args.path}`",
        f"Bucket: `{args.bucket_ms:g} ms`; search spikes: top `{args.search_top_pct:.0%}` "
        f"(min {args.min_spikes}); mutation bursts: top `{args.mutation_top_pct:.0%}` "
        f"(min {args.min_bursts}); near lag window: `-1..+1` buckets.",
        "",
        "Decision rule: `strong` requires near-lag hit rate >= 0.50, enrichment >= 2.0, "
        "and circular-shift p <= 0.05. `weak` requires hit rate >= 0.35, "
        "enrichment >= 1.5, and p <= 0.10.",
        "",
        "## Metric Summary",
        "",
        "| metric | strong | weak | unsupported | median near enrichment | median near hit rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in [m[0] for m in METRICS]:
        rs = by_metric.get(metric, [])
        enrich = sorted(float(r["near_enrichment"]) for r in rs)
        hit = sorted(float(r["near_hit_rate"]) for r in rs)
        mid = len(rs) // 2
        med_enrich = enrich[mid] if rs else 0.0
        med_hit = hit[mid] if rs else 0.0
        lines.append(
            f"| {metric} | {sum(1 for r in rs if r['support'] == 'strong')} "
            f"| {sum(1 for r in rs if r['support'] == 'weak')} "
            f"| {sum(1 for r in rs if r['support'] == 'unsupported')} "
            f"| {med_enrich:.2f} | {med_hit:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Strongest Near-Lag Cases",
            "",
            "| label | metric | support | lag | hit_rate | enrichment | p | search_p99_ms |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if r["support"] == "strong" else 1 if r["support"] == "weak" else 2,
            float(r["near_p_value"]),
            -float(r["near_enrichment"]),
        ),
    )
    for r in ranked[:20]:
        lines.append(
            f"| {r['label']} | {r['metric']} | {r['support']} | {r['near_lag']} "
            f"| {float(r['near_hit_rate']):.2f} | {float(r['near_enrichment']):.2f} "
            f"| {float(r['near_p_value']):.3f} | {float(r['search_p99_ms']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A strong result for `critical_ms` / `updates` would support the specific "
            "claim that neighbor-list critical-section mutation bursts cause search spikes.",
            "- A strong result for `connect_ms` but not `critical_ms` supports a broader "
            "writer graph-connect or memory-interference claim, not the narrower "
            "neighbor-list-critical-section claim.",
            "- Unsupported near-lag alignment means this trace does not justify the "
            "specific mutation-burst cause; the explanation should stay conservative.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="case dir or result dir containing case dirs")
    parser.add_argument("--bucket-ms", type=float, default=200.0)
    parser.add_argument("--search-top-pct", type=float, default=0.05)
    parser.add_argument("--mutation-top-pct", type=float, default=0.10)
    parser.add_argument("--min-spikes", type=int, default=5)
    parser.add_argument("--min-bursts", type=int, default=10)
    parser.add_argument("--max-abs-lag", type=int, default=5)
    parser.add_argument("--out-prefix", default="m3_mutation_spike_alignment")
    args = parser.parse_args()

    if (args.path / "raw_latency").is_dir():
        cases = [args.path]
        out_dir = args.path
    else:
        cases = [p for p in sorted(args.path.iterdir()) if p.is_dir()]
        out_dir = args.path

    rows: list[dict[str, object]] = []
    for case in cases:
        try:
            rows.extend(analyze_case(case, args))
        except Exception as exc:
            print(f"{case.name}: ERROR {exc}")

    csv_path = out_dir / f"{args.out_prefix}.csv"
    md_path = out_dir / f"{args.out_prefix}.md"
    fields = [
        "label",
        "sync_mode",
        "batch",
        "query_mode",
        "write_rate",
        "read_rate",
        "metric",
        "num_buckets",
        "num_spikes",
        "num_bursts",
        "search_p99_ms",
        "search_max_ms",
        "mutation_p90",
        "mutation_p99",
        "spike_mutation_mean",
        "nonspike_mutation_mean",
        "near_lag",
        "near_hit_rate",
        "near_hits",
        "near_expected_hits",
        "near_enrichment",
        "near_p_value",
        "best_lag",
        "best_hit_rate",
        "best_enrichment",
        "best_p_value",
        "support",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(md_path, rows, args)

    print(f"[m3-align] wrote {csv_path}")
    print(f"[m3-align] wrote {md_path}")
    for metric in [m[0] for m in METRICS]:
        rs = [r for r in rows if r["metric"] == metric]
        print(
            f"{metric}: strong={sum(1 for r in rs if r['support'] == 'strong')} "
            f"weak={sum(1 for r in rs if r['support'] == 'weak')} "
            f"unsupported={sum(1 for r in rs if r['support'] == 'unsupported')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
