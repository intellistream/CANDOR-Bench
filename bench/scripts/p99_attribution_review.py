#!/usr/bin/env python3
"""
Rigorous p99 attribution review for MVCC vs NoLock experiments.

Design choices:
- Only uses rows with explicit CC labels inferred from directory names.
- Excludes ambiguous directories (e.g., mvcc_vs_nolock_* mixed outputs).
- Uses strict 1:1 pairing on identical configs, then computes deltas (mvcc - nolock).
- Reports bootstrap confidence intervals and stratified summaries.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


P99_OP_COL = "search_p99_op_latency_ms (per-batch)"
P99_E2E_COL = "search_p99_e2e_latency_ms (per-batch)"
SEARCH_QPS_COL = "search_qps (per-point)"

PAIR_KEY_COLS = [
    "algorithm",
    "threads",
    "write_batch_size",
    "read_batch_size",
    "dataset_name",
    "with_external_rw_lock",
    "use_node_lock",
    "insert_input_rate",
    "search_input_rate",
    "index_params",
    "query_params",
]


@dataclass
class Row:
    path: Path
    rel_path: str
    row_index: int
    cc_label: Optional[str]
    cc_label_explicit: bool
    raw: Dict[str, str]
    stats: Dict[str, float]


def to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    v = str(value).strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def split_top_level_stats(stats: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    bracket_depth = 0
    for ch in stats:
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        if ch == "," and bracket_depth == 0:
            s = "".join(cur).strip()
            if s:
                parts.append(s)
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_stats(stats_text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not stats_text:
        return out
    for part in split_top_level_stats(stats_text):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        key = k.strip()
        token = v.strip().split(" ")[0]
        try:
            out[key] = float(token)
        except ValueError:
            continue
    return out


def has_token(part: str, token: str) -> bool:
    # token boundary on /_-. to avoid false positives in long words.
    return re.search(rf"(^|[_\-.]){re.escape(token)}([_\-.]|$)", part) is not None


def infer_cc_label(rel_path: Path) -> Tuple[Optional[str], bool]:
    labels: set[str] = set()
    explicit = False
    for part in rel_path.parts[:-1]:
        p = part.lower()
        # Known ambiguous experiment bucket.
        if "mvcc_vs_nolock" in p:
            continue

        has_mvcc = has_token(p, "mvcc")
        has_nolock = has_token(p, "nolock")
        has_lock = has_token(p, "lock")

        if has_mvcc and not has_nolock:
            labels.add("mvcc")
            explicit = True
        elif has_nolock and not has_mvcc:
            labels.add("nolock")
            explicit = True
        elif has_lock and not has_nolock and not has_mvcc:
            labels.add("lock")
            explicit = True

    if len(labels) == 1:
        return next(iter(labels)), explicit
    return None, False


def canonical_path_group(rel_path: Path) -> str:
    """
    Canonical experiment-family id from path.
    mvcc/nolock tokens are normalized to `{cc}` so only sibling families pair.
    """
    out_parts: List[str] = []
    for part in rel_path.parts[:-1]:
        p = part.lower()
        # token-level replacement only.
        p = re.sub(r"(^|[_\-.])mvcc([_\-.]|$)", r"\1{cc}\2", p)
        p = re.sub(r"(^|[_\-.])nolock([_\-.]|$)", r"\1{cc}\2", p)
        out_parts.append(p)
    return "/".join(out_parts)


def load_rows(result_root: Path) -> List[Row]:
    rows: List[Row] = []
    for csv_path in sorted(result_root.glob("**/benchmark_results.csv")):
        rel = csv_path.relative_to(result_root)
        label, explicit = infer_cc_label(rel)
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for idx, raw in enumerate(reader):
                rows.append(
                    Row(
                        path=csv_path,
                        rel_path=str(rel),
                        row_index=idx,
                        cc_label=label,
                        cc_label_explicit=explicit,
                        raw=dict(raw),
                        stats=parse_stats(raw.get("stats", "")),
                    )
                )
    return rows


def get_stat_ratio(stats: Dict[str, float], num: str, den: str) -> Optional[float]:
    a = stats.get(num)
    b = stats.get(den)
    if a is None or b is None or b == 0:
        return None
    return a / b


def row_pair_key(row: Row) -> Tuple[str, ...]:
    return tuple(str(row.raw.get(c, "")).strip() for c in PAIR_KEY_COLS)


def bootstrap_ci(
    values: List[float], rng: random.Random, n_boot: int = 20000, alpha: float = 0.05
) -> Dict[str, float]:
    if not values:
        return {}
    n = len(values)
    means: List[float] = []
    medians: List[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        sample.sort()
        means.append(sum(sample) / n)
        medians.append(sample[n // 2])
    means.sort()
    medians.sort()
    lo = int((alpha / 2) * n_boot)
    hi = int((1 - alpha / 2) * n_boot)
    if hi >= n_boot:
        hi = n_boot - 1
    return {
        "mean": sum(values) / len(values),
        "median": sorted(values)[len(values) // 2],
        "min": min(values),
        "max": max(values),
        "ci_mean_low": means[lo],
        "ci_mean_high": means[hi],
        "ci_median_low": medians[lo],
        "ci_median_high": medians[hi],
    }


def trimmed_mean(values: List[float], proportion: float = 0.1) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = int(len(s) * proportion)
    if 2 * k >= len(s):
        return sum(s) / len(s)
    core = s[k : len(s) - k]
    return sum(core) / len(core)


def sign_test_two_sided(n_pos: int, n_neg: int) -> Optional[float]:
    n = n_pos + n_neg
    if n == 0:
        return None
    k = min(n_pos, n_neg)
    # Two-sided exact binomial p-value for p=0.5.
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    p = min(1.0, 2.0 * tail)
    return p


def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None

    def ranks(vals: List[float]) -> List[float]:
        idx = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[idx[j + 1]] == vals[idx[i]]:
                j += 1
            r = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[idx[k]] = r
            i = j + 1
        return out

    rx = ranks(xs)
    ry = ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)


def analyze_pairs(rows: List[Row], rng: random.Random) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    eligible = [
        r
        for r in rows
        if r.cc_label_explicit and r.cc_label in ("mvcc", "nolock")
    ]
    groups: Dict[Tuple[str, ...], Dict[str, List[Row]]] = defaultdict(lambda: {"mvcc": [], "nolock": []})
    for row in eligible:
        path_group = canonical_path_group(Path(row.rel_path))
        groups[row_pair_key(row) + (path_group,)][row.cc_label].append(row)

    pair_records: List[Dict[str, object]] = []
    skipped_ambiguous = 0
    for key, bucket in groups.items():
        mvcc_rows = bucket["mvcc"]
        nolock_rows = bucket["nolock"]
        if len(mvcc_rows) != 1 or len(nolock_rows) != 1:
            skipped_ambiguous += 1
            continue
        mvcc = mvcc_rows[0]
        nolock = nolock_rows[0]

        rec = {
            "pair_key": "|".join(key),
            "dataset_name": mvcc.raw.get("dataset_name", ""),
            "threads": mvcc.raw.get("threads", ""),
            "write_batch_size": mvcc.raw.get("write_batch_size", ""),
            "insert_input_rate": mvcc.raw.get("insert_input_rate", ""),
            "search_input_rate": mvcc.raw.get("search_input_rate", ""),
            "mvcc_path": mvcc.rel_path,
            "nolock_path": nolock.rel_path,
            "path_group": canonical_path_group(Path(mvcc.rel_path)),
        }

        for col, short in [
            (P99_OP_COL, "p99_op"),
            (P99_E2E_COL, "p99_e2e"),
            (SEARCH_QPS_COL, "search_qps"),
            ("insert_qps (per-point)", "insert_qps"),
            ("recall", "recall"),
        ]:
            a = to_float(mvcc.raw.get(col))
            b = to_float(nolock.raw.get(col))
            rec[f"mvcc_{short}"] = a
            rec[f"nolock_{short}"] = b
            rec[f"delta_{short}"] = (a - b) if (a is not None and b is not None) else None

        for short, num, den in [
            ("rec_per_modified", "m2_recovered", "m2_search_expand_modified"),
            ("useful_ratio", "m2_useful", "m2_recovered"),
            ("lazy_skip_node_ratio", "m2_lazy_skip_nodes", "m2_total_nodes"),
        ]:
            a = get_stat_ratio(mvcc.stats, num, den)
            b = get_stat_ratio(nolock.stats, num, den)
            rec[f"mvcc_{short}"] = a
            rec[f"nolock_{short}"] = b
            rec[f"delta_{short}"] = (a - b) if (a is not None and b is not None) else None

        for short, stat_key in [
            ("undo_max_edges", "undo_max_edges"),
            ("m2_recovered", "m2_recovered"),
            ("m2_useful", "m2_useful"),
        ]:
            a = mvcc.stats.get(stat_key)
            b = nolock.stats.get(stat_key)
            rec[f"mvcc_{short}"] = a
            rec[f"nolock_{short}"] = b
            rec[f"delta_{short}"] = (a - b) if (a is not None and b is not None) else None

        pair_records.append(rec)

    delta_p99_op = [x["delta_p99_op"] for x in pair_records if isinstance(x.get("delta_p99_op"), float)]
    delta_p99_e2e = [x["delta_p99_e2e"] for x in pair_records if isinstance(x.get("delta_p99_e2e"), float)]
    delta_search_qps = [x["delta_search_qps"] for x in pair_records if isinstance(x.get("delta_search_qps"), float)]
    abs_delta_p99 = [abs(v) for v in delta_p99_op]

    def corr_with_delta(mechanism_col: str) -> Optional[float]:
        xs: List[float] = []
        ys: List[float] = []
        for rec in pair_records:
            x = rec.get(mechanism_col)
            y = rec.get("delta_p99_op")
            if isinstance(x, float) and isinstance(y, float):
                xs.append(x)
                ys.append(y)
        return spearman(xs, ys)

    per_dataset: Dict[str, List[float]] = defaultdict(list)
    for rec in pair_records:
        d = rec.get("dataset_name", "")
        v = rec.get("delta_p99_op")
        if isinstance(d, str) and isinstance(v, float):
            per_dataset[d].append(v)

    dataset_summary: Dict[str, Dict[str, float]] = {}
    for ds, vals in sorted(per_dataset.items()):
        if not vals:
            continue
        dataset_summary[ds] = bootstrap_ci(vals, rng, n_boot=10000)
        dataset_summary[ds]["n_pairs"] = len(vals)

    out = {
        "eligible_rows": len(eligible),
        "pair_group_count": len(groups),
        "skipped_ambiguous_groups": skipped_ambiguous,
        "strict_pair_count": len(pair_records),
        "delta_p99_op_positive_count": sum(1 for v in delta_p99_op if v > 0),
        "delta_p99_op_negative_count": sum(1 for v in delta_p99_op if v < 0),
        "delta_p99_op_summary": bootstrap_ci(delta_p99_op, rng),
        "delta_p99_e2e_summary": bootstrap_ci(delta_p99_e2e, rng),
        "delta_search_qps_summary": bootstrap_ci(delta_search_qps, rng),
        "abs_delta_p99_op_summary": bootstrap_ci(abs_delta_p99, rng),
        "mechanism_spearman": {
            "delta_p99_op_vs_delta_rec_per_modified": corr_with_delta("delta_rec_per_modified"),
            "delta_p99_op_vs_delta_useful_ratio": corr_with_delta("delta_useful_ratio"),
            "delta_p99_op_vs_delta_lazy_skip_node_ratio": corr_with_delta("delta_lazy_skip_node_ratio"),
            "delta_p99_op_vs_delta_undo_max_edges": corr_with_delta("delta_undo_max_edges"),
        },
        "per_dataset": dataset_summary,
    }
    out["delta_p99_op_trimmed_mean_10pct"] = trimmed_mean(delta_p99_op, 0.1)
    out["delta_p99_op_sign_test_pvalue"] = sign_test_two_sided(
        out["delta_p99_op_positive_count"], out["delta_p99_op_negative_count"]
    )
    ds_means = [blk["mean"] for blk in dataset_summary.values() if isinstance(blk.get("mean"), float)]
    out["delta_p99_op_macro_mean_across_datasets"] = (sum(ds_means) / len(ds_means)) if ds_means else None
    return pair_records, out


def analyze_freshness(rows: List[Row]) -> Dict[str, object]:
    records = []
    for row in rows:
        e2e = to_float(row.raw.get(P99_E2E_COL))
        op = to_float(row.raw.get(P99_OP_COL))
        q = to_float(row.raw.get("freshness_queue_pts_p99"))
        c = to_float(row.raw.get("freshness_contiguity_pts_p99"))
        t = to_float(row.raw.get("freshness_total_pts_p99"))
        if None in (e2e, op, q, c, t):
            continue
        records.append((e2e, op, q, c, t, row.rel_path))

    if not records:
        return {"n_rows": 0}

    e2e_vals = [r[0] for r in records]
    op_vals = [r[1] for r in records]
    q_vals = [r[2] for r in records]
    c_vals = [r[3] for r in records]
    d_vals = [r[0] - r[1] for r in records]

    out: Dict[str, object] = {
        "n_rows": len(records),
        "spearman_e2e_vs_op": spearman(e2e_vals, op_vals),
        "spearman_delta_e2e_minus_op_vs_queue_p99": spearman(d_vals, q_vals),
        "spearman_delta_e2e_minus_op_vs_contiguity_p99": spearman(d_vals, c_vals),
        "delta_e2e_minus_op_mean_ms": sum(d_vals) / len(d_vals),
        "delta_e2e_minus_op_median_ms": sorted(d_vals)[len(d_vals) // 2],
        "delta_e2e_minus_op_max_ms": max(d_vals),
    }

    top = sorted(records, key=lambda x: x[0] - x[1], reverse=True)[:5]
    out["top_delta_cases"] = [
        {
            "path": r[5],
            "p99_e2e_ms": r[0],
            "p99_op_ms": r[1],
            "delta_ms": r[0] - r[1],
            "queue_p99_pts": r[2],
            "contiguity_p99_pts": r[3],
            "total_p99_pts": r[4],
        }
        for r in top
    ]

    if np is not None and len(records) >= 5:
        # Standardized OLS: delta = beta_q * queue + beta_c * contiguity.
        y = np.array(d_vals, dtype=float)
        x = np.array([[r[2], r[3]] for r in records], dtype=float)
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x_std[x_std == 0] = 1.0
        xz = (x - x_mean) / x_std
        y_mean = y.mean()
        y_std = y.std()
        yz = (y - y_mean) / (y_std if y_std != 0 else 1.0)
        X = np.column_stack([np.ones(len(xz)), xz])
        beta = np.linalg.lstsq(X, yz, rcond=None)[0]
        yhat = X @ beta
        ss_res = float(((yz - yhat) ** 2).sum())
        ss_tot = float(((yz - yz.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
        out["delta_regression"] = {
            "std_beta_queue": float(beta[1]),
            "std_beta_contiguity": float(beta[2]),
            "r2": r2,
        }

    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as f:
            f.write("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_report(summary: Dict[str, object]) -> str:
    pair = summary["pair_analysis"]
    fresh = summary["freshness_analysis"]

    lines: List[str] = []
    lines.append("# p99 Attribution Review")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Labels are inferred only from explicit directory tokens (`mvcc`, `nolock`).")
    lines.append("- Mixed-label directories (e.g., `mvcc_vs_nolock_*`) are excluded from strict pairing.")
    lines.append("- Pairing is strict 1:1 on identical config fields.")
    lines.append("")
    lines.append("## Pairing Summary")
    lines.append(f"- Eligible labeled rows: {pair['eligible_rows']}")
    lines.append(f"- Pair groups: {pair['pair_group_count']}")
    lines.append(f"- Ambiguous groups skipped: {pair['skipped_ambiguous_groups']}")
    lines.append(f"- Strict pairs used: {pair['strict_pair_count']}")
    lines.append(
        f"- Sign count on Δp99_op: positive={pair['delta_p99_op_positive_count']}, "
        f"negative={pair['delta_p99_op_negative_count']}"
    )
    lines.append("")

    def emit_block(title: str, block: Dict[str, float]) -> None:
        if not block:
            lines.append(f"- {title}: n/a")
            return
        lines.append(
            f"- {title}: mean={block['mean']:.4f}, median={block['median']:.4f}, "
            f"95% CI(mean)=[{block['ci_mean_low']:.4f}, {block['ci_mean_high']:.4f}]"
        )

    lines.append("## Pair Effect (mvcc - nolock)")
    emit_block("Δ search p99 op (ms)", pair.get("delta_p99_op_summary", {}))
    emit_block("Δ search p99 e2e (ms)", pair.get("delta_p99_e2e_summary", {}))
    emit_block("Δ search qps", pair.get("delta_search_qps_summary", {}))
    emit_block("|Δ search p99 op| (ms)", pair.get("abs_delta_p99_op_summary", {}))
    tm = pair.get("delta_p99_op_trimmed_mean_10pct")
    if isinstance(tm, float):
        lines.append(f"- Δ search p99 op (10% trimmed mean): {tm:.4f}")
    macro = pair.get("delta_p99_op_macro_mean_across_datasets")
    if isinstance(macro, float):
        lines.append(f"- Δ search p99 op (dataset-macro mean): {macro:.4f}")
    pval = pair.get("delta_p99_op_sign_test_pvalue")
    if isinstance(pval, float):
        lines.append(f"- Sign test two-sided p-value: {pval:.6f}")
    lines.append("")

    lines.append("## Mechanism Attribution (Pair Delta Correlation)")
    ms = pair.get("mechanism_spearman", {})
    for k, v in ms.items():
        if v is None:
            lines.append(f"- {k}: n/a")
        else:
            lines.append(f"- {k}: spearman={v:.4f}")
    lines.append("")

    lines.append("## Dataset-Stratified Pair Effect")
    per_ds = pair.get("per_dataset", {})
    if not per_ds:
        lines.append("- n/a")
    else:
        for ds, blk in per_ds.items():
            lines.append(
                f"- {ds}: n={int(blk['n_pairs'])}, mean={blk['mean']:.4f}, median={blk['median']:.4f}, "
                f"95% CI(mean)=[{blk['ci_mean_low']:.4f}, {blk['ci_mean_high']:.4f}]"
            )
    lines.append("")

    lines.append("## Freshness Decomposition")
    lines.append(f"- Rows with freshness metrics: {fresh.get('n_rows', 0)}")
    if fresh.get("n_rows", 0) > 0:
        lines.append(f"- spearman(e2e, op): {fresh.get('spearman_e2e_vs_op'):.4f}")
        lines.append(
            f"- spearman(e2e-op, queue_p99): {fresh.get('spearman_delta_e2e_minus_op_vs_queue_p99'):.4f}"
        )
        lines.append(
            f"- spearman(e2e-op, contiguity_p99): {fresh.get('spearman_delta_e2e_minus_op_vs_contiguity_p99'):.4f}"
        )
        lines.append(
            f"- delta(e2e-op) ms: mean={fresh.get('delta_e2e_minus_op_mean_ms'):.4f}, "
            f"median={fresh.get('delta_e2e_minus_op_median_ms'):.4f}, max={fresh.get('delta_e2e_minus_op_max_ms'):.4f}"
        )
        reg = fresh.get("delta_regression")
        if isinstance(reg, dict):
            lines.append(
                f"- standardized OLS on delta(e2e-op): beta_queue={reg['std_beta_queue']:.4f}, "
                f"beta_contiguity={reg['std_beta_contiguity']:.4f}, R2={reg['r2']:.4f}"
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("- This report provides controlled association evidence, not formal causality proof.")
    lines.append("- For definitive causality, run repeated randomized controlled experiments with fixed seeds.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rigorous p99 attribution review")
    parser.add_argument(
        "--result-root",
        default="bench/result",
        help="Root directory containing benchmark result subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for artifacts; default: results/p99_attribution_review_<timestamp>",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap")
    args = parser.parse_args()

    result_root = Path(args.result_root).resolve()
    if not result_root.exists():
        raise SystemExit(f"result root does not exist: {result_root}")

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results").resolve() / f"p99_attribution_review_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows = load_rows(result_root)
    pair_rows, pair_summary = analyze_pairs(rows, rng)
    freshness_summary = analyze_freshness(rows)

    summary = {
        "result_root": str(result_root),
        "output_dir": str(out_dir),
        "seed": args.seed,
        "row_count_total": len(rows),
        "pair_analysis": pair_summary,
        "freshness_analysis": freshness_summary,
    }

    write_csv(out_dir / "pair_deltas.csv", pair_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "report.md").write_text(make_report(summary))

    print(f"rows_total={len(rows)}")
    print(f"strict_pairs={pair_summary['strict_pair_count']}")
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
