#!/usr/bin/env python3
import csv
import pathlib
import re
import sys


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except ValueError:
        return default


def parse_stats(raw):
    out = {}
    for match in re.finditer(r"([A-Za-z0-9_]+):([-+0-9.eE]+)", raw or ""):
        key, value = match.groups()
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


def method_from_path(path):
    parts = path.parts
    if "path_onehop" in parts:
        return "path_onehop"
    if "brute" in parts:
        return "brute"
    return "unknown"


def load_rows(root):
    rows = []
    for csv_path in sorted(root.glob("*/*/benchmark_results.csv")):
        method = method_from_path(csv_path)
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                stats = parse_stats(row.get("stats", ""))
                winners = stats.get("m2_path_onehop_validate_fn_fresh_winners", 0.0)
                missed = stats.get("m2_path_onehop_validate_fn_missed_fresh_winners", 0.0)
                fn_rate = missed / winners if winners > 0 else 0.0
                rows.append(
                    {
                        "dataset": row.get("dataset_name", ""),
                        "method": method,
                        "write_batch": row.get("write_batch_size", ""),
                        "read_batch": row.get("read_batch_size", ""),
                        "query_mode": row.get("query_mode", ""),
                        "write_rate": row.get("insert_input_rate", ""),
                        "search_rate": row.get("search_input_rate", ""),
                        "search_qps": as_float(row, "search_qps (per-point)"),
                        "search_p99_ms": as_float(row, "search_p99_op_latency_ms (per-batch)"),
                        "search_e2e_p99_ms": as_float(row, "search_p99_e2e_latency_ms (per-batch)"),
                        "occ_mean_ms": as_float(row, "inflight_occ_merge_mean_ms"),
                        "occ_p99_ms": as_float(row, "inflight_occ_merge_p99_ms"),
                        "occ_l2": as_float(row, "inflight_occ_l2_dists_mean"),
                        "occ_filter_tests": as_float(row, "inflight_occ_filter_tests_mean"),
                        "occ_candidates": as_float(row, "inflight_occ_candidates_mean"),
                        "recall": as_float(row, "recall"),
                        "fn_winners": winners,
                        "fn_missed": missed,
                        "fn_rate": fn_rate,
                    }
                )
    return rows


def key(row):
    return (
        row["dataset"],
        row["write_batch"],
        row["read_batch"],
        row["query_mode"],
        row["write_rate"],
        row["search_rate"],
    )


def main():
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
    rows = load_rows(root)
    by_method = {(key(row), row["method"]): row for row in rows}
    summary = []
    for row in rows:
        if row["method"] != "path_onehop":
            continue
        brute = by_method.get((key(row), "brute"))
        qps_speedup = row["search_qps"] / brute["search_qps"] if brute and brute["search_qps"] > 0 else 0.0
        p99_ratio = row["search_p99_ms"] / brute["search_p99_ms"] if brute and brute["search_p99_ms"] > 0 else 0.0
        occ_l2_reduction = 1.0 - row["occ_l2"] / brute["occ_l2"] if brute and brute["occ_l2"] > 0 else 0.0
        summary.append(
            {
                **row,
                "brute_search_qps": brute["search_qps"] if brute else 0.0,
                "brute_search_p99_ms": brute["search_p99_ms"] if brute else 0.0,
                "brute_occ_l2": brute["occ_l2"] if brute else 0.0,
                "qps_speedup_vs_brute": qps_speedup,
                "p99_ratio_vs_brute": p99_ratio,
                "occ_l2_reduction_vs_brute": occ_l2_reduction,
            }
        )

    out_csv = root / "summary.csv"
    fields = [
        "dataset",
        "write_batch",
        "read_batch",
        "query_mode",
        "write_rate",
        "search_rate",
        "search_qps",
        "brute_search_qps",
        "qps_speedup_vs_brute",
        "search_p99_ms",
        "brute_search_p99_ms",
        "p99_ratio_vs_brute",
        "occ_mean_ms",
        "occ_p99_ms",
        "occ_l2",
        "brute_occ_l2",
        "occ_l2_reduction_vs_brute",
        "occ_filter_tests",
        "occ_candidates",
        "fn_winners",
        "fn_missed",
        "fn_rate",
        "recall",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(summary, key=key):
            writer.writerow({field: row.get(field, "") for field in fields})

    out_md = root / "summary.md"
    with out_md.open("w") as f:
        f.write("# M2 Path-One-Hop Full Summary\n\n")
        f.write(
            "| Dataset | W Batch | R Batch | Mode | W Rate | R Rate | QPS Speedup | P99 Ratio | OCC L2 Reduction | FN Rate | Winners | Missed |\n"
        )
        f.write("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in sorted(summary, key=key):
            f.write(
                f"| {row['dataset']} | {row['write_batch']} | {row['read_batch']} | {row['query_mode']} | "
                f"{float(row['write_rate']):.0f} | {float(row['search_rate']):.0f} | "
                f"{row['qps_speedup_vs_brute']:.3f} | {row['p99_ratio_vs_brute']:.3f} | "
                f"{row['occ_l2_reduction_vs_brute']:.3f} | {row['fn_rate']:.4f} | "
                f"{row['fn_winners']:.0f} | {row['fn_missed']:.0f} |\n"
            )

    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
