#!/usr/bin/env python3
import csv
import pathlib
import statistics
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "notes" / "results" / "m2_path_corridor_full_0508"


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row, key):
    try:
        return float(row.get(key, 0) or 0)
    except ValueError:
        return 0.0


def mean(rows, key):
    vals = [f(r, key) for r in rows]
    return statistics.mean(vals) if vals else 0.0


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    groups = {}
    for csv_path in out.glob("*/benchmark_results.csv"):
        name = csv_path.parent.name
        parts = name.split("_")
        if len(parts) < 3:
            continue
        if len(parts) >= 4 and parts[-2] == "pathsig" and parts[-1] == "tight":
            variant = "pathsig_tight"
            batch = parts[-3]
            dataset = "_".join(parts[:-3])
        else:
            variant = parts[-1]
            batch = parts[-2]
            dataset = "_".join(parts[:-2])
        groups[(dataset, batch, variant)] = read_rows(csv_path)

    rows = []
    for (dataset, batch, variant), data in groups.items():
        rows.append({
            "dataset": dataset,
            "batch": batch,
            "variant": variant,
            "reps": len(data),
            "search_ms": mean(data, "search_mean_op_latency_ms (per-batch)"),
            "merge_ms": mean(data, "inflight_occ_merge_mean_ms"),
            "l2": mean(data, "inflight_occ_l2_dists_mean"),
            "filter_tests": mean(data, "inflight_occ_filter_tests_mean"),
            "candidates": mean(data, "inflight_occ_candidates_mean"),
            "fresh_pts": mean(data, "inflight_bf_pts_mean"),
            "recall": mean(data, "recall"),
        })

    by_key = {(r["dataset"], r["batch"], r["variant"]): r for r in rows}
    for r in rows:
        brute = by_key.get((r["dataset"], r["batch"], "brute"))
        if brute and r["variant"] != "brute":
            r["search_speedup_vs_brute"] = brute["search_ms"] / r["search_ms"] if r["search_ms"] else 0
            r["merge_speedup_vs_brute"] = brute["merge_ms"] / r["merge_ms"] if r["merge_ms"] else 0
            r["recall_delta_vs_brute"] = r["recall"] - brute["recall"]
        else:
            r["search_speedup_vs_brute"] = 1.0 if r["variant"] == "brute" else 0
            r["merge_speedup_vs_brute"] = 1.0 if r["variant"] == "brute" else 0
            r["recall_delta_vs_brute"] = 0

    rows.sort(key=lambda r: (r["dataset"], r["batch"], r["variant"]))
    out_csv = out / "summary.csv"
    fields = [
        "dataset",
        "batch",
        "variant",
        "reps",
        "search_ms",
        "merge_ms",
        "search_speedup_vs_brute",
        "merge_speedup_vs_brute",
        "fresh_pts",
        "l2",
        "filter_tests",
        "candidates",
        "recall",
        "recall_delta_vs_brute",
    ]
    with out_csv.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
    print(out_csv)


if __name__ == "__main__":
    main()
