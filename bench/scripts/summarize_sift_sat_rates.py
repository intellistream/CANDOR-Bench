#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict
from pathlib import Path


def f(row, key):
    return float(row[key])


def family_of(write_rate, read_rate):
    if write_rate == read_rate:
        return "BL"
    return "HR" if read_rate > write_rate else "HW"


def offered_label(write_rate, read_rate):
    return f"{int(write_rate/1000)}k:{int(read_rate/1000)}k"


def metric_of(family, row):
    if family == "HR":
        return f(row, "search_qps (per-point)")
    if family == "HW":
        return f(row, "insert_qps (per-point)")
    return f(row, "search_qps (per-point)") + f(row, "insert_qps (per-point)")


def main():
    if len(sys.argv) != 3:
        print("usage: summarize_sift_sat_rates.py <benchmark_results.csv> <out.csv>", file=sys.stderr)
        sys.exit(2)

    in_csv = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])

    rows = []
    with in_csv.open() as fobj:
        for row in csv.DictReader(fobj):
            wr = f(row, "insert_input_rate")
            rr = f(row, "search_input_rate")
            fam = family_of(wr, rr)
            row["_family"] = fam
            row["_metric"] = metric_of(fam, row)
            rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        key = (int(row["threads"]), row["_family"])
        groups[key].append(row)

    summary = []
    for key, group in sorted(groups.items()):
        threads, fam = key
        group.sort(key=lambda r: (f(r, "insert_input_rate"), f(r, "search_input_rate")))
        max_metric = max(r["_metric"] for r in group)
        threshold = 0.95 * max_metric
        picked = None
        for row in group:
            if row["_metric"] >= threshold:
                picked = row
                break
        if picked is None:
            picked = max(group, key=lambda r: r["_metric"])

        summary.append({
            "threads": threads,
            "family": fam,
            "recommended_rate": offered_label(f(picked, "insert_input_rate"), f(picked, "search_input_rate")),
            "insert_input_rate": int(f(picked, "insert_input_rate")),
            "search_input_rate": int(f(picked, "search_input_rate")),
            "actual_insert_qps": round(f(picked, "insert_qps (per-point)"), 2),
            "actual_search_qps": round(f(picked, "search_qps (per-point)"), 2),
            "actual_total_qps": round(f(picked, "insert_qps (per-point)") + f(picked, "search_qps (per-point)"), 2),
            "selection_metric": round(picked["_metric"], 2),
            "max_metric_in_family": round(max_metric, 2),
            "threshold_95pct": round(threshold, 2),
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=[
                "threads",
                "family",
                "recommended_rate",
                "insert_input_rate",
                "search_input_rate",
                "actual_insert_qps",
                "actual_search_qps",
                "actual_total_qps",
                "selection_metric",
                "max_metric_in_family",
                "threshold_95pct",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    print(out_csv)


if __name__ == "__main__":
    main()
