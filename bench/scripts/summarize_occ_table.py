#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


def family_of(write_rate: float, search_rate: float) -> str:
    if write_rate == search_rate:
        return "BL"
    if search_rate > write_rate:
        return "HR"
    return "HW"


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: summarize_occ_table.py <dataset> <out_csv>", file=sys.stderr)
        return 2

    dataset = sys.argv[1]
    out_csv = Path(sys.argv[2])
    root = Path("/home/junyao/code/ANN-CC-Bench")
    inputs = [
        root / "results" / f"{dataset}_occ_t16" / "benchmark_results.csv",
        root / "results" / f"{dataset}_occ_t32" / "benchmark_results.csv",
        root / "results" / f"{dataset}_occ_t64" / "benchmark_results.csv",
    ]

    rows = []
    for path in inputs:
        if not path.exists():
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                write_rate = float(row["insert_input_rate"])
                search_rate = float(row["search_input_rate"])
                rows.append(
                    {
                        "threads": int(row["threads"]),
                        "batch": int(row["write_batch_size"]),
                        "family": family_of(write_rate, search_rate),
                        "rate": f"{int(write_rate/1000)}k:{int(search_rate/1000)}k",
                        "search_qps": f"{float(row['search_qps (per-point)']):.2f}",
                        "occ_pct": f"{100.0 * float(row['inflight_occ_share_search_op']):.2f}",
                        "diff_pts": f"{float(row['inflight_bf_pts_mean']):.2f}",
                    }
                )

    rows.sort(key=lambda r: (r["threads"], r["batch"], {"BL": 0, "HR": 1, "HW": 2}[r["family"]]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threads", "batch", "family", "rate", "search_qps", "occ_pct", "diff_pts"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
