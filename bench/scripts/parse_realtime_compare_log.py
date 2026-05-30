#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path


VARIANT_RE = re.compile(r"==== Running config variant: (.+?) \(rep ")
MODE_RE = re.compile(
    r"Realtime (wolock|wlock): throughput insert=([0-9.]+) pts/s, search=([0-9.]+) pts/s .* queries=([0-9]+)"
)


def parse_variant(name):
    threads = None
    write_rate = None
    read_rate = None
    for part in name.split("__"):
        if part.startswith("w") and "-r" in part:
            left, right = part.split("-r", 1)
            write_rate = left[1:]
            read_rate = right
        elif part.startswith("num_threads"):
            threads = part.replace("num_threads", "")
    return threads, write_rate, read_rate


def main():
    if len(sys.argv) != 3:
        print("usage: parse_realtime_compare_log.py <run.log> <out.csv>", file=sys.stderr)
        sys.exit(2)

    log_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    current_variant = None
    rows = {}

    for line in log_path.read_text().splitlines():
        m = VARIANT_RE.search(line)
        if m:
            current_variant = m.group(1)
            rows.setdefault(current_variant, {})
            continue
        m = MODE_RE.search(line)
        if m and current_variant is not None:
            mode, ins, sea, q = m.groups()
            rows[current_variant][mode] = {
                "insert_qps": float(ins),
                "search_qps": float(sea),
                "queries": int(q),
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant",
            "threads",
            "write_rate",
            "read_rate",
            "wolock_insert_qps",
            "wolock_search_qps",
            "wlock_insert_qps",
            "wlock_search_qps",
            "search_speedup",
            "insert_speedup",
            "queries_replayed",
        ])
        for variant in sorted(rows):
            entry = rows[variant]
            if "wolock" not in entry or "wlock" not in entry:
                continue
            threads, write_rate, read_rate = parse_variant(variant)
            wol = entry["wolock"]
            wlk = entry["wlock"]
            writer.writerow([
                variant,
                threads,
                write_rate,
                read_rate,
                f"{wol['insert_qps']:.4f}",
                f"{wol['search_qps']:.4f}",
                f"{wlk['insert_qps']:.4f}",
                f"{wlk['search_qps']:.4f}",
                f"{(wol['search_qps'] / wlk['search_qps']):.4f}" if wlk["search_qps"] else "",
                f"{(wol['insert_qps'] / wlk['insert_qps']):.4f}" if wlk["insert_qps"] else "",
                wol["queries"],
            ])

    print(out_path)


if __name__ == "__main__":
    main()
