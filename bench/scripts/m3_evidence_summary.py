#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path


def weighted_incr_recall(rc_path: Path):
    if not rc_path or not rc_path.exists():
        return None, 0
    total_q = 0
    total_weighted = 0.0
    with rc_path.open() as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                avg = float(parts[1])
                q = int(parts[3])
            except ValueError:
                continue
            total_q += q
            total_weighted += avg * q
    if total_q == 0:
        return None, 0
    return total_weighted / total_q, total_q


def parse_stats(stats: str):
    out = {}
    for k in [
        "slipstream_publish",
        "slipstream_hit",
        "slipstream_miss",
        "slipstream_seeds_injected",
        "slipstream_quality_reject",
        "slipstream_gate",
        "slipstream_skip_ratio",
    ]:
        m = re.search(rf"\b{k}:([-+0-9.eE]+)", stats)
        if m:
            out[k] = m.group(1)
    return out


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_root>")
        return 1

    root = Path(sys.argv[1]).resolve()
    csv_paths = sorted(root.glob("*/benchmark_results.csv"))
    if not csv_paths:
        print(f"No benchmark_results.csv under {root}")
        return 2

    rows = []
    for p in csv_paths:
        with p.open() as f:
            dr = csv.DictReader(f)
            data = list(dr)
        if not data:
            continue
        r = data[-1]

        rc_files = sorted(p.parent.glob("**/*.rc"))
        rc = rc_files[0] if rc_files else None
        winc, qcnt = weighted_incr_recall(rc)
        s = parse_stats(r.get("stats", ""))

        rows.append(
            {
                "run": p.parent.name,
                "algo": r.get("algorithm", ""),
                "dataset": r.get("dataset_name", ""),
                "insert_qps": float(r.get("insert_qps (per-point)", "0") or 0),
                "search_qps": float(r.get("search_qps (per-point)", "0") or 0),
                "recall": float(r.get("recall", "0") or 0),
                "weighted_incr_recall": winc,
                "weighted_query_count": qcnt,
                "slipstream_publish": s.get("slipstream_publish", ""),
                "slipstream_hit": s.get("slipstream_hit", ""),
                "slipstream_miss": s.get("slipstream_miss", ""),
                "slipstream_seeds_injected": s.get("slipstream_seeds_injected", ""),
                "slipstream_quality_reject": s.get("slipstream_quality_reject", ""),
                "slipstream_gate": s.get("slipstream_gate", ""),
                "slipstream_skip_ratio": s.get("slipstream_skip_ratio", ""),
                "rc_file": str(rc) if rc else "",
                "csv_file": str(p),
            }
        )

    rows.sort(key=lambda x: (x["dataset"], x["algo"]))

    out_csv = root / "m3_evidence_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_md = root / "m3_evidence_summary.md"
    with out_md.open("w") as f:
        f.write("# M3 Evidence Summary\n\n")
        f.write("| dataset | algo | insert_qps | search_qps | recall | weighted_incr_recall(%) | q_count | slipstream_publish | hit | miss | quality_reject |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            winc = "" if r["weighted_incr_recall"] is None else f"{r['weighted_incr_recall']:.4f}"
            f.write(
                f"| {r['dataset']} | {r['algo']} | {r['insert_qps']:.2f} | {r['search_qps']:.2f} | {r['recall']:.3f} | {winc} | {r['weighted_query_count']} | {r['slipstream_publish']} | {r['slipstream_hit']} | {r['slipstream_miss']} | {r['slipstream_quality_reject']} |\n"
            )

    print(f"wrote: {out_csv}")
    print(f"wrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
