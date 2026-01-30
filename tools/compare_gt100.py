#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Tuple

import numpy as np

# Reuse the parser from inspect_gt100.py.
# `tools/` is not a Python package, so add it to sys.path for direct import.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from inspect_gt100 import read_gt100


def compare_first_n(
    a_path: str,
    b_path: str,
    n_queries: int,
    k: int,
    compare_dist: bool,
    dist_atol: float,
) -> Tuple[bool, str]:
    nq_a, k_a, ids_a, dists_a = read_gt100(a_path)
    nq_b, k_b, ids_b, dists_b = read_gt100(b_path)

    if nq_a != nq_b or k_a != k_b:
        return False, f"Shape mismatch: A(nq={nq_a},k={k_a}) vs B(nq={nq_b},k={k_b})"

    nq = nq_a
    k_eff = min(k, k_a)
    q_eff = min(n_queries, nq)

    all_ok = True
    lines = []
    lines.append(f"A: {a_path}")
    lines.append(f"B: {b_path}")
    lines.append(f"Compare: queries [0,{q_eff}) topK={k_eff} (fileK={k_a})")
    lines.append(f"Compare distances: {compare_dist} (atol={dist_atol})")

    for q in range(q_eff):
        row_ids_a = ids_a[q, :k_eff]
        row_ids_b = ids_b[q, :k_eff]

        if not np.array_equal(row_ids_a, row_ids_b):
            all_ok = False
            diff_idx = int(np.flatnonzero(row_ids_a != row_ids_b)[0])
            lines.append(
                f"q{q}: ID mismatch at rank {diff_idx}: A={int(row_ids_a[diff_idx])} B={int(row_ids_b[diff_idx])}"
            )
            continue

        if compare_dist:
            row_dist_a = dists_a[q, :k_eff]
            row_dist_b = dists_b[q, :k_eff]
            if not np.allclose(row_dist_a, row_dist_b, atol=dist_atol, rtol=0.0):
                all_ok = False
                diff_idx = int(np.flatnonzero(np.abs(row_dist_a - row_dist_b) > dist_atol)[0])
                lines.append(
                    f"q{q}: DIST mismatch at rank {diff_idx}: A={float(row_dist_a[diff_idx])} B={float(row_dist_b[diff_idx])}"
                )
                continue

        lines.append(f"q{q}: OK")

    if all_ok:
        lines.append("RESULT: ALL MATCH")
    else:
        lines.append("RESULT: MISMATCHES FOUND")

    return all_ok, "\n".join(lines)


def overlap_first_n(
    a_path: str,
    b_path: str,
    n_queries: int,
    k: int,
) -> Tuple[bool, str]:
    """Compute overlap percentage for each query.

    Overlap is computed on neighbor IDs as:
      |set(A_topk) ∩ set(B_topk)| / k
    """
    nq_a, k_a, ids_a, _ = read_gt100(a_path)
    nq_b, k_b, ids_b, _ = read_gt100(b_path)

    if nq_a != nq_b or k_a != k_b:
        return False, f"Shape mismatch: A(nq={nq_a},k={k_a}) vs B(nq={nq_b},k={k_b})"

    nq = nq_a
    k_eff = min(k, k_a)
    q_eff = min(n_queries, nq)

    lines = []
    lines.append(f"A: {a_path}")
    lines.append(f"B: {b_path}")
    lines.append(f"Overlap: queries [0,{q_eff}) topK={k_eff} (fileK={k_a})")

    overlaps = []
    for q in range(q_eff):
        a_row = ids_a[q, :k_eff]
        b_row = ids_b[q, :k_eff]
        inter = np.intersect1d(a_row, b_row, assume_unique=False)
        ratio = float(inter.size) / float(k_eff) if k_eff > 0 else 1.0
        overlaps.append(ratio)
        lines.append(f"q{q}: {inter.size}/{k_eff} = {ratio*100:.2f}%")

    if overlaps:
        mean_ratio = float(np.mean(overlaps))
        lines.append(f"MEAN: {mean_ratio*100:.2f}%")

    return True, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Compare two DiskANN .gt100 truthset files")
    ap.add_argument("--a", required=True, help="First .gt100 file")
    ap.add_argument("--b", required=True, help="Second .gt100 file")
    ap.add_argument("--nq", type=int, default=10, help="How many queries from start to compare")
    ap.add_argument("--k", type=int, default=100, help="Compare top-k neighbors")
    ap.add_argument(
        "--mode",
        choices=["exact", "overlap"],
        default="exact",
        help="Comparison mode: exact (IDs, optional distances) or overlap (set intersection ratio)",
    )
    ap.add_argument(
        "--compare_dist",
        action="store_true",
        help="Also compare distances (default compares only neighbor IDs)",
    )
    ap.add_argument("--dist_atol", type=float, default=0.0, help="Abs tolerance for distance compare")
    args = ap.parse_args()

    if args.mode == "overlap":
        ok, report = overlap_first_n(
            a_path=args.a,
            b_path=args.b,
            n_queries=args.nq,
            k=args.k,
        )
    else:
        ok, report = compare_first_n(
            a_path=args.a,
            b_path=args.b,
            n_queries=args.nq,
            k=args.k,
            compare_dist=args.compare_dist,
            dist_atol=args.dist_atol,
        )
    print(report)
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
