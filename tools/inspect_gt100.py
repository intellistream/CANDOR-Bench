#!/usr/bin/env python3
import argparse
import os
import struct
from typing import List, Tuple

import numpy as np


def read_gt100(gt_path: str) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """Read DiskANN truthset file written by compute_groundtruth.

    Expected layout (as indicated by DiskANN logs):
      uint32 npts, uint32 dim(K)
      uint32 id_matrix[npts * dim]
      float32 dist_matrix[npts * dim]

    Returns: (npts, k, ids[npts,k], dists[npts,k])
    """
    with open(gt_path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise RuntimeError(f"{gt_path}: file too small for header")
        npts, k = struct.unpack("<II", header)

        ids_flat = np.fromfile(f, dtype=np.uint32, count=npts * k)
        if ids_flat.size != npts * k:
            raise RuntimeError(
                f"{gt_path}: truncated while reading ids: expected {npts*k}, got {ids_flat.size}"
            )

        dists_flat = np.fromfile(f, dtype=np.float32, count=npts * k)
        if dists_flat.size != npts * k:
            raise RuntimeError(
                f"{gt_path}: truncated while reading dists: expected {npts*k}, got {dists_flat.size}"
            )

    ids = ids_flat.reshape(npts, k)
    dists = dists_flat.reshape(npts, k)
    return npts, k, ids, dists


def format_one_query(qid: int, ids_row: np.ndarray, dists_row: np.ndarray, limit_k: int) -> str:
    pairs = []
    for j in range(min(limit_k, ids_row.shape[0])):
        pairs.append(f"{int(ids_row[j])}:{float(dists_row[j]):.6f}")
    return f"q{qid}\t" + ", ".join(pairs)


def main():
    ap = argparse.ArgumentParser(description="Inspect DiskANN compute_groundtruth .gt100 files")
    ap.add_argument("--gt_file", required=True, help="Path to .gt100 truthset")
    ap.add_argument("--query", type=int, action="append", default=None, help="Query index to print (repeatable)")
    ap.add_argument("--start", type=int, default=0, help="Start query index (used when --query not set)")
    ap.add_argument("--count", type=int, default=5, help="How many queries to print (used when --query not set)")
    ap.add_argument("--k", type=int, default=10, help="How many neighbors to print per query")
    ap.add_argument("--out", default=None, help="Optional output path (TSV)")
    args = ap.parse_args()

    gt_path = args.gt_file
    if not os.path.exists(gt_path):
        raise SystemExit(f"gt_file not found: {gt_path}")

    npts, k, ids, dists = read_gt100(gt_path)

    if args.query is not None and len(args.query) > 0:
        query_indices: List[int] = args.query
    else:
        query_indices = list(range(args.start, min(args.start + args.count, npts)))

    lines: List[str] = []
    lines.append(f"file={gt_path}")
    lines.append(f"nq={npts} k={k} (printing top {min(args.k, k)})")

    for qid in query_indices:
        if qid < 0 or qid >= npts:
            lines.append(f"q{qid}\t<out of range>")
            continue
        lines.append(format_one_query(qid, ids[qid], dists[qid], args.k))

    text = "\n".join(lines)
    print(text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for line in lines[2:]:
                f.write(line)
                f.write("\n")


if __name__ == "__main__":
    main()
