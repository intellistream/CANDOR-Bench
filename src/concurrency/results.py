"""Result file writers (binary layouts shared with the C++ tools).

Two binary layouts, both little-endian:

* overall ``.res``: ``int32 n, int32 k`` then ``n*k uint32`` result tags.
* incremental ``.res``: ``uint64 num_entries`` then per entry
  ``uint64 insert_offset, uint64 query_tag, uint64 num_tags,
  num_tags * uint32 tags`` — sorted by (insert_offset, query_tag).

Plus the per-experiment directory scheme from the README:
``<output_dir>/<index_type>/<dataset>/<query_mode>/files/``.
"""

from __future__ import annotations

import csv
import os
import struct
from typing import Dict, Iterable, List, Tuple

import numpy as np

# One search record: (insert_offset, query_tag, result tag array).
Record = Tuple[int, int, np.ndarray]


def result_dir(cfg: dict) -> str:
    base = cfg["result"].get("output_dir") or "."
    return os.path.join(
        base,
        cfg["index"].get("index_type", "hnsw"),
        cfg["data"].get("dataset_name", "dataset"),
        cfg["workload"].get("query_mode", "round_robin"),
        "files",
    )


def write_overall_res(path: str, tags: np.ndarray) -> None:
    """tags: (n, k) uint32 search results with full visibility."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", tags.shape[0], tags.shape[1]))
        tags.astype("<u4").tofile(f)


def write_incr_res(path: str, records: Iterable[Record]) -> None:
    entries = sorted(records, key=lambda r: (r[0], r[1]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(entries)))
        for offset, query_tag, tags in entries:
            tags = np.asarray(tags, dtype="<u4")
            f.write(struct.pack("<QQQ", offset, query_tag, len(tags)))
            tags.tofile(f)
            # trailing num_dists, always zero — readers of the format
            # expect it per entry
            f.write(struct.pack("<Q", 0))


def records_from_run(records: dict) -> List[Record]:
    """Convert concurrency_native run records to the Record list shape."""
    offsets = np.asarray(records["insert_offsets"])
    qtags = np.asarray(records["query_tags"])
    rtags = np.asarray(records["result_tags"])
    return [
        (int(offsets[i]), int(qtags[i]), rtags[i]) for i in range(len(offsets))
    ]


def write_partial_diff_csv(path: str, diff: Dict[int, list]) -> None:
    """Stage-bucketed recall-diff summaries."""
    if not path or not diff:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stages = sorted(diff)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stage_offset", "diff (baseline - partial)", "count",
            "percentage", "avg_recall_base", "avg_recall_partial",
            "avg_future_tags",
        ])
        for idx, stage in enumerate(stages):
            rows = diff[stage]
            for r in rows:
                writer.writerow([
                    r["stage_offset"], f"{r['diff']:.6f}", r["count"],
                    f"{r['percentage']:.4f}",
                    f"{r['sum_recall_base'] / r['count']:.6f}",
                    f"{r['sum_recall_partial'] / r['count']:.6f}",
                    f"{r['sum_future_tags'] / r['count']:.6f}",
                ])
            if rows and idx < len(stages) - 1 and any(
                diff[s] for s in stages[idx + 1:]
            ):
                writer.writerow([""] * 7)
