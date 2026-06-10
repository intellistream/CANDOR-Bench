"""Incremental ground-truth loading and recall scoring.

File format (produced by the GT generation tooling):

* Index file (text): one line per split, ``<filename> <start_offset> <end_offset>``;
  blank lines and ``#`` comments ignored. Offsets are the committed-insert
  offsets the split covers (inclusive).
* Split file (binary, little-endian):
  ``int32 n_per_batch, int32 k, int32 num_batches`` then per batch:
  ``uint64 offset, [uint32 query_tags[n] if present], float32 dists[n*k],
  uint32 indices[n*k]``. The optional query-tag block is detected from the
  payload size.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

_HEADER = struct.Struct("<iii")
_OFFSET = struct.Struct("<Q")


@dataclass
class SplitFileInfo:
    filename: str
    start_offset: int
    end_offset: int


class IncrementalGTProvider:
    """Serves ground-truth top-k for (committed insert offset, query tag).

    Only one split file is held in memory at a time, so callers should
    group lookups by offset (``evaluate_records`` sorts for you).
    """

    def __init__(self, index_path: str, recall_at: int):
        self.base_dir = os.path.dirname(os.path.abspath(index_path))
        self.recall_at = int(recall_at)
        self.splits = self._parse_index(index_path)
        if not self.splits:
            raise ValueError(
                f"incremental ground truth index {index_path} contained no entries"
            )
        self._cached_file: Optional[str] = None
        # cache: offset -> query_tag -> np.ndarray[uint32] (top recall_at tags)
        self._cache: Dict[int, Dict[int, np.ndarray]] = {}

    @staticmethod
    def _parse_index(index_path: str) -> List[SplitFileInfo]:
        splits: List[SplitFileInfo] = []
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 3:
                    continue
                start, end = int(fields[1]), int(fields[2])
                splits.append(
                    SplitFileInfo(fields[0], max(start, 0), max(end, 0))
                )
        return splits

    def top_k(self, offset: int, query_tag: int) -> Optional[np.ndarray]:
        """Ground-truth tags for query_tag at committed offset, or None."""
        split = self._find_split(offset)
        if split is None:
            return None
        self._ensure_file_loaded(split.filename)
        stage = self._cache.get(offset)
        if stage is None:
            return None
        return stage.get(query_tag)

    def stage_map(self, offset: int) -> Optional[Dict[int, np.ndarray]]:
        """All ground-truth entries for one committed offset, or None."""
        split = self._find_split(offset)
        if split is None:
            return None
        self._ensure_file_loaded(split.filename)
        return self._cache.get(offset)

    def _find_split(self, offset: int) -> Optional[SplitFileInfo]:
        for split in self.splits:
            if split.start_offset <= offset <= split.end_offset:
                return split
        return None

    def _ensure_file_loaded(self, filename: str) -> None:
        if self._cached_file == filename:
            return
        self._load_split_file(filename)
        self._cached_file = filename

    def _load_split_file(self, filename: str) -> None:
        path = os.path.join(self.base_dir, filename)
        with open(path, "rb") as f:
            header = f.read(_HEADER.size)
            if len(header) != _HEADER.size:
                raise ValueError(f"split {path} too short for header")
            n_per_batch, k_file, num_batches = _HEADER.unpack(header)
            if n_per_batch <= 0 or k_file <= 0 or num_batches <= 0:
                raise ValueError(f"split {path} has invalid header values")

            payload_size = os.fstat(f.fileno()).st_size - _HEADER.size
            per_batch_base = 8 + n_per_batch * k_file * (4 + 4)
            expected = per_batch_base * num_batches
            query_tag_bytes = n_per_batch * 4 * num_batches
            if payload_size == expected + query_tag_bytes:
                has_query_tags = True
            elif payload_size == expected:
                has_query_tags = False
            else:
                raise ValueError(
                    f"split {path} has unexpected size: got {payload_size}, "
                    f"expected {expected} or {expected + query_tag_bytes}"
                )

            limit = self.recall_at
            if limit <= 0 or limit > k_file:
                limit = k_file

            self._cache = {}
            nk = n_per_batch * k_file
            for _ in range(num_batches):
                (offset,) = _OFFSET.unpack(f.read(_OFFSET.size))
                if has_query_tags:
                    query_tags = np.frombuffer(
                        f.read(n_per_batch * 4), dtype="<u4"
                    ).astype(np.int64)
                else:
                    query_tags = np.arange(n_per_batch, dtype=np.int64)
                f.seek(nk * 4, os.SEEK_CUR)  # distances, unused
                indices = np.frombuffer(f.read(nk * 4), dtype="<u4").reshape(
                    n_per_batch, k_file
                )
                stage = self._cache.setdefault(int(offset), {})
                topk = indices[:, :limit]
                for i in range(n_per_batch):
                    stage[int(query_tags[i])] = topk[i]


def overall_recall_against_gt(
    result_tags: np.ndarray, gt: np.ndarray, recall_at: int
) -> float:
    """Mean recall of (n, k) results against (n, >=k) ground truth rows."""
    n = len(result_tags)
    if n == 0:
        return 0.0
    return sum(
        compute_recall_against_gt(gt[i], result_tags[i], recall_at)
        for i in range(n)
    ) / n


def compute_recall_against_gt(
    gt_tags: np.ndarray, result_tags: np.ndarray, recall_at: int
) -> float:
    """Recall@k, clamped to the available ground-truth depth."""
    limit = min(int(recall_at), len(gt_tags))
    if limit <= 0:
        return 0.0
    ref = set(np.asarray(gt_tags[:limit]).tolist())
    compare = np.asarray(result_tags[:limit]).tolist()
    matches = sum(1 for t in compare if t in ref)
    return matches / limit


def evaluate_records(
    records: dict, provider: IncrementalGTProvider, recall_at: int = 0
) -> dict:
    """Score the search records returned by concurrency_native.run_benchmark.

    Returns overall mean recall plus per-stage (committed offset) means and
    the number of records that had no ground-truth entry.
    """
    offsets = np.asarray(records["insert_offsets"])
    tags = np.asarray(records["query_tags"])
    results = np.asarray(records["result_tags"])
    k = int(recall_at) if recall_at else int(records["k"])

    order = np.argsort(offsets, kind="stable")
    per_stage_sum: Dict[int, float] = {}
    per_stage_cnt: Dict[int, int] = {}
    missing = 0
    total = 0.0
    scored = 0
    for i in order:
        off = int(offsets[i])
        gt = provider.top_k(off, int(tags[i]))
        if gt is None or len(gt) == 0:
            missing += 1
            continue
        r = compute_recall_against_gt(gt, results[i], k)
        total += r
        scored += 1
        per_stage_sum[off] = per_stage_sum.get(off, 0.0) + r
        per_stage_cnt[off] = per_stage_cnt.get(off, 0) + 1

    return {
        "recall": total / scored if scored else 0.0,
        "scored": scored,
        "missing_gt": missing,
        "per_stage": {
            off: per_stage_sum[off] / per_stage_cnt[off]
            for off in sorted(per_stage_sum)
        },
    }


def compute_partial_diff_distribution(
    consistent, partial, recall_at: int, provider: IncrementalGTProvider
) -> Dict[int, list]:
    """Recall-diff histogram between consistent and partial-insert runs.

    Both inputs are lists of (insert_offset, query_tag, tags) records; the
    output buckets per stage by rounded (baseline - partial) recall diff,
    counting how many of the partial result tags point past the stage
    offset ("future" neighbors).
    """
    if recall_at <= 0:
        raise ValueError("recall_at must be greater than zero")

    baseline: Dict[int, Dict[int, np.ndarray]] = {}
    base_recall: Dict[int, Dict[int, float]] = {}
    for offset, qtag, tags in consistent:
        baseline.setdefault(offset, {})[qtag] = tags
        gt = provider.top_k(offset, qtag)
        if gt is None or len(gt) == 0:
            continue
        base_recall.setdefault(offset, {})[qtag] = compute_recall_against_gt(
            gt, tags, recall_at
        )

    stage_totals: Dict[int, int] = {}
    buckets: Dict[int, Dict[float, dict]] = {}
    for offset, qtag, tags in partial:
        stage_base = base_recall.get(offset)
        if stage_base is None or qtag not in baseline.get(offset, {}):
            continue
        gt = provider.top_k(offset, qtag)
        if gt is None or len(gt) == 0:
            continue
        if qtag not in stage_base:
            stage_base[qtag] = compute_recall_against_gt(
                gt, baseline[offset][qtag], recall_at
            )
        b = stage_base[qtag]
        p = compute_recall_against_gt(gt, tags, recall_at)
        diff = round(b - p, 6)

        stage_totals[offset] = stage_totals.get(offset, 0) + 1
        bucket = buckets.setdefault(offset, {}).setdefault(
            diff,
            {
                "stage_offset": offset,
                "diff": diff,
                "count": 0,
                "sum_recall_base": 0.0,
                "sum_recall_partial": 0.0,
                "sum_future_tags": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["sum_recall_base"] += b
        bucket["sum_recall_partial"] += p
        bucket["sum_future_tags"] += int(
            (np.asarray(tags, dtype=np.int64) >= offset).sum()
        )

    out: Dict[int, list] = {}
    for stage, by_diff in buckets.items():
        total = stage_totals.get(stage, 0)
        if not total:
            continue
        rows = list(by_diff.values())
        for r in rows:
            r["percentage"] = r["count"] / total * 100.0
        rows.sort(key=lambda r: (-r["diff"], -r["count"]))
        out[stage] = rows
    return out
