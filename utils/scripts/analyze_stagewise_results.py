#!/usr/bin/env python3

import argparse
import math
import os
import struct
import sys
from array import array
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


@dataclass
class SplitInfo:
    filename: str
    start_offset: int
    end_offset: int


@dataclass
class ResultEntry:
    insert_offset: int
    query_tag: int
    tags: List[int]


@dataclass
class BucketStats:
    count: int = 0
    raw_recall_sum: float = 0.0
    visible_recall_sum: float = 0.0
    raw_top1_hits: int = 0
    visible_top1_hits: int = 0
    future_hit_queries: int = 0
    future_top1_queries: int = 0
    future_tags_total: int = 0
    compare_count: int = 0
    raw_overlap_sum: float = 0.0
    visible_overlap_sum: float = 0.0
    raw_exact_matches: int = 0
    visible_exact_matches: int = 0
    raw_top1_matches: int = 0
    visible_top1_matches: int = 0
    compare_partner_raw_recall_sum: float = 0.0
    compare_partner_visible_recall_sum: float = 0.0
    compare_partner_raw_top1_hits: int = 0
    compare_partner_visible_top1_hits: int = 0
    worst_visible_recall: float = 1.0
    worst_visible_offset: int = -1
    worst_visible_query_tag: int = -1


@dataclass
class SummaryStats:
    total: int = 0
    gt_missing: int = 0
    raw_recall_sum: float = 0.0
    visible_recall_sum: float = 0.0
    raw_top1_hits: int = 0
    visible_top1_hits: int = 0
    zero_raw_queries: int = 0
    zero_visible_queries: int = 0
    future_hit_queries: int = 0
    future_top1_queries: int = 0
    future_tags_total: int = 0
    future_subset_count: int = 0
    future_subset_raw_recall_sum: float = 0.0
    future_subset_visible_recall_sum: float = 0.0
    clean_subset_count: int = 0
    clean_subset_raw_recall_sum: float = 0.0
    clean_subset_visible_recall_sum: float = 0.0
    compare_count: int = 0
    raw_overlap_sum: float = 0.0
    visible_overlap_sum: float = 0.0
    raw_exact_matches: int = 0
    visible_exact_matches: int = 0
    raw_top1_matches: int = 0
    visible_top1_matches: int = 0
    compare_primary_raw_recall_sum: float = 0.0
    compare_primary_visible_recall_sum: float = 0.0
    compare_primary_raw_top1_hits: int = 0
    compare_primary_visible_top1_hits: int = 0
    compare_partner_raw_recall_sum: float = 0.0
    compare_partner_visible_recall_sum: float = 0.0
    compare_partner_raw_top1_hits: int = 0
    compare_partner_visible_top1_hits: int = 0
    worst_query_visible_recall: float = 1.0
    worst_query_offset: int = -1
    worst_query_tag: int = -1
    worst_query_raw_tags: Tuple[int, ...] = ()
    worst_query_visible_tags: Tuple[int, ...] = ()
    worst_query_gt_tags: Tuple[int, ...] = ()
    buckets: Dict[int, BucketStats] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze stagewise incremental ANN result files. "
            "Reports raw recall, visible-only recall after filtering future IDs, "
            "future-hit rates, and optional A/B disagreement against a second result file."
        )
    )
    parser.add_argument("--res", required=True, help="Primary stagewise result file")
    parser.add_argument(
        "--gt-index",
        required=True,
        help="Incremental GT index file (for example data/sift/sift_stream_i20_offset_index.txt)",
    )
    parser.add_argument("--k", type=int, default=10, help="Recall/top-k cutoff")
    parser.add_argument(
        "--compare-res",
        help=(
            "Optional second stagewise result file. When provided, the script "
            "also reports A/B overlap and exact-match metrics."
        ),
    )
    parser.add_argument(
        "--bucket",
        type=int,
        default=20,
        help="Offset bucket size for window summaries. Use 0 to keep exact offsets.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Optional cap on processed queries. 0 means process all.",
    )
    parser.add_argument(
        "--top-buckets",
        type=int,
        default=10,
        help="Number of worst/future-heavy buckets to print.",
    )
    return parser.parse_args()


def read_exact(handle, nbytes: int) -> bytes:
    data = handle.read(nbytes)
    if len(data) != nbytes:
        raise EOFError(f"expected {nbytes} bytes, got {len(data)}")
    return data


def read_u64(handle) -> int:
    return struct.unpack("<Q", read_exact(handle, 8))[0]


def read_i32(handle) -> int:
    return struct.unpack("<i", read_exact(handle, 4))[0]


def read_u32_array(handle, count: int) -> List[int]:
    if count <= 0:
        return []
    raw = read_exact(handle, count * 4)
    values = array("I")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


class StagewiseResultReader:
    def __init__(self, path: str, k_limit: int):
        self.path = path
        self.k_limit = k_limit
        self.handle = open(path, "rb")
        self.num_queries = read_u64(self.handle)
        self.read_queries = 0

    def close(self) -> None:
        self.handle.close()

    def __iter__(self) -> Iterator[ResultEntry]:
        return self

    def __next__(self) -> ResultEntry:
        if self.read_queries >= self.num_queries:
            raise StopIteration
        insert_offset = read_u64(self.handle)
        query_tag = read_u64(self.handle)
        num_tags = read_u64(self.handle)
        tags = read_u32_array(self.handle, num_tags)
        num_dists = read_u64(self.handle)
        if num_dists > 0:
            self.handle.seek(num_dists * 4, os.SEEK_CUR)
        self.read_queries += 1
        return ResultEntry(insert_offset, query_tag, tags[: self.k_limit])


def result_key(entry: ResultEntry) -> Tuple[int, int]:
    return (entry.insert_offset, entry.query_tag)


def parse_gt_index(index_path: str) -> List[SplitInfo]:
    splits: List[SplitInfo] = []
    with open(index_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            splits.append(SplitInfo(parts[0], int(parts[1]), int(parts[2])))
    if not splits:
        raise ValueError(f"no GT slices found in {index_path}")
    return splits


def detect_gt_query_tags(path: str, n_per_batch: int, k_file: int, num_batches: int) -> bool:
    header_size = 3 * 4
    file_size = os.path.getsize(path)
    payload_size = file_size - header_size
    per_batch_base = 8 + n_per_batch * k_file * (4 + 4)
    expected_payload_size = per_batch_base * num_batches
    query_tag_bytes = n_per_batch * 4 * num_batches
    if payload_size == expected_payload_size + query_tag_bytes:
        return True
    if payload_size == expected_payload_size:
        return False
    raise ValueError(
        f"GT file {path} has unexpected payload size: {payload_size} "
        f"(expected {expected_payload_size} or {expected_payload_size + query_tag_bytes})"
    )


def load_gt_slice(index_dir: str, split: SplitInfo, k_limit: int) -> Dict[int, Dict[int, Tuple[int, ...]]]:
    path = os.path.join(index_dir, split.filename)
    by_offset: Dict[int, Dict[int, Tuple[int, ...]]] = {}
    with open(path, "rb") as handle:
        n_per_batch = read_i32(handle)
        k_file = read_i32(handle)
        num_batches = read_i32(handle)
        if n_per_batch <= 0 or k_file <= 0 or num_batches <= 0:
            raise ValueError(f"invalid GT header in {path}")
        has_query_tags = detect_gt_query_tags(path, n_per_batch, k_file, num_batches)
        for _ in range(num_batches):
            current_offset = read_u64(handle)
            if has_query_tags:
                query_ids = read_u32_array(handle, n_per_batch)
            else:
                query_ids = list(range(n_per_batch))
            handle.seek(n_per_batch * k_file * 4, os.SEEK_CUR)
            batch_indices = read_u32_array(handle, n_per_batch * k_file)
            offset_map = by_offset.setdefault(current_offset, {})
            for i, query_id in enumerate(query_ids):
                start = i * k_file
                tags = batch_indices[start : start + min(k_limit, k_file)]
                offset_map[int(query_id)] = tuple(tags)
    return by_offset


def matches_at_k(tags: List[int], gt_tags: Tuple[int, ...], k: int) -> int:
    if k <= 0:
        return 0
    gt_limit = min(k, len(gt_tags))
    if gt_limit <= 0:
        return 0
    left = sorted(tags[:k])
    right = sorted(gt_tags[:gt_limit])
    i = 0
    j = 0
    matches = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            i += 1
        elif right[j] < left[i]:
            j += 1
        else:
            matches += 1
            i += 1
            j += 1
    return matches


def visible_tags(tags: List[int], insert_offset: int, k: int) -> List[int]:
    visible = [tag for tag in tags[:k] if tag < insert_offset]
    return visible[:k]


def top1_hit(tags: List[int], gt_tags: Tuple[int, ...]) -> bool:
    return bool(tags) and bool(gt_tags) and tags[0] == gt_tags[0]


def overlap_at_k(left: List[int], right: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(left[:k]).intersection(right[:k])) / float(k)


def exact_set_match(left: List[int], right: List[int], k: int) -> bool:
    return set(left[:k]) == set(right[:k])


def bucket_key(offset: int, bucket_size: int) -> int:
    if bucket_size <= 0:
        return offset
    return (offset // bucket_size) * bucket_size


def avg_or_nan(total: float, count: int) -> float:
    if count <= 0:
        return float("nan")
    return total / count


def find_split_for_offset(splits: List[SplitInfo], start_idx: int, offset: int) -> Tuple[int, SplitInfo]:
    idx = start_idx
    while idx < len(splits):
        split = splits[idx]
        if split.start_offset <= offset <= split.end_offset:
            return idx, split
        if offset < split.start_offset:
            break
        idx += 1
    for idx, split in enumerate(splits):
        if split.start_offset <= offset <= split.end_offset:
            return idx, split
    raise KeyError(f"offset {offset} not covered by GT index")


def add_bucket_metrics(
    bucket: BucketStats,
    raw_recall: float,
    visible_recall: float,
    raw_top1: bool,
    visible_top1: bool,
    future_count: int,
    future_top1: bool,
    offset: int,
    query_tag: int,
    compare_raw_overlap: Optional[float],
    compare_visible_overlap: Optional[float],
    compare_raw_exact: Optional[bool],
    compare_visible_exact: Optional[bool],
    compare_raw_top1: Optional[bool],
    compare_visible_top1: Optional[bool],
    compare_partner_raw_recall: Optional[float],
    compare_partner_visible_recall: Optional[float],
    compare_partner_raw_top1_hit: Optional[bool],
    compare_partner_visible_top1_hit: Optional[bool],
) -> None:
    bucket.count += 1
    bucket.raw_recall_sum += raw_recall
    bucket.visible_recall_sum += visible_recall
    bucket.raw_top1_hits += int(raw_top1)
    bucket.visible_top1_hits += int(visible_top1)
    bucket.future_hit_queries += int(future_count > 0)
    bucket.future_top1_queries += int(future_top1)
    bucket.future_tags_total += future_count
    if visible_recall < bucket.worst_visible_recall:
        bucket.worst_visible_recall = visible_recall
        bucket.worst_visible_offset = offset
        bucket.worst_visible_query_tag = query_tag
    if compare_raw_overlap is not None:
        bucket.compare_count += 1
        bucket.raw_overlap_sum += compare_raw_overlap
        bucket.visible_overlap_sum += compare_visible_overlap or 0.0
        bucket.raw_exact_matches += int(bool(compare_raw_exact))
        bucket.visible_exact_matches += int(bool(compare_visible_exact))
        bucket.raw_top1_matches += int(bool(compare_raw_top1))
        bucket.visible_top1_matches += int(bool(compare_visible_top1))
        bucket.compare_partner_raw_recall_sum += compare_partner_raw_recall or 0.0
        bucket.compare_partner_visible_recall_sum += compare_partner_visible_recall or 0.0
        bucket.compare_partner_raw_top1_hits += int(bool(compare_partner_raw_top1_hit))
        bucket.compare_partner_visible_top1_hits += int(bool(compare_partner_visible_top1_hit))


def analyze(args: argparse.Namespace) -> SummaryStats:
    splits = parse_gt_index(args.gt_index)
    gt_index_dir = os.path.dirname(args.gt_index)

    primary = StagewiseResultReader(args.res, args.k)
    secondary = StagewiseResultReader(args.compare_res, args.k) if args.compare_res else None

    stats = SummaryStats()
    split_idx = 0
    current_split: Optional[SplitInfo] = None
    current_gt: Optional[Dict[int, Dict[int, Tuple[int, ...]]]] = None
    other_current: Optional[ResultEntry] = None
    if secondary is not None:
        try:
            other_current = next(secondary)
        except StopIteration:
            other_current = None

    try:
        for idx, entry in enumerate(primary):
            if args.max_queries > 0 and idx >= args.max_queries:
                break

            other = None
            if secondary is not None and other_current is not None:
                primary_key = result_key(entry)
                while other_current is not None and result_key(other_current) < primary_key:
                    try:
                        other_current = next(secondary)
                    except StopIteration:
                        other_current = None
                        break
                if other_current is not None and result_key(other_current) == primary_key:
                    other = other_current
                    try:
                        other_current = next(secondary)
                    except StopIteration:
                        other_current = None

            if current_split is None or not (
                current_split.start_offset <= entry.insert_offset <= current_split.end_offset
            ):
                split_idx, current_split = find_split_for_offset(
                    splits, split_idx, entry.insert_offset
                )
                current_gt = load_gt_slice(gt_index_dir, current_split, args.k)

            assert current_gt is not None
            gt_by_tag = current_gt.get(entry.insert_offset)
            if gt_by_tag is None:
                stats.gt_missing += 1
                continue
            gt_tags = gt_by_tag.get(entry.query_tag)
            if gt_tags is None:
                stats.gt_missing += 1
                continue

            raw_tags = entry.tags[: args.k]
            vis_tags = visible_tags(raw_tags, entry.insert_offset, args.k)
            future_count = sum(1 for tag in raw_tags if tag >= entry.insert_offset)
            future_top1 = bool(raw_tags) and raw_tags[0] >= entry.insert_offset

            raw_matches = matches_at_k(raw_tags, gt_tags, args.k)
            vis_matches = matches_at_k(vis_tags, gt_tags, args.k)
            raw_recall = raw_matches / float(args.k)
            vis_recall = vis_matches / float(args.k)
            raw_top1 = top1_hit(raw_tags, gt_tags)
            vis_top1 = top1_hit(vis_tags, gt_tags)

            stats.total += 1
            stats.raw_recall_sum += raw_recall
            stats.visible_recall_sum += vis_recall
            stats.raw_top1_hits += int(raw_top1)
            stats.visible_top1_hits += int(vis_top1)
            stats.zero_raw_queries += int(raw_matches == 0)
            stats.zero_visible_queries += int(vis_matches == 0)
            stats.future_hit_queries += int(future_count > 0)
            stats.future_top1_queries += int(future_top1)
            stats.future_tags_total += future_count
            if future_count > 0:
                stats.future_subset_count += 1
                stats.future_subset_raw_recall_sum += raw_recall
                stats.future_subset_visible_recall_sum += vis_recall
            else:
                stats.clean_subset_count += 1
                stats.clean_subset_raw_recall_sum += raw_recall
                stats.clean_subset_visible_recall_sum += vis_recall
            if vis_recall < stats.worst_query_visible_recall:
                stats.worst_query_visible_recall = vis_recall
                stats.worst_query_offset = entry.insert_offset
                stats.worst_query_tag = entry.query_tag
                stats.worst_query_raw_tags = tuple(raw_tags)
                stats.worst_query_visible_tags = tuple(vis_tags)
                stats.worst_query_gt_tags = tuple(gt_tags[: args.k])

            if other is not None:
                other_raw = other.tags[: args.k]
                other_vis = visible_tags(other_raw, other.insert_offset, args.k)
                raw_overlap = overlap_at_k(raw_tags, other_raw, args.k)
                vis_overlap = overlap_at_k(vis_tags, other_vis, args.k)
                raw_exact = exact_set_match(raw_tags, other_raw, args.k)
                vis_exact = exact_set_match(vis_tags, other_vis, args.k)
                raw_top1_match = bool(raw_tags) and bool(other_raw) and raw_tags[0] == other_raw[0]
                vis_top1_match = bool(vis_tags) and bool(other_vis) and vis_tags[0] == other_vis[0]
                other_raw_matches = matches_at_k(other_raw, gt_tags, args.k)
                other_vis_matches = matches_at_k(other_vis, gt_tags, args.k)
                other_raw_recall = other_raw_matches / float(args.k)
                other_vis_recall = other_vis_matches / float(args.k)
                other_raw_top1 = top1_hit(other_raw, gt_tags)
                other_vis_top1 = top1_hit(other_vis, gt_tags)
                stats.compare_count += 1
                stats.raw_overlap_sum += raw_overlap
                stats.visible_overlap_sum += vis_overlap
                stats.raw_exact_matches += int(raw_exact)
                stats.visible_exact_matches += int(vis_exact)
                stats.raw_top1_matches += int(raw_top1_match)
                stats.visible_top1_matches += int(vis_top1_match)
                stats.compare_primary_raw_recall_sum += raw_recall
                stats.compare_primary_visible_recall_sum += vis_recall
                stats.compare_primary_raw_top1_hits += int(raw_top1)
                stats.compare_primary_visible_top1_hits += int(vis_top1)
                stats.compare_partner_raw_recall_sum += other_raw_recall
                stats.compare_partner_visible_recall_sum += other_vis_recall
                stats.compare_partner_raw_top1_hits += int(other_raw_top1)
                stats.compare_partner_visible_top1_hits += int(other_vis_top1)
            else:
                raw_overlap = None
                vis_overlap = None
                raw_exact = None
                vis_exact = None
                raw_top1_match = None
                vis_top1_match = None
                other_raw_recall = None
                other_vis_recall = None
                other_raw_top1 = None
                other_vis_top1 = None

            bucket = stats.buckets.setdefault(
                bucket_key(entry.insert_offset, args.bucket), BucketStats()
            )
            add_bucket_metrics(
                bucket,
                raw_recall,
                vis_recall,
                raw_top1,
                vis_top1,
                future_count,
                future_top1,
                entry.insert_offset,
                entry.query_tag,
                raw_overlap,
                vis_overlap,
                raw_exact,
                vis_exact,
                raw_top1_match,
                vis_top1_match,
                other_raw_recall,
                other_vis_recall,
                other_raw_top1,
                other_vis_top1,
            )
    finally:
        primary.close()
        if secondary is not None:
            secondary.close()

    return stats


def format_pct(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value * 100.0:.2f}%"


def print_summary(args: argparse.Namespace, stats: SummaryStats) -> None:
    print(f"primary_res: {args.res}")
    print(f"gt_index: {args.gt_index}")
    if args.compare_res:
        print(f"compare_res: {args.compare_res}")
    print(f"k: {args.k}")
    print(f"bucket: {args.bucket}")
    print(f"processed_queries: {stats.total}")
    print(f"missing_gt_queries: {stats.gt_missing}")
    print()

    print("Overall:")
    print(f"  raw_recall@{args.k}: {format_pct(avg_or_nan(stats.raw_recall_sum, stats.total))}")
    print(
        f"  visible_only_recall@{args.k}: "
        f"{format_pct(avg_or_nan(stats.visible_recall_sum, stats.total))}"
    )
    print(f"  raw_top1_hit_rate: {format_pct(avg_or_nan(stats.raw_top1_hits, stats.total))}")
    print(
        f"  visible_only_top1_hit_rate: "
        f"{format_pct(avg_or_nan(stats.visible_top1_hits, stats.total))}"
    )
    print(f"  zero_raw_query_rate: {format_pct(avg_or_nan(stats.zero_raw_queries, stats.total))}")
    print(
        f"  zero_visible_query_rate: "
        f"{format_pct(avg_or_nan(stats.zero_visible_queries, stats.total))}"
    )
    print()

    print("Future Leakage:")
    print(
        f"  future_hit_query_rate: "
        f"{format_pct(avg_or_nan(stats.future_hit_queries, stats.total))}"
    )
    print(
        f"  future_top1_rate: "
        f"{format_pct(avg_or_nan(stats.future_top1_queries, stats.total))}"
    )
    print(
        f"  avg_future_tags_per_query@{args.k}: "
        f"{avg_or_nan(stats.future_tags_total, stats.total):.4f}"
    )
    print(
        f"  raw_recall_on_future_hit_queries: "
        f"{format_pct(avg_or_nan(stats.future_subset_raw_recall_sum, stats.future_subset_count))}"
    )
    print(
        f"  visible_recall_on_future_hit_queries: "
        f"{format_pct(avg_or_nan(stats.future_subset_visible_recall_sum, stats.future_subset_count))}"
    )
    print(
        f"  raw_recall_on_clean_queries: "
        f"{format_pct(avg_or_nan(stats.clean_subset_raw_recall_sum, stats.clean_subset_count))}"
    )
    print(
        f"  visible_recall_on_clean_queries: "
        f"{format_pct(avg_or_nan(stats.clean_subset_visible_recall_sum, stats.clean_subset_count))}"
    )
    print()

    if stats.compare_count > 0:
        print("A/B Disagreement:")
        print(f"  matched_queries: {stats.compare_count}")
        print(
            f"  raw_set_overlap@{args.k}: "
            f"{format_pct(avg_or_nan(stats.raw_overlap_sum, stats.compare_count))}"
        )
        print(
            f"  visible_set_overlap@{args.k}: "
            f"{format_pct(avg_or_nan(stats.visible_overlap_sum, stats.compare_count))}"
        )
        print(
            f"  raw_exact_set_match_rate: "
            f"{format_pct(avg_or_nan(stats.raw_exact_matches, stats.compare_count))}"
        )
        print(
            f"  visible_exact_set_match_rate: "
            f"{format_pct(avg_or_nan(stats.visible_exact_matches, stats.compare_count))}"
        )
        print(
            f"  raw_top1_match_rate: "
            f"{format_pct(avg_or_nan(stats.raw_top1_matches, stats.compare_count))}"
        )
        print(
            f"  visible_top1_match_rate: "
            f"{format_pct(avg_or_nan(stats.visible_top1_matches, stats.compare_count))}"
        )
        print(
            f"  primary_matched_raw_recall@{args.k}: "
            f"{format_pct(avg_or_nan(stats.compare_primary_raw_recall_sum, stats.compare_count))}"
        )
        print(
            f"  primary_matched_visible_recall@{args.k}: "
            f"{format_pct(avg_or_nan(stats.compare_primary_visible_recall_sum, stats.compare_count))}"
        )
        print(
            f"  primary_matched_raw_top1_hit_rate: "
            f"{format_pct(avg_or_nan(stats.compare_primary_raw_top1_hits, stats.compare_count))}"
        )
        print(
            f"  primary_matched_visible_top1_hit_rate: "
            f"{format_pct(avg_or_nan(stats.compare_primary_visible_top1_hits, stats.compare_count))}"
        )
        print(
            f"  compare_partner_raw_recall@{args.k}: "
            f"{format_pct(avg_or_nan(stats.compare_partner_raw_recall_sum, stats.compare_count))}"
        )
        print(
            f"  compare_partner_visible_recall@{args.k}: "
            f"{format_pct(avg_or_nan(stats.compare_partner_visible_recall_sum, stats.compare_count))}"
        )
        print(
            f"  compare_partner_raw_top1_hit_rate: "
            f"{format_pct(avg_or_nan(stats.compare_partner_raw_top1_hits, stats.compare_count))}"
        )
        print(
            f"  compare_partner_visible_top1_hit_rate: "
            f"{format_pct(avg_or_nan(stats.compare_partner_visible_top1_hits, stats.compare_count))}"
        )
        print()

    if stats.worst_query_offset >= 0:
        print("Worst Query (visible-only recall):")
        print(f"  offset={stats.worst_query_offset} query_tag={stats.worst_query_tag}")
        print(f"  visible_only_recall={format_pct(stats.worst_query_visible_recall)}")
        print(f"  raw_tags={list(stats.worst_query_raw_tags)}")
        print(f"  visible_tags={list(stats.worst_query_visible_tags)}")
        print(f"  gt_tags={list(stats.worst_query_gt_tags)}")
        print()

    if not stats.buckets:
        return

    buckets = list(stats.buckets.items())

    def bucket_visible_avg(item: Tuple[int, BucketStats]) -> float:
        _, bucket = item
        return avg_or_nan(bucket.visible_recall_sum, bucket.count)

    def bucket_future_rate(item: Tuple[int, BucketStats]) -> float:
        _, bucket = item
        return avg_or_nan(bucket.future_hit_queries, bucket.count)

    worst_buckets = sorted(buckets, key=bucket_visible_avg)[: args.top_buckets]
    future_buckets = sorted(buckets, key=bucket_future_rate, reverse=True)[: args.top_buckets]

    print(f"Worst {len(worst_buckets)} Buckets by visible-only recall:")
    for bucket_id, bucket in worst_buckets:
        print(
            "  "
            f"bucket={bucket_id} count={bucket.count} "
            f"raw={format_pct(avg_or_nan(bucket.raw_recall_sum, bucket.count))} "
            f"visible={format_pct(avg_or_nan(bucket.visible_recall_sum, bucket.count))} "
            f"future_hit={format_pct(avg_or_nan(bucket.future_hit_queries, bucket.count))} "
            f"future_top1={format_pct(avg_or_nan(bucket.future_top1_queries, bucket.count))} "
            f"worst_visible={format_pct(bucket.worst_visible_recall)} "
            f"worst_query=({bucket.worst_visible_offset},{bucket.worst_visible_query_tag})"
        )
    print()

    print(f"Top {len(future_buckets)} Buckets by future-hit rate:")
    for bucket_id, bucket in future_buckets:
        line = (
            "  "
            f"bucket={bucket_id} count={bucket.count} "
            f"future_hit={format_pct(avg_or_nan(bucket.future_hit_queries, bucket.count))} "
            f"future_top1={format_pct(avg_or_nan(bucket.future_top1_queries, bucket.count))} "
            f"avg_future_tags={avg_or_nan(bucket.future_tags_total, bucket.count):.4f} "
            f"raw={format_pct(avg_or_nan(bucket.raw_recall_sum, bucket.count))} "
            f"visible={format_pct(avg_or_nan(bucket.visible_recall_sum, bucket.count))}"
        )
        if bucket.compare_count > 0:
            line += (
                f" visible_overlap={format_pct(avg_or_nan(bucket.visible_overlap_sum, bucket.compare_count))}"
                f" visible_exact={format_pct(avg_or_nan(bucket.visible_exact_matches, bucket.compare_count))}"
            )
        print(line)


def main() -> int:
    args = parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.bucket < 0:
        raise ValueError("--bucket must be >= 0")
    stats = analyze(args)
    print_summary(args, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
