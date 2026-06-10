"""Concurrent benchmark runner for the concurrency track.

Usage:
    python3 -m concurrency.runner --config concurrency/configs/foo.yaml

Drives the native concurrency_native module (C++ hot path, GIL released),
then evaluates recall offline: overall recall against a DiskANN-style
ground-truth bin, and incremental recall against the split ground-truth
files. The YAML schema is unchanged from earlier releases.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)

from concurrency.config import load_config_variants  # noqa: E402
from concurrency.io import (  # noqa: E402
    load_aligned_bin,
    load_groundtruth_bin,
)
from concurrency.incremental_gt import (  # noqa: E402
    IncrementalGTProvider,
    evaluate_records,
    overall_recall_against_gt,
)
from concurrency.results import (  # noqa: E402
    records_from_run,
    result_dir,
    write_incr_res,
    write_overall_res,
)

# File name kept stable across harness generations (see README).
STATS_FILE = "benchmark_results.csv"

# Search-section flags the runner does not forward to the index yet;
# warn instead of silently ignoring them.
UNWIRED_SEARCH_FLAGS = (
    "enable_warm_start", "s3_adaptive_enabled", "enable_mvcc",
)

CSV_HEADER = [
    "algorithm", "threads", "write_batch_size", "read_batch_size",
    "dataset_name", "with_external_rw_lock", "use_node_lock",
    "insert_input_rate", "search_input_rate", "index_params", "query_params",
    "insert_qps (per-point)",
    "insert_mean_op_latency_ms (per-batch)",
    "insert_p95_op_latency_ms (per-batch)",
    "insert_p99_op_latency_ms (per-batch)",
    "insert_mean_e2e_latency_ms (per-batch)",
    "insert_p95_e2e_latency_ms (per-batch)",
    "insert_p99_e2e_latency_ms (per-batch)",
    "search_qps (per-point)",
    "search_mean_op_latency_ms (per-batch)",
    "search_p95_op_latency_ms (per-batch)",
    "search_p99_op_latency_ms (per-batch)",
    "search_mean_e2e_latency_ms (per-batch)",
    "search_p95_e2e_latency_ms (per-batch)",
    "search_p99_e2e_latency_ms (per-batch)",
    "result_below_k_count", "recall", "incr_recall", "peak_memory_mb",
    "avg_memory_mb", "stats",
]


def _import_native():
    here = os.path.dirname(os.path.abspath(__file__))
    build_lib = os.path.join(here, "..", "index", "concurrent", "build", "lib")
    sys.path.insert(0, build_lib)
    import concurrency_native

    return concurrency_native


def overall_recall(
    index, queries: np.ndarray, gt_path: str, recall_at: int,
    res_path: Optional[str] = None,
) -> float:
    """Post-run full-visibility recall, like calcOverallRecallForIndex.

    When res_path is given, the raw result tags are also written in the
    Go .res layout for any downstream tooling.
    """
    gt = load_groundtruth_bin(gt_path)
    if len(queries) < len(gt):
        raise ValueError(
            f"overall recall: only {len(queries)} queries available but "
            f"ground truth expects {len(gt)}"
        )
    n = len(gt)
    tags = index.batch_search(queries[:n], recall_at)
    if res_path:
        write_overall_res(res_path, tags)
    return overall_recall_against_gt(tags, gt, recall_at)


def effective_use_node_lock(cfg: Dict[str, Any]) -> bool:
    """Node-lock setting that actually applies: defaults on for hnsw,
    forced off under an external RW lock to avoid double locking."""
    index_type = cfg["index"].get("index_type", "hnsw").lower()
    if index_type not in ("hnsw", "segmented"):
        return False
    value = cfg["search"].get("use_node_lock")
    use = True if value is None else bool(value)
    if cfg["workload"].get("with_external_rw_lock"):
        use = False
    return use


def build_index(cc, cfg: Dict[str, Any], dim: int, num_points: int):
    """Construct the configured index (shared by runs and simulate)."""
    index_cfg, search_cfg = cfg["index"], cfg["search"]
    workload = cfg["workload"]
    index = cc.Index(
        index_cfg.get("index_type", "hnsw"),
        dim=dim,
        max_elements=num_points,
        M=int(index_cfg.get("m", 16)),
        ef_construction=int(index_cfg.get("ef_construction", 200)),
        num_threads=int(workload.get("num_threads", 4)),
        level_m=float(index_cfg.get("level_m", 0) or 0),
        alpha=float(index_cfg.get("alpha", 0) or 1.2),
        visit_limit=int(index_cfg.get("visit_limit", 0) or 0),
        use_node_lock=effective_use_node_lock(cfg),
        seal_threshold=int(index_cfg.get("seal_threshold", 0) or 0),
        sealed_type=index_cfg.get("sealed_type") or "hnsw",
    )
    index.set_search_features(
        enable_s3=bool(search_cfg.get("enable_s3")),
        enable_search_sharing=bool(search_cfg.get("enable_search_sharing")),
        enable_path_skip=bool(search_cfg.get("enable_path_skip")),
        enable_candidate_injection=bool(
            search_cfg.get("enable_candidate_injection")
        ),
    )
    index.set_query_params(
        ef_search=int(search_cfg.get("ef_search", 100)),
        beam_width=int(search_cfg.get("beam_width", 0) or 4),
        alpha=float(search_cfg.get("alpha", 0) or 1.2),
        visit_limit=int(search_cfg.get("visit_limit", 0) or 0),
    )
    return index


def build_driver_config(cfg: Dict[str, Any], num_points: int) -> Dict[str, Any]:
    data, workload = cfg["data"], cfg["workload"]
    profile, search = cfg["profile"], cfg["search"]
    # Bounds clamping: out-of-range max_elements falls back to the dataset
    # size; begin_num < 0 means "build everything up front" (insert phase
    # skipped); begin_num is capped at max_elements.
    max_elements = int(data.get("max_elements") or 0)
    if max_elements <= 0 or max_elements > num_points:
        max_elements = num_points
    begin_num = int(data.get("begin_num", 0))
    if begin_num < 0 or begin_num > max_elements:
        begin_num = max_elements
    driver = {
        "begin_num": begin_num,
        "max_elements": max_elements,
        "batch_size": int(workload["batch_size"]),
        # 0 (the configs' default) = unbuffered rendezvous queue.
        "queue_size": int(workload.get("queue_size", 0)),
        "num_threads": int(workload.get("num_threads", 4)),
        "insert_event_rate": float(workload.get("insert_event_rate", 0) or 0),
        "search_event_rate": float(workload.get("search_event_rate", 0) or 0),
        # Absent recall_at disables the search path.
        "recall_at": int(search.get("recall_at", 0)),
        "query_mode": workload.get("query_mode", "round_robin"),
        "zipfian_skew": float(workload.get("zipfian_skew", 0) or 0.99),
        "external_rw_lock": bool(workload.get("with_external_rw_lock", False)),
        "per_query_latency": bool(workload.get("per_query_latency") or False),
        # measurement starts after this many inserted points (0 = from t0)
        "warmup_points": int(workload.get("warmup_points", 0) or 0),
    }
    if profile.get("enable_memory_profile"):
        driver["mem_sample_interval_ms"] = int(
            profile.get("memory_monitor_interval") or 100
        )
    return driver


def run_variant(
    cc,
    name: str,
    cfg: Dict[str, Any],
    extra_driver: Optional[Dict[str, Any]] = None,
    snapshot: Optional[bytes] = None,
    take_snapshot: bool = False,
) -> Dict[str, Any]:
    data_cfg, search_cfg = cfg["data"], cfg["search"]

    # Paths resolve relative to the CWD.
    data = load_aligned_bin(data_cfg["data_path"])
    queries = load_aligned_bin(data_cfg["incr_query_path"])
    if queries.shape[1] != data.shape[1]:
        raise ValueError("query dim != data dim")

    for flag in UNWIRED_SEARCH_FLAGS:
        if search_cfg.get(flag):
            print(
                f"[{name}] WARNING: search.{flag} is not forwarded to the "
                "index by this runner; ignored"
            )

    driver_cfg = build_driver_config(cfg, len(data))
    if extra_driver:
        driver_cfg.update(extra_driver)
    begin = driver_cfg["begin_num"]

    index = build_index(
        cc, cfg, data.shape[1], max(driver_cfg["max_elements"], len(data))
    )

    baseline_snapshot = None
    if snapshot is not None:
        print(f"[{name}] restoring index from baseline snapshot ...")
        index.restore(snapshot)
    elif begin > 0:
        print(f"[{name}] building index with first {begin} points ...")
        t0 = time.time()
        index.build(data[:begin])
        print(f"[{name}] build took {time.time() - t0:.1f}s")
    if take_snapshot:
        try:
            baseline_snapshot = index.snapshot()
        except RuntimeError:
            # Index type without snapshots: later runs rebuild from
            # scratch instead.
            print(f"[{name}] snapshot unsupported; runs will rebuild")

    print(f"[{name}] running concurrent benchmark ...")
    out = cc.run_benchmark(index, data, queries, driver_cfg)
    stats = out["stats"]

    # Per-experiment raw outputs (.res files, Go layout) for any external
    # tooling that still consumes them. The variant name and lock mode go
    # into the file name so compare phases and sweep steps don't overwrite
    # each other.
    files_dir = result_dir(cfg)
    lock_suffix = "_wl" if driver_cfg["external_rw_lock"] else "_wol"
    safe_name = "".join(
        ch if ch.isalnum() or ch in "-_." else "_" for ch in name
    )
    base_name = (
        f"{cfg['index'].get('index_type', 'hnsw')}_{safe_name}{lock_suffix}"
    )
    if out["records"]["insert_offsets"].size:
        write_incr_res(
            os.path.join(files_dir, f"{base_name}_incr.res"),
            records_from_run(out["records"]),
        )

    recall = None
    gt_path = data_cfg.get("overall_gt_path")
    if gt_path and os.path.exists(gt_path):
        overall_queries = (
            load_aligned_bin(data_cfg["overall_query_path"])
            if data_cfg.get("overall_query_path")
            and data_cfg["overall_query_path"] != data_cfg["incr_query_path"]
            else queries
        )
        recall = overall_recall(
            index, overall_queries, gt_path, driver_cfg["recall_at"],
            res_path=os.path.join(files_dir, f"{base_name}_overall.res"),
        )
        print(f"[{name}] overall recall@{driver_cfg['recall_at']}: {recall:.4f}")

    incr = None
    incr_gt = data_cfg.get("incr_gt_path")
    if incr_gt and os.path.exists(incr_gt):
        provider = IncrementalGTProvider(incr_gt, driver_cfg["recall_at"])
        incr = evaluate_records(out["records"], provider)
        print(
            f"[{name}] incremental recall: {incr['recall']:.4f} "
            f"({incr['scored']} scored, {incr['missing_gt']} missing GT)"
        )

    return {
        "stats": stats,
        "recall": recall,
        "incr": incr,
        "index": index,
        "driver_cfg": driver_cfg,
        "out": out,
        "snapshot": baseline_snapshot,
    }


def append_stats_csv(
    path: str, cfg: Dict[str, Any], result: Dict[str, Any]
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path, newline="") as f:
            existing = next(csv.reader(f), None)
        if existing != CSV_HEADER:
            raise ValueError(
                f"{path} has a different column layout (likely written by "
                "another harness version); move it aside before appending"
            )
    stats, driver = result["stats"], result["driver_cfg"]
    index_cfg, search_cfg, workload = cfg["index"], cfg["search"], cfg["workload"]
    incr = result["incr"]
    row = [
        index_cfg.get("index_type", "hnsw"),
        driver["num_threads"], driver["batch_size"], driver["batch_size"],
        cfg["data"].get("dataset_name", ""),
        workload.get("with_external_rw_lock", False),
        effective_use_node_lock(cfg),
        driver["insert_event_rate"], driver["search_event_rate"],
        json.dumps({k: index_cfg.get(k) for k in ("m", "ef_construction")}),
        json.dumps({"ef_search": search_cfg.get("ef_search")}),
        f"{stats['insert_qps']:.2f}",
        f"{stats['mean_insert_op_latency_ms']:.3f}",
        f"{stats['p95_insert_op_latency_ms']:.3f}",
        f"{stats['p99_insert_op_latency_ms']:.3f}",
        f"{stats['mean_insert_e2e_latency_ms']:.3f}",
        f"{stats['p95_insert_e2e_latency_ms']:.3f}",
        f"{stats['p99_insert_e2e_latency_ms']:.3f}",
        f"{stats['search_qps']:.2f}",
        f"{stats['mean_search_op_latency_ms']:.3f}",
        f"{stats['p95_search_op_latency_ms']:.3f}",
        f"{stats['p99_search_op_latency_ms']:.3f}",
        f"{stats['mean_search_e2e_latency_ms']:.3f}",
        f"{stats['p95_search_e2e_latency_ms']:.3f}",
        f"{stats['p99_search_e2e_latency_ms']:.3f}",
        0,
        f"{result['recall']:.4f}" if result["recall"] is not None else "",
        f"{incr['recall']:.4f}" if incr else "",
        f"{stats['peak_memory_mb']:.1f}",
        f"{stats['avg_memory_mb']:.1f}",
        result["index"].dump_stats(),
    ]
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--variant", help="only run the variant with this name", default=None
    )
    args = parser.parse_args()

    cc = _import_native()
    variants = load_config_variants(args.config)
    if args.variant:
        variants = [(n, c) for n, c in variants if n == args.variant]
        if not variants:
            print(f"no variant named {args.variant}", file=sys.stderr)
            return 1

    from concurrency.modes import run_compare, run_throughput_sweep

    print(f"{len(variants)} variant(s) to run")
    for name, cfg in variants:
        print(f"==== variant: {name} ====")
        if cfg.get("throughput_sweep", {}).get("enabled"):
            run_throughput_sweep(cc, cfg, run_variant)
            continue
        # Same gate as Go's ShouldSimulate: simulate requires the compare
        # block itself to be enabled.
        compare_cfg = cfg.get("compare") or {}
        if compare_cfg.get("enabled") and (
            compare_cfg.get("simulate") or {}
        ).get("enabled"):
            from concurrency.simulate import run_simulate

            data = load_aligned_bin(cfg["data"]["data_path"])
            queries = load_aligned_bin(cfg["data"]["incr_query_path"])
            run_simulate(
                cc, cfg, data, queries,
                lambda: build_index(cc, cfg, data.shape[1], len(data)),
            )
            continue
        if compare_cfg.get("enabled"):
            run_compare(cc, cfg, run_variant)
            continue
        result = run_variant(cc, name, cfg)
        output_dir = cfg["result"].get("output_dir") or "./results/concurrent"
        append_stats_csv(
            os.path.join(output_dir, STATS_FILE), cfg, result
        )
        print(f"[{name}] results appended to {output_dir}/{STATS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
