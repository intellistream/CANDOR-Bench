"""Lag-timeline simulation.

A sequential (non-concurrent) experiment: a lead index applies inserts
batch by batch while a lag index replays the previous stage's snapshot,
so each stage yields a consistent result set and a one-stage-stale one.
Optionally a partial-insert trial reinserts the stage batch with randomly
truncated neighbor-update budgets to study partially-applied inserts;
recall diffs against incremental ground truth land in a CSV.

Every index call here is a single GIL-released C++ operation; there is
no concurrency in this pipeline, by design.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from concurrency.incremental_gt import (
    IncrementalGTProvider,
    compute_partial_diff_distribution,
)
from concurrency.results import (
    Record,
    result_dir,
    write_incr_res,
    write_partial_diff_csv,
)

PARTIAL_MODE_LAG_INSERT = 1
PARTIAL_MODE_LEAD_INSERT = 2
PARTIAL_MODE_CARRY_INSERT = 3
_MODE_NAMES = {1: "lag_insert", 2: "lead_insert", 3: "carry_insert"}


def _append_records(
    dst: List[Record], tags: np.ndarray, query_tags: List[int], offset: int
) -> None:
    for i, qtag in enumerate(query_tags):
        dst.append((offset, qtag, tags[i]))


def run_simulate(cc, cfg: dict, data: np.ndarray, queries: np.ndarray,
                 make_index) -> dict:
    """make_index() builds a fresh, empty index for the configured type."""
    data_cfg, workload = cfg["data"], cfg["workload"]
    search_cfg = cfg["search"]
    sim_cfg = (cfg.get("compare") or {}).get("simulate") or {}
    partial_cfg = sim_cfg.get("partial_update") or {}
    partial_enabled = bool(sim_cfg.get("enabled")) and bool(
        partial_cfg.get("enabled")
    )
    seed = int(partial_cfg.get("seed") or 1)
    mode = int(partial_cfg.get("mode") or 0)
    if mode not in _MODE_NAMES:
        mode = PARTIAL_MODE_LAG_INSERT

    query_mode = workload.get("query_mode", "round_robin")
    if query_mode in ("chasing", "peeking"):
        print("simulate: chasing/peeking modes preserve original order; "
              "nothing to do")
        return {}

    recall_at = int(search_cfg.get("recall_at", 10))
    begin = int(data_cfg.get("begin_num", 0))
    max_elements = min(int(data_cfg.get("max_elements") or len(data)),
                       len(data))
    batch_size = int(workload["batch_size"])

    provider: Optional[IncrementalGTProvider] = None
    if partial_enabled and data_cfg.get("incr_gt_path") and recall_at > 0:
        provider = IncrementalGTProvider(
            data_cfg["incr_gt_path"], recall_at
        )

    lead = make_index()
    lag = make_index()
    if begin > 0:
        lead.build(data[:begin])
        lag.build(data[:begin])
    if partial_enabled:
        lead.enable_insert_telemetry(True)

    # The driver's own stage-query generator, so simulate uses identical
    # windowing and RNG semantics to concurrent runs.
    source = cc.QuerySource(
        query_mode,
        np.ascontiguousarray(queries, dtype=np.float32),
        batch_size,
        int(workload.get("stage_query_window") or batch_size),
        float(workload.get("zipfian_skew") or 0.99),
    )

    prev_snapshot = lead.snapshot()
    consistent: List[Record] = []
    lagging: List[Record] = []
    partial: List[Record] = []
    carry_snapshot = None

    offset, stage = begin, 0
    all_tags = np.arange(max_elements, dtype=np.uint32)
    while offset < max_elements:
        next_off = min(offset + batch_size, max_elements)
        stage_queries, stage_qtags = source.next_batch(next_off)
        stage_qtags = stage_qtags.tolist()
        if len(stage_qtags) == 0:
            offset, stage = next_off, stage + 1
            continue

        batch = data[offset:next_off]
        batch_tags = all_tags[offset:next_off]

        lag.restore(prev_snapshot)

        if partial_enabled:
            stage_stats = lead.batch_partial_insert(
                batch, batch_tags, collect_stats=True
            )
            stage_max = int(stage_stats.max()) if len(stage_stats) else 0
        else:
            stage_stats, stage_max = None, 0
            lead.batch_insert(batch, batch_tags)

        consistent_raw = lead.batch_search(stage_queries, recall_at)
        _append_records(consistent, consistent_raw, stage_qtags, next_off)

        lagging_raw = lag.batch_search(stage_queries, recall_at)
        _append_records(lagging, lagging_raw, stage_qtags, next_off)

        full_snapshot = lead.snapshot()

        if partial_enabled:
            # Pick the trial batch and restore template per mode.
            trial_data, trial_tags = batch, batch_tags
            trial_stats, trial_max = stage_stats, stage_max
            trial_offset = next_off
            template = prev_snapshot
            ready = True
            if mode == PARTIAL_MODE_LEAD_INSERT:
                future_end = min(next_off + batch_size, max_elements)
                if future_end > next_off:
                    trial_data = data[next_off:future_end]
                    trial_tags = all_tags[next_off:future_end]
                    lag.restore(full_snapshot)
                    trial_stats = lag.batch_partial_insert(
                        trial_data, trial_tags, collect_stats=True
                    )
                    trial_max = (
                        int(trial_stats.max()) if len(trial_stats) else 0
                    )
                    template = full_snapshot
                else:
                    ready = False
            elif mode == PARTIAL_MODE_LAG_INSERT:
                trial_offset = offset + len(batch)

            if ready and len(trial_data):
                rng = np.random.default_rng(seed + stage * 1000)
                if mode == PARTIAL_MODE_CARRY_INSERT:
                    base = carry_snapshot if carry_snapshot else prev_snapshot
                else:
                    base = template
                lag.restore(base)
                # Per-point budget: the recorded update count when known,
                # else the stage max, else 1.
                if trial_stats is not None:
                    stats_arr = np.asarray(trial_stats, dtype=np.uint64)
                    caps = np.where(
                        stats_arr > 0, stats_arr, max(trial_max, 1)
                    ).astype(np.uint64)
                else:
                    caps = np.ones(len(trial_data), dtype=np.uint64)
                limits = rng.integers(0, caps + 1).astype(np.uint64)
                lag.batch_partial_insert(trial_data, trial_tags,
                                         limits=limits)
                partial_raw = lag.batch_search(stage_queries, recall_at)
                _append_records(partial, partial_raw, stage_qtags,
                                trial_offset)
                if mode == PARTIAL_MODE_CARRY_INSERT:
                    carry_snapshot = lag.snapshot()

        prev_snapshot = full_snapshot
        offset, stage = next_off, stage + 1

    out_dir = result_dir(cfg)
    name = cfg["index"].get("index_type", "hnsw")
    paths = {
        "incr_res": os.path.join(out_dir, f"{name}_incr.res"),
        "lagging_incr_res": os.path.join(out_dir, f"{name}_incr_lag.res"),
        "partial_incr_res": os.path.join(
            out_dir, f"{name}_incr_partial.res"
        ),
        "partial_diff": os.path.join(
            out_dir, f"{name}_partial_diff.csv"
        ),
    }
    write_incr_res(paths["incr_res"], consistent)
    write_incr_res(paths["lagging_incr_res"], lagging)
    if partial:
        write_incr_res(paths["partial_incr_res"], partial)
    if partial and provider is not None:
        diff = compute_partial_diff_distribution(
            consistent, partial, recall_at, provider
        )
        write_partial_diff_csv(paths["partial_diff"], diff)
        print(f"simulate: partial diff written to {paths['partial_diff']}")

    print(f"simulate: completed {stage} stages, results under {out_dir}")
    return {
        "stages": stage,
        "consistent": consistent,
        "lagging": lagging,
        "partial": partial,
        "paths": paths,
    }
