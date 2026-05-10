"""Rich workloads designed to need the per-vector / heat-aware mechanisms
in gamma_py_rich.py.

  - mixed_lifetime: each insert is randomly tagged "short" (lifetime ~2K
                     ops) or "long" (~50K ops). Cost-model admit should
                     learn to send short → buffer, long → direct-graph.
  - zipfian_queries: standard streaming insert/delete (random pattern),
                     BUT the query set is biased — queries are sampled
                     Zipf-weighted from a Q pool, so a small fraction of
                     queries dominates. Heat-aware tier should pin the
                     popular candidates.
  - phase_shift:    insert-heavy phase, then delete-heavy phase, then
                     query-heavy phase. Adaptive maint should adjust to
                     each phase.

These deliberately stress the rich modules in ways the uniform
sequential/random/cluster/partial_reset patterns cannot.
"""
from __future__ import annotations
import time
import numpy as np
from typing import Optional

from .data import compute_gt
from .metrics import recall_at_k


# ----------------------------------------------------------------------
# Mixed-lifetime workload
# ----------------------------------------------------------------------
def run_mixed_lifetime(idx, name, data, queries, init_n, batch, qstride,
                        *, short_lifetime_ops: int = 2000,
                        long_lifetime_ops: int = 50000,
                        short_fraction: float = 0.5,
                        seed: int = 42,
                        use_gamma_rich: bool = False,
                        has_mark_deleted: bool = False):
    """Mixed-lifetime streaming workload.

    For each inserted vector, we randomly assign:
      - tag = "short" with probability `short_fraction`  → deleted at
        insert_op + Uniform(0.5×T_short, 1.5×T_short) where T = short_lifetime_ops
      - tag = "long" otherwise  → deleted at + Uniform(0.5×T_long, 1.5×T_long)

    The router (if rich) gets the tag in `lifetime_tags` so cost-model
    admit can use it for per-vector decisions. Otherwise the tag is
    ignored.
    """
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    op = 0
    # Schedule: id -> (deletion_op, tag)
    schedule: dict[int, tuple[int, str]] = {}
    alive_set: set[int] = set()

    # Initial load
    init_ids = np.arange(init_n, dtype=np.int64)
    init_tags = ["init"] * init_n
    if use_gamma_rich:
        idx.initial_load(init_ids, np.ascontiguousarray(data[:init_n]),
                          lifetime_tags=init_tags)
    elif hasattr(idx, "initial_load"):  # gamma_v2
        idx.initial_load(init_ids, np.ascontiguousarray(data[:init_n]))
    else:
        idx.add(np.ascontiguousarray(data[:init_n]), init_ids)
    alive_set.update(int(i) for i in init_ids)
    for i in init_ids.tolist():
        schedule[int(i)] = (10**12, "init")  # never delete init
    op += init_n

    query_lat = []
    deleted_so_far = 0
    t0 = time.perf_counter()

    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        new_ids = np.arange(lo, hi, dtype=np.int64)
        new_vecs = np.ascontiguousarray(data[lo:hi])
        n_new = len(new_ids)

        # Assign tags + schedule deletes
        tags = []
        for i in new_ids.tolist():
            if rng.random() < short_fraction:
                tag = "short"
                death = op + int(rng.uniform(0.5, 1.5) * short_lifetime_ops)
            else:
                tag = "long"
                death = op + int(rng.uniform(0.5, 1.5) * long_lifetime_ops)
            tags.append(tag)
            schedule[int(i)] = (death, tag)

        if use_gamma_rich:
            idx.add(new_ids, new_vecs, lifetime_tags=tags)
        elif hasattr(idx, "initial_load"):  # gamma_v2-style
            idx.add(new_ids, new_vecs)
        else:
            idx.add(new_vecs, new_ids)
        alive_set.update(int(i) for i in new_ids)
        op += n_new

        # Apply scheduled deletes
        to_delete = [vid for vid, (death, _) in schedule.items()
                      if death <= op and vid in alive_set]
        if to_delete:
            arr = np.array(to_delete, dtype=np.int64)
            if use_gamma_rich or hasattr(idx, "initial_load"):
                idx.delete(arr)
            elif has_mark_deleted:
                for did in arr:
                    idx.mark_deleted(int(did))
            for vid in to_delete:
                alive_set.discard(vid)
                schedule.pop(vid, None)
            deleted_so_far += len(to_delete)

        if (lo - init_n) % qstride == 0:
            if use_gamma_rich or hasattr(idx, "initial_load"):
                idx.maintain()
            tq = time.perf_counter()
            idx.search(queries, 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))

    if use_gamma_rich or hasattr(idx, "initial_load"):
        idx.maintain()

    alive_arr = sorted(alive_set)
    sgt = compute_gt(data[alive_arr], queries, 10)
    sgt_orig = np.array(alive_arr, dtype=np.int64)[sgt]
    nbrs, _ = idx.search(queries, 10)
    return {
        "name": name,
        "total_s": time.perf_counter() - t0,
        "n_delete": deleted_so_far,
        "n_alive": len(alive_set),
        "query_latency_ms_p95": float(np.percentile(np.asarray(query_lat) * 1000, 95)) if query_lat else 0.0,
        "recall": recall_at_k(nbrs.astype(np.int64), sgt_orig.astype(np.int64)),
    }


# ----------------------------------------------------------------------
# Zipfian-query workload
# ----------------------------------------------------------------------
def run_zipfian_queries(idx, name, data, queries, init_n, batch, qstride,
                         *, zipf_alpha: float = 1.2,
                         seed: int = 42,
                         use_gamma_rich: bool = False,
                         has_mark_deleted: bool = False,
                         delete_every_n_inserts: int = 2):
    """Streaming insert/delete + Zipfian-distributed queries.

    Workload:
      - Inserts streamed in order (like sequential), but queries fired
        every qstride hit a Zipf-weighted subset of the Q query pool.
        Specifically we resample a subset of `len(queries)` queries
        from the pool with weights p_i ∝ 1/(i+1)^α; the same i-th query
        gets fired multiple times within a query batch.
      - Deletes happen FIFO every `delete_every_n_inserts` inserts (so
        a moderate churn).

    The rich router's heat tracker should pin the popular query targets
    in the hot tier, accelerating subsequent queries that hit the same
    candidates.
    """
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    op = 0

    # Pre-compute Zipf weights over query pool
    n_q = len(queries)
    weights = 1.0 / (np.arange(1, n_q + 1) ** zipf_alpha)
    weights /= weights.sum()

    init_ids = np.arange(init_n, dtype=np.int64)
    if use_gamma_rich:
        idx.initial_load(init_ids, np.ascontiguousarray(data[:init_n]))
    elif hasattr(idx, "initial_load"):
        idx.initial_load(init_ids, np.ascontiguousarray(data[:init_n]))
    else:
        idx.add(np.ascontiguousarray(data[:init_n]), init_ids)
    alive_set: set[int] = set(int(i) for i in init_ids)
    op += init_n

    deleted_so_far = 0
    query_lat = []
    insert_count_since_delete = 0
    t0 = time.perf_counter()

    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        new_ids = np.arange(lo, hi, dtype=np.int64)
        new_vecs = np.ascontiguousarray(data[lo:hi])
        if use_gamma_rich:
            idx.add(new_ids, new_vecs)
        elif hasattr(idx, "initial_load"):
            idx.add(new_ids, new_vecs)
        else:
            idx.add(new_vecs, new_ids)
        alive_set.update(int(i) for i in new_ids)
        op += len(new_ids)
        insert_count_since_delete += len(new_ids)

        # FIFO deletes
        if insert_count_since_delete >= delete_every_n_inserts * batch:
            n_to_del = insert_count_since_delete // delete_every_n_inserts
            # Delete oldest n_to_del
            del_ids = np.arange(deleted_so_far, deleted_so_far + n_to_del,
                                 dtype=np.int64)
            if use_gamma_rich or hasattr(idx, "initial_load"):
                idx.delete(del_ids)
            elif has_mark_deleted:
                for did in del_ids:
                    idx.mark_deleted(int(did))
            for vid in del_ids:
                alive_set.discard(int(vid))
            deleted_so_far += len(del_ids)
            insert_count_since_delete = 0

        if (lo - init_n) % qstride == 0:
            if use_gamma_rich or hasattr(idx, "initial_load"):
                idx.maintain()
            # Sample n_q queries Zipf-weighted from the pool
            sample_idx = rng.choice(n_q, size=n_q, replace=True, p=weights)
            zipf_q = np.ascontiguousarray(queries[sample_idx])
            tq = time.perf_counter()
            idx.search(zipf_q, 10)
            query_lat.append((time.perf_counter() - tq) / len(zipf_q))

    if use_gamma_rich or hasattr(idx, "initial_load"):
        idx.maintain()

    alive_arr = sorted(alive_set)
    sgt = compute_gt(data[alive_arr], queries, 10)
    sgt_orig = np.array(alive_arr, dtype=np.int64)[sgt]
    # Final search uses uniform queries to evaluate recall against ground truth
    nbrs, _ = idx.search(queries, 10)
    return {
        "name": name,
        "total_s": time.perf_counter() - t0,
        "n_delete": deleted_so_far,
        "n_alive": len(alive_set),
        "query_latency_ms_p95": float(np.percentile(np.asarray(query_lat) * 1000, 95)) if query_lat else 0.0,
        "recall": recall_at_k(nbrs.astype(np.int64), sgt_orig.astype(np.int64)),
    }
