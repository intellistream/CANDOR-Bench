"""Instrumented workload runner — separate timing for each op type.

The standard `run_workload` reports a single `total_s`. That conflates four
things: insert latency, delete latency, maintain (flush) cost, and search
latency. The user-facing claim of the buffer architecture is that
**insert and delete are dramatically faster** — neither is visible in
`total_s` alone.

This runner records, per-batch:
  - t_insert_per_vec_us  : avg microseconds per inserted vector (insert path only)
  - t_delete_per_vec_us  : avg microseconds per delete (cancel + mark_deleted path only)
  - t_maintain_us        : total microseconds spent in maintain() across all calls
  - t_search_per_q_us    : avg microseconds per query (search path only)
  - n_maintain_calls     : how many maintain() calls
  - cumulative wall time : the sum of the above

Plus the same metrics aggregated over the whole workload.

The conclusion from this is the right paper claim: "the buffer makes
insert P× faster and delete Q× faster, at the amortized cost of
periodic maintain passes that recover R% of the savings; net win is
S%."
"""
from __future__ import annotations
import time
import numpy as np
from typing import Optional

from .data import compute_gt
from .metrics import recall_at_k


def run_instrumented(idx, name, data, queries, init_n, batch, qstride,
                      delete_fn, *, use_gamma: bool, has_mark_deleted: bool,
                      max_deletes_per_batch: Optional[int] = None,
                      seed: int = 42):
    """Run streaming workload and record per-op-type timing.

    Returns the standard fields (total_s, recall, etc.) PLUS:
      n_inserts, n_deletes, n_searches, n_maintain_calls
      sum_t_insert_s, sum_t_delete_s, sum_t_maintain_s, sum_t_search_s
      mean_t_insert_per_vec_us, mean_t_delete_per_vec_us,
      mean_t_maintain_s, mean_t_search_per_q_us

    Note: delete timing only includes the synchronous delete-call cost
    (gamma's cancel-in-buffer or hnswlib's mark_deleted call). It does
    NOT include later overhead from carrying the tombstone (which shows
    up as inflated search latency).
    """
    n = data.shape[0]
    rng = np.random.default_rng(seed)

    # Initial load (timed separately, not folded into ops measurement)
    t0_init = time.perf_counter()
    if use_gamma:
        idx.initial_load(np.arange(init_n, dtype=np.int64),
                          np.ascontiguousarray(data[:init_n]))
    else:
        idx.add(np.ascontiguousarray(data[:init_n]),
                np.arange(init_n, dtype=np.int64))
    t_init_s = time.perf_counter() - t0_init

    alive_set = set(range(init_n))
    deleted_so_far = 0
    batch_idx = 0
    n_del_per = max_deletes_per_batch if max_deletes_per_batch else batch

    sum_t_insert_s = 0.0
    sum_t_delete_s = 0.0
    sum_t_maintain_s = 0.0
    sum_t_search_s = 0.0
    n_inserts = 0
    n_deletes = 0
    n_searches = 0
    n_maintain_calls = 0
    query_lat_per_q = []

    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        new_ids = np.arange(lo, hi, dtype=np.int64)
        new_vecs = np.ascontiguousarray(data[lo:hi])

        # ---- INSERT (timed) ----
        ti0 = time.perf_counter()
        if use_gamma:
            idx.add(new_ids, new_vecs)
        else:
            idx.add(new_vecs, new_ids)
        sum_t_insert_s += time.perf_counter() - ti0
        n_inserts += len(new_ids)
        alive_set.update(new_ids.tolist())

        # ---- DELETE (timed) ----
        if batch_idx >= 1 and len(alive_set) > n_del_per:
            del_ids = delete_fn(alive_set, n_del_per, rng, data, deleted_so_far,
                                batch_idx=batch_idx)
            del_ids = np.asarray(del_ids, dtype=np.int64)
            if len(del_ids) > 0:
                td0 = time.perf_counter()
                if use_gamma:
                    idx.delete(del_ids)
                elif has_mark_deleted:
                    for did in del_ids:
                        idx.mark_deleted(int(did))
                sum_t_delete_s += time.perf_counter() - td0
                n_deletes += len(del_ids)
                for did in del_ids:
                    alive_set.discard(int(did))
                deleted_so_far += len(del_ids)

        # ---- MAINTAIN + SEARCH (timed separately, only at qstride boundary) ----
        if (lo - init_n) % qstride == 0:
            if use_gamma:
                tm0 = time.perf_counter()
                idx.maintain()
                sum_t_maintain_s += time.perf_counter() - tm0
                n_maintain_calls += 1
            ts0 = time.perf_counter()
            idx.search(queries, 10)
            ds = time.perf_counter() - ts0
            sum_t_search_s += ds
            n_searches += len(queries)
            query_lat_per_q.append(ds / max(1, len(queries)))

        batch_idx += 1

    # Final maintain + recall measurement (not counted toward op timing — it's
    # a one-shot finalize)
    t_final = time.perf_counter()
    if use_gamma:
        tm0 = time.perf_counter()
        idx.maintain()
        sum_t_maintain_s += time.perf_counter() - tm0
        n_maintain_calls += 1

    alive_arr = sorted(alive_set)
    sgt = compute_gt(data[alive_arr], queries, 10)
    sgt_orig = np.array(alive_arr, dtype=np.int64)[sgt]
    nbrs, _ = idx.search(queries, 10)
    rec = recall_at_k(nbrs.astype(np.int64), sgt_orig.astype(np.int64))

    total_s = time.perf_counter() - t0
    return {
        "name": name,
        "total_s": total_s,
        "t_init_s": t_init_s,
        "n_delete": deleted_so_far,
        "n_alive": len(alive_set),
        "recall": rec,
        # Op-type cumulative time:
        "sum_t_insert_s": sum_t_insert_s,
        "sum_t_delete_s": sum_t_delete_s,
        "sum_t_maintain_s": sum_t_maintain_s,
        "sum_t_search_s": sum_t_search_s,
        # Op counts:
        "n_inserts": n_inserts,
        "n_deletes": n_deletes,
        "n_searches": n_searches,
        "n_maintain_calls": n_maintain_calls,
        # Per-op latencies (microseconds):
        "mean_t_insert_per_vec_us": (sum_t_insert_s / max(1, n_inserts)) * 1e6,
        "mean_t_delete_per_vec_us": (sum_t_delete_s / max(1, n_deletes)) * 1e6,
        "mean_t_maintain_per_call_ms": (sum_t_maintain_s / max(1, n_maintain_calls)) * 1e3,
        "mean_t_search_per_q_us": (sum_t_search_s / max(1, n_searches)) * 1e6,
        # Search p95 in milliseconds (matches old workloads.py):
        "query_latency_ms_p95": float(np.percentile(np.asarray(query_lat_per_q) * 1000, 95)) if query_lat_per_q else 0.0,
    }
