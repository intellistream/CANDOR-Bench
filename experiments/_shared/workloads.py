"""Shared streaming insert/delete workload runners + delete-pattern generators.

Used by e15-e19 to keep methodology identical across experiments.

Delete patterns
---------------
- sequential : delete oldest-inserted ID first (FIFO/LRU). hnswlib's best case.
- random     : delete uniformly-random alive ID. Realistic for many workloads.
- cluster    : pick a pivot, delete its k-NN among alive. Hot-spot / category churn.
- partial_reset : every 8 batches, big random delete burst. TTL cleanup pattern.
"""
import time
import numpy as np

from .data import compute_gt
from .metrics import recall_at_k


def deletes_sequential(alive, n_to_del, rng, data, deleted_so_far, **kw):
    return np.arange(deleted_so_far, deleted_so_far + n_to_del, dtype=np.int64)


def deletes_random(alive, n_to_del, rng, data, deleted_so_far, **kw):
    arr = np.fromiter(alive, dtype=np.int64, count=len(alive))
    if len(arr) < n_to_del:
        return arr
    return rng.choice(arr, size=n_to_del, replace=False)


def deletes_cluster(alive, n_to_del, rng, data, deleted_so_far, **kw):
    arr = np.fromiter(alive, dtype=np.int64, count=len(alive))
    if len(arr) <= n_to_del:
        return arr
    pivot = data[int(rng.choice(arr))]
    sample_size = min(len(arr), 5000)
    sampled = rng.choice(arr, size=sample_size, replace=False)
    diffs = data[sampled] - pivot
    dists = (diffs * diffs).sum(axis=1)
    return sampled[np.argpartition(dists, min(n_to_del, len(sampled) - 1))[:n_to_del]]


def deletes_partial_reset(alive, n_to_del, rng, data, deleted_so_far, batch_idx, **kw):
    if batch_idx % 8 == 0:
        big = n_to_del * 8
        arr = np.fromiter(alive, dtype=np.int64, count=len(alive))
        if len(arr) <= big:
            return arr
        return rng.choice(arr, size=big, replace=False)
    return np.array([], dtype=np.int64)


PATTERNS = {
    "sequential": deletes_sequential,
    "random": deletes_random,
    "cluster": deletes_cluster,
    "partial_reset": deletes_partial_reset,
}


def make_pattern(kind: str):
    return PATTERNS[kind]


def run_workload(idx, name, data, queries, init_n, batch, qstride,
                  delete_fn, use_gamma, has_mark_deleted,
                  max_deletes_per_batch=None, seed: int = 42):
    """Stream insert + delete + periodic query. Returns metrics dict.

    Parameters
    ----------
    idx : index object (gamma_v2 if use_gamma else a backend exposing add/mark_deleted/search)
    data : (N, d) float32. Streaming order assumed = id order.
    queries : (Q, d) float32.
    init_n : initial bulk-load size.
    batch : insert batch size per step.
    qstride : query every `qstride` inserted-rows.
    delete_fn : function (alive_set, n_to_del, rng, data, deleted_so_far, batch_idx=...) → np.ndarray of ids.
    use_gamma : True for GammaPyHybridV2 (id-first add API), False for direct backend.
    has_mark_deleted : True if backend supports mark_deleted (skip delete otherwise).
    max_deletes_per_batch : if set, override `batch` as the per-step delete count.
    seed : RNG seed for reproducible delete patterns.
    """
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    if use_gamma:
        idx.initial_load(np.arange(init_n, dtype=np.int64),
                          np.ascontiguousarray(data[:init_n]))
    else:
        idx.add(np.ascontiguousarray(data[:init_n]),
                np.arange(init_n, dtype=np.int64))
    alive_set = set(range(init_n))
    deleted_so_far = 0
    batch_idx = 0
    query_lat = []
    n_del_per = max_deletes_per_batch if max_deletes_per_batch else batch
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        new_ids = np.arange(lo, hi, dtype=np.int64)
        new_vecs = np.ascontiguousarray(data[lo:hi])
        if use_gamma:
            idx.add(new_ids, new_vecs)
        else:
            idx.add(new_vecs, new_ids)
        alive_set.update(new_ids.tolist())
        if batch_idx >= 1 and len(alive_set) > n_del_per:
            del_ids = delete_fn(alive_set, n_del_per, rng, data, deleted_so_far,
                                batch_idx=batch_idx)
            del_ids = np.asarray(del_ids, dtype=np.int64)
            if len(del_ids) > 0:
                if use_gamma:
                    idx.delete(del_ids)
                elif has_mark_deleted:
                    for did in del_ids:
                        idx.mark_deleted(int(did))
                for did in del_ids:
                    alive_set.discard(int(did))
                deleted_so_far += len(del_ids)
        if (lo - init_n) % qstride == 0:
            if use_gamma:
                idx.maintain()
            tq = time.perf_counter()
            idx.search(queries, 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1
    if use_gamma:
        idx.maintain()
    alive_arr = sorted(alive_set)
    sgt = compute_gt(data[alive_arr], queries, 10)
    alive_arr_np = np.array(alive_arr, dtype=np.int64)
    sgt_orig = alive_arr_np[sgt]
    nbrs, _ = idx.search(queries, 10)
    return {
        "name": name,
        "total_s": time.perf_counter() - t0,
        "n_delete": deleted_so_far,
        "n_alive": len(alive_set),
        "query_latency_ms_p95": float(np.percentile(np.asarray(query_lat) * 1000, 95)) if query_lat else 0.0,
        "recall": recall_at_k(nbrs.astype(np.int64), sgt_orig.astype(np.int64)),
    }
