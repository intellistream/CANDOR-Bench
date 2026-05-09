"""e14 — Optimized buffer + diverse delete patterns.

Tests gamma's hybrid (with vectorized buffer) against direct backends
under DIVERSE delete patterns. Goal: find scenarios where gamma still
wins (e.g., where backend's mark_deleted falls short).
"""
import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, compute_gt, recall_at_k
from _shared.gamma_py_v2 import GammaPyHybridV2
from _shared.gamma_py import HnswlibBackend, FaissHnswBackend
import numpy as np


# === Delete patterns ===

def deletes_sequential(n_alive, deleted_so_far, n_to_del):
    """Delete oldest n_to_del IDs (FIFO, news-feed style)."""
    return np.arange(deleted_so_far, deleted_so_far + n_to_del, dtype=np.int64)


def deletes_random(n_alive_set, n_to_del, rng):
    """Pick n_to_del random IDs from currently alive set."""
    alive = np.array(list(n_alive_set))
    if len(alive) < n_to_del:
        return alive
    chosen = rng.choice(alive, size=n_to_del, replace=False)
    return chosen


def deletes_cluster(data, alive_set, n_to_del, rng):
    """Pick a random pivot vector and delete its k nearest alive neighbors.
    Simulates 'delete a hot-spot' (e.g., user removed all photos from a trip)."""
    alive = np.array(list(alive_set), dtype=np.int64)
    if len(alive) <= n_to_del:
        return alive
    pivot_idx = rng.choice(alive)
    pivot = data[int(pivot_idx)]
    diffs = data[alive] - pivot
    dists = (diffs * diffs).sum(axis=1)
    nearest = alive[np.argpartition(dists, n_to_del)[:n_to_del]]
    return nearest


# === Workload runners ===

def run_workload(idx, name, data, queries, init_n, batch, qstride,
                  delete_fn, use_gamma, has_maint, has_mark_deleted):
    """Generic streaming workload with arbitrary delete pattern."""
    n = data.shape[0]
    rng = np.random.default_rng(42)
    # Initialize
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
        # Delete
        n_del = batch  # delete same volume as inserted (sliding window)
        if batch_idx >= 1 and len(alive_set) > n_del:
            del_ids = delete_fn(alive_set, n_del, rng, data, deleted_so_far)
            del_ids = np.asarray(del_ids, dtype=np.int64)
            if use_gamma:
                idx.delete(del_ids)
            elif has_mark_deleted:
                for did in del_ids:
                    idx.mark_deleted(int(did))
            for did in del_ids:
                alive_set.discard(int(did))
            deleted_so_far += len(del_ids)
        # Query checkpoint
        if (lo - init_n) % qstride == 0:
            if use_gamma and has_maint:
                idx.maintain()
            tq = time.perf_counter()
            idx.search(queries, 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1
    if use_gamma and has_maint:
        idx.maintain()
    # Compute final recall against alive set
    alive_arr = sorted(alive_set)
    alive_data = data[alive_arr]
    sgt = compute_gt(alive_data, queries, 10)
    # Map sgt indices (in alive_data) to original ids
    alive_arr_np = np.array(alive_arr, dtype=np.int64)
    sgt_orig = alive_arr_np[sgt]
    nbrs, _ = idx.search(queries, 10)
    recall = recall_at_k(nbrs.astype(np.int64), sgt_orig.astype(np.int64))
    return {
        "name": name,
        "total_s": time.perf_counter() - t0,
        "n_delete": deleted_so_far,
        "query_latency_ms_p95": float(np.percentile(np.asarray(query_lat) * 1000, 95)) if query_lat else 0.0,
        "recall": recall,
    }


# === Delete pattern wrappers (closure to match callsite signature) ===

def make_pattern(kind):
    if kind == "sequential":
        return lambda alive, n, rng, data, deleted: deletes_sequential(
            len(alive), deleted, n)
    if kind == "random":
        return lambda alive, n, rng, data, deleted: deletes_random(alive, n, rng)
    if kind == "cluster":
        return lambda alive, n, rng, data, deleted: deletes_cluster(
            data, alive, n, rng)
    raise ValueError(kind)


def main():
    data, queries = load_dataset("sift", n_queries=500, slice_n=200_000)
    init_n, batch, qstride = 20_000, 2_500, 10_000
    n, d = data.shape
    rows = []

    for pattern in ["sequential", "random", "cluster"]:
        print(f"\n========== delete_pattern = {pattern} ==========", flush=True)
        delete_fn = make_pattern(pattern)
        for be_name, be_factory in [("hnswlib", lambda: HnswlibBackend(d, max_elements=n)),
                                     ("faiss_hnsw", lambda: FaissHnswBackend(d, max_elements=n))]:
            # gamma_v2 + this backend
            be = be_factory()
            g = GammaPyHybridV2(be, d, buf_capacity=batch * 50)
            r = run_workload(g, f"gamma_v2+{be_name}", data, queries, init_n, batch, qstride,
                             delete_fn, use_gamma=True, has_maint=True, has_mark_deleted=False)
            rows.append({"pattern": pattern, **r})
            print(f"  {r['name']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
            # backend direct
            be2 = be_factory()
            r = run_workload(be2, f"{be_name} direct", data, queries, init_n, batch, qstride,
                             delete_fn, use_gamma=False, has_maint=False, has_mark_deleted=True)
            rows.append({"pattern": pattern, **r})
            print(f"  {r['name']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)
            out = os.path.join(os.path.dirname(__file__), "output.json")
            with open(out, "w") as f:
                json.dump(rows, f, indent=2)
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
