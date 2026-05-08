"""Scenario sweep: GammaFresh vs FaissHNSW vs AdaIVF on multiple workloads + datasets.

Each scenario reports total wall time, total ops, throughput, and final recall.
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, "/home/rprp/CANDOR-Bench")
sys.path.insert(0, "/home/rprp/CANDOR-Bench/algorithms_impl/GammaFresh/build/src/bindings/python")
sys.path.insert(0, "/home/rprp/CANDOR-Bench/algorithms_impl/GammaFresh/python")
from candy import Index


def load_sift_small():
    from datasets.registry import get_dataset
    ds = get_dataset("sift-small")
    data = np.asarray(ds.get_dataset()).astype(np.float32)
    queries = np.asarray(ds.get_queries()).astype(np.float32)
    gt = np.load("/home/rprp/CANDOR-Bench/raw_data/sift-small/sift_small_gt10.npy")
    return data, queries, gt


def load_random_100k():
    return np.load("/tmp/r100k_data.npy"), np.load("/tmp/r100k_q.npy"), np.load("/tmp/r100k_gt.npy")


def load_clustered_50k():
    return np.load("/tmp/c50k_data.npy"), np.load("/tmp/c50k_q.npy"), np.load("/tmp/c50k_gt.npy")


def make_gamma(dim, **ovr):
    base = {"backend": "FaissHNSW",
            "split_factor": 4.0, "gamma_split_threshold": 0.4,
            "candidate_reserve_size": 8,
            "maintenance_interval": 4, "maintenance_query_interval": 4,
            "maintenance_query_threshold_scale": 0.5}
    base.update(ovr)
    return {"global": {"dim": dim},
            "indexes": {"gamma_fresh": base,
                        "faiss_hnsw": {"m": 24, "ef_construction": 100, "ef_search": 64,
                                       "tombstone_rebuild_threshold": 512}}}


def make_faiss(dim):
    return {"global": {"dim": dim},
            "indexes": {"faiss_hnsw": {"m": 24, "ef_construction": 100, "ef_search": 64,
                                       "tombstone_rebuild_threshold": 512}}}


def make_ivf(dim):
    return {"global": {"dim": dim},
            "indexes": {"ada_ivf": {"target_posting_size": 64, "min_nprobe": 4, "max_nprobe": 64,
                                    "min_clusters": 16, "brute_force_threshold": 1000,
                                    "recluster_min_vectors": 5000, "max_training_points": 10000,
                                    "probe_ratio": 0.01, "nprobe_multiplier": 2,
                                    "kmeans_iterations": 12, "kmeans_max_iterations": 100,
                                    "kmeans_min_iterations": 1, "candidate_fanout": 4,
                                    "centroid_small_multiplier": 2, "default_seed": 17,
                                    "max_clusters": 65536}}}


def build_index(name, dim):
    if name == "gamma":
        return Index("GammaFresh", dim, config=make_gamma(dim))
    if name == "faiss":
        return Index("FaissHNSW", dim, config=make_faiss(dim))
    if name == "ivf":
        return Index("AdaIVF", dim, config=make_ivf(dim))
    raise ValueError(name)


def maintain(idx, name, vector_budget=0):
    """Call maintain on indexes that support it (no-op otherwise). Bench-driven:
    bench picks WHEN and HOW MUCH (vector_budget); the algorithm picks WHICH
    vectors (cold-and-old first, hot/recent stay in buffer for hybrid scan).
    vector_budget=0 means unbounded — drain everything migratable.
    """
    if name == "gamma":
        idx.maintain(int(vector_budget), 0, False)


def recall_at_k(pred, gt, k=10):
    return float(np.mean([len(set(p) & set(g)) / k for p, g in zip(pred, gt)]))


def scenario_streaming(idx, data, queries, gt, init_n, algo_name, batch=100, qstride=100):
    """Pure streaming insert + periodic query, with explicit maintenance between batches."""
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    n = len(data)
    t = time.perf_counter()
    ops = 0
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        idx.add(np.arange(lo, hi, dtype=np.uint64), np.ascontiguousarray(data[lo:hi]))
        ops += (hi - lo)
        if (lo - init_n) % qstride == 0:
            maintain(idx, algo_name)  # bench-driven maintain before query
            idx.search(np.ascontiguousarray(queries), 10)
            ops += len(queries)
    maintain(idx, algo_name)  # final drain before final search
    t = time.perf_counter() - t
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    return t, ops, recall_at_k(nbrs.astype(np.uint32), gt)


def scenario_bulk_delete(idx, data, queries, gt, init_n, algo_name, delete_frac=0.3):
    """Insert all, then bulk-delete a chunk, then query.
    NO explicit maint at the end — this scenario tests the algorithm's natural
    post-delete state. GammaFresh's hybrid graph+buffer-scan should keep recall
    high even with leftover buffered vectors; HNSW pays for its tombstones.
    """
    n = len(data)
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    t = time.perf_counter()
    idx.add(np.arange(init_n, n, dtype=np.uint64),
            np.ascontiguousarray(data[init_n:n]))
    n_del = int(n * delete_frac)
    # delete the OLDEST chunk (challenges concept drift)
    idx.delete(np.arange(0, n_del, dtype=np.uint64))
    t = time.perf_counter() - t
    # GT after delete: set neighbors that point to deleted ids → 0 hit.
    surviving = np.arange(n_del, n, dtype=np.uint32)
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    nbrs = nbrs.astype(np.uint32)
    # Re-compute GT on surviving subset
    surviving_data = data[n_del:]
    qn = (queries**2).sum(1, keepdims=True); dn = (surviving_data**2).sum(1)
    d2 = qn + dn - 2 * (queries @ surviving_data.T)
    gt_local = np.argpartition(d2, 10, axis=1)[:, :10]
    for i in range(gt_local.shape[0]):
        gt_local[i] = gt_local[i, np.argsort(d2[i, gt_local[i]])]
    gt_local = (gt_local + n_del).astype(np.uint32)
    return t, n + n_del + len(queries), recall_at_k(nbrs, gt_local)


def scenario_drift(idx, data, queries, gt, init_n, algo_name, drift_chunks=4, batch=100):
    """Insert in chunks, delete-oldest after each chunk → simulates drift."""
    n = len(data)
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    chunk_size = (n - init_n) // drift_chunks
    delete_each = chunk_size // 2
    t = time.perf_counter()
    ops = 0
    cur_lo = init_n
    last_delete = 0
    for c in range(drift_chunks):
        chunk_end = min(cur_lo + chunk_size, n)
        for lo in range(cur_lo, chunk_end, batch):
            hi = min(lo + batch, chunk_end)
            idx.add(np.arange(lo, hi, dtype=np.uint64),
                    np.ascontiguousarray(data[lo:hi]))
            ops += (hi - lo)
        # delete a slice of older vectors
        del_start = last_delete
        del_end = del_start + delete_each
        idx.delete(np.arange(del_start, del_end, dtype=np.uint64))
        last_delete = del_end
        ops += (del_end - del_start)
        maintain(idx, algo_name)  # bench-driven maint between drift cycles
        idx.search(np.ascontiguousarray(queries), 10)
        ops += len(queries)
        cur_lo = chunk_end
    t = time.perf_counter() - t
    surviving_data = data[last_delete:]
    qn = (queries**2).sum(1, keepdims=True); dn = (surviving_data**2).sum(1)
    d2 = qn + dn - 2 * (queries @ surviving_data.T)
    gt_local = np.argpartition(d2, 10, axis=1)[:, :10]
    for i in range(gt_local.shape[0]):
        gt_local[i] = gt_local[i, np.argsort(d2[i, gt_local[i]])]
    gt_local = (gt_local + last_delete).astype(np.uint32)
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    return t, ops, recall_at_k(nbrs.astype(np.uint32), gt_local)


def scenario_churn(idx, data, queries, gt, init_n, algo_name, cycles=20, churn_per_cycle=0.05):
    """Repeated insert+delete+query cycles. Each cycle deletes oldest chunk and
    inserts new chunk of equal size. Aggressive churn stresses HNSW tombstones
    and rebuilds; GammaFresh can drop ghosted partitions cheaply."""
    n = len(data)
    # Pre-load up to half the data so we have room to churn through the rest
    base_end = min(init_n + (n - init_n) // 2, n)
    idx.initial_load(np.arange(base_end, dtype=np.uint64),
                     np.ascontiguousarray(data[:base_end]))
    chunk = max(1, int(base_end * churn_per_cycle))
    cur_lo = base_end       # next id to insert (data[cur_lo:cur_lo+chunk])
    del_lo = 0              # oldest live id
    t = time.perf_counter()
    ops = 0
    for c in range(cycles):
        if cur_lo + chunk > n:
            break
        # delete oldest chunk
        idx.delete(np.arange(del_lo, del_lo + chunk, dtype=np.uint64))
        del_lo += chunk
        ops += chunk
        # insert new chunk
        idx.add(np.arange(cur_lo, cur_lo + chunk, dtype=np.uint64),
                np.ascontiguousarray(data[cur_lo:cur_lo + chunk]))
        cur_lo += chunk
        ops += chunk
        # Bench picks the budget — half a chunk per cycle drains stable vectors
        # incrementally without paying for a full HNSW rebuild every cycle. The
        # algorithm picks WHICH vectors (cold-and-old first); freshly inserted
        # hot vectors stay in buffer so query's hybrid scan still finds them.
        maintain(idx, algo_name, vector_budget=chunk // 2)
        # query
        idx.search(np.ascontiguousarray(queries), 10)
        ops += len(queries)
    t = time.perf_counter() - t
    surviving_data = data[del_lo:cur_lo]
    qn = (queries**2).sum(1, keepdims=True); dn = (surviving_data**2).sum(1)
    d2 = qn + dn - 2 * (queries @ surviving_data.T)
    gt_local = np.argpartition(d2, 10, axis=1)[:, :10]
    for i in range(gt_local.shape[0]):
        gt_local[i] = gt_local[i, np.argsort(d2[i, gt_local[i]])]
    gt_local = (gt_local + del_lo).astype(np.uint32)
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    return t, ops, recall_at_k(nbrs.astype(np.uint32), gt_local)


def scenario_burst(idx, data, queries, gt, init_n, algo_name, burst=2500):
    """Large-batch insert then query — IVF strength territory."""
    n = len(data)
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    t = time.perf_counter()
    ops = 0
    for lo in range(init_n, n, burst):
        hi = min(lo + burst, n)
        idx.add(np.arange(lo, hi, dtype=np.uint64),
                np.ascontiguousarray(data[lo:hi]))
        ops += (hi - lo)
    maintain(idx, algo_name)  # drain before final query
    idx.search(np.ascontiguousarray(queries), 10)
    ops += len(queries)
    t = time.perf_counter() - t
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    return t, ops, recall_at_k(nbrs.astype(np.uint32), gt)


SCENARIOS = {
    "streaming(b=100)":   (scenario_streaming, dict(batch=100)),
    "burst(b=2500)":      (scenario_burst, dict(burst=2500)),
    "drift(4chunk)":      (scenario_drift, dict(drift_chunks=4)),
    "bulk_delete(30%)":   (scenario_bulk_delete, dict(delete_frac=0.3)),
    "churn(20cyc,5%)":    (scenario_churn, dict(cycles=20, churn_per_cycle=0.05)),
}

# random-100k churn is dominated by per-maintain bookkeeping on a churning workload
# (a known optimization area). Skip the 50-cycle variant to keep the sweep reasonable.

DATASETS = [
    ("sift-small (10K, 128d)", load_sift_small, 1000),
    ("clustered-50k (50K, 128d, 50 clusters)", load_clustered_50k, 2500),
    ("random-100k (100K, 128d)", load_random_100k, 5000),
]

ALGOS = ["gamma", "faiss", "ivf"]


def main():
    rows = []
    for ds_name, loader, init_n in DATASETS:
        data, queries, gt = loader()
        print(f"\n========== {ds_name} ==========")
        for sc_name, (sc_fn, kwargs) in SCENARIOS.items():
            print(f"  --- {sc_name} ---")
            for algo in ALGOS:
                idx = build_index(algo, data.shape[1])
                t, ops, r = sc_fn(idx, data, queries, gt, init_n, algo, **kwargs)
                qps = ops / t if t > 0 else 0
                print(f"    {algo:8s}  time={t*1000:>6.0f}ms  ops/s={qps:>7.0f}  recall@10={r:.4f}")
                rows.append({"dataset": ds_name, "scenario": sc_name, "algo": algo,
                             "time_ms": t*1000, "qps": qps, "recall": r})
    print("\n=========== SUMMARY ===========")
    by_scenario = {}
    for r in rows:
        key = (r["dataset"], r["scenario"])
        by_scenario.setdefault(key, []).append(r)
    for key, group in by_scenario.items():
        print(f"\n{key[0]} | {key[1]}")
        winner = min(group, key=lambda x: x["time_ms"])
        for g in group:
            mark = " 🏆" if g is winner else ""
            print(f"  {g['algo']:8s}  {g['time_ms']:>6.0f}ms  {g['qps']:>7.0f} ops/s  recall {g['recall']:.4f}{mark}")
    with open("/tmp/scenario_results.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
