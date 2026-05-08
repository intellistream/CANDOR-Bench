"""SIFT 1M (1,000,000 × 128d) full-scenario benchmark.

Same scenarios as scenario_bench.py but on a real 1M-vector workload, plus
detailed per-phase metrics: insert latency, query latency p50/p95/p99,
end-to-end QPS, recall@10. Uses 1000 queries (subset of SIFT's 10K) for
faster GT/recall computation.
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
from datasets.registry import get_dataset


# ---------------------------- dataset prep ---------------------------- #
def load_sift_1m(n_queries: int = 1000):
    ds = get_dataset("sift")
    data = np.asarray(ds.get_dataset(), dtype=np.float32)  # (1M, 128)
    queries = np.asarray(ds.get_queries(), dtype=np.float32)  # (10K, 128)
    queries = queries[:n_queries]
    return data, queries


def compute_gt(data: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    """Brute force GT over (data, queries). Memory-aware blocked compute."""
    nq = queries.shape[0]
    qn = (queries**2).sum(1, keepdims=True)
    gt = np.empty((nq, k), dtype=np.uint32)
    block = 50000
    dn_full = (data**2).sum(1)
    for i0 in range(0, nq, 100):
        i1 = min(i0 + 100, nq)
        q = queries[i0:i1]
        qni = qn[i0:i1]
        # block over data to avoid huge cross matrix
        best_dist = np.full((i1 - i0, k), np.inf, dtype=np.float32)
        best_idx = np.full((i1 - i0, k), -1, dtype=np.int64)
        for j0 in range(0, data.shape[0], block):
            j1 = min(j0 + block, data.shape[0])
            d2 = qni + dn_full[j0:j1] - 2 * (q @ data[j0:j1].T)
            # combine with running top-k
            combined_dist = np.concatenate([best_dist, d2.astype(np.float32)], axis=1)
            combined_idx = np.concatenate(
                [best_idx, np.broadcast_to(np.arange(j0, j1, dtype=np.int64), (i1 - i0, j1 - j0))],
                axis=1,
            )
            order = np.argpartition(combined_dist, k, axis=1)[:, :k]
            row_idx = np.arange(i1 - i0)[:, None]
            best_dist = combined_dist[row_idx, order]
            best_idx = combined_idx[row_idx, order]
        # final sort within top-k
        for r in range(i1 - i0):
            o = np.argsort(best_dist[r])
            gt[i0 + r] = best_idx[r, o].astype(np.uint32)
    return gt


# ---------------------------- algo wrappers --------------------------- #
def make_gamma(dim, **ovr):
    base = {
        "backend": "FaissHNSW",
        "split_factor": 4.0, "gamma_split_threshold": 0.4,
        "candidate_reserve_size": 8,
        "maintenance_interval": 4, "maintenance_query_interval": 4,
        "maintenance_query_threshold_scale": 0.5,
    }
    base.update(ovr)
    return {"global": {"dim": dim},
            "indexes": {"gamma_fresh": base,
                        "faiss_hnsw": {"m": 32, "ef_construction": 120, "ef_search": 80,
                                       "tombstone_rebuild_threshold": 512}}}


def make_faiss(dim):
    return {"global": {"dim": dim},
            "indexes": {"faiss_hnsw": {"m": 32, "ef_construction": 120, "ef_search": 80,
                                       "tombstone_rebuild_threshold": 512}}}


def make_ivf(dim):
    # SIFT 1M: target ~256 clusters
    return {"global": {"dim": dim},
            "indexes": {"ada_ivf": {"target_posting_size": 4096, "min_nprobe": 16, "max_nprobe": 256,
                                    "min_clusters": 64, "brute_force_threshold": 5000,
                                    "recluster_min_vectors": 100000, "max_training_points": 100000,
                                    "probe_ratio": 0.02, "nprobe_multiplier": 2,
                                    "kmeans_iterations": 12, "kmeans_max_iterations": 100,
                                    "kmeans_min_iterations": 1, "candidate_fanout": 4,
                                    "centroid_small_multiplier": 2, "default_seed": 17,
                                    "max_clusters": 65536}}}


def build(name, dim):
    if name == "gamma":
        return Index("GammaFresh", dim, config=make_gamma(dim))
    if name == "faiss":
        return Index("FaissHNSW", dim, config=make_faiss(dim))
    if name == "ivf":
        return Index("AdaIVF", dim, config=make_ivf(dim))
    raise ValueError(name)


def maintain(idx, name, vector_budget=0):
    if name == "gamma":
        idx.maintain(int(vector_budget), 0, False)


# ---------------------------- metrics --------------------------------- #
def percentile(arr, p):
    if not arr:
        return 0.0
    return float(np.percentile(np.asarray(arr) * 1000, p))  # to ms


def recall_at_k(pred, gt, k=10):
    return float(np.mean([len(set(p) & set(g)) / k for p, g in zip(pred, gt)]))


# ---------------------------- scenarios ------------------------------- #
def scenario_streaming(idx, data, queries, gt, init_n, algo_name, batch=2500, qstride=10000):
    """Initial bulk + streaming insert with periodic queries."""
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    n = len(data)
    insert_lat = []
    query_lat = []
    t = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64), np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
        if (lo - init_n) % qstride == 0:
            maintain(idx, algo_name)
            tq = time.perf_counter()
            idx.search(np.ascontiguousarray(queries), 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
    maintain(idx, algo_name)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    query_lat.append((time.perf_counter() - tq) / len(queries))
    total = time.perf_counter() - t
    return {
        "total_s": total,
        "n_insert": n - init_n,
        "n_query": len(queries) * len(query_lat),
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "insert_latency_us_p99": percentile(insert_lat, 99) * 1000,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000,
        "query_latency_ms_p50": percentile(query_lat, 50),
        "query_latency_ms_p95": percentile(query_lat, 95),
        "query_latency_ms_p99": percentile(query_lat, 99),
        "recall": recall_at_k(nbrs.astype(np.uint32), gt),
    }


def scenario_burst(idx, data, queries, gt, init_n, algo_name, burst=10000):
    """Large-batch insert + final query."""
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    n = len(data)
    insert_lat = []
    t = time.perf_counter()
    for lo in range(init_n, n, burst):
        hi = min(lo + burst, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64), np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
    maintain(idx, algo_name)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    final_q_s = time.perf_counter() - tq
    total = time.perf_counter() - t
    return {
        "total_s": total,
        "n_insert": n - init_n,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "insert_latency_us_p99": percentile(insert_lat, 99) * 1000,
        "final_query_qps": len(queries) / final_q_s,
        "final_query_latency_ms_avg": (final_q_s / len(queries)) * 1000,
        "recall": recall_at_k(nbrs.astype(np.uint32), gt),
    }


def scenario_drift(idx, data, queries, gt, init_n, algo_name, drift_chunks=5):
    """Insert-then-delete-oldest cycles."""
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    n = len(data)
    chunk_size = (n - init_n) // drift_chunks
    delete_each = chunk_size // 2
    insert_lat, delete_lat, query_lat = [], [], []
    cur_lo, last_delete = init_n, 0
    t = time.perf_counter()
    for c in range(drift_chunks):
        chunk_end = min(cur_lo + chunk_size, n)
        for lo in range(cur_lo, chunk_end, 2500):
            hi = min(lo + 2500, chunk_end)
            ti = time.perf_counter()
            idx.add(np.arange(lo, hi, dtype=np.uint64), np.ascontiguousarray(data[lo:hi]))
            insert_lat.append((time.perf_counter() - ti) / (hi - lo))
        del_start, del_end = last_delete, last_delete + delete_each
        td = time.perf_counter()
        idx.delete(np.arange(del_start, del_end, dtype=np.uint64))
        delete_lat.append((time.perf_counter() - td) / (del_end - del_start))
        last_delete = del_end
        maintain(idx, algo_name)
        tq = time.perf_counter()
        idx.search(np.ascontiguousarray(queries), 10)
        query_lat.append((time.perf_counter() - tq) / len(queries))
        cur_lo = chunk_end

    surviving = data[last_delete:]
    surviving_gt = compute_gt(surviving, queries, 10)
    surviving_gt = (surviving_gt + last_delete).astype(np.uint32)
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    total = time.perf_counter() - t
    return {
        "total_s": total,
        "n_insert": n - init_n,
        "n_delete": last_delete,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": float(np.mean(delete_lat)) * 1e6,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000,
        "query_latency_ms_p95": percentile(query_lat, 95),
        "recall": recall_at_k(nbrs.astype(np.uint32), surviving_gt),
    }


def scenario_bulk_delete(idx, data, queries, gt, init_n, algo_name, delete_frac=0.3):
    """Insert all + bulk-delete oldest 30% + query (NO maint at end)."""
    n = len(data)
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat = []
    t = time.perf_counter()
    ti = time.perf_counter()
    idx.add(np.arange(init_n, n, dtype=np.uint64), np.ascontiguousarray(data[init_n:n]))
    insert_lat.append((time.perf_counter() - ti) / (n - init_n))
    n_del = int(n * delete_frac)
    td = time.perf_counter()
    idx.delete(np.arange(0, n_del, dtype=np.uint64))
    delete_s = time.perf_counter() - td
    total = time.perf_counter() - t

    surviving = data[n_del:]
    surviving_gt = compute_gt(surviving, queries, 10)
    surviving_gt = (surviving_gt + n_del).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    final_q_s = time.perf_counter() - tq
    return {
        "total_s": total,
        "n_insert": n - init_n,
        "n_delete": n_del,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": (delete_s / n_del) * 1e6,
        "final_query_qps": len(queries) / final_q_s,
        "final_query_latency_ms_avg": (final_q_s / len(queries)) * 1000,
        "recall": recall_at_k(nbrs.astype(np.uint32), surviving_gt),
    }


def scenario_churn(idx, data, queries, gt, init_n, algo_name, cycles=10, churn_frac=0.05):
    """Sustained insert+delete+query cycles."""
    n = len(data)
    base_end = init_n + (n - init_n) // 2
    idx.initial_load(np.arange(base_end, dtype=np.uint64),
                     np.ascontiguousarray(data[:base_end]))
    chunk = max(1, int(base_end * churn_frac))
    cur_lo, del_lo = base_end, 0
    insert_lat, delete_lat, query_lat = [], [], []
    t = time.perf_counter()
    for c in range(cycles):
        if cur_lo + chunk > n:
            break
        td = time.perf_counter()
        idx.delete(np.arange(del_lo, del_lo + chunk, dtype=np.uint64))
        delete_lat.append((time.perf_counter() - td) / chunk)
        del_lo += chunk
        ti = time.perf_counter()
        idx.add(np.arange(cur_lo, cur_lo + chunk, dtype=np.uint64),
                np.ascontiguousarray(data[cur_lo:cur_lo + chunk]))
        insert_lat.append((time.perf_counter() - ti) / chunk)
        cur_lo += chunk
        maintain(idx, algo_name, vector_budget=chunk // 2)
        tq = time.perf_counter()
        idx.search(np.ascontiguousarray(queries), 10)
        query_lat.append((time.perf_counter() - tq) / len(queries))

    surviving = data[del_lo:cur_lo]
    surviving_gt = compute_gt(surviving, queries, 10)
    surviving_gt = (surviving_gt + del_lo).astype(np.uint32)
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    total = time.perf_counter() - t
    return {
        "total_s": total,
        "n_cycle_inserts": chunk * cycles,
        "n_cycle_deletes": chunk * cycles,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": float(np.mean(delete_lat)) * 1e6,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000,
        "query_latency_ms_p95": percentile(query_lat, 95),
        "recall": recall_at_k(nbrs.astype(np.uint32), surviving_gt),
    }


SCENARIOS = {
    "streaming(b=2500)":   (scenario_streaming, dict(batch=2500)),
    "burst(b=10000)":      (scenario_burst, dict(burst=10000)),
    "drift(5cycles)":      (scenario_drift, dict(drift_chunks=5)),
    "bulk_delete(30%)":    (scenario_bulk_delete, dict(delete_frac=0.3)),
    "churn(10cyc,5%)":     (scenario_churn, dict(cycles=10, churn_frac=0.05)),
}
ALGOS = ["gamma", "faiss", "ivf"]


def main():
    print("Loading SIFT 1M …")
    data_full, queries = load_sift_1m(n_queries=1000)
    print(f"  data {data_full.shape}  queries {queries.shape}")

    gt_full_path = "/tmp/sift1m_gt10_q1000.npy"
    if os.path.exists(gt_full_path):
        gt_full = np.load(gt_full_path)
        print(f"  cached full-1M GT: {gt_full.shape}")
    else:
        print("Computing brute-force GT for full 1M …")
        gt_full = compute_gt(data_full, queries, k=10)
        np.save(gt_full_path, gt_full)
        print(f"  GT computed and cached: {gt_full.shape}")

    # GT for 200K slice (used by streaming/burst/drift/churn — keeps bench tractable)
    data_200k = data_full[:200000]
    gt_200k_path = "/tmp/sift1m_slice200k_gt10_q1000.npy"
    if os.path.exists(gt_200k_path):
        gt_200k = np.load(gt_200k_path)
        print(f"  cached 200K-slice GT: {gt_200k.shape}")
    else:
        print("Computing brute-force GT for 200K slice …")
        gt_200k = compute_gt(data_200k, queries, k=10)
        np.save(gt_200k_path, gt_200k)
        print(f"  GT computed: {gt_200k.shape}")

    # Plan: bulk_delete on full 1M (GammaFresh's showcase). Other scenarios
    # on 200K slice so the sweep finishes in reasonable wall time.
    runs = [
        ("streaming(b=2500)/200K",   data_200k, gt_200k, 20000, scenario_streaming, dict(batch=2500, qstride=20000)),
        ("burst(b=10000)/200K",      data_200k, gt_200k, 20000, scenario_burst,     dict(burst=10000)),
        ("drift(5cycles)/200K",      data_200k, gt_200k, 20000, scenario_drift,     dict(drift_chunks=5)),
        ("bulk_delete(30%)/1M",      data_full, gt_full, 100000, scenario_bulk_delete, dict(delete_frac=0.3)),
        ("churn(10cyc,5%)/200K",     data_200k, gt_200k, 20000, scenario_churn,     dict(cycles=10, churn_frac=0.05)),
    ]

    rows = []
    for sc_name, ds, ds_gt, init_n, sc_fn, kwargs in runs:
        print(f"\n========== {sc_name} ==========")
        for algo in ALGOS:
            idx = build(algo, ds.shape[1])
            try:
                m = sc_fn(idx, ds, queries, ds_gt, init_n, algo, **kwargs)
            except Exception as e:
                print(f"  {algo:8s}  ERROR: {type(e).__name__}: {e}")
                continue
            row = {"scenario": sc_name, "algo": algo, **m}
            rows.append(row)
            metric_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in m.items())
            print(f"  {algo:8s}  {metric_str}")

    out = "/home/rprp/CANDOR-Bench/results/scenarios/sift1m_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n✓ Saved: {out}")


if __name__ == "__main__":
    main()
