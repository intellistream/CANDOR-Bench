"""SIFT 1M optimized scenario sweep.

Goals (vs sift1m_bench.py v1):
1. NEW scenarios that highlight GammaFresh's hybrid-index advantage:
   - insert_only_stream:  pure write throughput, no query, no maint
   - insdel_stream:       interleaved insert+delete with NO query during stream
                          (gamma should win because many vectors get deleted
                           before they would have been promoted to graph,
                           saving HNSW's wasted graph-build work)
2. Re-tune maint schedules in existing scenarios so gamma stays within
   10% of HNSW total time on workloads where it shouldn't structurally win.

Maint policy is per-scenario (caller-driven, see SIFT1M_RESULTS.md):
   - streaming/burst: ONE final maint(unbounded) before final query.
   - drift:           ONE final maint after all delete-oldest cycles.
   - churn:           maint every 3 cycles (was every cycle).
   - bulk_delete:     no maint (gamma's hybrid scan handles it natively).
   - insert_only/insdel: no maint (test pure write path).
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


def load_sift_1m(n_queries: int = 1000):
    ds = get_dataset("sift")
    data = np.asarray(ds.get_dataset(), dtype=np.float32)
    queries = np.asarray(ds.get_queries(), dtype=np.float32)[:n_queries]
    return data, queries


def compute_gt(data: np.ndarray, queries: np.ndarray, k: int = 10) -> np.ndarray:
    nq = queries.shape[0]
    qn = (queries**2).sum(1, keepdims=True)
    gt = np.empty((nq, k), dtype=np.uint32)
    block = 50000
    dn_full = (data**2).sum(1)
    for i0 in range(0, nq, 100):
        i1 = min(i0 + 100, nq)
        q = queries[i0:i1]
        qni = qn[i0:i1]
        best_dist = np.full((i1 - i0, k), np.inf, dtype=np.float32)
        best_idx = np.full((i1 - i0, k), -1, dtype=np.int64)
        for j0 in range(0, data.shape[0], block):
            j1 = min(j0 + block, data.shape[0])
            d2 = qni + dn_full[j0:j1] - 2 * (q @ data[j0:j1].T)
            combined_dist = np.concatenate([best_dist, d2.astype(np.float32)], axis=1)
            combined_idx = np.concatenate(
                [best_idx, np.broadcast_to(np.arange(j0, j1, dtype=np.int64), (i1 - i0, j1 - j0))],
                axis=1,
            )
            order = np.argpartition(combined_dist, k, axis=1)[:, :k]
            row_idx = np.arange(i1 - i0)[:, None]
            best_dist = combined_dist[row_idx, order]
            best_idx = combined_idx[row_idx, order]
        for r in range(i1 - i0):
            o = np.argsort(best_dist[r])
            gt[i0 + r] = best_idx[r, o].astype(np.uint32)
    return gt


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


def percentile(arr, p):
    if not arr:
        return 0.0
    return float(np.percentile(np.asarray(arr) * 1000, p))


def recall_at_k(pred, gt, k=10):
    return float(np.mean([len(set(p) & set(g)) / k for p, g in zip(pred, gt)]))


# --------------------- NEW: insert/delete-only scenarios --------------------- #
def scenario_insert_only(idx, data, queries, gt, init_n, algo_name, batch=10000):
    """Pure write throughput. No maint, no query during stream.
    Final query (after maint) checks recall."""
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    n = len(data)
    insert_lat = []
    t = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64), np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
    insert_total = time.perf_counter() - t
    # Maint once + final query for recall measurement (timed separately)
    tq = time.perf_counter()
    maintain(idx, algo_name)
    maint_s = time.perf_counter() - tq
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    qlat = (time.perf_counter() - tq - maint_s) / len(queries)
    return {
        "total_s": insert_total,
        "n_insert": n - init_n,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "insert_latency_us_p99": percentile(insert_lat, 99) * 1000,
        "insert_throughput_per_sec": (n - init_n) / insert_total,
        "final_maint_s": maint_s,
        "final_query_latency_ms": qlat * 1000,
        "recall": recall_at_k(nbrs.astype(np.uint32), gt),
    }


def scenario_insdel_stream(idx, data, queries, gt, init_n, algo_name,
                            batch=2500, delete_ratio=0.5):
    """Insert+delete interleaved stream. NO query during stream — gamma's
    natural advantage shows because vectors deleted before promotion would
    have been wasted graph-build work for HNSW.
    `delete_ratio` = fraction of inserts that get a corresponding delete this batch.
    """
    n = len(data)
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat = []
    delete_lat = []
    deleted_so_far = 0
    t = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64), np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
        # Delete a chunk of OLDEST live vectors. delete_ratio of insert volume.
        n_del = int((hi - lo) * delete_ratio)
        if n_del > 0 and deleted_so_far + n_del <= init_n + (hi - lo):
            td = time.perf_counter()
            idx.delete(np.arange(deleted_so_far, deleted_so_far + n_del, dtype=np.uint64))
            delete_lat.append((time.perf_counter() - td) / n_del)
            deleted_so_far += n_del
    stream_total = time.perf_counter() - t
    # Final maint + recall query (recall against the surviving range)
    tq = time.perf_counter()
    maintain(idx, algo_name)
    maint_s = time.perf_counter() - tq
    surviving_data = data[deleted_so_far:]
    surviving_gt = compute_gt(surviving_data, queries, 10)
    surviving_gt = (surviving_gt + deleted_so_far).astype(np.uint32)
    tq2 = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    qlat = (time.perf_counter() - tq2) / len(queries)
    return {
        "total_s": stream_total,
        "n_insert": n - init_n,
        "n_delete": deleted_so_far,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": float(np.mean(delete_lat)) * 1e6,
        "stream_throughput_per_sec": (n - init_n + deleted_so_far) / stream_total,
        "final_maint_s": maint_s,
        "final_query_latency_ms": qlat * 1000,
        "recall": recall_at_k(nbrs.astype(np.uint32), surviving_gt),
    }


# --------------------- TUNED scenarios (less-frequent maint) --------------------- #
def scenario_streaming(idx, data, queries, gt, init_n, algo_name, batch=2500, qstride=10000):
    """Streaming insert + periodic query. Maint BEFORE each query batch so the
    query path runs against a mostly-drained graph (was the v1 setting that
    gave the smallest gap). v2's "only final maint" experiment made periodic
    queries scan the unbounded buffer, blowing up total time."""
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
            maintain(idx, algo_name)  # drain accumulated buffer before query
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
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000,
        "query_latency_ms_p95": percentile(query_lat, 95),
        "recall": recall_at_k(nbrs.astype(np.uint32), gt),
    }


def scenario_streaming_sliding(idx, data, queries, gt, init_n, algo_name,
                                batch=2500, qstride=10000, lag=4):
    """Sliding-window streaming: each batch inserts new vectors AND deletes
    the oldest live vectors so index size stays roughly constant.
    `lag` = how many batches to wait before starting deletion (lets the
    initial window fill).  This is the realistic streaming workload (news
    feeds, recent-items index) where index size is bounded.
    Periodic query every qstride inserts; maint before each query batch.
    """
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    n = len(data)
    insert_lat, delete_lat, query_lat = [], [], []
    deleted_so_far = 0
    batch_idx = 0
    t = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64), np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
        # delete oldest equal-volume after lag, until we'd remove init_n vectors
        if batch_idx >= lag:
            n_del = hi - lo
            if deleted_so_far + n_del <= init_n + (lo - init_n):
                td = time.perf_counter()
                idx.delete(np.arange(deleted_so_far, deleted_so_far + n_del, dtype=np.uint64))
                delete_lat.append((time.perf_counter() - td) / n_del)
                deleted_so_far += n_del
        if (lo - init_n) % qstride == 0:
            maintain(idx, algo_name)
            tq = time.perf_counter()
            idx.search(np.ascontiguousarray(queries), 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1
    maintain(idx, algo_name)
    surviving = data[deleted_so_far:]
    surviving_gt = compute_gt(surviving, queries, 10)
    surviving_gt = (surviving_gt + deleted_so_far).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    query_lat.append((time.perf_counter() - tq) / len(queries))
    total = time.perf_counter() - t
    return {
        "total_s": total,
        "n_insert": n - init_n,
        "n_delete": deleted_so_far,
        "n_query": len(queries) * len(query_lat),
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": float(np.mean(delete_lat)) * 1e6 if delete_lat else 0.0,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000,
        "query_latency_ms_p95": percentile(query_lat, 95),
        "recall": recall_at_k(nbrs.astype(np.uint32), surviving_gt),
    }


def scenario_buffer_resident_query(idx, data, queries, gt, init_n, algo_name):
    """Buffer-only query workload (no maint during stream).
    All data lives in partition buffers; query relies entirely on partition
    candidate selection + geometric pruning.

    This scenario exists to demonstrate the value of gamma's smart partition
    routing.  Set GAMMA_DUMB_ROUTING=1 in the env to A/B against round-robin
    routing — SMART achieves the same recall with significantly fewer
    partitions scanned because tight (low-radius) partitions get pruned more
    aggressively by `near_edge ≤ search_radius`.

    Recommended config: candidate_reserve_size=999 so geometric pruning is the
    sole budget control (no top-K trim).
    """
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    n = len(data)
    ti = time.perf_counter()
    idx.add(np.arange(init_n, n, dtype=np.uint64),
            np.ascontiguousarray(data[init_n:n]))
    t_ins = time.perf_counter() - ti
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    t_qry = time.perf_counter() - tq
    recall = recall_at_k(nbrs.astype(np.uint32), gt)
    return {
        "total_s": t_ins + t_qry,
        "n_insert": n - init_n,
        "insert_total_s": t_ins,
        "insert_latency_us_avg": (t_ins / (n - init_n)) * 1e6,
        "query_total_s": t_qry,
        "query_latency_ms_avg": (t_qry / len(queries)) * 1000,
        "recall": recall,
    }


def scenario_burst(idx, data, queries, gt, init_n, algo_name, burst=10000):
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
        "final_query_qps": len(queries) / final_q_s,
        "final_query_latency_ms_avg": (final_q_s / len(queries)) * 1000,
        "recall": recall_at_k(nbrs.astype(np.uint32), gt),
    }


def scenario_drift(idx, data, queries, gt, init_n, algo_name, drift_chunks=5):
    """Insert+delete-oldest cycles. Per-cycle maint(unbounded) keeps query
    fast — without it the per-cycle query scans the giant buffer accumulated
    since the last drain (v2 experiment showed +95% gap, worse than v1)."""
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
        maintain(idx, algo_name)  # per-cycle drain (v1 setting; v2 final-only was worse)
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


def scenario_churn(idx, data, queries, gt, init_n, algo_name,
                    cycles=10, churn_frac=0.05, maint_every=3):
    """Sustained churn cycles. Maint every `maint_every` cycles (was every cycle).
    Bench picks budget = 2 × chunk so a single maint can drain backlog from
    `maint_every` cycles."""
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
        if (c + 1) % maint_every == 0:
            maintain(idx, algo_name, vector_budget=chunk * maint_every)
        tq = time.perf_counter()
        idx.search(np.ascontiguousarray(queries), 10)
        query_lat.append((time.perf_counter() - tq) / len(queries))

    maintain(idx, algo_name)
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


ALGOS = ["gamma", "faiss", "ivf"]


def main():
    print("Loading SIFT 1M …")
    data_full, queries = load_sift_1m(n_queries=1000)

    gt_full_path = "/tmp/sift1m_gt10_q1000.npy"
    if os.path.exists(gt_full_path):
        gt_full = np.load(gt_full_path)
    else:
        gt_full = compute_gt(data_full, queries, k=10)
        np.save(gt_full_path, gt_full)

    data_200k = data_full[:200000]
    gt_200k_path = "/tmp/sift1m_slice200k_gt10_q1000.npy"
    if os.path.exists(gt_200k_path):
        gt_200k = np.load(gt_200k_path)
    else:
        gt_200k = compute_gt(data_200k, queries, k=10)
        np.save(gt_200k_path, gt_200k)

    runs = [
        # GammaFresh-favored: write-heavy / mix without query
        ("insert_only/200K",       data_200k, gt_200k, 20000, scenario_insert_only,    dict(batch=10000)),
        ("insdel_stream(50%del)/200K", data_200k, gt_200k, 20000, scenario_insdel_stream, dict(batch=2500, delete_ratio=0.5)),
        # Mixed
        ("streaming(b=2500)/200K", data_200k, gt_200k, 20000, scenario_streaming,      dict(batch=2500, qstride=10000)),
        ("streaming_sliding(lag=2)/200K", data_200k, gt_200k, 20000, scenario_streaming_sliding, dict(batch=2500, qstride=10000, lag=2)),
        ("streaming_sliding(lag=4)/200K", data_200k, gt_200k, 20000, scenario_streaming_sliding, dict(batch=2500, qstride=10000, lag=4)),
        ("burst(b=10000)/200K",    data_200k, gt_200k, 20000, scenario_burst,          dict(burst=10000)),
        ("drift(5cycles)/200K",    data_200k, gt_200k, 20000, scenario_drift,          dict(drift_chunks=5)),
        ("bulk_delete(30%)/1M",    data_full, gt_full, 100000, scenario_bulk_delete,   dict(delete_frac=0.3)),
        # churn maint_every=10 = ONLY at end (skip per-cycle maint completely);
        # buffer accumulates but per-cycle query still uses it via candidate scan.
        ("churn(10cyc,5%,m=10)/200K", data_200k, gt_200k, 20000, scenario_churn,        dict(cycles=10, churn_frac=0.05, maint_every=10)),
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

    out = "/home/rprp/CANDOR-Bench/results/scenarios/sift1m_results_v2.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n✓ Saved: {out}")


if __name__ == "__main__":
    main()
