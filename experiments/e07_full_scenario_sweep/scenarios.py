"""Scenario implementations shared by e07 (and reusable elsewhere).
Each scenario is a function `(idx, name, data, queries, init_n, **kwargs) -> dict`.
"""
import time
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from _shared import compute_gt, recall_at_k, percentile_ms, maintain


def scenario_insert_only(idx, name, data, queries, gt, init_n, batch=10000):
    n = data.shape[0]
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat = []
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64),
                np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
    stream_s = time.perf_counter() - t0
    tm = time.perf_counter(); maintain(idx, name); maint_s = time.perf_counter() - tm
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    qry_s = time.perf_counter() - tq
    return {
        "scenario": "insert_only", "stream_s": stream_s, "maint_s": maint_s,
        "wall_s": stream_s + maint_s + qry_s,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "insert_throughput_per_sec": (n - init_n) / stream_s,
        "final_query_latency_ms": (qry_s / len(queries)) * 1000,
        "recall": recall_at_k(nbrs.astype(np.uint32), gt),
    }


def scenario_insdel_stream(idx, name, data, queries, gt, init_n, batch=2500, delete_ratio=0.5):
    n = data.shape[0]
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat, delete_lat = [], []
    deleted = 0
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64),
                np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
        n_del = int((hi - lo) * delete_ratio)
        if n_del > 0 and deleted + n_del <= init_n + (hi - lo):
            td = time.perf_counter()
            idx.delete(np.arange(deleted, deleted + n_del, dtype=np.uint64))
            delete_lat.append((time.perf_counter() - td) / n_del)
            deleted += n_del
    stream_s = time.perf_counter() - t0
    tm = time.perf_counter(); maintain(idx, name); maint_s = time.perf_counter() - tm
    sgt = (compute_gt(data[deleted:], queries, 10) + deleted).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    qry_s = time.perf_counter() - tq
    return {
        "scenario": "insdel_stream", "stream_s": stream_s, "maint_s": maint_s,
        "wall_s": stream_s + maint_s + qry_s,
        "n_delete": deleted,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": (float(np.mean(delete_lat)) * 1e6) if delete_lat else 0.0,
        "stream_throughput_per_sec": (n - init_n + deleted) / stream_s,
        "final_query_latency_ms": (qry_s / len(queries)) * 1000,
        "recall": recall_at_k(nbrs.astype(np.uint32), sgt),
    }


def scenario_streaming(idx, name, data, queries, gt, init_n, batch=2500, qstride=10000):
    n = data.shape[0]
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat, query_lat = [], []
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64),
                np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
        if (lo - init_n) % qstride == 0:
            maintain(idx, name)
            tq = time.perf_counter()
            idx.search(np.ascontiguousarray(queries), 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
    maintain(idx, name)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    final_qlat = (time.perf_counter() - tq) / len(queries)
    wall_s = time.perf_counter() - t0
    return {
        "scenario": "streaming", "wall_s": wall_s,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000 if query_lat else 0.0,
        "query_latency_ms_p95": percentile_ms(query_lat, 95),
        "recall": recall_at_k(nbrs.astype(np.uint32), gt),
    }


def scenario_streaming_sliding(idx, name, data, queries, gt, init_n,
                                batch=2500, qstride=10000, lag=2):
    n = data.shape[0]
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat, delete_lat, query_lat = [], [], []
    deleted, batch_idx = 0, 0
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64),
                np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
        if batch_idx >= lag:
            n_del = hi - lo
            if deleted + n_del <= init_n + (lo - init_n):
                td = time.perf_counter()
                idx.delete(np.arange(deleted, deleted + n_del, dtype=np.uint64))
                delete_lat.append((time.perf_counter() - td) / n_del)
                deleted += n_del
        if (lo - init_n) % qstride == 0:
            maintain(idx, name)
            tq = time.perf_counter()
            idx.search(np.ascontiguousarray(queries), 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1
    maintain(idx, name)
    sgt = (compute_gt(data[deleted:], queries, 10) + deleted).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    final_qlat = (time.perf_counter() - tq) / len(queries)
    wall_s = time.perf_counter() - t0
    return {
        "scenario": f"streaming_sliding(lag={lag})", "wall_s": wall_s,
        "n_delete": deleted,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": (float(np.mean(delete_lat)) * 1e6) if delete_lat else 0.0,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000 if query_lat else 0.0,
        "query_latency_ms_p95": percentile_ms(query_lat, 95),
        "recall": recall_at_k(nbrs.astype(np.uint32), sgt),
    }


def scenario_burst(idx, name, data, queries, gt, init_n, burst=10000):
    n = data.shape[0]
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat = []
    t0 = time.perf_counter()
    for lo in range(init_n, n, burst):
        hi = min(lo + burst, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64),
                np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
    maintain(idx, name)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    final_q_s = time.perf_counter() - tq
    wall_s = time.perf_counter() - t0
    return {
        "scenario": "burst", "wall_s": wall_s,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "final_query_latency_ms": (final_q_s / len(queries)) * 1000,
        "recall": recall_at_k(nbrs.astype(np.uint32), gt),
    }


def scenario_drift(idx, name, data, queries, gt, init_n, drift_chunks=5):
    n = data.shape[0]
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    chunk = (n - init_n) // drift_chunks
    delete_each = chunk // 2
    insert_lat, delete_lat, query_lat = [], [], []
    cur_lo, last_delete = init_n, 0
    t0 = time.perf_counter()
    for c in range(drift_chunks):
        chunk_end = min(cur_lo + chunk, n)
        for lo in range(cur_lo, chunk_end, 2500):
            hi = min(lo + 2500, chunk_end)
            ti = time.perf_counter()
            idx.add(np.arange(lo, hi, dtype=np.uint64),
                    np.ascontiguousarray(data[lo:hi]))
            insert_lat.append((time.perf_counter() - ti) / (hi - lo))
        del_start, del_end = last_delete, last_delete + delete_each
        td = time.perf_counter()
        idx.delete(np.arange(del_start, del_end, dtype=np.uint64))
        delete_lat.append((time.perf_counter() - td) / (del_end - del_start))
        last_delete = del_end
        maintain(idx, name)
        tq = time.perf_counter()
        idx.search(np.ascontiguousarray(queries), 10)
        query_lat.append((time.perf_counter() - tq) / len(queries))
        cur_lo = chunk_end
    sgt = (compute_gt(data[last_delete:], queries, 10) + last_delete).astype(np.uint32)
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    wall_s = time.perf_counter() - t0
    return {
        "scenario": "drift", "wall_s": wall_s,
        "n_delete": last_delete,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": float(np.mean(delete_lat)) * 1e6,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000,
        "query_latency_ms_p95": percentile_ms(query_lat, 95),
        "recall": recall_at_k(nbrs.astype(np.uint32), sgt),
    }


def scenario_bulk_delete(idx, name, data, queries, gt, init_n, delete_frac=0.3):
    n = data.shape[0]
    n_del = int(n * delete_frac)
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat = []
    t0 = time.perf_counter()
    ti = time.perf_counter()
    idx.add(np.arange(init_n, n, dtype=np.uint64),
            np.ascontiguousarray(data[init_n:n]))
    insert_lat.append((time.perf_counter() - ti) / (n - init_n))
    td = time.perf_counter()
    idx.delete(np.arange(0, n_del, dtype=np.uint64))
    del_s = time.perf_counter() - td
    op_total = time.perf_counter() - t0
    sgt = (compute_gt(data[n_del:], queries, 10) + n_del).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    final_q_s = time.perf_counter() - tq
    return {
        "scenario": "bulk_delete", "wall_s": op_total + final_q_s,
        "op_s": op_total,
        "n_delete": n_del,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": (del_s / n_del) * 1e6,
        "final_query_latency_ms": (final_q_s / len(queries)) * 1000,
        "recall": recall_at_k(nbrs.astype(np.uint32), sgt),
    }


def scenario_churn(idx, name, data, queries, gt, init_n, cycles=10,
                    churn_frac=0.05, maint_every=10):
    n = data.shape[0]
    base_end = init_n + (n - init_n) // 2
    idx.initial_load(np.arange(base_end, dtype=np.uint64),
                     np.ascontiguousarray(data[:base_end]))
    chunk = max(1, int(base_end * churn_frac))
    cur_lo, del_lo = base_end, 0
    insert_lat, delete_lat, query_lat = [], [], []
    t0 = time.perf_counter()
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
            maintain(idx, name)
        tq = time.perf_counter()
        idx.search(np.ascontiguousarray(queries), 10)
        query_lat.append((time.perf_counter() - tq) / len(queries))
    maintain(idx, name)
    sgt = (compute_gt(data[del_lo:cur_lo], queries, 10) + del_lo).astype(np.uint32)
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    wall_s = time.perf_counter() - t0
    return {
        "scenario": "churn", "wall_s": wall_s,
        "insert_latency_us_avg": float(np.mean(insert_lat)) * 1e6,
        "delete_latency_us_avg": float(np.mean(delete_lat)) * 1e6,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000,
        "query_latency_ms_p95": percentile_ms(query_lat, 95),
        "recall": recall_at_k(nbrs.astype(np.uint32), sgt),
    }
