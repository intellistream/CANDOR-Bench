"""e13 — Fair comparison: gamma+X vs X-direct for each backend X."""
import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import load_dataset, compute_gt, recall_at_k, percentile_ms
from _shared.gamma_py import GammaPyHybrid, HnswlibBackend, FaissHnswBackend
import numpy as np


def run_gamma_streaming_sliding(backend_factory, data, queries, init_n, batch, qstride, lag):
    n, d = data.shape
    backend = backend_factory(d, n)
    g = GammaPyHybrid(backend, d)
    g.initial_load(np.arange(init_n, dtype=np.int64),
                    np.ascontiguousarray(data[:init_n]))
    deleted, batch_idx, query_lat = 0, 0, []
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        g.add(np.arange(lo, hi, dtype=np.int64),
              np.ascontiguousarray(data[lo:hi]))
        if batch_idx >= lag:
            n_del = hi - lo
            if deleted + n_del <= init_n + (lo - init_n):
                g.delete(np.arange(deleted, deleted + n_del))
                deleted += n_del
        if (lo - init_n) % qstride == 0:
            g.maintain()
            tq = time.perf_counter()
            g.search(queries, 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1
    g.maintain()
    sgt = (compute_gt(data[deleted:], queries, 10) + deleted).astype(np.uint32)
    nbrs, _ = g.search(queries, 10)
    return {
        "total_s": time.perf_counter() - t0, "n_delete": deleted,
        "query_latency_ms_p95": percentile_ms(query_lat, 95),
        "recall": recall_at_k(nbrs.astype(np.uint32), sgt),
    }


def run_xdirect_streaming_sliding(backend_factory, data, queries, init_n, batch, qstride, lag):
    n, d = data.shape
    b = backend_factory(d, n)
    b.add(np.ascontiguousarray(data[:init_n]), np.arange(init_n, dtype=np.int64))
    deleted, batch_idx, query_lat = 0, 0, []
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        b.add(np.ascontiguousarray(data[lo:hi]), np.arange(lo, hi, dtype=np.int64))
        if batch_idx >= lag:
            n_del = hi - lo
            if deleted + n_del <= init_n + (lo - init_n):
                for did in range(deleted, deleted + n_del):
                    b.mark_deleted(did)
                deleted += n_del
        if (lo - init_n) % qstride == 0:
            tq = time.perf_counter()
            b.search(queries, 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1
    sgt = (compute_gt(data[deleted:], queries, 10) + deleted).astype(np.uint32)
    nbrs, _ = b.search(queries, 10)
    return {
        "total_s": time.perf_counter() - t0, "n_delete": deleted,
        "query_latency_ms_p95": percentile_ms(query_lat, 95),
        "recall": recall_at_k(nbrs.astype(np.uint32), sgt),
    }


def run_gamma_bulk_delete(backend_factory, data, queries, init_n, n_del):
    n, d = data.shape
    backend = backend_factory(d, n)
    g = GammaPyHybrid(backend, d)
    g.initial_load(np.arange(init_n, dtype=np.int64),
                    np.ascontiguousarray(data[:init_n]))
    t0 = time.perf_counter()
    g.add(np.arange(init_n, n, dtype=np.int64),
          np.ascontiguousarray(data[init_n:n]))
    g.delete(np.arange(0, n_del))
    g.maintain()
    op = time.perf_counter() - t0
    sgt = (compute_gt(data[n_del:], queries, 10) + n_del).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = g.search(queries, 10)
    qry = (time.perf_counter() - tq) / len(queries) * 1000
    return {"op_s": op, "qry_ms": qry,
            "recall": recall_at_k(nbrs.astype(np.uint32), sgt)}


def run_xdirect_bulk_delete(backend_factory, data, queries, init_n, n_del):
    n, d = data.shape
    b = backend_factory(d, n)
    t0 = time.perf_counter()
    b.add(np.ascontiguousarray(data), np.arange(n, dtype=np.int64))
    for i in range(n_del):
        b.mark_deleted(i)
    op = time.perf_counter() - t0
    sgt = (compute_gt(data[n_del:], queries, 10) + n_del).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = b.search(queries, 10)
    qry = (time.perf_counter() - tq) / len(queries) * 1000
    return {"op_s": op, "qry_ms": qry,
            "recall": recall_at_k(nbrs.astype(np.uint32), sgt)}


def main():
    data, queries = load_dataset("sift", n_queries=500, slice_n=200_000)
    init_n, batch, qstride = 20_000, 2_500, 10_000
    n_del = int(data.shape[0] * 0.3)

    backends = [
        ("hnswlib", lambda d, n: HnswlibBackend(d, max_elements=n)),
        ("faiss_hnsw", lambda d, n: FaissHnswBackend(d, max_elements=n)),
    ]
    rows = []

    # bulk_delete
    print(f"\n========== bulk_delete(30%) ==========", flush=True)
    for be_name, be_factory in backends:
        gr = run_gamma_bulk_delete(be_factory, data, queries, init_n, n_del)
        xr = run_xdirect_bulk_delete(be_factory, data, queries, init_n, n_del)
        delta_op = (xr["op_s"] - gr["op_s"]) / xr["op_s"] * 100
        print(f"  {be_name}:")
        print(f"    gamma+{be_name:12s}  op={gr['op_s']:6.1f}s  qry={gr['qry_ms']:.2f}ms  recall={gr['recall']:.4f}", flush=True)
        print(f"    {be_name+' direct':18s}  op={xr['op_s']:6.1f}s  qry={xr['qry_ms']:.2f}ms  recall={xr['recall']:.4f}  → gamma {delta_op:+.1f}%", flush=True)
        rows.append({"workload": "bulk_delete", "backend": be_name,
                     "gamma_op": gr["op_s"], "gamma_qry": gr["qry_ms"], "gamma_recall": gr["recall"],
                     "direct_op": xr["op_s"], "direct_qry": xr["qry_ms"], "direct_recall": xr["recall"]})

    # streaming_sliding lag=2 and lag=4
    for lag in [2, 4]:
        print(f"\n========== streaming_sliding(lag={lag}) ==========", flush=True)
        for be_name, be_factory in backends:
            gr = run_gamma_streaming_sliding(be_factory, data, queries, init_n, batch, qstride, lag)
            xr = run_xdirect_streaming_sliding(be_factory, data, queries, init_n, batch, qstride, lag)
            delta_t = (xr["total_s"] - gr["total_s"]) / xr["total_s"] * 100
            print(f"  {be_name}:")
            print(f"    gamma+{be_name:12s}  total={gr['total_s']:6.1f}s  qry_p95={gr['query_latency_ms_p95']:.3f}ms  recall={gr['recall']:.4f}", flush=True)
            print(f"    {be_name+' direct':18s}  total={xr['total_s']:6.1f}s  qry_p95={xr['query_latency_ms_p95']:.3f}ms  recall={xr['recall']:.4f}  → gamma {delta_t:+.1f}%", flush=True)
            rows.append({"workload": f"streaming_sliding_lag{lag}", "backend": be_name,
                         "gamma_total": gr["total_s"], "gamma_qry_p95": gr["query_latency_ms_p95"], "gamma_recall": gr["recall"],
                         "direct_total": xr["total_s"], "direct_qry_p95": xr["query_latency_ms_p95"], "direct_recall": xr["recall"]})

    out = os.path.join(os.path.dirname(__file__), "output.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
