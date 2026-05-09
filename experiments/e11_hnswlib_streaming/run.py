"""e11 — hnswlib in streaming_sliding workload.

Tests whether hnswlib (which beat gamma on offline-rebuild bulk_delete in e10)
also suffers tombstone-pollution under streaming with no rebuild window.
"""
import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import build, load_dataset, compute_gt, recall_at_k, percentile_ms, maintain
import numpy as np


def run_gamma_streaming(data, queries, init_n, batch, qstride, lag):
    n = data.shape[0]
    idx = build("gamma", data.shape[1])
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    insert_lat, query_lat = [], []
    deleted, batch_idx = 0, 0
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        ti = time.perf_counter()
        idx.add(np.arange(lo, hi, dtype=np.uint64), np.ascontiguousarray(data[lo:hi]))
        insert_lat.append((time.perf_counter() - ti) / (hi - lo))
        if batch_idx >= lag:
            n_del = hi - lo
            if deleted + n_del <= init_n + (lo - init_n):
                idx.delete(np.arange(deleted, deleted + n_del, dtype=np.uint64))
                deleted += n_del
        if (lo - init_n) % qstride == 0:
            maintain(idx, "gamma")
            tq = time.perf_counter()
            idx.search(np.ascontiguousarray(queries), 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1
    maintain(idx, "gamma")
    sgt = (compute_gt(data[deleted:], queries, 10) + deleted).astype(np.uint32)
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    return {
        "algo": "gamma", "total_s": time.perf_counter() - t0,
        "n_delete": deleted,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000 if query_lat else 0.0,
        "query_latency_ms_p95": percentile_ms(query_lat, 95),
        "recall": recall_at_k(nbrs.astype(np.uint32), sgt),
    }


def run_hnswlib_streaming(data, queries, init_n, batch, qstride, lag,
                           rebuild_every=0):
    """hnswlib supports mark_deleted (tombstone). With rebuild_every>0, do
    a full rebuild every N batches (matches gamma's maint frequency)."""
    import hnswlib
    n, d = data.shape
    idx = hnswlib.Index(space='l2', dim=d)
    idx.init_index(max_elements=n, ef_construction=120, M=32)
    idx.set_ef(80)
    idx.add_items(data[:init_n], np.arange(init_n))
    deleted, batch_idx = 0, 0
    query_lat = []
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        idx.add_items(data[lo:hi], np.arange(lo, hi))
        if batch_idx >= lag:
            n_del = hi - lo
            if deleted + n_del <= init_n + (lo - init_n):
                for did in range(deleted, deleted + n_del):
                    idx.mark_deleted(did)
                deleted += n_del
        # periodic rebuild
        if rebuild_every > 0 and batch_idx > 0 and batch_idx % rebuild_every == 0:
            # Rebuild from surviving alive vectors
            alive_ids = np.arange(deleted, hi)
            alive_data = data[deleted:hi]
            new_idx = hnswlib.Index(space='l2', dim=d)
            new_idx.init_index(max_elements=n, ef_construction=120, M=32)
            new_idx.set_ef(80)
            new_idx.add_items(alive_data, alive_ids)
            idx = new_idx
        if (lo - init_n) % qstride == 0:
            tq = time.perf_counter()
            idx.knn_query(queries, k=10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1
    sgt = (compute_gt(data[deleted:], queries, 10) + deleted).astype(np.uint32)
    labels, _ = idx.knn_query(queries, k=10)
    label = "hnswlib" + (f"+rebuild_every_{rebuild_every}" if rebuild_every else "")
    return {
        "algo": label, "total_s": time.perf_counter() - t0,
        "n_delete": deleted,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000 if query_lat else 0.0,
        "query_latency_ms_p95": percentile_ms(query_lat, 95),
        "recall": recall_at_k(labels.astype(np.uint32), sgt.astype(np.uint32)),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["200K", "1M"], default="200K")
    args = p.parse_args()
    if args.scale == "200K":
        slice_n, init_n, batch, qstride, n_queries = 200_000, 20_000, 2_500, 10_000, 500
    else:
        slice_n, init_n, batch, qstride, n_queries = 1_000_000, 100_000, 10_000, 50_000, 1000
    print(f"Loading SIFT {args.scale} …", flush=True)
    data, queries = load_dataset("sift", n_queries=n_queries, slice_n=slice_n)
    lag = 2

    print(f"\n========== streaming_sliding(lag={lag})/{args.scale} ==========", flush=True)
    rows = []

    print("[gamma] running...", flush=True)
    r = run_gamma_streaming(data, queries, init_n, batch, qstride, lag)
    rows.append(r); print(f"  {r['algo']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)

    print("[hnswlib no rebuild] running...", flush=True)
    r = run_hnswlib_streaming(data, queries, init_n, batch, qstride, lag, rebuild_every=0)
    rows.append(r); print(f"  {r['algo']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)

    print("[hnswlib + rebuild every 4] running...", flush=True)
    r = run_hnswlib_streaming(data, queries, init_n, batch, qstride, lag, rebuild_every=4)
    rows.append(r); print(f"  {r['algo']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)

    out = os.path.join(os.path.dirname(__file__), f"output_{args.scale}.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
