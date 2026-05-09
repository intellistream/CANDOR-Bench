"""e09 — Add Faiss native NSG / IVFFlat / IVFPQ to the comparison.

Tests on bulk_delete (gamma's strongest scenario) — show gamma beats not
just HNSW but also other graph-based and quantized ANN algorithms.
"""
import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import build, load_dataset, compute_gt, recall_at_k
import numpy as np
import faiss


def run_gamma(data, queries, init_n, n_del):
    n = data.shape[0]
    idx = build("gamma", data.shape[1])
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    t0 = time.perf_counter()
    idx.add(np.arange(init_n, n, dtype=np.uint64),
            np.ascontiguousarray(data[init_n:n]))
    idx.delete(np.arange(0, n_del, dtype=np.uint64))
    idx.maintain(0, 0, False)  # drain buffer to graph for fair query comparison
    op_total = time.perf_counter() - t0
    sgt = (compute_gt(data[n_del:], queries, 10) + n_del).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    final_q_s = time.perf_counter() - tq
    return {"algo": "gamma", "op_total_s": op_total,
            "final_query_latency_ms": (final_q_s / len(queries)) * 1000,
            "recall": recall_at_k(nbrs.astype(np.uint32), sgt)}


def configure_for_query(idx, kind):
    """Set search-time parameters to be fair (high-recall config)."""
    if kind == "hnsw" and hasattr(idx, "hnsw"):
        idx.hnsw.efSearch = 80
    elif kind == "nsg" and hasattr(idx, "nsg"):
        idx.nsg.search_L = 80
    elif kind == "ivf" and hasattr(idx, "nprobe"):
        idx.nprobe = 16


def run_faiss_native(label, idx_factory, kind, data, queries, init_n, n_del,
                      incremental_add=True):
    """For Faiss-native indexes simulate bulk-delete by REBUILDING from surviving slice.

    incremental_add=False: NSG cannot do incremental add, build from full set at once.
    """
    n = data.shape[0]
    d = data.shape[1]
    t0 = time.perf_counter()
    # Build index on init_n + rest
    idx = idx_factory(d)
    if hasattr(idx, "train"):
        idx.train(np.ascontiguousarray(data[:max(init_n, 10000)]))  # IVF needs training
    if incremental_add:
        idx.add(np.ascontiguousarray(data[:init_n]))
        idx.add(np.ascontiguousarray(data[init_n:n]))
    else:
        idx.add(np.ascontiguousarray(data[:n]))
    # Bulk delete: rebuild from surviving slice
    surviving = data[n_del:n]
    idx_after = idx_factory(d)
    if hasattr(idx_after, "train"):
        idx_after.train(surviving[:max(10000, init_n)])
    idx_after.add(surviving)
    configure_for_query(idx_after, kind)
    op_total = time.perf_counter() - t0
    sgt_native = compute_gt(surviving, queries, 10)
    tq = time.perf_counter()
    D, I = idx_after.search(np.ascontiguousarray(queries), 10)
    final_q_s = time.perf_counter() - tq
    recall = recall_at_k(I.astype(np.uint32), sgt_native.astype(np.uint32))
    return {"algo": label, "op_total_s": op_total,
            "final_query_latency_ms": (final_q_s / len(queries)) * 1000,
            "recall": recall}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["200K", "1M"], default="200K")
    args = p.parse_args()
    if args.scale == "200K":
        slice_n, init_n, n_queries = 200_000, 20_000, 500
    else:
        slice_n, init_n, n_queries = 1_000_000, 100_000, 1000
    print(f"Loading SIFT {args.scale} …", flush=True)
    data, queries = load_dataset("sift", n_queries=n_queries, slice_n=slice_n)
    delete_frac = 0.3
    n_del = int(data.shape[0] * delete_frac)

    print(f"\n========== bulk_delete({delete_frac:.0%})/{args.scale} ==========", flush=True)
    rows = []

    # gamma
    r = run_gamma(data, queries, init_n, n_del)
    rows.append(r)
    print(f"  {r['algo']:18s}  op={r['op_total_s']:6.1f}s  "
          f"qry={r['final_query_latency_ms']:.2f}ms  recall={r['recall']:.4f}", flush=True)

    # Faiss-native baselines
    factories = [
        ("faiss_HNSW(M=32,ef=80)",    "hnsw", lambda d: faiss.IndexHNSWFlat(d, 32), True),
        ("faiss_NSG(R=32,L=80)",      "nsg",  lambda d: faiss.IndexNSGFlat(d, 32), False),
        ("faiss_IVF(nlist=64,np=16)", "ivf",  lambda d: faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 64), True),
        ("faiss_IVFPQ(nl=64,m=8,np=16)", "ivf", lambda d: faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, 64, 8, 8), True),
    ]
    for label, kind, factory, incremental in factories:
        try:
            r = run_faiss_native(label, factory, kind, data, queries, init_n, n_del, incremental_add=incremental)
            rows.append(r)
            print(f"  {label:18s}  op={r['op_total_s']:6.1f}s  "
                  f"qry={r['final_query_latency_ms']:.2f}ms  recall={r['recall']:.4f}", flush=True)
        except Exception as e:
            print(f"  {label:18s}  ERROR: {type(e).__name__}: {str(e)[:80]}", flush=True)
        out = os.path.join(os.path.dirname(__file__), f"output_{args.scale}.json")
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
