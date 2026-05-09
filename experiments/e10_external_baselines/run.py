"""e10 — Compare gamma against hnswlib / scann / annoy / faiss native indexes
on bulk_delete (gamma's strongest scenario)."""
import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import build, load_dataset, compute_gt, recall_at_k
import numpy as np


def run_gamma(data, queries, init_n, n_del):
    n = data.shape[0]
    idx = build("gamma", data.shape[1])
    idx.initial_load(np.arange(init_n, dtype=np.uint64),
                     np.ascontiguousarray(data[:init_n]))
    t0 = time.perf_counter()
    idx.add(np.arange(init_n, n, dtype=np.uint64),
            np.ascontiguousarray(data[init_n:n]))
    idx.delete(np.arange(0, n_del, dtype=np.uint64))
    idx.maintain(0, 0, False)
    op_total = time.perf_counter() - t0
    sgt = (compute_gt(data[n_del:], queries, 10) + n_del).astype(np.uint32)
    tq = time.perf_counter()
    nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    final_q_s = time.perf_counter() - tq
    return {"algo": "gamma", "op_total_s": op_total,
            "final_query_latency_ms": (final_q_s / len(queries)) * 1000,
            "recall": recall_at_k(nbrs.astype(np.uint32), sgt)}


def run_hnswlib(data, queries, n_del):
    import hnswlib
    n, d = data.shape
    surviving = data[n_del:n]
    t0 = time.perf_counter()
    idx = hnswlib.Index(space='l2', dim=d)
    idx.init_index(max_elements=n, ef_construction=120, M=32)
    idx.add_items(data, np.arange(n))
    # Simulate bulk_delete via rebuild from surviving
    idx2 = hnswlib.Index(space='l2', dim=d)
    idx2.init_index(max_elements=len(surviving), ef_construction=120, M=32)
    idx2.add_items(surviving, np.arange(len(surviving)))
    idx2.set_ef(80)
    op_total = time.perf_counter() - t0
    sgt_native = compute_gt(surviving, queries, 10)
    tq = time.perf_counter()
    labels, distances = idx2.knn_query(queries, k=10)
    final_q_s = time.perf_counter() - tq
    return {"algo": "hnswlib(M=32,ef=80)", "op_total_s": op_total,
            "final_query_latency_ms": (final_q_s / len(queries)) * 1000,
            "recall": recall_at_k(labels.astype(np.uint32), sgt_native.astype(np.uint32))}


def run_scann(data, queries, n_del):
    import scann
    n, d = data.shape
    surviving = data[n_del:n].astype(np.float32)
    t0 = time.perf_counter()
    # scann is build-only; build directly on surviving slice
    builder = scann.scann_ops_pybind.builder(surviving, num_neighbors=10, distance_measure="squared_l2")
    searcher = (builder
                .tree(num_leaves=int(np.sqrt(len(surviving))), num_leaves_to_search=16, training_sample_size=min(50000, len(surviving)))
                .score_ah(2, anisotropic_quantization_threshold=0.2)
                .reorder(100)
                .build())
    op_total = time.perf_counter() - t0
    sgt_native = compute_gt(surviving, queries, 10)
    tq = time.perf_counter()
    neighbors, distances = searcher.search_batched(queries.astype(np.float32))
    final_q_s = time.perf_counter() - tq
    return {"algo": "scann", "op_total_s": op_total,
            "final_query_latency_ms": (final_q_s / len(queries)) * 1000,
            "recall": recall_at_k(neighbors.astype(np.uint32), sgt_native.astype(np.uint32))}


def run_annoy(data, queries, n_del):
    from annoy import AnnoyIndex
    n, d = data.shape
    surviving = data[n_del:n]
    t0 = time.perf_counter()
    idx = AnnoyIndex(d, 'euclidean')
    for i, vec in enumerate(surviving):
        idx.add_item(i, vec.tolist())
    idx.build(50)  # 50 trees
    op_total = time.perf_counter() - t0
    sgt_native = compute_gt(surviving, queries, 10)
    tq = time.perf_counter()
    results = np.array([idx.get_nns_by_vector(q.tolist(), 10) for q in queries])
    final_q_s = time.perf_counter() - tq
    return {"algo": "annoy(50trees)", "op_total_s": op_total,
            "final_query_latency_ms": (final_q_s / len(queries)) * 1000,
            "recall": recall_at_k(results.astype(np.uint32), sgt_native.astype(np.uint32))}


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
    print("[gamma] running...", flush=True)
    r = run_gamma(data, queries, init_n, n_del)
    rows.append(r)
    print(f"  {r['algo']:25s}  op={r['op_total_s']:7.1f}s  qry={r['final_query_latency_ms']:.2f}ms  recall={r['recall']:.4f}", flush=True)

    for label, runner in [("hnswlib", run_hnswlib), ("scann", run_scann), ("annoy", run_annoy)]:
        print(f"[{label}] running...", flush=True)
        try:
            r = runner(data, queries, n_del)
            rows.append(r)
            print(f"  {r['algo']:25s}  op={r['op_total_s']:7.1f}s  qry={r['final_query_latency_ms']:.2f}ms  recall={r['recall']:.4f}", flush=True)
        except Exception as e:
            print(f"  {label:25s}  ERROR: {type(e).__name__}: {str(e)[:80]}", flush=True)
        out = os.path.join(os.path.dirname(__file__), f"output_{args.scale}.json")
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)

    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
