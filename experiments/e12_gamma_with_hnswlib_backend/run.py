"""e12 — POC: gamma architecture (buffer + maint-driven graph migration)
with hnswlib as backend instead of Faiss HNSW.

Tests whether gamma's structural design is independent of backend choice.
"""
import os, sys, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _shared
from _shared import build, load_dataset, compute_gt, recall_at_k, percentile_ms, maintain
import numpy as np
import hnswlib


class GammaPyHnswlib:
    """Python POC: buffer + hnswlib graph with explicit maint."""

    def __init__(self, dim: int, max_elements: int, M=32, ef_c=120, ef_s=80):
        self.dim = dim
        self.graph = hnswlib.Index(space='l2', dim=dim)
        self.graph.init_index(max_elements=max_elements, ef_construction=ef_c, M=M)
        self.graph.set_ef(ef_s)
        # Buffer: id -> vector, kept as a dict for O(1) lookup
        self.buffer_ids: list[int] = []
        self.buffer_vecs: list[np.ndarray] = []
        self.buffer_set: set[int] = set()  # for fast membership
        self.in_graph: set[int] = set()  # ids already migrated to graph
        self.deleted_in_graph: set[int] = set()  # tombstoned in graph

    def initial_load(self, ids, vectors):
        # init_load goes to graph directly (matches gamma's semantics)
        self.graph.add_items(vectors, ids)
        for i in ids:
            self.in_graph.add(int(i))

    def add(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self.buffer_ids.append(int(i))
            self.buffer_vecs.append(v)
            self.buffer_set.add(int(i))

    def delete(self, ids):
        ids = [int(i) for i in ids]
        for i in ids:
            if i in self.buffer_set:
                # Just mark for skip during query; we'll do a sweep later
                self.buffer_set.discard(i)
            elif i in self.in_graph:
                self.graph.mark_deleted(i)
                self.deleted_in_graph.add(i)

    def maintain(self):
        """Drain buffer (alive ids) to graph, clear buffer."""
        # Filter buffer to alive ids only
        alive_pairs = [(i, v) for i, v in zip(self.buffer_ids, self.buffer_vecs)
                       if i in self.buffer_set]
        if alive_pairs:
            ids = np.array([p[0] for p in alive_pairs], dtype=np.uint64)
            vecs = np.stack([p[1] for p in alive_pairs])
            self.graph.add_items(vecs, ids)
            for i in ids:
                self.in_graph.add(int(i))
        # Clear buffer
        self.buffer_ids = []
        self.buffer_vecs = []
        self.buffer_set = set()

    def search(self, queries, k):
        """Search graph + scan buffer; merge top-k by distance."""
        # Graph query
        graph_labels, graph_dists = self.graph.knn_query(queries, k=k)
        if not self.buffer_ids:
            return graph_labels, graph_dists
        # Buffer scan: compute distance to all buffer vectors
        buf_alive_idx = [j for j, i in enumerate(self.buffer_ids) if i in self.buffer_set]
        if not buf_alive_idx:
            return graph_labels, graph_dists
        buf_vecs = np.stack([self.buffer_vecs[j] for j in buf_alive_idx])
        buf_ids = np.array([self.buffer_ids[j] for j in buf_alive_idx])
        # L2 distances
        buf_d = np.linalg.norm(buf_vecs[None, :, :] - queries[:, None, :], axis=2) ** 2
        # Merge: for each query, take top-k from union of graph results and buffer
        result_labels = np.empty_like(graph_labels)
        result_dists = np.empty_like(graph_dists)
        for q_i in range(len(queries)):
            all_ids = np.concatenate([graph_labels[q_i], buf_ids])
            all_dists = np.concatenate([graph_dists[q_i], buf_d[q_i]])
            order = np.argpartition(all_dists, k)[:k]
            order = order[np.argsort(all_dists[order])]
            result_labels[q_i] = all_ids[order]
            result_dists[q_i] = all_dists[order]
        return result_labels, result_dists


def run_streaming_sliding(idx, name, data, queries, init_n, batch, qstride, lag,
                            is_py_gamma=False, is_hnswlib=False):
    n = data.shape[0]
    if is_py_gamma:
        idx.initial_load(np.arange(init_n, dtype=np.uint64),
                          np.ascontiguousarray(data[:init_n]))
    elif is_hnswlib:
        idx.add_items(data[:init_n], np.arange(init_n))
    else:
        idx.initial_load(np.arange(init_n, dtype=np.uint64),
                          np.ascontiguousarray(data[:init_n]))
    deleted, batch_idx = 0, 0
    query_lat = []
    t0 = time.perf_counter()
    for lo in range(init_n, n, batch):
        hi = min(lo + batch, n)
        if is_py_gamma:
            idx.add(np.arange(lo, hi, dtype=np.uint64),
                    np.ascontiguousarray(data[lo:hi]))
        elif is_hnswlib:
            idx.add_items(data[lo:hi], np.arange(lo, hi))
        else:
            idx.add(np.arange(lo, hi, dtype=np.uint64),
                    np.ascontiguousarray(data[lo:hi]))
        if batch_idx >= lag:
            n_del = hi - lo
            if deleted + n_del <= init_n + (lo - init_n):
                if is_py_gamma:
                    idx.delete(np.arange(deleted, deleted + n_del))
                elif is_hnswlib:
                    for did in range(deleted, deleted + n_del):
                        idx.mark_deleted(did)
                else:
                    idx.delete(np.arange(deleted, deleted + n_del, dtype=np.uint64))
                deleted += n_del
        if (lo - init_n) % qstride == 0:
            if is_py_gamma:
                idx.maintain()
            elif name == "gamma":
                maintain(idx, "gamma")
            tq = time.perf_counter()
            if is_py_gamma or is_hnswlib:
                idx.search(queries, 10) if is_py_gamma else idx.knn_query(queries, k=10)
            else:
                idx.search(np.ascontiguousarray(queries), 10)
            query_lat.append((time.perf_counter() - tq) / len(queries))
        batch_idx += 1
    if is_py_gamma:
        idx.maintain()
    elif name == "gamma":
        maintain(idx, "gamma")
    sgt = (compute_gt(data[deleted:], queries, 10) + deleted).astype(np.uint32)
    if is_py_gamma:
        nbrs, _ = idx.search(queries, 10)
    elif is_hnswlib:
        nbrs, _ = idx.knn_query(queries, k=10)
    else:
        nbrs, _ = idx.search(np.ascontiguousarray(queries), 10)
    return {
        "algo": name, "total_s": time.perf_counter() - t0,
        "n_delete": deleted,
        "query_latency_ms_avg": float(np.mean(query_lat)) * 1000 if query_lat else 0.0,
        "query_latency_ms_p95": percentile_ms(query_lat, 95),
        "recall": recall_at_k(nbrs.astype(np.uint32), sgt),
    }


def main():
    data, queries = load_dataset("sift", n_queries=500, slice_n=200_000)
    init_n, batch, qstride, lag = 20_000, 2_500, 10_000, 2

    print(f"\n========== streaming_sliding(lag={lag})/200K ==========", flush=True)
    rows = []

    print("[gamma_py_hnswlib (POC)]", flush=True)
    gpy = GammaPyHnswlib(dim=data.shape[1], max_elements=data.shape[0])
    r = run_streaming_sliding(gpy, "gamma_py_hnswlib", data, queries, init_n, batch, qstride, lag, is_py_gamma=True)
    rows.append(r); print(f"  {r['algo']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)

    print("[hnswlib (raw, with mark_deleted)]", flush=True)
    h = hnswlib.Index(space='l2', dim=data.shape[1])
    h.init_index(max_elements=data.shape[0], ef_construction=120, M=32)
    h.set_ef(80)
    r = run_streaming_sliding(h, "hnswlib_raw", data, queries, init_n, batch, qstride, lag, is_hnswlib=True)
    rows.append(r); print(f"  {r['algo']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)

    print("[gamma (FaissHNSW backend, for reference)]", flush=True)
    g = build("gamma", data.shape[1])
    r = run_streaming_sliding(g, "gamma", data, queries, init_n, batch, qstride, lag)
    rows.append(r); print(f"  {r['algo']:25s} total={r['total_s']:6.1f}s recall={r['recall']:.4f} qry_p95={r['query_latency_ms_p95']:.3f}ms", flush=True)

    out = os.path.join(os.path.dirname(__file__), "output_200K.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n✓ saved {out}")


if __name__ == "__main__":
    main()
