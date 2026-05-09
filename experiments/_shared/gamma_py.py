"""Generic Python POC of gamma's hybrid architecture, parameterized by graph backend.

Usage:
    g = GammaPyHybrid(backend=HnswlibBackend(...))
    g.initial_load(ids, vecs)
    g.add(ids, vecs)        # → buffer
    g.delete(ids)           # → buffer remove or graph mark_deleted
    g.maintain()            # → flush alive buffer to graph
    labels, dists = g.search(queries, k)

Lets us A/B "gamma+X" vs "X direct" for any backend X to isolate the
architectural value (vs the backend's own performance).
"""
from __future__ import annotations
import abc
import numpy as np


class GraphBackend(abc.ABC):
    @abc.abstractmethod
    def add(self, vectors: np.ndarray, ids: np.ndarray) -> None: ...
    @abc.abstractmethod
    def mark_deleted(self, vec_id: int) -> None: ...
    @abc.abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (labels, distances), shape (Q, k). distances are SQUARED L2."""


class HnswlibBackend(GraphBackend):
    """hnswlib (Qdrant/ChromaDB default backend)."""
    def __init__(self, dim, max_elements, M=32, ef_c=120, ef_s=80):
        import hnswlib
        self.idx = hnswlib.Index(space='l2', dim=dim)
        self.idx.init_index(max_elements=max_elements, ef_construction=ef_c, M=M)
        self.idx.set_ef(ef_s)
    def add(self, vectors, ids):
        self.idx.add_items(vectors, ids)
    def mark_deleted(self, vec_id):
        self.idx.mark_deleted(vec_id)
    def search(self, queries, k):
        return self.idx.knn_query(queries, k=k)


class FaissHnswBackend(GraphBackend):
    """Faiss HNSW (the original gamma backend)."""
    def __init__(self, dim, max_elements, M=32, ef_c=120, ef_s=80):
        import faiss
        self.idx = faiss.IndexHNSWFlat(dim, M)
        self.idx.hnsw.efConstruction = ef_c
        self.idx.hnsw.efSearch = ef_s
        # Faiss IndexHNSWFlat assigns sequential IDs; we need to track
        # external_id → internal_id ourselves
        self._ext2int: dict[int, int] = {}
        self._deleted: set[int] = set()
    def add(self, vectors, ids):
        start_internal = self.idx.ntotal
        self.idx.add(np.ascontiguousarray(vectors, dtype=np.float32))
        for offset, ext_id in enumerate(ids):
            self._ext2int[int(ext_id)] = start_internal + offset
    def mark_deleted(self, vec_id):
        self._deleted.add(int(vec_id))
    def search(self, queries, k):
        # Search with a larger k to compensate for filtered tombstones, then trim.
        over_k = k * 4 if self._deleted else k
        D, I = self.idx.search(np.ascontiguousarray(queries, dtype=np.float32), over_k)
        # Map internal back to external + filter deleted
        # Build inverse map (cached for performance, rebuild on add)
        if not hasattr(self, '_int2ext_cache') or self._int2ext_cache_size != len(self._ext2int):
            self._int2ext_cache = {v: k for k, v in self._ext2int.items()}
            self._int2ext_cache_size = len(self._ext2int)
        labels = np.full((len(queries), k), -1, dtype=np.int64)
        dists = np.full((len(queries), k), np.inf, dtype=np.float32)
        for q_i in range(len(queries)):
            kept = 0
            for j in range(over_k):
                internal = I[q_i, j]
                if internal < 0:
                    continue
                ext = self._int2ext_cache.get(internal, -1)
                if ext < 0 or ext in self._deleted:
                    continue
                labels[q_i, kept] = ext
                dists[q_i, kept] = D[q_i, j]
                kept += 1
                if kept >= k:
                    break
        return labels, dists


class GammaPyHybrid:
    """Python POC of gamma's hybrid architecture: buffer + (any graph backend)."""

    def __init__(self, backend: GraphBackend, dim: int):
        self.dim = dim
        self.graph: GraphBackend = backend
        self.buffer_ids: list[int] = []
        self.buffer_vecs: list[np.ndarray] = []
        self.buffer_set: set[int] = set()
        self.in_graph: set[int] = set()

    def initial_load(self, ids, vectors):
        # Goes directly to graph (matches gamma semantics)
        self.graph.add(vectors, ids)
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
                self.buffer_set.discard(i)  # alive sweep filters this out
            elif i in self.in_graph:
                self.graph.mark_deleted(i)

    def maintain(self):
        """Flush alive buffer → graph."""
        if not self.buffer_ids:
            return
        alive_idx = [j for j, i in enumerate(self.buffer_ids) if i in self.buffer_set]
        if alive_idx:
            ids = np.array([self.buffer_ids[j] for j in alive_idx], dtype=np.int64)
            vecs = np.stack([self.buffer_vecs[j] for j in alive_idx])
            self.graph.add(vecs, ids)
            for i in ids:
                self.in_graph.add(int(i))
        self.buffer_ids, self.buffer_vecs, self.buffer_set = [], [], set()

    def search(self, queries, k):
        """Hybrid: graph search + buffer scan, merge top-k."""
        graph_labels, graph_dists = self.graph.search(queries, k)
        if not self.buffer_set:
            return graph_labels, graph_dists
        # Buffer scan over alive only
        alive_idx = [j for j, i in enumerate(self.buffer_ids) if i in self.buffer_set]
        if not alive_idx:
            return graph_labels, graph_dists
        buf_vecs = np.stack([self.buffer_vecs[j] for j in alive_idx])
        buf_ids = np.array([self.buffer_ids[j] for j in alive_idx])
        diff = buf_vecs[None, :, :] - queries[:, None, :]
        buf_d = (diff * diff).sum(axis=2)  # squared L2
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
