"""Optimized Python POC of gamma's hybrid architecture.

v2 changes from v1:
- Buffer is pre-allocated numpy array (no Python list growth)
- Alive tracking via numpy boolean mask
- Buffer scan fully vectorized (single matmul per query batch)
- ID lookup via flat array (not Python dict)

Goal: minimize Python overhead so the architecture's TRUE cost vs
benefit is what we measure (not Python's slow loops).
"""
from __future__ import annotations
import numpy as np
from .backends import GraphBackend


class GammaRouter:
    def __init__(self, backend: GraphBackend, dim: int, buf_capacity: int):
        self.dim = dim
        self.graph = backend
        # Pre-allocate buffer arrays
        self.buf_capacity = buf_capacity
        self.buf_vecs = np.empty((buf_capacity, dim), dtype=np.float32)
        self.buf_ids = np.empty(buf_capacity, dtype=np.int64)
        self.buf_alive = np.zeros(buf_capacity, dtype=np.bool_)
        self.buf_size = 0  # number of slots used
        # ID -> position in buffer (for fast delete by id)
        self._id_to_pos: dict[int, int] = {}
        # IDs already in graph (so delete can route correctly)
        self._in_graph: set[int] = set()

    def initial_load(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        self.graph.add(np.ascontiguousarray(vectors), ids_arr)
        for i in ids_arr:
            self._in_graph.add(int(i))

    def add(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        n_new = len(ids_arr)
        # Append to buffer (vectorized)
        end = self.buf_size + n_new
        if end > self.buf_capacity:
            # Resize buffer
            new_cap = max(end, self.buf_capacity * 2)
            new_vecs = np.empty((new_cap, self.dim), dtype=np.float32)
            new_ids = np.empty(new_cap, dtype=np.int64)
            new_alive = np.zeros(new_cap, dtype=np.bool_)
            new_vecs[:self.buf_size] = self.buf_vecs[:self.buf_size]
            new_ids[:self.buf_size] = self.buf_ids[:self.buf_size]
            new_alive[:self.buf_size] = self.buf_alive[:self.buf_size]
            self.buf_vecs = new_vecs
            self.buf_ids = new_ids
            self.buf_alive = new_alive
            self.buf_capacity = new_cap
        self.buf_vecs[self.buf_size:end] = vectors
        self.buf_ids[self.buf_size:end] = ids_arr
        self.buf_alive[self.buf_size:end] = True
        for i, id_int in enumerate(ids_arr):
            self._id_to_pos[int(id_int)] = self.buf_size + i
        self.buf_size = end

    def delete(self, ids):
        # Vectorized: route ids by membership
        for i in ids:
            id_int = int(i)
            pos = self._id_to_pos.get(id_int, -1)
            if pos >= 0 and self.buf_alive[pos]:
                self.buf_alive[pos] = False
                del self._id_to_pos[id_int]
            elif id_int in self._in_graph:
                self.graph.mark_deleted(id_int)

    def maintain(self):
        if self.buf_size == 0:
            return
        # Collect alive vectors and push to graph in one batch
        alive_mask = self.buf_alive[:self.buf_size]
        if alive_mask.any():
            ids = self.buf_ids[:self.buf_size][alive_mask]
            vecs = self.buf_vecs[:self.buf_size][alive_mask]
            self.graph.add(np.ascontiguousarray(vecs), ids)
            for i in ids:
                self._in_graph.add(int(i))
        # Reset buffer
        self.buf_size = 0
        self.buf_alive[:] = False
        self._id_to_pos.clear()

    def search(self, queries, k):
        n_alive = int(self.buf_alive[:self.buf_size].sum())
        # Graph search
        graph_labels, graph_dists = self.graph.search(queries, k)
        if n_alive == 0:
            return graph_labels, graph_dists
        # Vectorized buffer scan
        alive_mask = self.buf_alive[:self.buf_size]
        buf_vecs = self.buf_vecs[:self.buf_size][alive_mask]
        buf_ids = self.buf_ids[:self.buf_size][alive_mask]
        # Squared L2 distance via matmul: ||q-v||^2 = ||q||^2 - 2 q.v + ||v||^2
        q_sq = (queries * queries).sum(axis=1, keepdims=True)        # (Q, 1)
        v_sq = (buf_vecs * buf_vecs).sum(axis=1)                       # (B,)
        cross = queries @ buf_vecs.T                                   # (Q, B)
        buf_d = q_sq + v_sq[None, :] - 2.0 * cross
        # Merge graph + buffer per query
        Q = len(queries)
        result_labels = np.empty((Q, k), dtype=graph_labels.dtype)
        result_dists = np.empty((Q, k), dtype=graph_dists.dtype)
        for q_i in range(Q):
            all_ids = np.concatenate([graph_labels[q_i], buf_ids])
            all_dists = np.concatenate([graph_dists[q_i], buf_d[q_i]])
            order = np.argpartition(all_dists, k)[:k]
            order = order[np.argsort(all_dists[order])]
            result_labels[q_i] = all_ids[order]
            result_dists[q_i] = all_dists[order]
        return result_labels, result_dists
