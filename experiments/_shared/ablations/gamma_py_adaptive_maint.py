"""GammaPyHybridAdaptiveMaint — gamma_v2 plus auto-trigger maintenance on
buffer fullness.

Inspired by the C++ scheduler that triggers maint based on observed
mutation/query rates. Here the rule is simpler: whenever the buffer is
fuller than `auto_maint_fill_threshold`, invoke maintain() at the end of
add(). The caller can still call maintain() explicitly (it's a no-op if
the buffer is empty).

Knob:
  auto_maint_fill_threshold : float in (0, 1]; fraction of buf_capacity
                                that triggers an auto-maint. Default 0.8.

Hypothesis: smaller threshold = more frequent flush = lower per-query
cost (smaller buffer scan) but higher maintenance overhead. Bigger
threshold = larger buffers, fewer flushes, more buffer-scan time per
query and higher migration burst.
"""
from __future__ import annotations
import numpy as np
from ..backends import GraphBackend


class GammaPyHybridAdaptiveMaint:
    def __init__(self, backend: GraphBackend, dim: int, buf_capacity: int,
                 *, auto_maint_fill_threshold: float = 0.8):
        if not (0.0 < auto_maint_fill_threshold <= 1.0):
            raise ValueError(f"auto_maint_fill_threshold {auto_maint_fill_threshold} out of (0, 1]")
        self.dim = dim
        self.graph = backend
        self.auto_threshold = auto_maint_fill_threshold
        # Buffer (mirrors v2)
        self.buf_capacity = buf_capacity
        self.buf_vecs = np.empty((buf_capacity, dim), dtype=np.float32)
        self.buf_ids = np.empty(buf_capacity, dtype=np.int64)
        self.buf_alive = np.zeros(buf_capacity, dtype=np.bool_)
        self.buf_size = 0
        self._id_to_pos: dict[int, int] = {}
        self._in_graph: set[int] = set()
        # Stats
        self.auto_maint_count = 0

    def initial_load(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        self.graph.add(np.ascontiguousarray(vectors), ids_arr)
        for i in ids_arr:
            self._in_graph.add(int(i))

    def add(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        n_new = len(ids_arr)
        end = self.buf_size + n_new
        if end > self.buf_capacity:
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
        # NEW: trigger auto-maint when buffer fills
        if self.buf_size >= self.auto_threshold * self.buf_capacity:
            self.maintain()
            self.auto_maint_count += 1

    def delete(self, ids):
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
        alive_mask = self.buf_alive[:self.buf_size]
        if alive_mask.any():
            ids = self.buf_ids[:self.buf_size][alive_mask]
            vecs = self.buf_vecs[:self.buf_size][alive_mask]
            self.graph.add(np.ascontiguousarray(vecs), ids)
            for i in ids:
                self._in_graph.add(int(i))
        self.buf_size = 0
        self.buf_alive[:] = False
        self._id_to_pos.clear()

    def search(self, queries, k):
        graph_labels, graph_dists = self.graph.search(queries, k)
        if self.buf_size == 0:
            return graph_labels, graph_dists
        alive_mask = self.buf_alive[:self.buf_size]
        n_alive = int(alive_mask.sum())
        if n_alive == 0:
            return graph_labels, graph_dists
        buf_vecs = self.buf_vecs[:self.buf_size][alive_mask]
        buf_ids = self.buf_ids[:self.buf_size][alive_mask]
        q_sq = (queries * queries).sum(axis=1, keepdims=True)
        v_sq = (buf_vecs * buf_vecs).sum(axis=1)
        cross = queries @ buf_vecs.T
        buf_d = q_sq + v_sq[None, :] - 2.0 * cross
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
