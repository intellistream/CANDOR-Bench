"""GammaPyHybridCombined — gamma_v2 + tombstone-rebuild + cost-model admit.

Combines the two modules that empirically helped (e25 rebuild and e27
cost-admit). The hypothesis is that they're orthogonal:
- cost-admit picks the right insert path per batch (buffer for short-
  lived patterns, direct-graph for long-lived).
- rebuild prunes accumulated tombstones in the graph.

Knobs:
  rebuild_threshold      : float, same as gamma_py_rebuild
  rebuild_factory        : callable, same
  admit_threshold_ops    : int, same as gamma_py_cost_admit
  history_window         : int, same
"""
from __future__ import annotations
from collections import deque
from typing import Callable
import numpy as np
from .gamma_py import GraphBackend


class GammaPyHybridCombined:
    def __init__(self, backend: GraphBackend,
                 rebuild_factory: Callable[[], GraphBackend],
                 dim: int, buf_capacity: int,
                 *, rebuild_threshold: float = 0.5,
                 min_alive_for_rebuild: int = 1000,
                 admit_threshold_ops: int = 1_000_000_000,
                 history_window: int = 1000,
                 default_pred_lifetime: int = 1_000_000):
        self.dim = dim
        self.graph = backend
        self.rebuild_factory = rebuild_factory
        self.rebuild_threshold = rebuild_threshold
        self.min_alive_for_rebuild = min_alive_for_rebuild
        self.admit_threshold = int(admit_threshold_ops)
        self.history_window = int(history_window)
        # Buffer
        self.buf_capacity = buf_capacity
        self.buf_vecs = np.empty((buf_capacity, dim), dtype=np.float32)
        self.buf_ids = np.empty(buf_capacity, dtype=np.int64)
        self.buf_alive = np.zeros(buf_capacity, dtype=np.bool_)
        self.buf_size = 0
        self._id_to_pos: dict[int, int] = {}
        self._in_graph: set[int] = set()
        self._graph_vecs: dict[int, np.ndarray] = {}
        # Rebuild tracking
        self._n_inserted_total = 0
        self._n_deleted_in_graph = 0
        self._deleted_set: set[int] = set()
        # Cost-admit tracking
        self.recent_lifetimes: deque[int] = deque(maxlen=self.history_window)
        self.pred_lifetime = int(default_pred_lifetime)
        self.global_op = 0
        self.insert_op_at: dict[int, int] = {}
        # Stats
        self.rebuild_count = 0
        self.admit_to_buffer_count = 0
        self.admit_direct_count = 0

    def initial_load(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        vec_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        self.graph.add(vec_arr, ids_arr)
        for i, v in zip(ids_arr, vec_arr):
            self._in_graph.add(int(i))
            self._graph_vecs[int(i)] = v
            self.insert_op_at[int(i)] = self.global_op
            self.global_op += 1
        self._n_inserted_total += len(ids_arr)

    def _decide_buffer(self) -> bool:
        return self.pred_lifetime < self.admit_threshold

    def add(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        vec_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        n_new = len(ids_arr)
        if self._decide_buffer():
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
            self.buf_vecs[self.buf_size:end] = vec_arr
            self.buf_ids[self.buf_size:end] = ids_arr
            self.buf_alive[self.buf_size:end] = True
            for i, id_int in enumerate(ids_arr):
                self._id_to_pos[int(id_int)] = self.buf_size + i
                self.insert_op_at[int(id_int)] = self.global_op + i
            self.buf_size = end
            self.admit_to_buffer_count += n_new
        else:
            # direct admit
            self.graph.add(vec_arr, ids_arr)
            for i, v in zip(ids_arr, vec_arr):
                self._in_graph.add(int(i))
                self._graph_vecs[int(i)] = v
                self.insert_op_at[int(i)] = self.global_op
            self._n_inserted_total += n_new
            self.admit_direct_count += n_new
        self.global_op += n_new

    def delete(self, ids):
        for i in ids:
            id_int = int(i)
            t_in = self.insert_op_at.pop(id_int, None)
            if t_in is not None:
                self.recent_lifetimes.append(self.global_op - t_in)
            pos = self._id_to_pos.get(id_int, -1)
            if pos >= 0 and self.buf_alive[pos]:
                self.buf_alive[pos] = False
                del self._id_to_pos[id_int]
            elif id_int in self._in_graph:
                self.graph.mark_deleted(id_int)
                self._n_deleted_in_graph += 1
                self._deleted_set.add(id_int)
            self.global_op += 1
        if self.recent_lifetimes:
            self.pred_lifetime = int(sum(self.recent_lifetimes) / len(self.recent_lifetimes))

    def _maybe_rebuild(self):
        n_alive = len(self._in_graph) - self._n_deleted_in_graph
        if n_alive < self.min_alive_for_rebuild:
            return
        denom = max(1, len(self._in_graph))
        if self._n_deleted_in_graph / denom < self.rebuild_threshold:
            return
        alive_ids = [i for i in self._in_graph if i not in self._deleted_set]
        if len(alive_ids) < self.min_alive_for_rebuild:
            return
        alive_ids_arr = np.array(alive_ids, dtype=np.int64)
        alive_vecs = np.stack([self._graph_vecs[i] for i in alive_ids])
        self.graph = self.rebuild_factory()
        self.graph.add(np.ascontiguousarray(alive_vecs), alive_ids_arr)
        self._in_graph = set(alive_ids)
        self._graph_vecs = {i: self._graph_vecs[i] for i in alive_ids}
        self._n_deleted_in_graph = 0
        self._n_inserted_total = len(alive_ids)
        self._deleted_set = set()
        self.rebuild_count += 1

    def maintain(self):
        if self.buf_size > 0:
            alive_mask = self.buf_alive[:self.buf_size]
            if alive_mask.any():
                ids = self.buf_ids[:self.buf_size][alive_mask]
                vecs = self.buf_vecs[:self.buf_size][alive_mask]
                vecs_c = np.ascontiguousarray(vecs)
                self.graph.add(vecs_c, ids)
                for i, v in zip(ids, vecs_c):
                    self._in_graph.add(int(i))
                    self._graph_vecs[int(i)] = v
                self._n_inserted_total += len(ids)
            self.buf_size = 0
            self.buf_alive[:] = False
            self._id_to_pos.clear()
        self._maybe_rebuild()

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
