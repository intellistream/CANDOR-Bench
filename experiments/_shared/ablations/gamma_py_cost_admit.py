"""GammaPyHybridCostAdmit — gamma_v2 plus a learned admit decision.

Inspired by the C++ placement_controller's newcomer_epsilon_ms knob.
The C++ predicts a per-vector cost and routes to buffer or graph.
Our Python version uses a simpler proxy: rolling mean of observed
lifetimes of deleted vectors. If recent lifetimes are short, new
admits go to buffer (likely to be canceled); if long, new admits
skip the buffer and go straight to the graph (no migration cost
later).

State:
  insert_op_at[id] : the global op index when an id was inserted
  global_op       : monotonically incrementing op counter
  recent_lifetimes : ring buffer of last L observed lifetimes
  pred_lifetime    : current rolling-mean prediction (in operations)

Decision rule (per new vector, applied to the WHOLE batch):
  if pred_lifetime < admit_threshold_ops: admit to buffer
  else: skip buffer, add straight to graph

Knobs:
  admit_threshold_ops : integer. Compare against rolling-mean lifetime.
                         Set very large (1e9) → always buffer (= v2).
                         Set 0 → always direct (= v3 no_buffer_inserts).
  history_window      : how many recent lifetimes to average over.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from ..backends import GraphBackend


class GammaPyHybridCostAdmit:
    def __init__(self, backend: GraphBackend, dim: int, buf_capacity: int,
                 *, admit_threshold_ops: int = 10000,
                 history_window: int = 1000,
                 default_pred_lifetime: int = 1_000_000):
        self.dim = dim
        self.graph = backend
        self.admit_threshold = int(admit_threshold_ops)
        self.history_window = int(history_window)
        self.recent_lifetimes: deque[int] = deque(maxlen=self.history_window)
        # Default prediction (used until we observe any deletes)
        self.pred_lifetime = int(default_pred_lifetime)
        # Buffer (mirrors v2)
        self.buf_capacity = buf_capacity
        self.buf_vecs = np.empty((buf_capacity, dim), dtype=np.float32)
        self.buf_ids = np.empty(buf_capacity, dtype=np.int64)
        self.buf_alive = np.zeros(buf_capacity, dtype=np.bool_)
        self.buf_size = 0
        self._id_to_pos: dict[int, int] = {}
        self._in_graph: set[int] = set()
        # Insertion tracking
        self.global_op = 0
        self.insert_op_at: dict[int, int] = {}
        # Stats
        self.admit_to_buffer_count = 0
        self.admit_direct_count = 0

    def initial_load(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        self.graph.add(np.ascontiguousarray(vectors), ids_arr)
        for i in ids_arr:
            self._in_graph.add(int(i))
            self.insert_op_at[int(i)] = self.global_op
            self.global_op += 1

    def _decide_buffer(self) -> bool:
        return self.pred_lifetime < self.admit_threshold

    def add(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        vec_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        n_new = len(ids_arr)
        # Decision per batch (uniform — we treat this as the cost-model
        # output for the batch). The C++ predicts per-vector; here we
        # batch-predict for speed.
        if self._decide_buffer():
            # buffer admit
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
            # direct admit to graph
            self.graph.add(vec_arr, ids_arr)
            for i, id_int in enumerate(ids_arr):
                self._in_graph.add(int(id_int))
                self.insert_op_at[int(id_int)] = self.global_op + i
            self.admit_direct_count += n_new
        self.global_op += n_new

    def delete(self, ids):
        for i in ids:
            id_int = int(i)
            # Track lifetime
            t_in = self.insert_op_at.pop(id_int, None)
            if t_in is not None:
                lifetime = self.global_op - t_in
                self.recent_lifetimes.append(lifetime)
            pos = self._id_to_pos.get(id_int, -1)
            if pos >= 0 and self.buf_alive[pos]:
                self.buf_alive[pos] = False
                del self._id_to_pos[id_int]
            elif id_int in self._in_graph:
                self.graph.mark_deleted(id_int)
            self.global_op += 1
        # Update prediction
        if self.recent_lifetimes:
            self.pred_lifetime = int(sum(self.recent_lifetimes) / len(self.recent_lifetimes))

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
