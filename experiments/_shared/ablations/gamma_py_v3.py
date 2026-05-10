"""Component-toggleable variant of GammaRouter — used for ablations.

Each toggle disables one mechanism of the router:

  use_buffer_scan        : if False, search() does graph-only (no buffer scan).
                            Tests whether buffer scan masks recall under churn.
  buffer_inserts          : if False, every add() goes straight to the graph
                            (no buffer admit). Tests whether buffering inserts
                            actually pays off.
  absorb_buffer_deletes  : if False, deletes that hit a buffered id go to the
                            graph anyway (mark_deleted; full pollution).
                            Tests whether buffer-absorbed deletes are the win.
  eager_maint            : if True, maintain() is forced after every add().
                            Tests whether deferred (lazy) maint matters.

Default is all four enabled = identical to v2. Disabling any one isolates that
mechanism's contribution.
"""
from __future__ import annotations
import numpy as np
from ..backends import GraphBackend


class GammaPyHybridV3:
    def __init__(self, backend: GraphBackend, dim: int, buf_capacity: int,
                 *, use_buffer_scan: bool = True,
                 buffer_inserts: bool = True,
                 absorb_buffer_deletes: bool = True,
                 eager_maint: bool = False):
        self.dim = dim
        self.graph = backend
        self.buf_capacity = buf_capacity
        self.buf_vecs = np.empty((buf_capacity, dim), dtype=np.float32)
        self.buf_ids = np.empty(buf_capacity, dtype=np.int64)
        self.buf_alive = np.zeros(buf_capacity, dtype=np.bool_)
        self.buf_size = 0
        self._id_to_pos: dict[int, int] = {}
        self._in_graph: set[int] = set()

        self.use_buffer_scan = use_buffer_scan
        self.buffer_inserts = buffer_inserts
        self.absorb_buffer_deletes = absorb_buffer_deletes
        self.eager_maint = eager_maint

    def initial_load(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        self.graph.add(np.ascontiguousarray(vectors), ids_arr)
        for i in ids_arr:
            self._in_graph.add(int(i))

    def add(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        if not self.buffer_inserts:
            # No buffer admit — go straight to graph
            self.graph.add(np.ascontiguousarray(vectors), ids_arr)
            for i in ids_arr:
                self._in_graph.add(int(i))
            return
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
        if self.eager_maint:
            self.maintain()

    def delete(self, ids):
        for i in ids:
            id_int = int(i)
            pos = self._id_to_pos.get(id_int, -1)
            if pos >= 0 and self.buf_alive[pos]:
                if self.absorb_buffer_deletes:
                    # Absorb in buffer — never reaches graph
                    self.buf_alive[pos] = False
                    del self._id_to_pos[id_int]
                else:
                    # Force into graph (worst-case, baseline)
                    # First flush this pos to graph, then mark_deleted
                    self.graph.add(np.ascontiguousarray(self.buf_vecs[pos:pos + 1]),
                                    self.buf_ids[pos:pos + 1])
                    self._in_graph.add(id_int)
                    self.buf_alive[pos] = False
                    del self._id_to_pos[id_int]
                    self.graph.mark_deleted(id_int)
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
        if not self.use_buffer_scan or self.buf_size == 0:
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
