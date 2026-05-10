"""GammaRouterWithRebuild — gamma_v2 plus a tombstone-driven graph rebuild.

Inspired by:
  - C++ GammaFresh's full maintenance pass (which rebuilds segments when
    tombstone-density crosses a threshold)
  - hnswlib's `tombstone_rebuild_threshold` (which triggers a full graph
    rebuild after enough deletions)

Mechanism:
  After every maintain(), check deleted_total / inserted_total. If it
  exceeds `rebuild_threshold`, drop the current backend and rebuild it
  from the remaining alive ids (taken from caller-provided alive_set
  via maintain_alive_set()).

Use case hypothesis: helps when the graph has been polluted with
tombstones (e.g., gamma's 1M sequential loss) by amortizing the per-
delete cost over occasional bulk rebuilds.

Knobs:
  rebuild_threshold : float in (0, 1]; trigger when tombstone fraction
                       exceeds this. Default 0.5.
  rebuild_factory   : callable returning a fresh GraphBackend for the
                       rebuild. Required so we can re-create the graph.
  min_alive_for_rebuild : skip rebuild if alive set is below this size
                           (rebuild not worthwhile on tiny graphs).
"""
from __future__ import annotations
import numpy as np
from typing import Callable
from .backends import GraphBackend


class GammaRouterWithRebuild:
    def __init__(self, backend: GraphBackend, rebuild_factory: Callable[[], GraphBackend],
                 dim: int, buf_capacity: int,
                 *, rebuild_threshold: float = 0.5,
                 min_alive_for_rebuild: int = 1000):
        self.dim = dim
        self.graph = backend
        self.rebuild_factory = rebuild_factory
        self.rebuild_threshold = rebuild_threshold
        self.min_alive_for_rebuild = min_alive_for_rebuild
        # Buffer (mirror of v2)
        self.buf_capacity = buf_capacity
        self.buf_vecs = np.empty((buf_capacity, dim), dtype=np.float32)
        self.buf_ids = np.empty(buf_capacity, dtype=np.int64)
        self.buf_alive = np.zeros(buf_capacity, dtype=np.bool_)
        self.buf_size = 0
        self._id_to_pos: dict[int, int] = {}
        self._in_graph: set[int] = set()
        # Tombstone tracking
        self._n_inserted_total = 0  # vectors that ever entered the graph
        self._n_deleted_in_graph = 0  # mark_deleted calls on graph
        # Mirror data for rebuilds: id -> vector
        self._graph_vecs: dict[int, np.ndarray] = {}
        # Stats
        self.rebuild_count = 0

    def initial_load(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        vec_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        self.graph.add(vec_arr, ids_arr)
        for i, v in zip(ids_arr, vec_arr):
            self._in_graph.add(int(i))
            self._graph_vecs[int(i)] = v
        self._n_inserted_total += len(ids_arr)

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

    def delete(self, ids):
        for i in ids:
            id_int = int(i)
            pos = self._id_to_pos.get(id_int, -1)
            if pos >= 0 and self.buf_alive[pos]:
                self.buf_alive[pos] = False
                del self._id_to_pos[id_int]
            elif id_int in self._in_graph:
                self.graph.mark_deleted(id_int)
                self._n_deleted_in_graph += 1

    def _maybe_rebuild(self):
        """Drop the graph and rebuild from alive vectors if tombstone density too high."""
        n_alive = len(self._in_graph) - self._n_deleted_in_graph
        if n_alive < self.min_alive_for_rebuild:
            return
        denom = max(1, len(self._in_graph))
        tomb_frac = self._n_deleted_in_graph / denom
        if tomb_frac < self.rebuild_threshold:
            return
        # Rebuild: collect alive vectors, create fresh backend, re-add.
        # We need to know which ids are alive in the graph. Track via
        # _in_graph minus the (implicit) deleted set; but we only counted
        # _n_deleted_in_graph, not WHICH ones. Use a deleted set instead.
        # To keep this simple, we'll rebuild from _graph_vecs filtered by
        # NOT-being-mark_deleted. To know that, we maintain a side set.
        alive_ids = [i for i in self._in_graph if i not in self._deleted_in_graph_set]
        if len(alive_ids) < self.min_alive_for_rebuild:
            return
        alive_ids_arr = np.array(alive_ids, dtype=np.int64)
        alive_vecs = np.stack([self._graph_vecs[i] for i in alive_ids])
        # Drop the old backend and replace
        self.graph = self.rebuild_factory()
        self.graph.add(np.ascontiguousarray(alive_vecs), alive_ids_arr)
        # Reset state — only what's truly in the new graph
        self._in_graph = set(alive_ids)
        # Drop dead-vector mirror entries to reclaim memory
        self._graph_vecs = {i: self._graph_vecs[i] for i in alive_ids}
        self._n_deleted_in_graph = 0
        self._n_inserted_total = len(alive_ids)
        self._deleted_in_graph_set = set()
        self.rebuild_count += 1

    @property
    def _deleted_in_graph_set(self):
        # Lazy-init for first use
        if not hasattr(self, "_deleted_set"):
            self._deleted_set = set()
        return self._deleted_set

    @_deleted_in_graph_set.setter
    def _deleted_in_graph_set(self, value):
        self._deleted_set = value

    def maintain(self):
        # Flush alive buffer entries to graph (same as v2)
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
        # NEW: maybe trigger a tombstone rebuild
        self._maybe_rebuild()

    def delete(self, ids):  # noqa: F811 — override to track set
        for i in ids:
            id_int = int(i)
            pos = self._id_to_pos.get(id_int, -1)
            if pos >= 0 and self.buf_alive[pos]:
                self.buf_alive[pos] = False
                del self._id_to_pos[id_int]
            elif id_int in self._in_graph:
                self.graph.mark_deleted(id_int)
                self._n_deleted_in_graph += 1
                self._deleted_in_graph_set.add(id_int)

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
