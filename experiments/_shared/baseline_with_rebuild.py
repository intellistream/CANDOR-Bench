"""HnswlibDirectWithRebuild — apples-to-apples baseline for the e25
tombstone-rebuild comparison.

Critique from peer review: e25 compares `gamma+rebuild` vs `hnswlib direct
without rebuild`, so the win could be entirely from the rebuild trigger
(which hnswlib doesn't ship with) rather than from the buffer/absorb router.

This wrapper takes any backend, exposes the same `add / mark_deleted /
search` interface, AND periodically drops the backend and rebuilds it
from alive ids. The caller is responsible for tracking which ids are
alive and feeding `(alive_ids, alive_vecs)` to the wrapper at rebuild
points.

Used in e29 — the missing baseline.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from .backends import GraphBackend


class HnswlibDirectWithRebuild(GraphBackend):
    """A direct backend that periodically rebuilds from alive vectors."""

    def __init__(self, backend_factory: Callable[[], GraphBackend],
                 rebuild_threshold: float = 0.5,
                 min_alive_for_rebuild: int = 1000):
        self.backend_factory = backend_factory
        self.idx = backend_factory()
        self.rebuild_threshold = rebuild_threshold
        self.min_alive_for_rebuild = min_alive_for_rebuild
        # Track the full population for rebuild
        self._all_vecs: dict[int, np.ndarray] = {}
        self._deleted: set[int] = set()
        self._n_inserted = 0
        # Stats
        self.rebuild_count = 0

    def add(self, vectors, ids):
        ids_arr = np.asarray(ids, dtype=np.int64)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self.idx.add(vecs, ids_arr)
        for vid, v in zip(ids_arr.tolist(), vecs):
            self._all_vecs[vid] = v
            self._n_inserted += 1

    def mark_deleted(self, vec_id):
        self.idx.mark_deleted(vec_id)
        self._deleted.add(int(vec_id))
        # Check rebuild trigger
        denom = max(1, self._n_inserted)
        if (len(self._deleted) / denom >= self.rebuild_threshold
                and self._n_inserted - len(self._deleted) >= self.min_alive_for_rebuild):
            self._rebuild()

    def _rebuild(self):
        alive_ids = [vid for vid in self._all_vecs if vid not in self._deleted]
        if len(alive_ids) < self.min_alive_for_rebuild:
            return
        alive_vecs = np.stack([self._all_vecs[vid] for vid in alive_ids])
        alive_arr = np.array(alive_ids, dtype=np.int64)
        self.idx = self.backend_factory()
        self.idx.add(np.ascontiguousarray(alive_vecs), alive_arr)
        # Reset state
        self._all_vecs = {vid: self._all_vecs[vid] for vid in alive_ids}
        self._deleted = set()
        self._n_inserted = len(alive_ids)
        self.rebuild_count += 1

    def search(self, queries, k):
        return self.idx.search(queries, k)
