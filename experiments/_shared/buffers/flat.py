"""FlatBuffer — pre-allocated numpy arrays with alive bitmask.

Algorithmically equivalent to the buffer inside `router.GammaRouter` —
extracted here so the same code path can be reused inside ClusterBuffer
(each cluster bucket is a small FlatBuffer)."""
from __future__ import annotations
import numpy as np
from .base import Buffer


class FlatBuffer(Buffer):
    def __init__(self, dim: int, init_capacity: int = 1024):
        self.dim = dim
        self.capacity = max(init_capacity, 16)
        self.vecs = np.empty((self.capacity, dim), dtype=np.float32)
        self.ids = np.empty(self.capacity, dtype=np.int64)
        self.alive = np.zeros(self.capacity, dtype=np.bool_)
        self.size = 0  # slots used (alive + dead)
        self._id_to_pos: dict[int, int] = {}

    def _grow(self, need: int) -> None:
        new_cap = max(need, self.capacity * 2)
        new_vecs = np.empty((new_cap, self.dim), dtype=np.float32)
        new_ids = np.empty(new_cap, dtype=np.int64)
        new_alive = np.zeros(new_cap, dtype=np.bool_)
        new_vecs[:self.size] = self.vecs[:self.size]
        new_ids[:self.size] = self.ids[:self.size]
        new_alive[:self.size] = self.alive[:self.size]
        self.vecs, self.ids, self.alive = new_vecs, new_ids, new_alive
        self.capacity = new_cap

    def add(self, ids, vecs):
        ids = np.asarray(ids, dtype=np.int64)
        vecs = np.asarray(vecs, dtype=np.float32)
        n = len(ids)
        end = self.size + n
        if end > self.capacity:
            self._grow(end)
        self.vecs[self.size:end] = vecs
        self.ids[self.size:end] = ids
        self.alive[self.size:end] = True
        for i, id_int in enumerate(ids):
            self._id_to_pos[int(id_int)] = self.size + i
        self.size = end

    def delete(self, id_int: int) -> bool:
        pos = self._id_to_pos.get(int(id_int), -1)
        if pos >= 0 and self.alive[pos]:
            self.alive[pos] = False
            del self._id_to_pos[int(id_int)]
            return True
        return False

    def search(self, queries: np.ndarray, k: int):
        Q = len(queries)
        if self.size == 0 or not self.alive[:self.size].any():
            ids = np.full((Q, k), -1, dtype=np.int64)
            dists = np.full((Q, k), np.inf, dtype=np.float32)
            return ids, dists
        mask = self.alive[:self.size]
        bvecs = self.vecs[:self.size][mask]
        bids = self.ids[:self.size][mask]
        # squared L2: ||q||^2 + ||v||^2 - 2 q.v
        q_sq = (queries * queries).sum(axis=1, keepdims=True)        # (Q, 1)
        v_sq = (bvecs * bvecs).sum(axis=1)                            # (B,)
        cross = queries @ bvecs.T                                     # (Q, B)
        d = q_sq + v_sq[None, :] - 2.0 * cross                        # (Q, B)
        n_alive = len(bids)
        kk = min(k, n_alive)
        out_ids = np.full((Q, k), -1, dtype=np.int64)
        out_d = np.full((Q, k), np.inf, dtype=np.float32)
        for q_i in range(Q):
            order = np.argpartition(d[q_i], kk - 1)[:kk]
            order = order[np.argsort(d[q_i][order])]
            out_ids[q_i, :kk] = bids[order]
            out_d[q_i, :kk] = d[q_i][order]
        return out_ids, out_d

    def drain_alive(self):
        if self.size == 0:
            empty_ids = np.empty(0, dtype=np.int64)
            empty_vecs = np.empty((0, self.dim), dtype=np.float32)
            self._reset()
            return empty_ids, empty_vecs
        mask = self.alive[:self.size]
        ids = self.ids[:self.size][mask].copy()
        vecs = self.vecs[:self.size][mask].copy()
        self._reset()
        return ids, vecs

    def _reset(self):
        self.size = 0
        self.alive[:] = False
        self._id_to_pos.clear()

    @property
    def n_alive(self) -> int:
        return int(self.alive[:self.size].sum())

    @property
    def n_dead(self) -> int:
        return self.size - self.n_alive
