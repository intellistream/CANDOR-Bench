"""GammaPyHybridSpatial — gamma_v2 extended with spatial multi-partition routing.

Mirrors the C++ design: K partitions, each with its own backend graph and
its own write buffer; new vectors routed to the nearest-centroid partition;
queries search the top-M nearest-centroid partitions and merge results.

Knobs:
  k_partitions       : number of partitions (1 = degenerate to gamma_v2)
  search_partitions  : how many nearest-centroid partitions to query per
                        request (1 ≤ M ≤ K). Smaller M = faster query but
                        worse cross-boundary recall.
  routing            : "centroid" (smart) or "round_robin" (dumb baseline,
                        mirrors the C++ GAMMA_DUMB_ROUTING ablation).
  centroid_update    : "on_maint" (recompute on every maintenance) or
                        "frozen" (keep initial centroids).

Initial centroids: mini-batch from the first-load, picked deterministically
by stride. Per-partition buffers are pre-allocated to buf_capacity_per_part
(default = buf_capacity / k_partitions, with 2× headroom).
"""
from __future__ import annotations
import numpy as np
from typing import Callable
from ..backends import GraphBackend


class GammaPyHybridSpatial:
    def __init__(self, backend_factory: Callable[[int], GraphBackend],
                 dim: int, buf_capacity: int,
                 *, k_partitions: int = 4,
                 search_partitions: int = 2,
                 routing: str = "centroid",
                 centroid_update: str = "on_maint",
                 max_elements_per_part: int | None = None):
        if routing not in ("centroid", "round_robin"):
            raise ValueError(f"unknown routing: {routing}")
        if centroid_update not in ("on_maint", "frozen"):
            raise ValueError(f"unknown centroid_update: {centroid_update}")
        if not (1 <= search_partitions <= k_partitions):
            raise ValueError(f"search_partitions {search_partitions} out of [1, {k_partitions}]")

        self.dim = dim
        self.k = k_partitions
        self.m_search = search_partitions
        self.routing = routing
        self.centroid_update = centroid_update
        self._round_robin_cursor = 0

        # One graph per partition.
        self.graphs: list[GraphBackend] = [backend_factory(p) for p in range(self.k)]

        # Pre-allocated per-partition buffer.
        per_part_cap = max(1, buf_capacity // self.k * 2)  # 2× headroom for skew
        self.buf_capacity_per_part = per_part_cap
        self.buf_vecs = [np.empty((per_part_cap, dim), dtype=np.float32) for _ in range(self.k)]
        self.buf_ids = [np.empty(per_part_cap, dtype=np.int64) for _ in range(self.k)]
        self.buf_alive = [np.zeros(per_part_cap, dtype=np.bool_) for _ in range(self.k)]
        self.buf_size = [0 for _ in range(self.k)]
        # Per-partition centroid (mean of vectors assigned).
        self.centroids = np.zeros((self.k, dim), dtype=np.float32)
        self._centroid_initialized = False
        # Per-partition vector counts (for centroid recomputation).
        self.part_vec_count = np.zeros(self.k, dtype=np.int64)
        self.part_vec_sum = np.zeros((self.k, dim), dtype=np.float64)

        # ID lookups: id -> (partition, "buffer"|"graph", buffer_pos_or_-1)
        self._id_to_loc: dict[int, tuple[int, str, int]] = {}

    # ------------------------------------------------------------------
    # Centroid + routing helpers
    # ------------------------------------------------------------------
    def _seed_centroids_from_data(self, data: np.ndarray):
        """Pick K seed centroids by uniform stride over data."""
        n = len(data)
        if n == 0:
            return
        if n < self.k:
            # Fewer points than partitions; just pad-repeat.
            self.centroids[:n] = data
            for j in range(n, self.k):
                self.centroids[j] = data[j % n]
        else:
            stride = n // self.k
            for j in range(self.k):
                self.centroids[j] = data[j * stride]
        self._centroid_initialized = True

    def _route(self, vectors: np.ndarray) -> np.ndarray:
        """Return partition assignments (length len(vectors))."""
        if self.routing == "round_robin":
            n = len(vectors)
            ids = (np.arange(n, dtype=np.int64) + self._round_robin_cursor) % self.k
            self._round_robin_cursor = (self._round_robin_cursor + n) % self.k
            return ids
        # centroid routing: find nearest centroid for each vector
        if not self._centroid_initialized:
            # Without centroids, fall back to round-robin
            return self._route_rr(len(vectors))
        # ||q - c||^2 = ||q||^2 + ||c||^2 - 2 q.c
        q_sq = (vectors * vectors).sum(axis=1, keepdims=True)
        c_sq = (self.centroids * self.centroids).sum(axis=1)[None, :]
        d2 = q_sq + c_sq - 2.0 * (vectors @ self.centroids.T)
        return d2.argmin(axis=1).astype(np.int64)

    def _route_rr(self, n: int) -> np.ndarray:
        ids = (np.arange(n, dtype=np.int64) + self._round_robin_cursor) % self.k
        self._round_robin_cursor = (self._round_robin_cursor + n) % self.k
        return ids

    def _update_partition_centroid_running(self, p: int, vectors: np.ndarray):
        """Online running-mean update of centroid p with `vectors` added."""
        if self.centroid_update != "on_maint":
            return
        n_new = len(vectors)
        if n_new == 0:
            return
        self.part_vec_sum[p] += vectors.sum(axis=0)
        self.part_vec_count[p] += n_new
        self.centroids[p] = (self.part_vec_sum[p] / max(1, self.part_vec_count[p])).astype(np.float32)
        self._centroid_initialized = True

    # ------------------------------------------------------------------
    # Public API: matches GammaRouter
    # ------------------------------------------------------------------
    def initial_load(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        vec_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        # Seed centroids before routing
        self._seed_centroids_from_data(vec_arr)
        assignments = self._route(vec_arr)
        for p in range(self.k):
            mask = assignments == p
            if not mask.any():
                continue
            sub_ids = ids_arr[mask]
            sub_vecs = np.ascontiguousarray(vec_arr[mask])
            self.graphs[p].add(sub_vecs, sub_ids)
            for i in sub_ids:
                self._id_to_loc[int(i)] = (p, "graph", -1)
            self._update_partition_centroid_running(p, sub_vecs)

    def add(self, ids, vectors):
        ids_arr = np.asarray(ids, dtype=np.int64)
        vec_arr = np.ascontiguousarray(vectors, dtype=np.float32)
        assignments = self._route(vec_arr)
        for p in range(self.k):
            mask = assignments == p
            if not mask.any():
                continue
            sub_ids = ids_arr[mask]
            sub_vecs = vec_arr[mask]
            n_new = len(sub_ids)
            cur = self.buf_size[p]
            end = cur + n_new
            if end > self.buf_capacity_per_part:
                # Grow this partition's buffer.
                new_cap = max(end, self.buf_capacity_per_part * 2)
                nb_vecs = np.empty((new_cap, self.dim), dtype=np.float32)
                nb_ids = np.empty(new_cap, dtype=np.int64)
                nb_alive = np.zeros(new_cap, dtype=np.bool_)
                nb_vecs[:cur] = self.buf_vecs[p][:cur]
                nb_ids[:cur] = self.buf_ids[p][:cur]
                nb_alive[:cur] = self.buf_alive[p][:cur]
                self.buf_vecs[p] = nb_vecs
                self.buf_ids[p] = nb_ids
                self.buf_alive[p] = nb_alive
                self.buf_capacity_per_part = max(self.buf_capacity_per_part, new_cap)
            self.buf_vecs[p][cur:end] = sub_vecs
            self.buf_ids[p][cur:end] = sub_ids
            self.buf_alive[p][cur:end] = True
            for offset, id_int in enumerate(sub_ids):
                self._id_to_loc[int(id_int)] = (p, "buffer", cur + offset)
            self.buf_size[p] = end

    def delete(self, ids):
        for i in ids:
            id_int = int(i)
            loc = self._id_to_loc.get(id_int)
            if loc is None:
                continue
            p, where, pos = loc
            if where == "buffer" and self.buf_alive[p][pos]:
                self.buf_alive[p][pos] = False
                del self._id_to_loc[id_int]
            elif where == "graph":
                self.graphs[p].mark_deleted(id_int)

    def maintain(self):
        for p in range(self.k):
            if self.buf_size[p] == 0:
                continue
            alive_mask = self.buf_alive[p][:self.buf_size[p]]
            if alive_mask.any():
                ids = self.buf_ids[p][:self.buf_size[p]][alive_mask]
                vecs = self.buf_vecs[p][:self.buf_size[p]][alive_mask]
                vecs_c = np.ascontiguousarray(vecs)
                self.graphs[p].add(vecs_c, ids)
                for i in ids:
                    self._id_to_loc[int(i)] = (p, "graph", -1)
                self._update_partition_centroid_running(p, vecs_c)
            self.buf_size[p] = 0
            self.buf_alive[p][:] = False

    def search(self, queries, k):
        Q = len(queries)
        # Decide which partitions to search per query
        if self.k == 1 or self.m_search >= self.k or not self._centroid_initialized:
            partitions_per_query = [list(range(self.k)) for _ in range(Q)]
        else:
            q_sq = (queries * queries).sum(axis=1, keepdims=True)
            c_sq = (self.centroids * self.centroids).sum(axis=1)[None, :]
            d2 = q_sq + c_sq - 2.0 * (queries @ self.centroids.T)
            # top-M nearest centroids per query
            top_idx = np.argpartition(d2, self.m_search, axis=1)[:, :self.m_search]
            partitions_per_query = top_idx.tolist()

        result_labels = np.full((Q, k), -1, dtype=np.int64)
        result_dists = np.full((Q, k), np.inf, dtype=np.float32)

        # Group queries by their selected-partition set so we can batch-search.
        # Simpler approach: for each query, search its partitions, merge.
        for q_i in range(Q):
            parts = partitions_per_query[q_i]
            cands_labels = []
            cands_dists = []
            q_single = queries[q_i:q_i + 1]
            for p in parts:
                if self.graphs[p] is None:
                    continue
                try:
                    lbls, dsts = self.graphs[p].search(q_single, k)
                except Exception:
                    continue
                cands_labels.append(lbls[0])
                cands_dists.append(dsts[0])
                # Buffer scan in this partition
                if self.buf_size[p] > 0:
                    alive_mask = self.buf_alive[p][:self.buf_size[p]]
                    n_alive = int(alive_mask.sum())
                    if n_alive > 0:
                        bv = self.buf_vecs[p][:self.buf_size[p]][alive_mask]
                        bi = self.buf_ids[p][:self.buf_size[p]][alive_mask]
                        v_sq = (bv * bv).sum(axis=1)
                        cross = q_single @ bv.T
                        bd = (q_single * q_single).sum(axis=1, keepdims=True) + v_sq[None, :] - 2.0 * cross
                        cands_labels.append(bi)
                        cands_dists.append(bd[0])
            if not cands_labels:
                continue
            all_lbls = np.concatenate(cands_labels)
            all_dsts = np.concatenate(cands_dists)
            valid = all_lbls >= 0
            if not valid.any():
                continue
            all_lbls = all_lbls[valid]
            all_dsts = all_dsts[valid]
            kk = min(k, len(all_lbls))
            order = np.argpartition(all_dsts, kk - 1)[:kk]
            order = order[np.argsort(all_dsts[order])]
            result_labels[q_i, :kk] = all_lbls[order]
            result_dists[q_i, :kk] = all_dsts[order]
        return result_labels, result_dists
