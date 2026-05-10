"""ClusterBuffer — IVF-style K-bucket buffer.

Each newly-inserted vector is assigned to its nearest centroid; each
bucket is itself a `FlatBuffer`. Search probes the top-`n_probe`
centroids and merges hits.

Centroid policy:
  - Lazily fitted on the first `add()` (or via `fit_centroids()` from
    init bulk-load).
  - Centroids are *fixed* once fitted unless `refit_centroids()` is
    called explicitly (e.g. by the router during a cluster-rebuild).
  - This is intentionally simple — split / merge / γ-aware repartition
    is the C++ design and was shown to under-perform a periodic full
    rebuild (see e21/e22). We keep the buffer dumb and let the router
    decide maintain strategy.

Per-cluster metadata maintained for `in_place_migrate` strategies:
  - `cluster_query_hits[k]` : number of search calls that probed cluster k
  - `cluster_inserts[k]`    : cumulative # of inserts to cluster k
  - `cluster_dead[k]`       : # of dead-but-still-occupying-slots in k
"""
from __future__ import annotations
import numpy as np
from typing import Iterable
import faiss

from .base import Buffer
from .flat import FlatBuffer


class ClusterBuffer(Buffer):
    def __init__(self, dim: int, n_clusters: int = 16,
                 init_capacity_per_cluster: int = 256,
                 n_probe: int = 4,
                 kmeans_seed: int = 1):
        self.dim = dim
        self.K = int(n_clusters)
        self.n_probe = max(1, min(int(n_probe), self.K))
        self._init_cap = int(init_capacity_per_cluster)
        self._kmeans_seed = int(kmeans_seed)

        self.centroids: np.ndarray | None = None  # (K, dim)
        self.clusters: list[FlatBuffer] = [
            FlatBuffer(dim, self._init_cap) for _ in range(self.K)
        ]
        self._id_to_cluster: dict[int, int] = {}

        # Telemetry for in_place_migrate / γ
        self.cluster_query_hits = np.zeros(self.K, dtype=np.int64)
        self.cluster_inserts = np.zeros(self.K, dtype=np.int64)

    # ----- centroid mgmt -----
    def fit_centroids(self, sample_vecs: np.ndarray) -> None:
        """Run faiss KMeans on `sample_vecs` to set centroids."""
        if len(sample_vecs) < self.K:
            # Fall back: pick first len() vectors as centroids, rest random
            cents = np.empty((self.K, self.dim), dtype=np.float32)
            cents[:len(sample_vecs)] = sample_vecs.astype(np.float32, copy=False)
            if len(sample_vecs) < self.K:
                rng = np.random.default_rng(self._kmeans_seed)
                cents[len(sample_vecs):] = rng.standard_normal(
                    (self.K - len(sample_vecs), self.dim)).astype(np.float32)
            self.centroids = cents
            return
        v = np.ascontiguousarray(sample_vecs, dtype=np.float32)
        km = faiss.Kmeans(self.dim, self.K, niter=20, verbose=False,
                          seed=self._kmeans_seed)
        km.train(v)
        self.centroids = km.centroids.astype(np.float32, copy=False)

    def refit_centroids(self) -> None:
        """Re-fit centroids from current alive vectors. Used by router during
        cluster-rebuild maintenance. After re-fit, all alive vectors are
        re-routed to their (possibly new) nearest centroid."""
        ids, vecs = self.drain_alive()
        if len(ids) == 0:
            return
        self.fit_centroids(vecs)
        # Reset per-cluster counters since cluster identity changed
        self.cluster_query_hits[:] = 0
        self.cluster_inserts[:] = 0
        self.add(ids, vecs)

    # ----- routing -----
    def _assign(self, vecs: np.ndarray) -> np.ndarray:
        """Return cluster_id for each row of vecs."""
        if self.centroids is None:
            self.fit_centroids(vecs[: max(self.K * 4, len(vecs))])
        # squared L2 to centroids
        c_sq = (self.centroids * self.centroids).sum(axis=1)         # (K,)
        v_sq = (vecs * vecs).sum(axis=1, keepdims=True)               # (N, 1)
        cross = vecs @ self.centroids.T                                # (N, K)
        d = v_sq + c_sq[None, :] - 2.0 * cross
        return np.argmin(d, axis=1)

    # ----- Buffer interface -----
    def add(self, ids, vecs):
        ids = np.asarray(ids, dtype=np.int64)
        vecs = np.asarray(vecs, dtype=np.float32)
        if len(ids) == 0:
            return
        cluster_ids = self._assign(vecs)
        for k in range(self.K):
            mask = cluster_ids == k
            if not mask.any():
                continue
            self.clusters[k].add(ids[mask], vecs[mask])
            kn = int(mask.sum())
            self.cluster_inserts[k] += kn
            for i in ids[mask]:
                self._id_to_cluster[int(i)] = k

    def delete(self, id_int: int) -> bool:
        k = self._id_to_cluster.get(int(id_int), -1)
        if k < 0:
            return False
        ok = self.clusters[k].delete(id_int)
        if ok:
            del self._id_to_cluster[int(id_int)]
        return ok

    def search(self, queries: np.ndarray, k: int):
        Q = len(queries)
        if self.centroids is None or self.n_alive == 0:
            ids = np.full((Q, k), -1, dtype=np.int64)
            dists = np.full((Q, k), np.inf, dtype=np.float32)
            return ids, dists

        # 1. find top-n_probe centroids per query
        q_sq = (queries * queries).sum(axis=1, keepdims=True)         # (Q, 1)
        c_sq = (self.centroids * self.centroids).sum(axis=1)          # (K,)
        cross = queries @ self.centroids.T                             # (Q, K)
        c_d = q_sq + c_sq[None, :] - 2.0 * cross                       # (Q, K)
        # top-n_probe nearest centroids (smallest distance)
        probe_idx = np.argpartition(c_d, self.n_probe - 1, axis=1)[:, :self.n_probe]  # (Q, P)

        out_ids = np.full((Q, k), -1, dtype=np.int64)
        out_d = np.full((Q, k), np.inf, dtype=np.float32)

        # Telemetry: count probe hits
        unique_probes, probe_counts = np.unique(probe_idx, return_counts=True)
        for c, cnt in zip(unique_probes, probe_counts):
            self.cluster_query_hits[c] += int(cnt)

        # 2. for each query, scan its probed clusters
        for q_i in range(Q):
            cands_ids: list[np.ndarray] = []
            cands_d: list[np.ndarray] = []
            for cid in probe_idx[q_i]:
                cb = self.clusters[int(cid)]
                if cb.size == 0 or not cb.alive[:cb.size].any():
                    continue
                mask = cb.alive[:cb.size]
                bvecs = cb.vecs[:cb.size][mask]
                bids = cb.ids[:cb.size][mask]
                v_sq = (bvecs * bvecs).sum(axis=1)
                d = (queries[q_i] * queries[q_i]).sum() + v_sq - 2.0 * (bvecs @ queries[q_i])
                cands_ids.append(bids)
                cands_d.append(d.astype(np.float32))
            if not cands_ids:
                continue
            all_ids = np.concatenate(cands_ids)
            all_d = np.concatenate(cands_d)
            n = len(all_ids)
            kk = min(k, n)
            order = np.argpartition(all_d, kk - 1)[:kk]
            order = order[np.argsort(all_d[order])]
            out_ids[q_i, :kk] = all_ids[order]
            out_d[q_i, :kk] = all_d[order]
        return out_ids, out_d

    def drain_alive(self):
        all_ids: list[np.ndarray] = []
        all_vecs: list[np.ndarray] = []
        for cb in self.clusters:
            ids, vecs = cb.drain_alive()
            if len(ids):
                all_ids.append(ids)
                all_vecs.append(vecs)
        self._id_to_cluster.clear()
        if not all_ids:
            return (np.empty(0, dtype=np.int64),
                    np.empty((0, self.dim), dtype=np.float32))
        return np.concatenate(all_ids), np.concatenate(all_vecs)

    @property
    def n_alive(self) -> int:
        return sum(c.n_alive for c in self.clusters)

    @property
    def n_dead(self) -> int:
        return sum(c.n_dead for c in self.clusters)

    # ----- per-cluster ops for in_place_migrate -----
    def per_cluster_alive(self) -> np.ndarray:
        return np.array([c.n_alive for c in self.clusters], dtype=np.int64)

    def per_cluster_dead(self) -> np.ndarray:
        return np.array([c.n_dead for c in self.clusters], dtype=np.int64)

    def drain_clusters(self, cluster_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        """Drain alive vectors from the given clusters. Used by the router
        when it decides to migrate selected clusters to the backend."""
        out_ids: list[np.ndarray] = []
        out_vecs: list[np.ndarray] = []
        for cid in cluster_ids:
            cb = self.clusters[int(cid)]
            ids, vecs = cb.drain_alive()
            for i in ids:
                self._id_to_cluster.pop(int(i), None)
            if len(ids):
                out_ids.append(ids)
                out_vecs.append(vecs)
            # Reset telemetry for the drained cluster
            self.cluster_query_hits[int(cid)] = 0
            self.cluster_inserts[int(cid)] = 0
        if not out_ids:
            return (np.empty(0, dtype=np.int64),
                    np.empty((0, self.dim), dtype=np.float32))
        return np.concatenate(out_ids), np.concatenate(out_vecs)
