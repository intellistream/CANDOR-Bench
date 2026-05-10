"""ModularRouter — pluggable buffer + pluggable maintenance strategy.

Single graph backend (the "long-term" structure) + any `Buffer`
(flat or cluster) for the streaming write path. The maintenance
strategy controls what `maintain()` does.

Strategies
----------
"lazy_flush"
    Flush all alive buffer vectors → backend.add. Mirrors
    GammaRouter (router.py).

"rebuild"
    Like lazy_flush, then check graph tombstone fraction; if above
    `rebuild_threshold`, drop the backend and rebuild from current
    alive set. Mirrors GammaRouterWithRebuild (router_with_rebuild.py).

"in_place_migrate"
    *Cluster-buffer only*. Per-cluster decision: a cluster whose
    "value" exceeds threshold migrates into the backend; clusters
    that are mostly dead get drained-and-dropped without going to the
    graph. Mimics the C++ design.

"cluster_rebuild"
    *Cluster-buffer only*. Like rebuild, but also re-fits the buffer
    centroids on the alive set. Tests "even the cluster identities
    are stale; periodic full rebuild handles it".

"lifetime_aware_migrate"
    *Cluster-buffer only*. Per-cluster decision based on observed
    cluster_avg_lifetime (EMA from absorbed deletes). Clusters whose
    expected lifetime exceeds `lifetime_horizon` migrate to graph
    (vectors will live long enough to amortize the migration cost);
    clusters with shorter expected lifetime are kept in buffer
    (vectors will likely be absorbed before next maintain). This is
    the C++ GammaFresh team's intended mechanism, finally given a
    workload that has lifetime structure (see e35 / workloads_lifetime.py).

"lifetime_aware_migrate_with_rebuild"
    *Cluster-buffer only*. lifetime_aware_migrate + tombstone-rebuild
    safety net. Combined variant — does the rebuild trigger still help
    when admission is already lifetime-aware?

"no_op"
    Do nothing. Used to confirm what happens if the buffer just keeps
    growing (worst-case sanity).
"""
from __future__ import annotations
import numpy as np
from typing import Callable, Optional

from .buffers import Buffer, ClusterBuffer
from .backends import GraphBackend


class ModularRouter:
    def __init__(self,
                 buffer: Buffer,
                 backend: GraphBackend,
                 dim: int,
                 *,
                 maintain_strategy: str = "lazy_flush",
                 rebuild_threshold: float = 0.5,
                 backend_factory: Optional[Callable[[], GraphBackend]] = None,
                 # in_place_migrate knobs
                 migrate_min_inserts: int = 256,
                 migrate_min_query_hits: int = 4,
                 migrate_dead_drop_fraction: float = 0.7,
                 # lifetime_aware_migrate knobs
                 lifetime_horizon: float = 5000.0,
                 lifetime_min_observations: int = 5,
                 lifetime_min_alive: int = 32,
                 lifetime_migrate_top_fraction: float = 0.5):
        self.dim = dim
        self.buffer = buffer
        self.backend = backend
        self.maintain_strategy = maintain_strategy
        self.rebuild_threshold = float(rebuild_threshold)
        self.backend_factory = backend_factory

        self.migrate_min_inserts = int(migrate_min_inserts)
        self.migrate_min_query_hits = int(migrate_min_query_hits)
        self.migrate_dead_drop_fraction = float(migrate_dead_drop_fraction)
        self.lifetime_horizon = float(lifetime_horizon)
        self.lifetime_min_observations = int(lifetime_min_observations)
        self.lifetime_min_alive = int(lifetime_min_alive)
        self.lifetime_migrate_top_fraction = float(lifetime_migrate_top_fraction)

        # State the router needs in order to rebuild
        self._graph_vecs: dict[int, np.ndarray] = {}      # id → vec (alive in graph)
        self._tombstones: int = 0                         # #marked-deleted in graph

        # Telemetry
        self.rebuild_count = 0
        self.cluster_migrate_count = 0
        self.cluster_drop_count = 0
        self.lifetime_kept_in_buffer_count = 0

        cluster_only_strategies = (
            "in_place_migrate",
            "cluster_rebuild",
            "lifetime_aware_migrate",
            "lifetime_aware_migrate_with_rebuild",
        )
        rebuild_strategies = (
            "rebuild",
            "cluster_rebuild",
            "lifetime_aware_migrate_with_rebuild",
        )
        if maintain_strategy in cluster_only_strategies:
            if not isinstance(buffer, ClusterBuffer):
                raise ValueError(
                    f"maintain_strategy='{maintain_strategy}' requires ClusterBuffer")
        if maintain_strategy in rebuild_strategies and backend_factory is None:
            raise ValueError(
                f"maintain_strategy='{maintain_strategy}' requires backend_factory")

    # ----- standard router API (matches router.GammaRouter) -----
    def initial_load(self, ids, vecs):
        """Bulk load directly into the backend (skips buffer)."""
        ids = np.asarray(ids, dtype=np.int64)
        vecs = np.asarray(vecs, dtype=np.float32)
        self.backend.add(np.ascontiguousarray(vecs), ids)
        for i, v in zip(ids, vecs):
            self._graph_vecs[int(i)] = v

    def add(self, ids, vecs):
        self.buffer.add(ids, vecs)

    def delete(self, ids):
        for i in ids:
            id_int = int(i)
            absorbed = self.buffer.delete(id_int)
            if not absorbed and id_int in self._graph_vecs:
                self.backend.mark_deleted(id_int)
                self._tombstones += 1
                del self._graph_vecs[id_int]

    def search(self, queries, k):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        # Backend search
        g_ids, g_d = self.backend.search(queries, k)
        # Buffer search
        if self.buffer.n_alive == 0:
            return g_ids, g_d
        b_ids, b_d = self.buffer.search(queries, k)
        # Merge per query (top-k by distance)
        Q = len(queries)
        out_ids = np.empty((Q, k), dtype=g_ids.dtype)
        out_d = np.empty((Q, k), dtype=g_d.dtype)
        for q_i in range(Q):
            cat_ids = np.concatenate([g_ids[q_i], b_ids[q_i]])
            cat_d = np.concatenate([g_d[q_i], b_d[q_i]])
            order = np.argpartition(cat_d, k - 1)[:k]
            order = order[np.argsort(cat_d[order])]
            out_ids[q_i] = cat_ids[order]
            out_d[q_i] = cat_d[order]
        return out_ids, out_d

    # ----- maintain dispatch -----
    def maintain(self):
        s = self.maintain_strategy
        if s == "no_op":
            return
        if s == "lazy_flush":
            self._flush_all_alive()
        elif s == "rebuild":
            self._flush_all_alive()
            self._maybe_rebuild_backend()
        elif s == "in_place_migrate":
            self._in_place_migrate()
        elif s == "cluster_rebuild":
            self._flush_all_alive()
            self._maybe_rebuild_backend(refit_buffer_centroids=True)
        elif s == "lifetime_aware_migrate":
            self._lifetime_aware_migrate()
        elif s == "lifetime_aware_migrate_with_rebuild":
            self._lifetime_aware_migrate()
            self._maybe_rebuild_backend()
        else:
            raise ValueError(f"unknown maintain_strategy: {s}")

    # ----- internals -----
    def _flush_all_alive(self):
        ids, vecs = self.buffer.drain_alive()
        if len(ids):
            self.backend.add(np.ascontiguousarray(vecs), ids)
            for i, v in zip(ids, vecs):
                self._graph_vecs[int(i)] = v

    def _tombstone_fraction(self) -> float:
        live = len(self._graph_vecs)
        total = live + self._tombstones
        return 0.0 if total == 0 else self._tombstones / total

    def _maybe_rebuild_backend(self, *, refit_buffer_centroids: bool = False):
        if self._tombstone_fraction() < self.rebuild_threshold:
            return
        # Drop backend, rebuild from alive set
        self.backend = self.backend_factory()
        if self._graph_vecs:
            ids = np.array(list(self._graph_vecs.keys()), dtype=np.int64)
            vecs = np.stack([self._graph_vecs[int(i)] for i in ids])
            self.backend.add(np.ascontiguousarray(vecs), ids)
        self._tombstones = 0
        self.rebuild_count += 1
        if refit_buffer_centroids and isinstance(self.buffer, ClusterBuffer):
            self.buffer.refit_centroids()

    def _in_place_migrate(self):
        """Per-cluster decide migrate-to-graph vs drop-without-graph.

        Heuristic (cost-min flavored, no fake zeros):
          - if cluster's dead/(alive+dead) ≥ drop_fraction → drop
            entirely (vectors that lived their whole life inside
            buffer; they were never queried much).
          - elif cluster's inserts ≥ migrate_min_inserts AND
                cluster_query_hits ≥ migrate_min_query_hits AND
                its alive count > 0 → migrate alive to graph
                (cluster has been heavily used).
          - else: leave it (low traffic, not worth doing anything).
        """
        assert isinstance(self.buffer, ClusterBuffer)
        cb = self.buffer
        K = cb.K
        per_alive = cb.per_cluster_alive()
        per_dead = cb.per_cluster_dead()
        per_total = per_alive + per_dead

        clusters_to_drop: list[int] = []
        clusters_to_migrate: list[int] = []

        for k in range(K):
            if per_total[k] == 0:
                continue
            dead_frac = per_dead[k] / per_total[k]
            if dead_frac >= self.migrate_dead_drop_fraction and per_alive[k] == 0:
                clusters_to_drop.append(k)
                continue
            inserts = int(cb.cluster_inserts[k])
            hits = int(cb.cluster_query_hits[k])
            if (inserts >= self.migrate_min_inserts
                    and hits >= self.migrate_min_query_hits
                    and per_alive[k] > 0):
                clusters_to_migrate.append(k)

        # Drop low-value clusters (no graph touch)
        if clusters_to_drop:
            cb.drain_clusters(clusters_to_drop)
            self.cluster_drop_count += len(clusters_to_drop)

        # Migrate high-value clusters into graph
        if clusters_to_migrate:
            ids, vecs = cb.drain_clusters(clusters_to_migrate)
            if len(ids):
                self.backend.add(np.ascontiguousarray(vecs), ids)
                for i, v in zip(ids, vecs):
                    self._graph_vecs[int(i)] = v
            self.cluster_migrate_count += len(clusters_to_migrate)

    def _lifetime_aware_migrate(self):
        """Per-cluster admission via *comparative* lifetime ranking.

        Of the eligible clusters (alive >= lifetime_min_alive), migrate
        the top `lifetime_migrate_top_fraction` ranked by observed
        avg_lifetime EMA. Clusters with insufficient observations get
        the horizon as a neutral default rank (so they're not
        permanently boxed out of migration before any signal arrives).

        Why comparative rather than absolute threshold? An absolute
        threshold can collapse to "migrate nothing" or "migrate
        everything" depending on workload mean, which collapses the
        experiment into "no admission" vs "always migrate" and tells
        us nothing about whether the lifetime *signal* is useful. The
        comparative rule isolates the question: given we're going to
        migrate ~half of the clusters per maintain, is the lifetime
        ranking the right one to pick by?
        """
        assert isinstance(self.buffer, ClusterBuffer)
        cb = self.buffer
        K = cb.K
        avg_life = cb.cluster_avg_lifetime           # (K,)
        n_obs = cb.cluster_n_observed                # (K,)
        per_alive = cb.per_cluster_alive()           # (K,)

        # For unobserved clusters: use horizon as neutral default
        eff_life = np.where(
            n_obs >= self.lifetime_min_observations,
            avg_life,
            self.lifetime_horizon,
        )

        eligible_mask = per_alive >= self.lifetime_min_alive
        if not eligible_mask.any():
            return
        eligible_idx = np.where(eligible_mask)[0]
        eligible_life = eff_life[eligible_mask]

        n_total = len(eligible_idx)
        n_top = max(1, int(np.ceil(self.lifetime_migrate_top_fraction * n_total)))
        # argpartition with -eligible_life picks the n_top largest values
        if n_top >= n_total:
            top_within = np.arange(n_total)
        else:
            top_within = np.argpartition(-eligible_life, n_top - 1)[:n_top]
        clusters_to_migrate = eligible_idx[top_within].tolist()

        self.lifetime_kept_in_buffer_count += (n_total - n_top)

        if clusters_to_migrate:
            ids, vecs = cb.drain_clusters(clusters_to_migrate)
            if len(ids):
                self.backend.add(np.ascontiguousarray(vecs), ids)
                for i, v in zip(ids, vecs):
                    self._graph_vecs[int(i)] = v
            self.cluster_migrate_count += len(clusters_to_migrate)
